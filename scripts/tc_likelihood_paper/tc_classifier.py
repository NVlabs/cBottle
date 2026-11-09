# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TC classifier probability in the Miami HPX3 pixel, added as ``tc_probability``
(``samples``) or ``era5_tc_probability`` (``era5``) to each
``RUN_DIR/likelihoods_rank*.csv``. ``--maps-dir`` also writes the HPX3 maps."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from earth2grid import healpix
from importance_sampling import contiguous_runs, rank_csvs
from torch.utils.data import DataLoader

import cbottle.inference
from cbottle.datasets.dataset_3d import get_dataset
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter

SIGMA = 0.002
LON, LAT = -80.0, 25.0
UNIX_TO_1900_S = 2208988800
NEST, RING = healpix.PixelOrder.NEST, healpix.PixelOrder.RING
PAD_XY = healpix.HEALPIX_PAD_XY


def miami_pixels(level: int, order) -> np.ndarray:
    lon, lat = (torch.tensor([v], dtype=torch.float64) for v in (LON, LAT))
    parent = int(healpix.Grid(3, pixel_order=NEST).ang2pix(lon, lat))
    n = 4 ** (level - 3)
    index = healpix.Grid(level, pixel_order=order).reorder(
        NEST, torch.arange(12 * 4**level, dtype=torch.float64)
    )
    return index.long().numpy()[parent * n : (parent + 1) * n]


PIXEL = miami_pixels(3, PAD_XY)[0]


@torch.no_grad()
def classify(model, x, condition, day_of_year, second_of_day) -> torch.Tensor:
    device = next(model.separate_classifier.parameters()).device
    logits = model.separate_classifier(
        x.to(device),
        torch.full((len(x), 1), SIGMA, device=device),
        condition=condition.to(device),
        day_of_year=day_of_year.to(device),
        second_of_day=second_of_day.to(device),
    ).logits
    return torch.sigmoid(logits)[:, 0, 0].float().cpu()


def add_column(csv: str, name: str, values: dict) -> None:
    df = pd.read_csv(csv)
    df[name] = pd.to_datetime(df["timestamp"], format="mixed").map(values)
    df.to_csv(csv, index=False)


def maps_writer(maps_dir: str, rank: int, source: str) -> NetCDFWriter:
    path = os.path.join(maps_dir, f"{rank}.nc")
    if os.path.exists(path):
        os.remove(path)
    attrs = {
        "description": "TC classifier probability",
        "sigma": SIGMA,
        "source": source,
    }
    config = NetCDFConfig(hpx_level=3, attrs=attrs)
    return NetCDFWriter(maps_dir, config, channels=["tc_probability"], rank=rank)


def write_maps(writer: NetCDFWriter, probs: torch.Tensor, times) -> None:
    seconds = pd.DatetimeIndex(times).asi8 // 10**9 + UNIX_TO_1900_S
    ring = healpix.reorder(probs, PAD_XY, RING).unsqueeze(1)
    writer.write_batch({"tc_probability": ring}, torch.as_tensor(seconds))


def classify_samples(model, run_dir: str, maps_dir, batch_size: int) -> None:
    info = model.batch_info
    center = torch.tensor(info.center, dtype=torch.float32).view(-1, 1)
    scales = torch.tensor(info.scales, dtype=torch.float32).view(-1, 1)
    channels = [str(c) for c in info.channels]
    amip = get_dataset(
        dataset="amip", rank=0, world_size=1, infinite=False, shuffle=False
    )
    for rank, csv in rank_csvs(run_dir).items():
        writer = maps_writer(maps_dir, rank, run_dir) if maps_dir else None
        at_pixel = {}
        with xr.open_dataset(os.path.join(run_dir, f"{rank}.nc")) as ds:
            times = pd.DatetimeIndex(ds["time"].values)
            amip.set_times(times)
            cond = next(iter(DataLoader(amip, batch_size=len(times))))
            for start in range(0, len(times), batch_size):
                sl = slice(start, start + batch_size)
                fields = np.stack([ds[c].isel(time=sl).values for c in channels], 1)
                x = (torch.from_numpy(fields).float() - center) / scales
                probs = classify(
                    model,
                    healpix.reorder(x, RING, PAD_XY).unsqueeze(2),
                    cond["condition"][sl],
                    cond["day_of_year"][sl],
                    cond["second_of_day"][sl],
                )
                at_pixel.update(zip(times[sl], probs[:, PIXEL].tolist()))
                if writer:
                    write_maps(writer, probs, times[sl])
        add_column(csv, "tc_probability", at_pixel)


def classify_era5(model, run_dirs: list[str], maps_dir, batch_size: int) -> None:
    csvs = [csv for d in run_dirs for csv in rank_csvs(d).values()]
    stamps = pd.concat(pd.read_csv(c, usecols=["timestamp"]) for c in csvs)
    ds = get_dataset(
        dataset="era5", rank=0, world_size=1, infinite=False, shuffle=False
    )
    writer = maps_writer(maps_dir, 0, "era5") if maps_dir else None
    at_pixel = {}
    for run in contiguous_runs(pd.to_datetime(stamps["timestamp"], format="mixed")):
        ds.set_times(run)
        for batch in DataLoader(ds, batch_size=batch_size):
            seconds = batch["timestamp"].double().round().long().numpy()
            times = pd.to_datetime(seconds, unit="s")
            probs = classify(
                model,
                torch.nan_to_num(batch["target"]),
                torch.nan_to_num(batch["condition"]),
                batch["day_of_year"],
                batch["second_of_day"],
            )
            at_pixel.update(zip(times, probs[:, PIXEL].tolist()))
            if writer:
                write_maps(writer, probs, times)
    for csv in csvs:
        add_column(csv, "era5_tc_probability", at_pixel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["samples", "era5"])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--maps-dir")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    model = cbottle.inference.load("cbottle-3d-moe-tc", root=args.checkpoint)
    if args.source == "era5":
        classify_era5(model, args.run_dirs, args.maps_dir, args.batch_size)
    else:
        for run_dir in args.run_dirs:
            classify_samples(model, run_dir, args.maps_dir, args.batch_size)


if __name__ == "__main__":
    main()
