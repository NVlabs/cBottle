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
"""Max 10 m wind and min MSLP in the Miami HPX3 pixel: ``physical_metrics_{run}.csv``
per run and ``era5_hpx3_metrics.csv`` at the timestamps of the first run."""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
from importance_sampling import contiguous_runs, load_likelihoods, rank_csvs
from tc_classifier import PAD_XY, RING, miami_pixels
from torch.utils.data import DataLoader

from cbottle.datasets.dataset_3d import get_dataset

CHANNELS = ["uas", "vas", "pres_msl"]


def metrics(times, u, v, p) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": times,
            "ws_hpx3_max": np.hypot(u, v).max(axis=1),
            "mslp_hpx3_min": p.min(axis=1),
        }
    )


def sample_metrics(run_dir: str) -> pd.DataFrame:
    child = miami_pixels(6, RING)
    frames = []
    for rank in rank_csvs(run_dir):
        with xr.open_dataset(os.path.join(run_dir, f"{rank}.nc")) as ds:
            fields = [ds[c].values[:, child] for c in CHANNELS]
            frames.append(metrics(ds["time"].values, *fields))
    lik = load_likelihoods(run_dir)[
        ["timestamp", "tc_probability", "era5_tc_probability"]
    ]
    return lik.merge(pd.concat(frames), on="timestamp", how="left")


def era5_metrics(times, batch_size: int = 16) -> pd.DataFrame:
    ds = get_dataset(
        dataset="era5", rank=0, world_size=1, infinite=False, shuffle=False
    )
    idx = [[str(c) for c in ds.batch_info.channels].index(c) for c in CHANNELS]
    center = np.asarray(ds.batch_info.center)[idx, None]
    scale = np.asarray(ds.batch_info.scales)[idx, None]
    child = miami_pixels(6, PAD_XY)
    frames = []
    for run in contiguous_runs(times):
        ds.set_times(run)
        for batch in DataLoader(ds, batch_size=batch_size):
            x = batch["target"][:, idx, 0].numpy()[..., child] * scale + center
            seconds = batch["timestamp"].double().round().long().numpy()
            frames.append(metrics(pd.to_datetime(seconds, unit="s"), *x.swapaxes(0, 1)))
    return pd.concat(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        sample_metrics(run_dir).to_csv(
            f"{args.output_dir}/physical_metrics_{name}.csv", index=False
        )
    era5_metrics(load_likelihoods(args.run_dirs[0])["timestamp"]).to_csv(
        f"{args.output_dir}/era5_hpx3_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
