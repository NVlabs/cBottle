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
import os

import cbottle.config.environment as config
import earth2grid
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from cbottle import visualizations
from cbottle.datasets import samplers
from cbottle.datasets.dataset_2d import HealpixDatasetV5, NetCDFWrapperV1
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from earth2grid import healpix
from cbottle.inference import (
    SuperResolutionModel,
    DistilledSuperResolutionModel,
    Coords,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse

import numpy as np
import tqdm
import xarray as xr


def diagnostics(pred, lr, target):
    titles = ["input", "prediction", "target"]
    for var in pred.keys():
        plt.figure(figsize=(50, 25))
        vmin = torch.min(pred[var][0, 0])
        vmax = torch.max(pred[var][0, 0])
        for idx, data, title in zip(
            np.arange(1, 4), [lr[var][0, 0], pred[var][0, 0], target[var][0, 0]], titles
        ):
            visualizations.visualize(
                data,
                pos=(1, 3, idx),
                title=title,
                nlat=1024,
                nlon=2048,
                vmin=vmin,
                vmax=vmax,
            )
        plt.tight_layout()
        plt.savefig(f"output_{var}")


def inference(arg_list=None, customized_dataset=None):
    parser = argparse.ArgumentParser(description="Distributed Deep Learning Task")
    parser.add_argument("state_path", type=str, help="Path to the model state file")
    parser.add_argument("output_path", type=str, help="Path to the output directory")
    parser.add_argument(
        "--input-path", type=str, default="", help="Path to the input data"
    )
    parser.add_argument("--plot-sample", action="store_true", help="Plot samples")
    parser.add_argument(
        "--min-samples", type=int, default=1, help="Number of samples to inference"
    )
    parser.add_argument(
        "--level", type=int, default=10, help="HPX level for high res input"
    )
    parser.add_argument(
        "--level-lr", type=int, default=6, help="HPX level for low res input"
    )
    parser.add_argument(
        "--patch-size", type=int, default=128, help="Patch size for multidiffusion"
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=32,
        help="Overlapping pixel number between patches",
    )
    parser.add_argument(
        "--num-steps", type=int, default=18, help="Sampler iteration number"
    )
    parser.add_argument("--sigma-max", type=int, default=800, help="Noise sigma max")
    parser.add_argument(
        "--save-data", action="store_true", help="Save target data without inference"
    )
    parser.add_argument(
        "--distill-inference",
        action="store_true",
        help="Inference with distilled model",
    )
    parser.add_argument(
        "--super-resolution-box",
        type=int,
        nargs=4,
        default=None,
        metavar=("lon_west", "lon_east", "lat_south", "lat_north"),
        help="Bounding box (lon_west lon_east lat_south lat_north) where super-resolution will be applied. "
        "Regions outside the box remain coarse.",
    )
    args = parser.parse_args(arg_list)
    input_path = args.input_path
    state_path = args.state_path
    output_path = args.output_path
    plot_sample = args.plot_sample
    distill_inference = args.distill_inference
    hpx_level = args.level
    min_samples = args.min_samples
    box = tuple(args.super_resolution_box) if args.super_resolution_box else None

    LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    WORLD_RANK = int(os.getenv("RANK", 0))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12345")

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=WORLD_SIZE, rank=WORLD_RANK
    )
    torch.cuda.set_device(LOCAL_RANK)

    if torch.cuda.is_available():
        if LOCAL_RANK is not None:
            device = torch.device(f"cuda:{LOCAL_RANK}")
        else:
            device = torch.device("cuda")
    if customized_dataset is not None:
        test_dataset = customized_dataset(
            split="test",
        )
        tasks = None
    elif input_path:
        ds = xr.open_dataset(input_path)
        test_dataset = NetCDFWrapperV1(
            ds, hpx_level=hpx_level, normalize=False, healpixpad_order=False
        )
        tasks = np.r_[WORLD_RANK : len(test_dataset) : WORLD_SIZE]
    else:
        test_dataset = HealpixDatasetV5(
            path=config.RAW_DATA_URL,
            land_path=config.LAND_DATA_URL_10,
            normalize=False,
            train=False,
            yield_index=True,
            healpixpad_order=False,
        )
        sampler = samplers.subsample(test_dataset, min_samples=min_samples)
        tasks = samplers.distributed_split(sampler)

    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, sampler=tasks
    )

    # Initialize netCDF writer
    nc_config = NetCDFConfig(
        hpx_level=hpx_level,
        time_units=test_dataset.time_units,
        calendar=test_dataset.calendar,
        attrs={},
    )
    writer = NetCDFWriter(
        output_path, nc_config, test_dataset.batch_info.channels, rank=WORLD_RANK
    )

    if distill_inference:
        model = DistilledSuperResolutionModel.from_pretrained(
            state_path,
            hpx_lr_level=args.level_lr,
            patch_size=args.patch_size,
            overlap_size=args.overlap_size,
            num_steps=args.num_steps,
            sigma_max=args.sigma_max,
            device=device,
        )
    else:
        model = SuperResolutionModel.from_pretrained(
            state_path,
            hpx_lr_level=args.level_lr,
            patch_size=args.patch_size,
            overlap_size=args.overlap_size,
            num_steps=args.num_steps,
            sigma_max=args.sigma_max,
            device=device,
        )

    high_res_level = model.high_res_grid.level
    low_res_level = model.low_res_grid.level

    coords = Coords(test_dataset.batch_info, model.low_res_grid)

    for batch in tqdm.tqdm(loader, disable=WORLD_RANK != 0):
        target = batch["target"]
        target = target[0, :, 0]
        with torch.no_grad():
            # coarsen target if input is hpx64 icon
            if not input_path:
                lr = target
                for _ in range(high_res_level - low_res_level):
                    npix = lr.size(-1)
                    shape = lr.shape[:-1]
                    lr = lr.view(shape + (npix // 4, 4)).mean(-1)
                inp = lr.cuda()[None, :, None]
            # load condition if input path is given
            else:
                inp = batch["condition"]
                inp = inp
                inp = inp.cuda(non_blocking=True)

        if args.save_data:
            pred = target[None, :, None]
        else:
            pred, _ = model(inp, coords=coords, extents=box)

        target = target[None, :, None]
        lr = lr[None, :, None]

        def prepare(x):
            ring_order = healpix.reorder(
                x,
                earth2grid.healpix.PixelOrder.NEST,
                earth2grid.healpix.PixelOrder.RING,
            )
            return {
                test_dataset.batch_info.channels[c]: ring_order[:, c].cpu()
                for c in range(x.shape[1])
            }

        output_data = prepare(pred)
        # Convert time data to timestamps
        timestamps = batch["timestamp"]
        writer.write_batch(output_data, timestamps)

    if WORLD_RANK == 0 and plot_sample:
        input_data = prepare(lr)
        target_data = prepare(target)
        diagnostics(
            output_data,
            input_data,
            target_data,
        )


if __name__ == "__main__":
    inference()
