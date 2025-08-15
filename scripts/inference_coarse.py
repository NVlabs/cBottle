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
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
import cbottle.distributed as dist
import torch
import torch.distributed
import tqdm
from cbottle.dataclass_parser import Help, a, parse_args
from cbottle.datasets import dataset_3d, samplers
from cbottle.datasets.dataset_3d import VARIABLE_CONFIGS
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
import cbottle.inference
import numpy as np
import warnings
import pandas as pd

logger = logging.getLogger(__name__)


class Sampler(Enum):
    all = auto()
    fibonacci = auto()
    random = auto()


class Dataset(Enum):
    icon = auto()
    era5 = auto()
    amip = auto()


@dataclass(frozen=True)
class SamplerArgs:
    test_fast: a[bool, Help("Run in fast test mode")] = False
    regression: a[bool, Help("Run regression tests")] = False
    validate_only: a[bool, Help("Only perform validation")] = False
    min_samples: a[int, Help("Minimum number of samples.")] = -1
    start_from_noisy_image: a[bool, Help("Start from a noisy image")] = False
    sigma_max: a[float, Help("Maximum sigma value")] = 80.0
    sampler: Sampler = Sampler.all
    mode: a[str, Help("options: infill, translate, sample, save_data")] = "sample"
    translate_dataset: a[
        str,
        Help(
            'Dataset to translate input to when using mode == "translate". era5 or icon.'
        ),
    ] = "icon"
    bf16: a[bool, Help("Use bf16")] = False
    batch_gpu: a[int, Help("Batch size per GPU")] = 4
    seed: int | None = None
    tc_location: a[
        str, Help("TC location(s) as 'lat,lon' or 'lat1,lon1;lat2,lon2;...'")
    ] = ""
    guidance_scale: a[float, Help("Guidance scale")] = 0.03


@dataclass
class CLI:
    state_path: a[
        str,
        Help(
            "Paths to the model state file (accept comma-separated paths for sigma-dependent MoE with descending order)"
        ),
    ]
    output_path: a[str, Help("Path to the output directory")]
    sigma_thresholds: a[str, Help("Comma-separated thresholds")] = "100.0,10.0"
    dataset: Dataset = Dataset.icon
    data_split: str = ""
    sst_offset: float = 0.0
    sample: SamplerArgs = SamplerArgs()
    hpx_level: int = 6
    start_time: a[str, Help("Start time")] = ""
    end_time: a[str, Help("End time")] = "2018-12-31"
    timestamp_frequency: a[str, Help("Timestamp frequency, out of h, D, M, Y")] = "h"


units = "seconds since 1900-1-1 0:0:0"
calendar = "proleptic_gregorian"


def parse_tc_location(tc_location: str):
    pairs = [p.strip() for p in tc_location.split(";") if p.strip()]
    if pairs:
        coords = [tuple(map(float, pair.split(","))) for pair in pairs]
        lats, lons = zip(*coords)
        return lats, lons
    return None, None


def get_requested_times(args):
    if args.start_time:
        return pd.date_range(
            start=args.start_time, end=args.end_time, freq=args.timestamp_frequency
        )


def save_inferences(
    model: cbottle.inference.CBottle3d,
    dataset,
    output_path,
    *,
    attrs=None,
    hpx_level: int,
    config: SamplerArgs,
    rank: int,
    world_size: int,
    tc_lats: list[float] | None = None,
    tc_lons: list[float] | None = None,
):
    attrs = attrs or {}
    tasks = None

    # Initialize netCDF writer
    nc_config = NetCDFConfig(
        hpx_level=hpx_level,
        time_units=dataset.time_units,
        calendar=dataset.calendar,
        attrs=attrs,
    )
    writer = NetCDFWriter(
        output_path, nc_config, dataset.batch_info.channels, rank=rank
    )

    if config.sampler == Sampler.fibonacci:
        sampler = samplers.subsample(dataset, min_samples=config.min_samples)
        tasks = samplers.distributed_split(sampler)
    elif config.sampler == Sampler.all:
        if config.min_samples > 0:
            dataset.set_times(dataset.times[: config.min_samples])

        if hasattr(dataset, "infinite"):
            dataset.infinite = False
        if hasattr(dataset, "shuffle"):
            dataset.shuffle = False

        # Skip times that have already been processed
        try:
            logger.info(
                f"Skipping {writer.time_index} times out of {len(dataset._times)}"
            )
            dataset._times = dataset._times[writer.time_index :]
        except AttributeError:
            pass

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=config.batch_gpu, sampler=tasks, drop_last=False
    )

    for batch in tqdm.tqdm(loader, disable=rank != 0):
        if (config.min_samples > 0) and (
            writer.time_index * world_size > config.min_samples
        ):
            break

        indices_where_tc = None
        if tc_lons is not None:
            indices_where_tc = model.get_guidance_pixels(tc_lons, tc_lats)
            np.save(
                os.path.join(output_path, "indices_where_tc.npy"),
                indices_where_tc.numpy(),
            )

        match config.mode:
            case "save_data":
                out, coords = model.denormalize(batch)
            case "translate":
                out, coords = model.translate(batch, dataset=config.translate_dataset)
            case "infill":
                out, coords = model.infill(batch)
            case "sample":
                out, coords = model.sample(
                    batch,
                    start_from_noisy_image=config.start_from_noisy_image,
                    guidance_pixels=indices_where_tc,
                    guidance_scale=config.guidance_scale,
                    bf16=config.bf16,
                )
            case _:
                raise NotImplementedError(config.mode)

        writer.write_target(out, coords, batch["timestamp"])


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", "Cannot do a zero-copy NCHW to NHWC")
    args = parse_args(CLI, convert_underscore_to_hyphen=False)

    state_paths = [s.strip() for s in args.state_path.split(",")]
    sigma_thresholds = [float(tok) for tok in args.sigma_thresholds.split(",")]
    sigma_thresholds = sigma_thresholds[: len(state_paths) - 1]

    if args.sample.tc_location:
        tc_lats, tc_lons = parse_tc_location(args.sample.tc_location)
        if tc_lats is None or tc_lons is None:
            raise ValueError("Invalid TC location format")
    else:
        tc_lats = tc_lons = None

    logging.info(f"Using {len(state_paths)} model state path(s)")
    for path in state_paths:
        logging.info(f" - {path}")

    dist.init()

    model = cbottle.inference.CBottle3d.from_pretrained(
        state_paths,
        sigma_thresholds=sigma_thresholds,
        sigma_max=args.sample.sigma_max,
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # get slurm vars if present
    id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))  # 1-indexed
    slurm_array_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    rank = rank + world_size * (id - 1)
    world_size = world_size * slurm_array_count

    # only one time step in this:
    # dataset = Era5Dataset(args.hpx_level, train=False)
    batch_info = model.coords.batch_info
    variables = dataset_3d.guess_variable_config(batch_info.channels)

    dataset = dataset_3d.get_dataset(
        rank=rank,
        world_size=world_size,
        split=args.data_split,
        dataset=args.dataset.name,
        sst_input=True,
        infinite=False,
        shuffle=False,
        variable_config=VARIABLE_CONFIGS[variables],
    )

    requested_times = get_requested_times(args)
    if requested_times is not None:
        dataset.set_times(requested_times)

    dataset.infinite = False
    dataset.batch_info = batch_info

    save_inferences(
        model,
        dataset,
        args.output_path,
        hpx_level=args.hpx_level,
        config=args.sample,
        rank=rank,
        world_size=world_size,
        tc_lats=tc_lats,
        tc_lons=tc_lons,
    )


if __name__ == "__main__":
    main()
