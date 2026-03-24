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
"""
Distributed correlated inference script using CorrelatedLatentGenerator.

This script performs distributed inference with temporally correlated latents,
ensuring proper correlation across time steps even when different ranks handle
different time windows.

Settings are hard-coded to match correlated_inference.py example.
"""

import logging
import math
import os
import cbottle.distributed as dist
import torch
import tqdm
from cbottle.datasets import dataset_3d
from cbottle.datasets.dataset_3d import VARIABLE_CONFIGS
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from cbottle.correlated_latents_generator import CorrelatedLatentGenerator
import cbottle.inference
import warnings
import pandas as pd

logger = logging.getLogger(__name__)

DATASET_NAME = "amip"
SEED = 42
BATCH_SIZE = 20
HPX_LEVEL = 6
START_TIME = "1978-10-01T00:00:00"
END_TIME = "2024-12-31T21:00:00"
FREQ = "3h"


def save_inferences_correlated(
    model: cbottle.inference.CBottle3d,
    dataset,
    output_path: str,
    rank: int,
    world_size: int,
    total_times: int,
    model_name: str = "",
    correlation_half_life: float = 8.0,
):
    """
    Save inferences with temporally correlated latents.

    This function handles distributed inference with proper temporal correlation
    by tracking global time indices and using CorrelatedLatentGenerator.
    """
    latent_generator = CorrelatedLatentGenerator(
        device=torch.device("cuda"),
        correlation_half_life=correlation_half_life,
        seed=SEED,
        rank=rank,
    )

    logger.info(
        f"Rank {rank}: Initialized CorrelatedLatentGenerator with half-life={correlation_half_life}, seed={SEED}"
    )

    # Initialize netCDF writer
    attrs = {
        "correlation_half_life": correlation_half_life,
        "seed": SEED,
        "model": model_name,
    }
    nc_config = NetCDFConfig(
        hpx_level=HPX_LEVEL,
        time_units=dataset.time_units,
        calendar=dataset.calendar,
        attrs=attrs,
    )
    writer = NetCDFWriter(
        output_path, nc_config, dataset.batch_info.channels, rank=rank
    )

    # Set dataset to not shuffle
    if hasattr(dataset, "infinite"):
        dataset.infinite = False
    if hasattr(dataset, "shuffle"):
        dataset.shuffle = False

    # Skip times that have already been processed
    try:
        logger.info(
            f"Rank {rank}: Skipping {writer.time_index} times out of {len(dataset._times)}"
        )
        dataset._times = dataset._times[writer.time_index :]
    except AttributeError:
        pass

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, sampler=None, drop_last=False
    )

    # Track global time index for correlation
    # Each rank processes a contiguous block of time steps
    # The dataset was split by rank via set_times(), using _split() which does:
    # - samples_per_rank = ceil(total_times / world_size)
    # - rank r gets times [r * samples_per_rank : (r+1) * samples_per_rank]

    # Calculate the starting global time index for this rank
    samples_per_rank = math.ceil(total_times / world_size)
    global_time_offset = rank * samples_per_rank + writer.time_index

    logger.info(
        f"Rank {rank}: Starting inference at global time offset {global_time_offset} "
        f"(total_times={total_times}, samples_per_rank={samples_per_rank})"
    )
    logger.info(f"Rank {rank}: Processing {len(dataset._times)} consecutive time steps")

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, disable=rank != 0)):
        images = batch["target"]
        batch_size_actual = images.shape[0]

        # Calculate the global time offset for this batch
        # Each batch contains consecutive time steps: offset + batch_idx * batch_size
        current_global_time = global_time_offset + batch_idx * batch_size_actual

        # Generate correlated latents for this batch
        pre_generated_latents = latent_generator.generate_latents(
            batch_size=batch_size_actual,
            channels=model.net.img_channels,
            time_length=model.net.time_length,
            spatial_dims=model.net.domain.numel(),
            time_offset=current_global_time,
        )

        logger.debug(
            f"Rank {rank}: Generated correlated latents for batch {batch_idx}, "
            f"global time range [{current_global_time}, {current_global_time + batch_size_actual})"
        )

        # Sample with pre-generated correlated latents
        out, coords = model._sample_with_latents(
            batch,
            pre_generated_latents=pre_generated_latents,
        )

        writer.write_target(out, coords, batch["timestamp"])

    logger.info(f"Rank {rank}: Completed inference")


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", "Cannot do a zero-copy NCHW to NHWC")

    import argparse

    parser = argparse.ArgumentParser(description="Distributed correlated inference")
    parser.add_argument("output_path", type=str)
    parser.add_argument(
        "--model",
        type=str,
        default="cbottle-3d-moe-aimip",
        help="Model name (e.g. cbottle-3d-moe-aimip-p1, -p2, -p3, -p4)",
    )
    parser.add_argument(
        "--half-life",
        type=float,
        default=8.0,
        help="Correlation half-life in time steps (default: 8.0)",
    )
    parser.add_argument(
        "--sst-offset",
        type=float,
        default=0.0,
        help="Uniform SST offset in Kelvin (e.g. 2.0 or 4.0 for warming experiments)",
    )
    args = parser.parse_args()

    output_path = args.output_path
    model_name = args.model
    correlation_half_life = args.half_life
    sst_offset = args.sst_offset

    dist.init()

    logger.info(f"Loading model: {model_name}")
    model = cbottle.inference.load(model_name)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Get SLURM vars if present
    id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))  # 1-indexed
    slurm_array_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    rank = rank + world_size * (id - 1)
    world_size = world_size * slurm_array_count

    logger.info(
        f"Rank {rank} of {world_size} (SLURM array task {id}/{slurm_array_count})"
    )

    # Setup dataset
    batch_info = model.coords.batch_info
    variables = dataset_3d.guess_variable_config(batch_info.channels)

    if sst_offset != 0.0:
        logger.info(f"Applying SST offset: {sst_offset} K")

    dataset = dataset_3d.get_dataset(
        rank=rank,
        world_size=world_size,
        split="",
        dataset=DATASET_NAME,
        sst_input=True,
        infinite=False,
        shuffle=False,
        variable_config=VARIABLE_CONFIGS[variables],
        sst_offset=sst_offset,
    )

    # Set times
    times = pd.date_range(start=START_TIME, end=END_TIME, freq=FREQ)
    total_times = len(times)
    logger.info(
        f"Total time steps across all ranks: {total_times} from {times[0]} to {times[-1]}"
    )

    dataset.set_times(times)
    logger.info(
        f"Rank {rank}: Processing {len(dataset._times)} consecutive time steps "
        f"(after dataset split)"
    )

    dataset.infinite = False
    dataset.batch_info = batch_info

    save_inferences_correlated(
        model,
        dataset,
        output_path,
        rank=rank,
        world_size=world_size,
        total_times=total_times,
        model_name=model_name,
        correlation_half_life=correlation_half_life,
    )


if __name__ == "__main__":
    main()
