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

import torch
import time
import tqdm
import logging
import warnings
import os
from enum import auto, Enum
from dataclasses import dataclass

import cbottle.distributed as dist
from cbottle.dataclass_parser import Help, a, parse_args
import cbottle.inference
from cbottle.datasets import dataset_3d, samplers
from cbottle.datasets.dataset_3d import VARIABLE_CONFIGS
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets.merged_dataset import TimeMergedMapStyle
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from inference_coarse import Dataset, Sampler, get_requested_times

logger = logging.getLogger(__name__)


class FrameSelectionStrategy(Enum):
    """Strategy for selecting which frames to condition on during inference."""

    unconditional = auto()  # No frames kept (fully unconditional)
    first_frame = auto()  # Keep only the first frame
    first_two = auto()  # Keep first two frames
    endpoints = auto()  # Keep first and last frames
    center_frame = auto()  # Keep the middle frame

    def get_keep_frames(self, time_length: int) -> list[int]:
        return {
            FrameSelectionStrategy.unconditional: [],
            FrameSelectionStrategy.first_frame: [0],
            FrameSelectionStrategy.first_two: [0, 1],
            FrameSelectionStrategy.endpoints: [0, time_length - 1],
            FrameSelectionStrategy.center_frame: [time_length // 2],
        }[self]

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class SamplerArgs:
    """Other than tc_guidance related arguments, accepts the same arguments as `inference_coarse.py`"""

    min_samples: a[int, Help("Minimum number of samples.")] = -1
    start_from_noisy_image: a[bool, Help("Start from a noisy image")] = False
    sigma_min: a[float, Help("Minimum sigma value")] = 0.02
    sigma_max: a[float, Help("Maximum sigma value")] = 1000.0
    sampler: Sampler = Sampler.fibonacci
    mode: a[str, Help("options: infill, translate, sample, save_data")] = "sample"
    translate_dataset: a[
        str,
        Help(
            'Dataset to translate input to when using mode == "translate". era5 or icon.'
        ),
    ] = "icon"
    bf16: a[bool, Help("Use bf16")] = False
    frame_selection_strategy: FrameSelectionStrategy = (
        FrameSelectionStrategy.unconditional
    )
    seed: int | None = None


@dataclass
class CLI:
    state_path: a[
        str,
        Help("Path to the model state file"),
    ]
    output_path: a[str, Help("Path to the output directory")]
    dataset: Dataset = Dataset.icon
    data_split: str = ""
    sample: SamplerArgs = SamplerArgs()
    hpx_level: int = 6
    start_time: a[str, Help("Start time")] = ""
    end_time: a[str, Help("End time")] = "2018-12-31"
    time_step: int = 6  # spacing of frames in hours


def save_inferences(
    model: cbottle.inference.CBottle3d,
    dataset: TimeMergedMapStyle,
    output_path: str,
    *,
    attrs=None,
    hpx_level: int,
    config: SamplerArgs,
    rank: int,
    world_size: int,
    keep_frames: list[int] = [],
) -> None:
    start_time = time.time()

    attrs = attrs or {}
    batch_info = dataset.batch_info

    # Setup netCDF files
    nc_config = NetCDFConfig(
        hpx_level=hpx_level,
        time_units=dataset.time_units,
        calendar=dataset.calendar,
        format="NETCDF4",
        attrs=attrs,
    )

    writer = NetCDFWriter(
        output_path,
        nc_config,
        batch_info.channels,
        rank=rank,
        add_video_variables=True,
    )

    # Setup tasks
    if config.sampler == Sampler.fibonacci:
        sampler = samplers.subsample(
            dataset, min_samples=max(config.min_samples, world_size)
        )
        tasks = samplers.distributed_split(sampler)
    elif config.sampler == Sampler.all:
        all_tasks = list(range(len(dataset)))
        rank_tasks = samplers.distributed_split(all_tasks)
        if config.min_samples > 0:
            rank_tasks = rank_tasks[: config.min_samples]

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

    batch_size = 1
    time_length = model.time_length
    loader = torch.utils.data.DataLoader(
        pin_memory=True,
        dataset=dataset,
        batch_size=batch_size,
        sampler=tasks,
    )

    for batch in tqdm.tqdm(loader, disable=rank != 0):
        if (config.min_samples > 0) and (
            writer.time_index * world_size > config.min_samples
        ):
            break

        # dataset provides a single timestamp per video, corresponding to the first frame
        first_frame_ts = batch.pop("timestamp").cpu()
        batch_size = first_frame_ts.shape[0]
        frame_offsets_sec = torch.tensor(
            [batch_info.get_time_delta(i).total_seconds() for i in range(time_length)],
        )
        lead_time_hours = (
            (frame_offsets_sec / 3600.0).unsqueeze(0).expand(batch_size, -1)
        )
        timestamps = first_frame_ts.unsqueeze(-1) + frame_offsets_sec.unsqueeze(0)

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
                    bf16=config.bf16,
                )
            case _:
                raise NotImplementedError(config.mode)

        if config.mode == "save_data":
            frame_source = torch.full((batch_size, time_length), 0, dtype=torch.int8)
        else:
            frame_source = torch.full((batch_size, time_length), 1, dtype=torch.int8)
            frame_source[:, keep_frames] = 2

        scalars = {
            "frame_source_flag": frame_source,
            "lead_time": lead_time_hours,
        }
        writer.write_target(out, coords, timestamps, scalars=scalars)

    time_end = time.time()
    logger.info(
        f"Inference completed in {time_end - start_time:.2f} sec for {len(loader)} batches on rank {rank}"
    )


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", "Cannot do a zero-copy NCHW to NHWC")
    args = parse_args(CLI, convert_underscore_to_hyphen=False)

    state_path = args.state_path
    logging.info(f"Using {state_path} model state path")

    dist.init()

    sigma_max = args.sample.sigma_max
    sigma_min = args.sample.sigma_min
    # Video model does not use MOE yet so just provide single path and no thresholds
    model = cbottle.inference.CBottle3d.from_pretrained(
        state_path,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # get slurm vars if present
    id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))  # 1-indexed
    slurm_array_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    rank = rank + world_size * (id - 1)
    world_size = world_size * slurm_array_count

    batch_info = model.coords.batch_info
    variables = dataset_3d.guess_variable_config(batch_info.channels)

    time_length = model.time_length
    time_step = args.time_step

    frame_selection_strategy = args.sample.frame_selection_strategy
    keep_frames_arr = frame_selection_strategy.get_keep_frames(time_length)
    frame_masker = FrameMasker(keep_frames=keep_frames_arr)

    if (
        args.dataset.name == "amip"
        and frame_selection_strategy != FrameSelectionStrategy.unconditional
    ):
        raise ValueError(
            "AMIP dataset only supports unconditional frame selection strategy"
        )

    dataset = dataset_3d.get_dataset(
        split=args.data_split,
        dataset=args.dataset.name,
        sst_input=True,
        infinite=False,
        shuffle=False,
        time_step=time_step,
        time_length=time_length,
        frame_masker=frame_masker,
        variable_config=VARIABLE_CONFIGS[variables],
        map_style=True,
    )

    requested_times = get_requested_times(args)
    if requested_times is not None:
        dataset.set_times(requested_times)

    description = (
        "Model inference data"
        if args.sample.mode != "save_data"
        else "Ground truth data"
    )
    attrs = {
        "description": description,
        "dataset": args.dataset.name,
        "frame_selection_strategy": frame_selection_strategy.name,
    }

    if args.sample.seed is not None:
        torch.manual_seed(args.sample.seed)

    if rank == 0:
        logger.info(
            "\nInference Configuration:"
            "\n------------------------"
            f"\nOutput path:  {args.output_path}"
            f"\nDataset:      {args.dataset}"
            f"\nSampler mode: {args.sample.mode}"
            "\nSampling settings:"
            f"\n  • sigma_max:   {sigma_max}"
            f"\n  • sigma_min:   {sigma_min}"
            f"\n  • sampler:     {args.sample.sampler}"
            f"\n  • min_samples: {args.sample.min_samples}"
            "\nVideo settings:"
            f"\n  • time_length: {time_length}"
            f"\n  • frame_step:  {time_step} (hours)"
            f"\n  • masking:     {args.sample.frame_selection_strategy}"
            f"\n  • keep frames: {keep_frames_arr}"
            "\n------------------------"
        )

    save_inferences(
        model=model,
        dataset=dataset,
        output_path=args.output_path,
        attrs=attrs,
        hpx_level=args.hpx_level,
        config=args.sample,
        rank=rank,
        world_size=world_size,
        keep_frames=keep_frames_arr,
    )


if __name__ == "__main__":
    main()
