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

import datetime
from typing import Optional

import torch
import time
import tqdm
import os
import cftime
from enum import auto, Enum
from dataclasses import dataclass

import cbottle.distributed as dist
from cbottle.dataclass_parser import Help, a, parse_args

from cbottle.datasets import dataset_3d, samplers
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets.merged_dataset import TimeMergedMapStyle
from cbottle.denoiser_factories import get_denoiser, DenoiserType
from cbottle.diffusion_samplers import edm_sampler_from_sigma
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from inference_coarse import (
    Dataset,
    SamplerArgs as BaseSamplerArgs,
    CLI as BaseCLI,
    prepare_for_saving,
    build_labels,
)
from train_coarse import TrainingLoop

# Constants
UNITS = "seconds since 1900-1-1 0:0:0"
CALENDAR = "proleptic_gregorian"
HPX_LEVEL = 6


class TasksType(Enum):
    subsample = auto()
    dec_may = auto()
    long_inference = auto()


class SaveMode(Enum):
    """Controls which data to save during inference."""

    data = auto()  # Save only ground truth
    inference = auto()  # Save only model outputs
    all = auto()  # Save both


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
class SamplerArgs(BaseSamplerArgs):
    sigma_min: a[float, Help("Minimum sigma value")] = 0.02
    save_mode: SaveMode = SaveMode.all
    frame_selection_strategy: FrameSelectionStrategy = (
        FrameSelectionStrategy.unconditional
    )
    tasks_type: TasksType = TasksType.subsample


@dataclass
class CLI(BaseCLI):
    data_split: str = "test"
    sample: SamplerArgs = SamplerArgs()
    output_path: Optional[str] = None
    time_step: int = 6


def _create_writer(
    base_path: str,
    description: str,
    dataset_type: Dataset,
    dataset: torch.utils.data.Dataset,
    rank: int,
    format: str = "NETCDF4",
    add_video_variables: bool = True,
) -> NetCDFWriter:
    nc_config = NetCDFConfig(
        hpx_level=HPX_LEVEL,
        time_units=UNITS,
        calendar=CALENDAR,
        format=format,
        attrs={"description": description, "dataset": dataset_type.name},
    )

    return NetCDFWriter(
        base_path,
        nc_config,
        dataset.batch_info.channels,
        rank=rank,
        add_video_variables=add_video_variables,
    )


def run_inference(
    net: torch.nn.Module,
    config: SamplerArgs,
    video_dataset: TimeMergedMapStyle,
    output_path: str,
    save_mode: SaveMode,
    min_samples: int,
    sigma_max: float,
    sigma_min: float,
    keep_frames: list[int] = [],
    tasks: Optional[list[int]] = None,
    dataset_type: Dataset = Dataset.icon,
) -> None:
    """Run inference on the dataset"""
    time_top = time.time()
    if tasks is None:
        tasks = samplers.subsample(video_dataset, min_samples=min_samples)

    tasks = samplers.distributed_split(tasks)

    # Create dataloader
    batch_size = 1
    time_length = net.time_length
    loader = torch.utils.data.DataLoader(
        pin_memory=True,
        dataset=video_dataset,
        batch_size=batch_size,
        sampler=tasks,
    )
    batch_info = video_dataset.batch_info

    # Setup netCDF files
    inference_path = os.path.join(output_path, "inferences")
    target_path = os.path.join(output_path, "target")

    rank = dist.get_rank()

    if save_mode in [SaveMode.inference, SaveMode.all]:
        inference_writer = _create_writer(
            inference_path,
            "Model inference data",
            dataset_type,
            video_dataset,
            rank,
            add_video_variables=True,
        )

    if save_mode in [SaveMode.data, SaveMode.all]:
        target_writer = _create_writer(
            target_path,
            "Ground truth data",
            dataset_type,
            video_dataset,
            rank,
            add_video_variables=True,
        )

    hpx = net.domain._grid
    torch.manual_seed(dist.get_rank())
    loop = TrainingLoop()
    loop.device = net.device

    for batch in tqdm.tqdm(loader, disable=dist.get_rank() != 0):
        batch = loop._stage_dict_batch(batch)

        timestamp = batch.pop("timestamp").cpu()
        first_frame_times = [
            round_to_half_hour(
                datetime.datetime.fromtimestamp(t.item(), datetime.timezone.utc)
            )
            for t in timestamp
        ]
        all_times_datetimes = [
            [
                first_frame_times[b] + batch_info.get_time_delta(i)
                for i in range(time_length)
            ]
            for b in range(batch_size)
        ]
        all_times_timestamps = torch.tensor(
            [
                [cftime.date2num(dt, units=UNITS, calendar=CALENDAR) for dt in sequence]
                for sequence in all_times_datetimes
            ]
        )
        lead_time_hours = (all_times_timestamps - all_times_timestamps[:, :1]) / 3600.0

        with torch.no_grad():
            # Save ground truth if requested
            if save_mode in [SaveMode.data, SaveMode.all]:
                truth_ring_denormalized = prepare_for_saving(
                    batch["target"], hpx, batch_info
                )
                frame_source = torch.full(
                    (batch_size, time_length), 0, dtype=torch.int8
                )

                target_writer.write_batch(
                    truth_ring_denormalized,
                    all_times_timestamps,
                    scalars={
                        "frame_source_flag": frame_source,
                        "lead_time": lead_time_hours,
                    },
                )

            # Generate and save inferences
            if save_mode in [SaveMode.inference, SaveMode.all]:
                images = batch["target"]
                labels = batch["labels"]
                condition = batch["condition"]
                second_of_day = batch["second_of_day"].to(images.device).float()
                day_of_year = batch["day_of_year"].to(images.device).float()

                labels_when_nan = None
                if config.denoiser_type == DenoiserType.mask_filling:
                    labels_when_nan = build_labels(labels, config.denoiser_when_nan)
                elif config.denoiser_type == DenoiserType.infill:
                    labels = build_labels(labels, config.denoiser_when_nan)

                # Get appropriate denoiser
                D = get_denoiser(
                    net,
                    images,
                    labels,
                    condition,
                    second_of_day,
                    day_of_year,
                    denoiser_type=config.denoiser_type,
                    sigma_max=sigma_max,
                    labels_when_nan=labels_when_nan,
                )

                latents = (
                    torch.randn(
                        (batch_size, net.img_channels, time_length, net.domain.numel()),
                        device=net.device,
                    )
                    * sigma_max
                )

                xT = (
                    latents + batch["target"]
                    if config.start_from_noisy_image
                    else latents
                )

                out = edm_sampler_from_sigma(
                    D,
                    xT,
                    randn_like=torch.randn_like,
                    sigma_max=int(sigma_max),
                    sigma_min=sigma_min,
                )

                inference_ring_denormalized = prepare_for_saving(out, hpx, batch_info)

                frame_source = torch.full(
                    (batch_size, time_length), 1, dtype=torch.int8
                )
                frame_source[:, keep_frames] = 2
                inference_writer.write_batch(
                    inference_ring_denormalized,
                    all_times_timestamps,
                    scalars={
                        "frame_source_flag": frame_source,
                        "lead_time": lead_time_hours,
                    },
                )

    time_end = time.time()
    print(
        f"Inference completed in {time_end - time_top:.2f} sec for {len(loader)} batches on rank {dist.get_rank()}"
    )


def build_video_tasks(
    dataset,
    start_date: cftime.DatetimeProlepticGregorian,
    end_date: cftime.DatetimeProlepticGregorian,
    time_length: int = 12,
    frame_step: int = 12,
) -> list[int]:
    """
    Return a list of valid starting indices in 'dataset' whose T-frame window
    lies entirely between [start_date .. end_date].
    """
    times = dataset.times
    valid_indices = []

    first_valid_idx = None
    for i in range(len(dataset)):
        if times[i] >= start_date:
            first_valid_idx = i
            break

    assert first_valid_idx is not None, "No valid index found"

    for i in range(first_valid_idx, len(dataset), time_length * frame_step):
        end_idx = (
            i + (time_length - 1) * frame_step
        )  # the idx of the last frame of the T-frame video

        if end_idx >= len(times):
            break

        start_time, end_time = times[i], times[end_idx]

        if start_time >= start_date and end_time <= end_date:
            valid_indices.append(i)

    return valid_indices


def long_inference_tasks(
    dataset,
    year_start: int = 1980,
    year_end: int = 2000,
    time_length: int = 12,
    frame_step: int = 12,
) -> list[int]:
    start_date = cftime.DatetimeProlepticGregorian(year_start, 1, 1, 0, 0, 0)
    end_date = cftime.DatetimeProlepticGregorian(year_end - 1, 12, 31, 23, 59, 0)
    return build_video_tasks(dataset, start_date, end_date, time_length, frame_step)


def build_dec_may_video_tasks(
    dataset,
    year_start: int = 2024,
    year_end: int = 2025,
    time_length: int = 12,
    frame_step: int = 12,
) -> list[int]:
    start_date = cftime.DatetimeProlepticGregorian(year_start, 12, 1, 0, 0, 0)
    end_date = cftime.DatetimeProlepticGregorian(year_end, 5, 31, 23, 59, 0)
    return build_video_tasks(dataset, start_date, end_date, time_length, frame_step)


def round_to_half_hour(dt):
    minutes = dt.hour * 60 + dt.minute
    rounded_minutes = round(minutes / 30) * 30
    return dt.replace(
        hour=rounded_minutes // 60, minute=rounded_minutes % 60, second=0, microsecond=0
    )


def main():
    args = parse_args(CLI, convert_underscore_to_hyphen=False)
    state_path = args.state_path
    run_name = os.path.basename(os.path.dirname(state_path))
    ckpt_name = os.path.basename(state_path).split(".")[0]
    ckpt_num = ckpt_name.split("-")[-1]

    dist.init()

    min_samples = args.sample.min_samples
    if min_samples <= 0:
        min_samples = dist.get_world_size()  # Default to world size

    sigma_max = args.sample.sigma_max
    sigma_min = args.sample.sigma_min

    from cbottle.checkpointing import Checkpoint

    with Checkpoint(state_path) as checkpoint:
        net = checkpoint.read_model()

    net.eval()
    net.requires_grad_(False)
    net.float()
    net.cuda()

    time_length = net.time_length
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

    denoiser_str = args.sample.denoiser_type.name

    if not args.output_path:
        args.output_path = f"inferences/video/{run_name}/{ckpt_num}/sigma_{sigma_min}_{int(sigma_max)}/{frame_selection_strategy}/{args.dataset.name}v6/denoiser_{denoiser_str}"

    dataset = dataset_3d.get_dataset(
        split=args.data_split,
        dataset=args.dataset.name,
        sst_input=True,
        infinite=False,
        shuffle=False,
        time_step=time_step,
        time_length=time_length,
        frame_masker=frame_masker,
    )

    tasks = None
    if args.sample.tasks_type == TasksType.dec_may:
        tasks = build_dec_may_video_tasks(
            dataset,
            year_start=2024,
            year_end=2025,
            time_length=time_length,
            frame_step=time_step,
        )
        args.output_path = f"{args.output_path}/dec_may_{len(tasks)}"
    elif args.sample.tasks_type == TasksType.long_inference:
        tasks = long_inference_tasks(
            dataset,
            year_start=1980,
            year_end=2000,
            time_length=time_length,
            frame_step=time_step,
        )
        args.output_path = f"{args.output_path}/long_inference_{len(tasks)}"

    if dist.get_rank() == 0:
        print("\nInference Configuration:")
        print("------------------------")
        print(f"Run name:     {run_name}")
        print(f"Output path:  {args.output_path}")
        print(f"Dataset:      {args.dataset} (version v6)")
        print(f"Save mode:    {args.sample.save_mode}")
        print("\nSampling settings:")
        print(f"  • sigma_max:   {sigma_max}")
        print(f"  • sigma_min:   {sigma_min}")
        print(f"  • denoiser:    {denoiser_str}")
        print(f"  • denoiser_when_nan: {args.sample.denoiser_when_nan}")
        print("\nVideo settings:")
        print(f"  • time_length: {time_length}")
        print(f"  • frame_step:  {time_step}")
        print(f"  • masking:     {args.sample.frame_selection_strategy}")
        print(f"  • keep frames: {keep_frames_arr}")

        if args.sample.tasks_type != TasksType.subsample:
            print(f"\nTask type: {args.sample.tasks_type} ({len(tasks)} tasks)")
        else:
            print(f"\nProcessing {min_samples} samples")
        print("------------------------")

    run_inference(
        net=net,
        video_dataset=dataset,
        output_path=args.output_path,
        save_mode=args.sample.save_mode,
        min_samples=min_samples,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        keep_frames=keep_frames_arr,
        tasks=tasks,
        config=args.sample,
        dataset_type=args.dataset,
    )


if __name__ == "__main__":
    main()
