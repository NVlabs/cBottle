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
import logging
import warnings
import time
from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd
import numpy as np
import tqdm

import cbottle.distributed as dist
import cbottle.inference
from cbottle.dataclass_parser import Help, a, parse_args
from cbottle.datasets import dataset_3d, samplers
from cbottle.datasets.dataset_3d import VARIABLE_CONFIGS
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from cbottle.video_rollout import VideoRollout, RolloutStep
from inference_coarse import Dataset, Sampler, parse_model_paths

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


class RolloutDuration(Enum):
    """Duration presets for rollouts."""

    single = auto()  # Single video (no rollout)
    two_weeks = auto()  # Every 2 weeks from start time
    one_month = auto()  # Every month start
    three_months = auto()  # Every 3 months start
    six_months = auto()  # Every 6 months start
    one_year = auto()  # Evert 6 months start


@dataclass(frozen=True)
class SamplerArgs:
    """Rollout and sampling configuration."""

    duration: RolloutDuration = RolloutDuration.one_month
    sigma_min: a[float, Help("Minimum sigma value")] = 0.02
    sigma_max: a[float, Help("Maximum sigma value")] = 1000.0
    bf16: a[bool, Help("Use bfloat16 precision")] = False
    torch_compile: a[bool, Help("Use torch.compile() for model")] = False
    seed: int | None = None
    min_samples: a[int, Help("Minimum number of samples from dataset (-1 for all)")] = 1
    sampler: Sampler = Sampler.fibonacci
    seed_conditioning: a[
        FrameSelectionStrategy, Help("Conditioning strategy for seed step")
    ] = FrameSelectionStrategy.unconditional
    rollout_conditioning: a[
        FrameSelectionStrategy, Help("Conditioning strategy for rollout steps")
    ] = FrameSelectionStrategy.first_frame


@dataclass
class CLI:
    """Video rollout inference CLI."""

    output_path: a[str, Help("Output netCDF path")]
    dataset: Dataset = Dataset.amip
    sample: SamplerArgs = SamplerArgs()
    state_path: a[
        str,
        Help(
            "Optional: Direct paths to model state file (comma-separated for MoE). "
            "If not provided, uses checkpoint_root + named model 'cbottle-3d-video'"
        ),
    ] = ""
    checkpoint_root: a[
        str, Help("Root directory for named models (used if state_path not provided)")
    ] = ""
    sigma_thresholds: a[str, Help("Comma-separated sigma thresholds for MoE")] = (
        "100.0,10.0"
    )
    data_split: str = ""
    hpx_level: int = 6
    start_time: a[str, Help("Start time (YYYY-MM-DD format)")] = ""
    end_time: a[str, Help("End time (YYYY-MM-DD format)")] = "2018-12-31"
    time_step: a[int, Help("Hours between frames")] = 6


def compute_rollout_tasks(
    dataset_times: pd.DatetimeIndex,
    duration: RolloutDuration,
    time_step_hours: int,
    time_length: int,
    frame_step: int,
    conditioning_frames: list[int],
    start_time: str = "",
    end_time: str = "",
) -> list[tuple[int, int, pd.Timestamp]]:
    """
    Compute rollout tasks: (start_idx, num_rollout_steps, target_end_time) for each rollout.
    """
    dataset_start = dataset_times[0]
    dataset_end = dataset_times[-1]

    if start_time and end_time:
        time_range_start = pd.Timestamp(start_time)
        time_range_end = pd.Timestamp(end_time)
    else:
        time_range_start = dataset_start
        time_range_end = dataset_end

    duration_freq_map = {
        RolloutDuration.single: f"{time_step_hours*time_length}H",  # One full video
        RolloutDuration.two_weeks: "14D",
        RolloutDuration.one_month: "MS",  # Month start
        RolloutDuration.three_months: "3MS",
        RolloutDuration.six_months: "6MS",
        RolloutDuration.one_year: "YS",
    }

    freq = duration_freq_map[duration]
    rollout_start_times = pd.date_range(
        start=time_range_start, end=time_range_end, freq=freq
    )

    if len(rollout_start_times) == 0:
        raise ValueError(
            f"No valid rollout start times in range {time_range_start} to {time_range_end}. "
            f"Dataset: {dataset_start} to {dataset_end}, duration: {duration.name}"
        )

    rollout_tasks = []
    for i, start_time_pd in enumerate(rollout_start_times):
        start_idx = dataset_times.get_indexer([start_time_pd], method="nearest")[0]

        if i + 1 < len(rollout_start_times):
            target_end_time = rollout_start_times[i + 1]
        else:
            # for the final rollout, use specified end or dataset end
            target_end_time = min(time_range_end, dataset_end)

        # Calculate how many video frames needed from start to target end
        time_delta = (target_end_time - start_time_pd).total_seconds() / 3600.0  # hours
        total_frames_needed = int(np.ceil(time_delta / time_step_hours))

        new_frames_per_step = time_length - len(conditioning_frames)
        if total_frames_needed <= time_length:
            # Just the initial window, no rollout steps
            num_rollout_steps = 0
        else:
            frames_after_seed = total_frames_needed - time_length
            # Use ceil to ensure we generate enough frames to cover the full duration
            num_rollout_steps = int(np.ceil(frames_after_seed / new_frames_per_step))

        # Verify we have enough dataset runway
        total_video_frames_needed = (
            time_length + num_rollout_steps * new_frames_per_step
        )
        last_dataset_idx_offset = (total_video_frames_needed - 1) * frame_step

        if start_idx + last_dataset_idx_offset >= len(dataset_times):
            # Not enough data - compute how many frames we can actually generate
            available_dataset_count = len(dataset_times) - start_idx

            if available_dataset_count <= 0:
                continue
            max_achievable_frames = ((available_dataset_count - 1) // frame_step) + 1

            if max_achievable_frames < time_length:
                continue

            frames_available_for_rollout = max_achievable_frames - time_length
            num_rollout_steps = frames_available_for_rollout // new_frames_per_step

        rollout_tasks.append((start_idx, num_rollout_steps, target_end_time))

    if len(rollout_tasks) == 0:
        raise ValueError(
            "No valid rollout tasks could be generated. "
            "Dataset may be too short for the specified duration."
        )

    return rollout_tasks


def write_step_frames(
    writer: NetCDFWriter,
    step: RolloutStep,
    target_end_timestamp: float,
) -> None:
    """Write frames from a rollout step, filtering by target end time."""
    frame_mask = step.new_timestamps[0] < target_end_timestamp
    writer.write_target(
        step.new_frames[:, :, frame_mask, :],
        step.coords,
        step.new_timestamps[:, frame_mask],
        scalars={
            "frame_source_flag": step.frame_source_flags[:, frame_mask],
            "lead_time": step.lead_time_hours[:, frame_mask],
        },
    )


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", "Cannot do a zero-copy NCHW to NHWC")

    args = parse_args(CLI, convert_underscore_to_hyphen=False)

    dist.init()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))
    array_size = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    rank = rank + world_size * (task_id - 1)
    world_size = world_size * array_size
    device = "cuda"

    if args.state_path:
        state_paths, sigma_thresholds = parse_model_paths(
            args.state_path, args.sigma_thresholds
        )

        model = cbottle.inference.CBottle3d.from_pretrained(
            state_paths,
            sigma_thresholds=sigma_thresholds,
            sigma_min=args.sample.sigma_min,
            sigma_max=args.sample.sigma_max,
            device=device,
            torch_compile=args.sample.torch_compile,
        )
    else:
        model = cbottle.inference.load(
            "cbottle-3d-video",
            root=args.checkpoint_root if args.checkpoint_root else None,
            device=device,
            sigma_min=args.sample.sigma_min,
            sigma_max=args.sample.sigma_max,
            torch_compile=args.sample.torch_compile,
        )

    batch_info = model.coords.batch_info
    variables = dataset_3d.guess_variable_config(batch_info.channels)
    time_length = model.time_length

    dataset = dataset_3d.get_dataset(
        split=args.data_split,
        dataset=args.dataset.name,
        sst_input=True,
        infinite=False,
        shuffle=False,
        time_step=args.time_step,
        time_length=time_length,
        variable_config=VARIABLE_CONFIGS[variables],
        map_style=True,
        frame_masker=None,  # masking handled by VideoRollout
    )

    seed_conditioning_frames = args.sample.seed_conditioning.get_keep_frames(
        time_length
    )
    rollout_conditioning_frames = args.sample.rollout_conditioning.get_keep_frames(
        time_length
    )

    if (
        args.sample.duration != RolloutDuration.single
        and args.sample.rollout_conditioning == FrameSelectionStrategy.unconditional
    ):
        raise ValueError(
            "Multi-step rollouts require rollout_conditioning (cannot be unconditional)."
        )

    dataset_times = dataset.times
    rollout_tasks = compute_rollout_tasks(
        dataset_times=dataset_times,
        duration=args.sample.duration,
        time_step_hours=args.time_step,
        time_length=time_length,
        frame_step=dataset.frame_step,
        conditioning_frames=rollout_conditioning_frames,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    attrs = {
        "description": "Video rollout inference",
        "dataset": args.dataset.name,
        "rollout_duration": args.sample.duration.name,
        "seed_conditioning": args.sample.seed_conditioning.name,
        "rollout_conditioning": args.sample.rollout_conditioning.name,
        "time_step_hours": args.time_step,
    }

    nc_config = NetCDFConfig(
        hpx_level=args.hpx_level,
        time_units=dataset.time_units,
        calendar=dataset.calendar,
        format="NETCDF4",
        attrs=attrs,
    )

    writer = NetCDFWriter(
        args.output_path,
        nc_config,
        batch_info.channels,
        rank=rank,
        add_video_variables=True,
    )

    if args.sample.sampler == Sampler.fibonacci:
        num_samples = (
            args.sample.min_samples
            if args.sample.min_samples > 0
            else len(rollout_tasks)
        )
        sampled_positions = samplers.subsample(rollout_tasks, min_samples=num_samples)
        selected_rollouts = [rollout_tasks[i] for i in sampled_positions]
    elif args.sample.sampler == Sampler.all:
        selected_rollouts = rollout_tasks
        if args.sample.min_samples > 0:
            selected_rollouts = selected_rollouts[: args.sample.min_samples]
    my_rollout_tasks = samplers.distributed_split(selected_rollouts)

    if rank == 0:
        logger.info("\n=== Rollout Configuration ===")
        logger.info(f"Duration: {args.sample.duration.name}")
        logger.info(f"Video: {time_length} frames @ {args.time_step}h")
        logger.info(
            f"Conditioning - Seed: {args.sample.seed_conditioning.name}, "
            f"Rollout: {args.sample.rollout_conditioning.name}"
        )
        logger.info(
            f"Sampling: {args.sample.sampler.name} - "
            f"Of possible {len(rollout_tasks)} rollout tasks → {len(selected_rollouts)} selected"
        )
        logger.info(
            f"Distribution: {len(selected_rollouts)} tasks / {world_size} workers "
            f"≈ {len(selected_rollouts) // world_size} per worker"
        )

        if len(rollout_tasks) > 0:
            start_idx, _, target_end = rollout_tasks[0]
            last_start_idx, _, last_target_end = rollout_tasks[-1]
            logger.info(
                f"Time range: {dataset_times[start_idx].strftime('%Y-%m-%d')} to "
                f"{last_target_end.strftime('%Y-%m-%d')}"
            )

    sample_kwargs = {
        "bf16": args.sample.bf16,
        "seed": args.sample.seed,
    }

    total_start_time = time.time()
    for task_idx, (start_idx, num_rollout_steps, target_end_time) in enumerate(
        my_rollout_tasks
    ):
        start_time_ts = dataset_times[start_idx]

        batch = dataset[start_idx]

        rollout = VideoRollout(
            model=model,
            dataset=dataset,
            sample_kwargs=sample_kwargs,
        )
        target_frames = batch["target"]
        seed_frames = {
            idx: target_frames[:, :, idx, :] for idx in seed_conditioning_frames
        }
        rollout.seed_generation(start_time_ts, frames=seed_frames)
        step = rollout.step_forward(conditioning_frames=seed_conditioning_frames)

        target_end_timestamp = target_end_time.value / 1e9
        write_step_frames(writer, step, target_end_timestamp)

        pbar = tqdm.tqdm(
            range(num_rollout_steps),
            desc=f"Rank {rank} Task {task_idx+1}/{len(my_rollout_tasks)}",
            disable=(rank != 0),
        )

        for _ in pbar:
            step = rollout.step_forward(conditioning_frames=rollout_conditioning_frames)
            write_step_frames(writer, step, target_end_timestamp)

    total_elapsed = time.time() - total_start_time
    if rank == 0:
        logger.info(
            f"\nCompleted {len(my_rollout_tasks)} rollouts in {total_elapsed/60:.1f} min. "
            f"Output: {args.output_path}"
        )


if __name__ == "__main__":
    main()
