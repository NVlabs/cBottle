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
from inference_coarse_video import FrameSelectionStrategy

logger = logging.getLogger(__name__)


class RolloutDuration(Enum):
    """Duration presets for rollouts."""

    single = auto()  # Single video (no rollout)
    two_weeks = auto()  # Every 2 weeks from start time
    one_month = auto()  # Every month start
    three_months = auto()  # Every 3 months start
    six_months = auto()  # Every 6 months start
    one_year = auto()  # Evert 6 months start

    def to_freq(self, time_step_hours: int, time_length: int) -> str:
        """Convert duration to pandas frequency string."""
        freq_map = {
            RolloutDuration.single: f"{time_step_hours * time_length}H",
            RolloutDuration.two_weeks: "14D",
            RolloutDuration.one_month: "MS",
            RolloutDuration.three_months: "3MS",
            RolloutDuration.six_months: "6MS",
            RolloutDuration.one_year: "YS",
        }
        return freq_map[self]


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
    dataset_start, dataset_end = dataset_times[0], dataset_times[-1]

    if start_time and end_time:
        time_range_start = pd.Timestamp(start_time)
        time_range_end = pd.Timestamp(end_time)
        if time_range_start < dataset_start or time_range_end > dataset_end:
            raise ValueError(
                f"Time range {time_range_start} to {time_range_end} is outside the dataset time range {dataset_start} to {dataset_end}"
            )
    else:
        range_start, range_end = dataset_start, dataset_end

    # Generate rollout start times based on duration
    rollout_starts = pd.date_range(
        start=range_start,
        end=range_end,
        freq=duration.to_freq(time_step_hours, time_length),
    )

    if len(rollout_starts) == 0:
        raise ValueError(
            f"No valid rollout start times in {range_start} to {range_end}. "
            f"Dataset: {dataset_start} to {dataset_end}, duration: {duration.name}"
        )

    new_frames_per_step = time_length - len(conditioning_frames)
    tasks = []

    for i, t0 in enumerate(rollout_starts):
        start_idx = dataset_times.get_indexer([t0], method="nearest")[0]

        if i + 1 < len(rollout_starts):
            target_end = rollout_starts[i + 1]
        else:
            target_end = range_end

        # number of video frames needed to cover [t0, target_end)
        hours = (target_end - t0).total_seconds() / 3600.0
        frames_needed = int(np.ceil(hours / time_step_hours))

        if frames_needed <= time_length:
            num_steps = 0
        else:
            # use ceil to ensure full coverage of duration
            num_steps = int(
                np.ceil((frames_needed - time_length) / new_frames_per_step)
            )

        # check dataset runway and reduce num_steps if needed
        total_video_frames = time_length + num_steps * new_frames_per_step
        last_idx_needed = start_idx + (total_video_frames - 1) * frame_step

        if last_idx_needed >= len(dataset_times):
            available_indices = len(dataset_times) - start_idx
            if available_indices <= 0:
                continue

            max_frames_achievable = ((available_indices - 1) // frame_step) + 1
            if max_frames_achievable < time_length:
                continue

            num_steps = (max_frames_achievable - time_length) // new_frames_per_step

        tasks.append((start_idx, num_steps, target_end))

    if len(tasks) == 0:
        raise ValueError(
            "No valid rollout tasks generated. Dataset may be too short for the specified duration."
        )

    return tasks


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
