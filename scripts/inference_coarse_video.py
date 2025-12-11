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
from dataclasses import dataclass
import numpy as np
import pandas as pd
from functools import partial
from enum import auto, Enum

import cbottle.distributed as dist
from cbottle.dataclass_parser import Help, a, parse_args
import cbottle.inference
from cbottle.datasets import dataset_3d, samplers
from cbottle.datasets.dataset_3d import VARIABLE_CONFIGS
from cbottle.datasets.merged_dataset import TimeMergedMapStyle
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from cbottle.moe_utils import parse_model_paths
from cbottle.inference import VideoAutoregression, AutoregressionDiagnostics
from inference_coarse import Dataset, Sampler

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


def compute_autoregression_tasks(
    dataset_times: pd.DatetimeIndex,
    duration: int,
    time_step_hours: int,
    time_length: int,
    frame_step: int,
    conditioning_frames: list[int],
    start_time: str = "",
    end_time: str = "",
) -> list[tuple[int, int, pd.Timestamp]]:
    """
    Compute autoregression tasks: (start_idx, num_autoregression_steps, target_end_time)
    for each autoregression window.
    """
    dataset_start, dataset_end = dataset_times[0], dataset_times[-1]
    if start_time and end_time:
        range_start = pd.Timestamp(start_time)
        range_end = pd.Timestamp(end_time)
        if range_start < dataset_start or range_end > dataset_end:
            raise ValueError(
                f"Time range {range_start} to {range_end} is outside the dataset "
                f"time range {dataset_start} to {dataset_end}"
            )
    else:
        range_start, range_end = dataset_start, dataset_end

    # Generate autoregression start times based on duration
    autoregression_starts = pd.date_range(
        start=range_start,
        end=range_end,
        freq=f"{duration}D",
    )

    if len(autoregression_starts) == 0:
        raise ValueError(
            f"No valid autoregression start times in {range_start} to {range_end}. "
            f"Dataset: {dataset_start} to {dataset_end}, duration: {duration}"
        )

    new_frames_per_step = time_length - len(conditioning_frames)
    tasks: list[tuple[int, int, pd.Timestamp]] = []

    for i, t0 in enumerate(autoregression_starts):
        start_idx = dataset_times.get_indexer([t0], method="nearest")[0]

        if i + 1 < len(autoregression_starts):
            target_end = autoregression_starts[i + 1]
        else:
            target_end = range_end

        # Number of video frames needed to cover [t0, target_end)
        hours = (target_end - t0).total_seconds() / 3600.0
        frames_needed = int(np.ceil(hours / time_step_hours))

        if frames_needed <= time_length:
            num_steps = 0
        else:
            # Use ceil to ensure full coverage of duration
            num_steps = int(
                np.ceil((frames_needed - time_length) / new_frames_per_step)
            )

        # Check dataset runway and reduce num_steps if needed
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
            "No valid autoregression tasks generated. "
            "Dataset may be too short for the specified duration."
        )

    return tasks


def write_step_frames(
    writer: NetCDFWriter,
    diags: AutoregressionDiagnostics,
    autoregression_start_timestamp: float,
    target_end_timestamp: float,
    is_initialization: bool,
) -> None:
    """
    Select and write the appropriate subset of frames from an autoregression step.

    Parameters
    ----------
    writer : NetCDFWriter
        Open NetCDF writer used to append new frames to disk.
    diags : AutoregressionDiagnostics
        Minimal diagnostic bundle from the model, containing raw frames,
        timestamps, coordinate metadata, and conditioning information.
    autoregression_start_timestamp : float
        Absolute timestamp (seconds since epoch) marking the start of the
        entire autoregression sequence. Used to compute lead-time hours.
    target_end_timestamp : float
        Upper bound (exclusive) on timestamps to write. Frames occurring at or
        after this time are omitted.
    is_initialization : bool
        If True, this is the first autoregression step, and all frames
        (including conditioning frames) are written. Otherwise, only
        newly generated frames (source=1) are written.
    """
    frames = diags.frames
    timestamps = diags.timestamps
    device = frames.device

    time_length = frames.shape[2]

    # Frame source for each frame: 0=GT, 1=generated, 2=conditioning.
    frame_source = [
        (0 if is_initialization else 2) if i in diags.conditioning_indices else 1
        for i in range(time_length)
    ]

    if is_initialization:
        frames_to_write_indices = list(range(time_length))
    else:
        frames_to_write_indices = [i for i, src in enumerate(frame_source) if src == 1]

    # Lead times in hours relative to autoregression start.
    lead_time_hours = ((timestamps - autoregression_start_timestamp) / 3600.0).to(
        device
    )

    new_frames = frames[:, :, frames_to_write_indices, :]
    new_frame_source_flags = torch.tensor(
        [[frame_source[i] for i in frames_to_write_indices]],
        dtype=torch.int8,
        device=device,
    )
    new_lead_time_hours = lead_time_hours[:, frames_to_write_indices]
    new_timestamps = timestamps[:, frames_to_write_indices].to(device)

    frame_mask = new_timestamps[0] < target_end_timestamp

    writer.write_target(
        new_frames[:, :, frame_mask, :],
        diags.coords,
        new_timestamps[:, frame_mask],
        scalars={
            "frame_source_flag": new_frame_source_flags[:, frame_mask],
            "lead_time": new_lead_time_hours[:, frame_mask],
        },
    )


@dataclass(frozen=True)
class SamplerArgs:
    """Sampling and autoregression configuration."""

    min_samples: a[int, Help("Minimum number of samples.")] = -1
    start_from_noisy_image: a[bool, Help("Start from a noisy image")] = False
    sigma_min: a[float, Help("Minimum sigma value")] = 0.02
    sigma_max: a[float, Help("Maximum sigma value")] = 1000.0
    sampler: Sampler = Sampler.fibonacci
    mode: a[str, Help("options: infill, translate, sample, save_data")] = "sample"
    translate_dataset: a[
        str,
        Help(
            'Dataset to translate input to when using mode == "translate". '
            "era5 or icon."
        ),
    ] = "icon"
    bf16: a[bool, Help("Use bf16")] = False
    initialization_conditioning: a[
        FrameSelectionStrategy, Help("Conditioning strategy for initialization")
    ] = FrameSelectionStrategy.unconditional
    seed: int | None = None
    autoregression_duration: a[int, Help("Number of days to cover")] = 1
    autoregression_num_conditioning_frames: a[
        int,
        Help(
            "Number of conditioning frames during autoregression steps "
            "(contiguous from start)"
        ),
    ] = 1


@dataclass
class CLI:
    state_path: a[
        str,
        Help(
            "Direct paths to model state file (comma-separated for MoE). "
            "If not provided, uses checkpoint_root + named model 'cbottle-3d-video'"
        ),
    ]
    output_path: a[str, Help("Path to the output directory")]
    checkpoint_root: a[
        str, Help("Root directory for named models (used if state_path not provided)")
    ] = ""
    sigma_thresholds: a[str, Help("Comma-separated sigma thresholds for MoE")] = (
        "100.0,10.0"
    )
    dataset: Dataset = Dataset.icon
    data_split: str = ""
    sample: SamplerArgs = SamplerArgs()
    hpx_level: int = 6
    start_time: a[str, Help("Start time")] = ""
    end_time: a[str, Help("End time")] = "2018-12-31"
    time_step: a[int, Help("Hours between frames")] = 6


def save_inferences(
    model: cbottle.inference.CBottle3d,
    dataset: TimeMergedMapStyle,
    output_path: str,
    *,
    attrs: dict | None = None,
    hpx_level: int,
    config: SamplerArgs,
    rank: int,
    world_size: int,
    autoregression_tasks: list[tuple[int, int, pd.Timestamp]] | None = None,
    autoregression_num_conditioning: int | None = None,
) -> None:
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

    if not autoregression_tasks:
        logger.info("No autoregression tasks provided; nothing to run.")
        return

    # Setup tasks
    if config.sampler == Sampler.fibonacci:
        if config.min_samples > 0:
            sampled_positions = samplers.subsample(
                autoregression_tasks,
                min_samples=max(config.min_samples, world_size),
            )
            selected_autoregression_tasks = [
                autoregression_tasks[i] for i in sampled_positions
            ]
        else:
            selected_autoregression_tasks = autoregression_tasks
    elif config.sampler == Sampler.all:
        selected_autoregression_tasks = autoregression_tasks
        if config.min_samples > 0:
            selected_autoregression_tasks = selected_autoregression_tasks[
                : config.min_samples
            ]
    else:
        raise ValueError(f"Unsupported sampler: {config.sampler}")

    my_autoregression_tasks = samplers.distributed_split(selected_autoregression_tasks)

    match config.mode:
        case "save_data":

            def evaluation(batch):
                out, coords = model.denormalize(batch)
                return out, coords, out
        case "translate":
            evaluation = partial(
                model.translate,
                dataset=config.translate_dataset,
                return_untransformed=True,
            )
        case "infill":
            evaluation = partial(
                model.infill,
                return_untransformed=True,
            )
        case "sample":
            evaluation = partial(
                model.sample,
                start_from_noisy_image=config.start_from_noisy_image,
                return_untransformed=True,
            )
        case _:
            raise NotImplementedError(config.mode)

    sample_kwargs = {}
    if config.mode != "save_data":
        sample_kwargs["bf16"] = config.bf16
    if config.mode == "sample" and config.seed is not None:
        sample_kwargs["seed"] = config.seed

    autoregression_engine = VideoAutoregression(
        model=model,
        dataset=dataset,
        evaluation=evaluation,
        sample_kwargs=sample_kwargs,
    )
    dataset_times = dataset.times
    initialization_frames = config.initialization_conditioning.get_keep_frames(
        model.time_length
    )

    total_start_time = time.time()
    for task_idx, (start_idx, num_autoregression_steps, target_end_time) in enumerate(
        my_autoregression_tasks
    ):
        start_time_ts = dataset_times[start_idx]
        batch = dataset[start_idx]

        target_frames = batch["target"]
        init_frames = {idx: target_frames[:, idx, :] for idx in initialization_frames}

        # Initialization step
        state, diags = autoregression_engine.initialize(
            start_time_ts, frames=init_frames
        )

        target_end_timestamp = target_end_time.value / 1e9
        # state.autoregression_start_time is assumed to be a pd.Timestamp
        autoregression_start_timestamp = state.autoregression_start_time.value / 1e9

        write_step_frames(
            writer,
            diags,
            autoregression_start_timestamp,
            target_end_timestamp,
            is_initialization=True,
        )

        pbar = tqdm.tqdm(
            range(num_autoregression_steps),
            desc=(f"Rank {rank} Task {task_idx + 1}/{len(my_autoregression_tasks)}"),
            disable=(rank != 0),
        )

        # Subsequent autoregressive steps
        for _ in pbar:
            state, diags = autoregression_engine.step_forward(
                state, num_conditioning_frames=autoregression_num_conditioning
            )
            write_step_frames(
                writer,
                diags,
                autoregression_start_timestamp,
                target_end_timestamp,
                is_initialization=False,
            )

    total_elapsed = time.time() - total_start_time
    logger.info(
        f"\nCompleted {len(my_autoregression_tasks)} autoregression windows on "
        f"rank {rank} in {total_elapsed / 60:.1f} min. "
        f"Output: {output_path}"
    )

    if dist.get_world_size() > 1:
        torch.distributed.barrier()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", "Cannot do a zero-copy NCHW to NHWC")
    args = parse_args(CLI, convert_underscore_to_hyphen=False)

    state_path = args.state_path
    logging.info(f"Using {state_path} model state path")

    dist.init()

    sigma_max = args.sample.sigma_max
    sigma_min = args.sample.sigma_min
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

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
        )
    else:
        model = cbottle.inference.load(
            "cbottle-3d-video",
            root=args.checkpoint_root if args.checkpoint_root else None,
            device=device,
            sigma_min=args.sample.sigma_min,
            sigma_max=args.sample.sigma_max,
        )

    batch_info = model.coords.batch_info
    variables = dataset_3d.guess_variable_config(batch_info.channels)

    time_length = model.time_length
    time_step = args.time_step

    initialization_conditioning = args.sample.initialization_conditioning

    if (
        args.dataset.name == "amip"
        and initialization_conditioning != FrameSelectionStrategy.unconditional
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
        time_length=model.time_length,
        frame_masker=None,
        variable_config=VARIABLE_CONFIGS[variables],
        map_style=True,
    )

    autoregression_num_conditioning = args.sample.autoregression_num_conditioning_frames

    autoregression_tasks = compute_autoregression_tasks(
        dataset_times=dataset.times,
        duration=args.sample.autoregression_duration,
        time_step_hours=args.time_step,
        time_length=time_length,
        frame_step=dataset.frame_step,
        conditioning_frames=list(range(autoregression_num_conditioning)),
        start_time=args.start_time,
        end_time=args.end_time,
    )

    description = (
        "Video inference" if args.sample.mode != "save_data" else "Ground truth data"
    )
    attrs = {
        "description": description,
        "dataset": args.dataset.name,
        "initialization_conditioning": initialization_conditioning.name,
        "autoregression_duration": args.sample.autoregression_duration,
        "autoregression_num_conditioning_frames": (autoregression_num_conditioning),
        "time_step_hours": args.time_step,
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
            f"\n  • masking:     {args.sample.initialization_conditioning}"
            f"\n  • number of autoregression tasks: {len(autoregression_tasks)}"
            f"\n  • number of autoregression steps per task: "
            f"{autoregression_tasks[0][1] + 1}"
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
        autoregression_tasks=autoregression_tasks,
        autoregression_num_conditioning=autoregression_num_conditioning,
    )


if __name__ == "__main__":
    main()
