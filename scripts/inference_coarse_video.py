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

import cbottle.distributed as dist
from cbottle.dataclass_parser import Help, a, parse_args
import cbottle.inference
from cbottle.datasets import dataset_3d, samplers
from cbottle.datasets.dataset_3d import VARIABLE_CONFIGS
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets.merged_dataset import TimeMergedMapStyle
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from cbottle.moe_utils import parse_model_paths
from cbottle.autoregression import FrameSelectionStrategy, AutoregressionRuntimeConfig
from inference_coarse import Dataset, Sampler, get_requested_times

logger = logging.getLogger(__name__)


def extract_batch_for_single_autoregression(
    batch, autoregression_cfg, autoregression_step
):
    """
    Extract a per-autoregression-window batch from a longer batch.

    Parameters
    ----------
    batch : dict
        A dictionary containing batched tensors for fields such as
        "target", "condition", "second_of_day", "day_of_year", "labels",
        "last_output", etc.
        Time-dependent tensors are assumed to have time as the last dimension.
    autoregression_cfg : AutoregressionRuntimeConfig
        Configuration object defining start/end indices and conditioning frames.
    autoregression_step : int
        Index of the current autoregression step.

    Returns
    -------
    dict
        A new batch dictionary containing tensors sliced according to the
        current autoregression window, and with the "condition" field updated
        to include model outputs from previous steps when autoregression_step > 0.
    """

    batch_window = {}
    start, end = autoregression_cfg.start_end_pairs[autoregression_step]
    num_conditioning_frames = autoregression_cfg.num_conditioning_frames
    for key, value in batch.items():
        # resemble conditions
        if key == "labels":
            batch_window[key] = value
        elif key in ["second_of_day", "day_of_year"]:
            batch_window[key] = value[:, start:end]
        elif key == "condition" and autoregression_step != 0:
            condition_window = value[:, :, start:end]
            num_channels = batch["target"].shape[1]
            # set mask
            condition_window[:, -1:, :num_conditioning_frames] = 1
            # clear frame condition
            condition_window[
                :,
                :num_channels,
            ] = 0
            # set frame condition
            condition_window[:, :num_channels, :num_conditioning_frames] = batch[
                "last_output"
            ][:, :, -num_conditioning_frames:]
            batch_window[key] = condition_window
        # target or condition for the first autoregression step
        else:
            batch_window[key] = value[:, :, start:end]
    return batch_window


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
    autoregression_enabled: a[bool, Help("Use sliding-window autoregression")] = False
    autoregression_duration: a[int, Help("Number of days to cover")] = False
    autoregression_num_conditioning_frames: a[
        int,
        Help(
            "Number of conditioning frames during autoregression steps (contiguous from start)"
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
    attrs=None,
    hpx_level: int,
    config: SamplerArgs,
    rank: int,
    world_size: int,
    keep_frames: list[int] = [],
    autoregression_cfg: AutoregressionRuntimeConfig | None = None,
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
    time_length = autoregression_cfg.dataset_time_length
    loader = torch.utils.data.DataLoader(
        pin_memory=True,
        dataset=dataset,
        batch_size=batch_size,
        sampler=tasks,
    )
    if autoregression_cfg is None:
        num_autoregressions = 1
    else:
        num_autoregressions = autoregression_cfg.num_autoregressions
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
        autoregression_outputs = []
        for autoregression_step in tqdm.tqdm(
            range(num_autoregressions), disable=rank != 0, leave=False
        ):
            if num_autoregressions == 1:
                batch_window = batch
            else:
                batch_window = extract_batch_for_single_autoregression(
                    batch, autoregression_cfg, autoregression_step
                )

            match config.mode:
                case "save_data":
                    out, coords = model.denormalize(batch_window)
                case "translate":
                    out, coords = model.translate(
                        batch_window, dataset=config.translate_dataset
                    )
                case "infill":
                    out, coords = model.infill(batch_window)
                case "sample":
                    out, coords = model.sample(
                        batch_window,
                        start_from_noisy_image=config.start_from_noisy_image,
                        bf16=config.bf16,
                    )
                case _:
                    raise NotImplementedError(config.mode)
            # save output in batch for auto-regression
            batch["last_output"] = model.normalize(out)
            if autoregression_step == 0:
                autoregression_outputs.append(out)
            else:
                autoregression_outputs.append(out[:, :, 1:])

        autoregression_outputs = torch.cat(autoregression_outputs, dim=2)
        if config.mode == "save_data":
            frame_source = torch.full((batch_size, time_length), 0, dtype=torch.int8)
        else:
            frame_source = torch.full((batch_size, time_length), 1, dtype=torch.int8)
            frame_source[:, keep_frames] = 2

        scalars = {
            "frame_source_flag": frame_source,
            "lead_time": lead_time_hours,
        }
        writer.write_target(autoregression_outputs, coords, timestamps, scalars=scalars)

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

    autoregression_cfg = AutoregressionRuntimeConfig(
        enabled=args.sample.autoregression_enabled,
        duration=args.sample.autoregression_duration,  # days
        num_conditioning_frames=args.sample.autoregression_num_conditioning_frames,
        model_time_length=time_length,
        time_step=time_step,
    )

    dataset = dataset_3d.get_dataset(
        split=args.data_split,
        dataset=args.dataset.name,
        sst_input=True,
        infinite=False,
        shuffle=False,
        time_step=time_step,
        time_length=autoregression_cfg.dataset_time_length,
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
            f"\n  • total time length: {autoregression_cfg.dataset_time_length}"
            f"\n  • number of autoregression steps: {autoregression_cfg.num_autoregressions}"
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
        autoregression_cfg=autoregression_cfg,
    )


if __name__ == "__main__":
    main()
