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
Autoregressive video generation with sliding window rollout.
"""

import torch
import logging
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from cbottle.inference import CBottle3d, Coords
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets.merged_dataset import TimeMergedMapStyle

logger = logging.getLogger(__name__)


@dataclass
class RolloutStep:
    """Result from one rollout step (batch_size=1)."""

    frames: torch.Tensor  # (1, C, T, npix) denormalized
    timestamps: torch.Tensor  # (1, T)
    frame_source: list[int]  # 0=GT, 1=generated, 2=conditioning
    coords: Coords
    step_index: int
    is_seed: bool = False
    cumulative_frames_written: int = 0  # number of frames written before this step

    @property
    def device(self) -> torch.device:
        return self.frames.device

    @property
    def batch_info(self):
        return self.coords.batch_info

    @property
    def frames_to_write_indices(self) -> list[int]:
        """Indices of frames to write (seed: all, rollout: only generated)."""
        if self.is_seed:
            return list(range(len(self.frame_source)))
        else:
            return [i for i, src in enumerate(self.frame_source) if src == 1]

    @property
    def new_frames(self) -> torch.Tensor:
        """Newly generated frames (when not seed step, exclude conditioning frames)

        Returns:
            (1, C, num_frames, npix)
        """
        return self.frames[:, :, self.frames_to_write_indices, :]

    @property
    def new_timestamps(self) -> torch.Tensor:
        """Timestamps for newly generated frames"""
        return self.timestamps[:, self.frames_to_write_indices].to(self.device)

    @property
    def frame_source_flags(self) -> torch.Tensor:
        """Frame source flags (1, num_frames)."""
        indices = self.frames_to_write_indices
        return torch.tensor(
            [[self.frame_source[i] for i in indices]], dtype=torch.int8
        ).to(self.device)

    @property
    def lead_time_hours(self) -> torch.Tensor:
        """Lead times in hours relative to rollout start."""
        indices = self.frames_to_write_indices
        num_frames = len(indices)

        time_delta_per_frame = self.batch_info.get_time_delta(1)
        hours_per_frame = time_delta_per_frame.total_seconds() / 3600.0
        base_offset_hours = self.cumulative_frames_written * hours_per_frame

        lead_times = torch.zeros((1, num_frames))
        for i, frame_idx in enumerate(indices):
            delta = self.batch_info.get_time_delta(frame_idx)
            lead_times[0, i] = delta.total_seconds() / 3600.0 + base_offset_hours

        return lead_times.to(self.device)


class VideoRollout:
    """
    Autoregressive video generation. Only supports batch_size=1.

    Example:
        rollout = VideoRollout(model, dataset)
        rollout.seed_generation(time, frames={0: frame_0, 1: frame_1})  # init with conditioning
        step = rollout.step_forward([0, 1])  # generate using frames 0,1 as conditioning

        for _ in range(10):
            step = rollout.step_forward([0]) # autoregressively use last frame of prev step as initial frame for next
    """

    def __init__(
        self,
        model: CBottle3d,
        dataset: TimeMergedMapStyle,
        sample_kwargs: Optional[dict] = None,
    ):
        """
        Args:
            model: CBottle3d inference model
            dataset: TimeMergedMapStyle dataset (without frame masking)
            sample_kwargs: Optional kwargs for model.sample()
        """
        self.model = model
        self.dataset = dataset
        self.sample_kwargs = sample_kwargs or {}

        self.time_length = model.time_length
        self.batch_info = dataset.batch_info
        self.device = next(model.net.parameters()).device

        self.dataset_times = dataset.times.to_numpy()

        # Current state (normalized, HPXPAD order)
        self.current_frames = None
        self.current_timestamps = None
        self.current_start_time = None
        self.step_count = 0
        self.cumulative_frames_written = 0
        self._is_seed = False

        self._seed_frames = {}
        self._seed_frame_source = None
        self.seed_state = None

    def seed_generation(
        self,
        start_time: pd.Timestamp,
        frames: Optional[dict[int, torch.Tensor]] = None,
    ) -> None:
        """
        Initialize rollout state.

        Args:
            start_time: Time for frame 0
            frames: Optional conditioning frames {idx: tensor (1, C, npix)}.
                   Tensors should be normalized and in HEALPIX_PAD_XY order.
        """
        frames = frames or {}

        for idx in frames.keys():
            if idx < 0 or idx >= self.time_length:
                raise ValueError(
                    f"Frame index {idx} out of range [0, {self.time_length})"
                )

        self._seed_frames = frames
        self.current_frames = None
        self.current_timestamps = None
        self.current_start_time = start_time
        self.step_count = 0
        self.cumulative_frames_written = 0
        self._is_seed = True

        self._seed_frame_source = [
            0 if i in frames else -1 for i in range(self.time_length)
        ]
        self.seed_state = (frames.copy(), start_time, 0)

        logger.debug(
            f"Seed generation initialized at {start_time}, {len(frames)} conditioning frames"
        )

    def step_forward(self, conditioning_frames: list[int]) -> RolloutStep:
        """Generate next frames moving forward in time."""
        return self._step("forward", conditioning_frames)

    def step_backward(self, conditioning_frames: list[int]) -> RolloutStep:
        """Generate frames moving backward in time."""
        return self._step("backward", conditioning_frames)

    def _validate_conditioning(self, conditioning_frames: list[int], direction: str):
        """Validate we have the frames needed for conditioning."""
        for idx in conditioning_frames:
            if idx < 0 or idx >= self.time_length:
                raise ValueError(f"Conditioning frame {idx} out of range")

        if self._is_seed:
            for idx in conditioning_frames:
                if idx not in self._seed_frames:
                    raise RuntimeError(
                        f"Missing conditioning frame {idx} for seed generation"
                    )
        else:
            # Rollout step: need frames that will slide to conditioning positions
            num_cond = len(conditioning_frames)
            if direction == "forward":
                needed = list(range(self.time_length - num_cond, self.time_length))
            else:
                needed = list(range(num_cond))

            for idx in needed:
                if self.current_frames is None:
                    raise RuntimeError(
                        f"Missing frame {idx} needed for {direction} rollout"
                    )

    def reset_to_seed(self):
        """Reset to seed generation state"""
        if self.seed_state is None:
            raise RuntimeError("No seed state. Call seed_generation() first.")
        frames, start_time, cum_frames = self.seed_state
        self._seed_frames = frames.copy()
        self.current_frames = None
        self.current_timestamps = None
        self.current_start_time = start_time
        self.cumulative_frames_written = cum_frames
        self.step_count = 0
        self._is_seed = True
        logger.info("Reset to seed generation")

    def _step(self, direction: str, conditioning_frames: list[int]) -> RolloutStep:
        """Execute generation step (seed or rollout)."""
        if self.current_start_time is None:
            raise RuntimeError("Must call seed_generation() first")

        self._validate_conditioning(conditioning_frames, direction)

        is_seed = self._is_seed

        if is_seed:
            new_start_time = self.current_start_time
            batch = self._load_batch_at_time(new_start_time)

            for idx in self._seed_frames.keys():
                batch["target"][:, :, idx, :] = self._seed_frames[idx]

            timestamps = self._compute_timestamps_from_batch(batch)
            frame_source = [
                0 if i in conditioning_frames else 1 for i in range(self.time_length)
            ]
            step_index = 0
        else:
            # Rollout step: advance time and slide window
            sign = 1 if direction == "forward" else -1
            self.step_count += sign

            num_new_frames = self.time_length - len(conditioning_frames)
            time_delta = (
                self.batch_info.get_time_delta(1).total_seconds() * num_new_frames
            )
            new_start_time = self.current_start_time + pd.Timedelta(
                seconds=time_delta * sign
            )

            batch = self._load_batch_at_time(new_start_time)

            for i, cond_idx in enumerate(conditioning_frames):
                source_idx = (
                    (self.time_length - len(conditioning_frames) + i)
                    if direction == "forward"
                    else i
                )
                batch["target"][:, :, cond_idx, :] = self.current_frames[
                    :, :, source_idx, :
                ]

            timestamps = self._compute_timestamps_from_batch(batch)
            frame_source = [
                2 if i in conditioning_frames else 1 for i in range(self.time_length)
            ]
            step_index = self.step_count

        batch = FrameMasker(keep_frames=conditioning_frames)(batch)
        frames_processed, coords, frames_raw = self.model.sample(
            batch, return_untransformed=True, **self.sample_kwargs
        )

        # Store raw frames for conditioning next step
        self.current_frames = frames_raw
        self.current_timestamps = timestamps
        self.current_start_time = new_start_time

        if is_seed:
            self._is_seed = False

        step = RolloutStep(
            frames_processed,
            timestamps,
            frame_source,
            coords,
            step_index,
            is_seed=is_seed,
            cumulative_frames_written=self.cumulative_frames_written,
        )

        self.cumulative_frames_written += len(step.frames_to_write_indices)
        return step

    def _load_batch_at_time(self, start_time: pd.Timestamp) -> dict:
        """Load batch from dataset at start_time."""
        start_time_np = start_time.to_datetime64()
        matching_indices = np.where(self.dataset_times == start_time_np)[0]

        if len(matching_indices) == 0:
            raise ValueError(f"Time {start_time} not found in dataset")

        index = matching_indices[0]
        sample = self.dataset[index]

        if sample["target"].ndim != 4:
            sample = torch.utils.data.default_collate([sample])

        return self._move_to_device(sample)

    def _move_to_device(self, batch: dict) -> dict:
        """Move all batch tensors to device."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _compute_timestamps_from_batch(self, batch: dict) -> torch.Tensor:
        """Compute timestamps (B, T) from batch."""
        ts = batch["timestamp"]
        frame_offsets = torch.tensor(
            [
                self.batch_info.get_time_delta(i).total_seconds()
                for i in range(self.time_length)
            ],
            device=ts.device,
        )
        return ts.unsqueeze(-1) + frame_offsets.unsqueeze(0)
