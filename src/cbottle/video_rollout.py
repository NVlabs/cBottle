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
import datetime
from typing import Optional
from dataclasses import dataclass

from cbottle.inference import CBottle3d, Coords
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets.merged_dataset import TimeMergedMapStyle

logger = logging.getLogger(__name__)


@dataclass
class RolloutDiagnostics:
    """Results from one rollout step (batch_size=1)."""

    frames: torch.Tensor  # (1, C, T, npix) denormalized
    timestamps: torch.Tensor  # (1, T) absolute timestamps in seconds
    conditioning_indices: list[int]  # Indices of conditioning frames
    coords: Coords
    rollout_start_timestamp: float  # Start of rollout in seconds
    is_initialization: bool = False

    def __post_init__(self):
        time_length = self.frames.shape[2]

        # Frame source for each frame: 0=GT, 1=generated, 2=conditioning.
        self.frame_source = [
            (0 if self.is_initialization else 2)
            if i in self.conditioning_indices
            else 1
            for i in range(time_length)
        ]

        if self.is_initialization:
            self.frames_to_write_indices = list(range(time_length))
        else:
            self.frames_to_write_indices = [
                i for i, src in enumerate(self.frame_source) if src == 1
            ]

    @property
    def device(self) -> torch.device:
        return self.frames.device

    @property
    def batch_info(self):
        return self.coords.batch_info

    @property
    def lead_time_hours(self) -> torch.Tensor:
        """Lead times in hours relative to rollout start."""
        return ((self.timestamps - self.rollout_start_timestamp) / 3600.0).to(
            self.device
        )

    @property
    def new_frames(self) -> torch.Tensor:
        """New frames from this rollout step (excludes previously generated frames used for conditioning).

        Returns:
            (1, C, num_frames, npix)
        """
        return self.frames[:, :, self.frames_to_write_indices, :]

    @property
    def new_timestamps(self) -> torch.Tensor:
        """Timestamps corresponding to new frames from this rollout step"""
        return self.timestamps[:, self.frames_to_write_indices].to(self.device)

    @property
    def new_frame_source_flags(self) -> torch.Tensor:
        """Frame source flags for new frames from this rollout step"""
        return torch.tensor(
            [[self.frame_source[i] for i in self.frames_to_write_indices]],
            dtype=torch.int8,
        ).to(self.device)

    @property
    def new_lead_time_hours(self) -> torch.Tensor:
        """Lead times for the new frames relative to start of this rollout step."""
        return self.lead_time_hours[:, self.frames_to_write_indices]


@dataclass
class VideoRolloutState:
    current_frames: torch.Tensor
    window_start_time: pd.Timestamp
    rollout_start_time: pd.Timestamp  # Fixed reference time for the entire rollout


class VideoRollout:
    """Autoregressive video generation forwards or backwards in time. Only supports batch_size=1.

    Example:
        rollout = VideoRollout(model, dataset)

        # Initialize with arbitrary conditioning frames
        state, diags = rollout.initialize(time, frames={0: frame_0, 1: frame_1})

        # Continue autoregressive rollout
        for _ in range(10):
            state, diags = rollout.step_forward(state, num_conditioning_frames=1)
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

    @property
    def time_step(self) -> datetime.timedelta:
        return self.batch_info.get_time_delta(1)

    def initialize(
        self,
        start_time: pd.Timestamp,
        frames: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[VideoRolloutState, RolloutDiagnostics]:
        """Initialize and generate first state/diagnostics.

        Args:
            start_time: Time for frame 0
            frames: Optional conditioning frames {idx: tensor (1, C, npix)}.
                   Tensors should be normalized and in HEALPIX_PAD_XY order.
                   Can be arbitrary indices for infilling.
        """
        frames = frames or {}

        for idx in frames.keys():
            if idx < 0 or idx >= self.time_length:
                raise ValueError(
                    f"Frame index {idx} out of range [0, {self.time_length})"
                )

        batch = self._load_batch_at_time(start_time)
        batch, conditioning_indices = self._insert_conditioning_frames(batch, frames)

        state, diags = self._generate_frames(
            batch=batch,
            start_time=start_time,
            rollout_start_time=start_time,
            conditioning_indices=conditioning_indices,
            is_initialization=True,
        )

        logger.debug(f"Initialized at {start_time}, {len(frames)} conditioning frames")

        return state, diags

    def step_forward(
        self,
        state: VideoRolloutState,
        num_conditioning_frames: int = 1,
    ) -> tuple[VideoRolloutState, RolloutDiagnostics]:
        """Step forward in time. Last N frames slide to first N positions"""
        return self._step(state, num_conditioning_frames, direction="forward")

    def step_backward(
        self,
        state: VideoRolloutState,
        num_conditioning_frames: int = 1,
    ) -> tuple[VideoRolloutState, RolloutDiagnostics]:
        """Step backward in time. First N frames slide to last N positions"""
        return self._step(state, num_conditioning_frames, direction="backward")

    def _step(
        self,
        state: VideoRolloutState,
        num_conditioning_frames: int,
        direction: str,
    ) -> tuple[VideoRolloutState, RolloutDiagnostics]:
        if num_conditioning_frames < 0 or num_conditioning_frames >= self.time_length:
            raise ValueError(
                f"num_conditioning_frames must be in [0, {self.time_length}), "
                f"got {num_conditioning_frames}"
            )

        sign = 1 if direction == "forward" else -1
        num_new_frames = self.time_length - num_conditioning_frames
        new_start_time = (
            state.window_start_time + self.time_step * num_new_frames * sign
        )

        batch = self._load_batch_at_time(new_start_time)
        batch, conditioning_indices = self._slide_conditioning_frames(
            batch, state.current_frames, num_conditioning_frames, direction
        )

        new_state, diags = self._generate_frames(
            batch=batch,
            start_time=new_start_time,
            rollout_start_time=state.rollout_start_time,
            conditioning_indices=conditioning_indices,
            is_initialization=False,
        )

        return new_state, diags

    def _insert_conditioning_frames(
        self, batch: dict, frames: dict[int, torch.Tensor]
    ) -> tuple[dict, list[int]]:
        conditioning_indices = list(frames.keys())
        for idx in conditioning_indices:
            batch["target"][:, :, idx, :] = frames[idx]
        return batch, conditioning_indices

    def _slide_conditioning_frames(
        self,
        batch: dict,
        prev_frames: torch.Tensor,
        num_conditioning_frames: int,
        direction: str,
    ) -> tuple[dict, list[int]]:
        """Slide conditioning frames from previous step into new batch."""
        if direction == "forward":
            indices = list(range(num_conditioning_frames))
            source_start = self.time_length - num_conditioning_frames
            batch["target"][:, :, :num_conditioning_frames, :] = prev_frames[
                :, :, source_start:, :
            ]
        else:
            target_start = self.time_length - num_conditioning_frames
            indices = list(range(target_start, self.time_length))
            batch["target"][:, :, target_start:, :] = prev_frames[
                :, :, :num_conditioning_frames, :
            ]

        return batch, indices

    def _generate_frames(
        self,
        batch: dict,
        start_time: pd.Timestamp,
        rollout_start_time: pd.Timestamp,
        conditioning_indices: list[int],
        is_initialization: bool,
    ) -> tuple[VideoRolloutState, RolloutDiagnostics]:
        """Generate frames and create both state and diagnostics."""
        timestamps = self._compute_timestamps_from_batch(batch)

        batch = self._apply_frame_masking(batch, conditioning_indices)
        frames_processed, coords, frames_raw = self.model.sample(
            batch, return_untransformed=True, **self.sample_kwargs
        )

        diags = RolloutDiagnostics(
            frames_processed,
            timestamps,
            conditioning_indices,
            coords,
            rollout_start_timestamp=rollout_start_time.value / 1e9,
            is_initialization=is_initialization,
        )

        state = VideoRolloutState(
            current_frames=frames_raw.clone(),
            window_start_time=start_time,
            rollout_start_time=rollout_start_time,
        )

        return state, diags

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
            [i * self.time_step.total_seconds() for i in range(self.time_length)],
            device=ts.device,
        )
        return ts.unsqueeze(-1) + frame_offsets.unsqueeze(0)

    def _apply_frame_masking(
        self, batch: dict, conditioning_indices: list[int]
    ) -> dict:
        return FrameMasker(keep_frames=conditioning_indices)(batch)
