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
Autoregressive video generation with sliding-window autoregression.
"""

import logging
import datetime
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import CBottle3d, Coords
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets.merged_dataset import TimeMergedMapStyle

logger = logging.getLogger(__name__)


@dataclass
class AutoregressionDiagnostics:
    """
    Minimal data bundle returned from a single autoregression step (batch_size=1).

    This structure intentionally contains only raw model outputs and metadata
    required for downstream writing.
    """

    frames: torch.Tensor  # (1, C, T, npix) denormalized
    timestamp: (
        torch.Tensor
    )  # (1, ) absolute start timestamp in seconds for the current autoregression step
    conditioning_indices: list[int]  # Indices of conditioning frames
    coords: "Coords"


@dataclass
class VideoAutoregressionState:
    """
    State of the sliding-window autoregression.

    Attributes
    ----------
    current_frames : torch.Tensor
        Tensor of shape (1, C, T, npix) representing the most recent window
        of frames in normalized model space.
    window_start_time : pd.Timestamp
        Timestamp corresponding to the first frame in `current_frames`.
    autoregression_start_time : pd.Timestamp
        Fixed reference time for the entire autoregression sequence. Used
        by downstream code to compute lead times.
    """

    current_frames: torch.Tensor
    window_start_time: pd.Timestamp
    autoregression_start_time: (
        pd.Timestamp
    )  # Fixed reference time for the entire autoregression


class VideoAutoregression:
    """
    Autoregressive video generation forwards or backwards in time.
    Only supports batch_size=1.

    Example
    -------
    >>> autoreg = VideoAutoregression(model, dataset, evaluation)
    >>> state, diags = autoreg.initialize(time, frames={0: frame_0, 1: frame_1})
    >>> for _ in range(10):
    ...     state, diags = autoreg.step_forward(state, num_conditioning_frames=1)
    """

    def __init__(
        self,
        model: "CBottle3d",
        evaluation: Callable[
            [Dict[str, torch.Tensor]], Tuple[torch.Tensor, "Coords", torch.Tensor]
        ],
        dataset: TimeMergedMapStyle,
        sample_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        model : CBottle3d
            CBottle3d inference model.
        evaluation : Callable
            Callable that takes a batch dict and returns
            (frames_processed, coords, frames_raw).
        dataset : TimeMergedMapStyle
            Dataset providing batches without frame masking.
        sample_kwargs : dict, optional
            Optional keyword arguments forwarded to the sampling call.
        """
        self.model = model
        self.evaluation = evaluation
        self.dataset = dataset
        self.sample_kwargs = sample_kwargs or {}
        self.time_length = model.time_length
        self.batch_info = dataset.batch_info
        self.device = next(model.net.parameters()).device

        self.dataset_times = dataset.times.to_numpy()

    @property
    def time_step(self) -> datetime.timedelta:
        """Time difference between consecutive frames in the dataset."""
        return self.batch_info.get_time_delta(1)

    def initialize(
        self,
        start_time: pd.Timestamp,
        frames: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[VideoAutoregressionState, AutoregressionDiagnostics]:
        """
        Initialize the autoregression and generate the first state/diagnostics.

        Parameters
        ----------
        start_time : pd.Timestamp
            Time corresponding to frame index 0 in the initial window.
        frames : dict[int, torch.Tensor], optional
            Optional conditioning frames {idx: tensor (1, C, npix)}. Tensors
            should be normalized and in HEALPIX_PAD_XY order. Indices can
            be arbitrary for infilling.

        Returns
        -------
        state : VideoAutoregressionState
            Initial autoregression state.
        diags : AutoregressionDiagnostics
            Diagnostics for the initial window.
        """
        frames = frames or {}
        for idx in frames.keys():
            if idx < 0 or idx >= self.time_length:
                raise ValueError(
                    f"Frame index {idx} out of range [0, {self.time_length})"
                )

        batch = self._load_batch_at_time(start_time)
        condition_to_insert, conditioning_indices = self._insert_conditioning_frames(
            batch, frames
        )

        state, diags = self._generate_frames(
            batch=batch,
            start_time=start_time,
            autoregression_start_time=start_time,
            conditioning_indices=conditioning_indices,
            condition_to_insert=condition_to_insert,
        )

        return state, diags

    def step_forward(
        self,
        state: VideoAutoregressionState,
        num_conditioning_frames: int = 1,
    ) -> tuple[VideoAutoregressionState, AutoregressionDiagnostics]:
        """
        Step forward in time.

        The last `num_conditioning_frames` from the previous window become the
        conditioning frames at the beginning of the new window.
        """
        return self._step(state, num_conditioning_frames, direction="forward")

    def step_backward(
        self,
        state: VideoAutoregressionState,
        num_conditioning_frames: int = 1,
    ) -> tuple[VideoAutoregressionState, AutoregressionDiagnostics]:
        """
        Step backward in time.

        The first `num_conditioning_frames` from the previous window become the
        conditioning frames at the end of the new window.
        """
        return self._step(state, num_conditioning_frames, direction="backward")

    def _step(
        self,
        state: VideoAutoregressionState,
        num_conditioning_frames: int,
        direction: str,
    ) -> tuple[VideoAutoregressionState, AutoregressionDiagnostics]:
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
        condition_to_insert, conditioning_indices = self._slide_conditioning_frames(
            state.current_frames, num_conditioning_frames, direction
        )

        new_state, diags = self._generate_frames(
            batch=batch,
            start_time=new_start_time,
            autoregression_start_time=state.autoregression_start_time,
            conditioning_indices=conditioning_indices,
            condition_to_insert=condition_to_insert,
        )
        return new_state, diags

    def _insert_conditioning_frames(
        self, batch: dict, frames: dict[int, torch.Tensor]
    ) -> tuple[dict, list[int]]:
        conditioning_indices = list(frames.keys())
        new_condition = torch.zeros_like(batch["target"])
        for idx in conditioning_indices:
            new_condition[:, :, idx, :] = frames[idx]
        return new_condition, conditioning_indices

    def _slide_conditioning_frames(
        self,
        prev_frames: torch.Tensor,
        num_conditioning_frames: int,
        direction: str,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Slide conditioning frames from the previous window into a new tensor.

        Returns
        -------
        condition_tensor : torch.Tensor
            Tensor with conditioning frames placed at the appropriate positions.
        indices : list[int]
            Indices in the new window that are considered conditioning frames.
        """
        slided_condition = torch.zeros_like(prev_frames)
        if num_conditioning_frames == 0:
            return slided_condition, []

        if direction == "forward":
            indices = list(range(num_conditioning_frames))
            source_start = self.time_length - num_conditioning_frames
            slided_condition[:, :, :num_conditioning_frames, :] = prev_frames[
                :, :, source_start:, :
            ]
        else:
            target_start = self.time_length - num_conditioning_frames
            indices = list(range(target_start, self.time_length))
            slided_condition[:, :, target_start:, :] = prev_frames[
                :, :, :num_conditioning_frames, :
            ]

        return slided_condition, indices

    def _generate_frames(
        self,
        batch: dict,
        start_time: pd.Timestamp,
        autoregression_start_time: pd.Timestamp,
        conditioning_indices: list[int],
        condition_to_insert: torch.Tensor | None,
    ) -> tuple[VideoAutoregressionState, AutoregressionDiagnostics]:
        """
        Generate frames for a single autoregression window and create both
        state and diagnostics.
        """
        batch = self._apply_frame_masking(
            batch, conditioning_indices, condition_to_insert
        )
        frames_processed, coords, frames_raw = self.evaluation(
            batch, **self.sample_kwargs
        )

        diags = AutoregressionDiagnostics(
            frames=frames_processed,
            timestamp=batch["timestamp"],
            conditioning_indices=conditioning_indices,
            coords=coords,
        )
        state = VideoAutoregressionState(
            current_frames=frames_raw.clone(),
            window_start_time=start_time,
            autoregression_start_time=autoregression_start_time,
        )

        return state, diags

    def _load_batch_at_time(self, start_time: pd.Timestamp) -> dict:
        """Load a batch from the dataset at the given start time."""
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
        """Move all batch tensors in the batch to the model device."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _apply_frame_masking(
        self,
        batch: dict,
        conditioning_indices: list[int],
        condition_to_insert: torch.Tensor | None,
    ) -> dict:
        """Apply frame masking with the given conditioning indices."""
        return FrameMasker(keep_frames=conditioning_indices)(batch, condition_to_insert)
