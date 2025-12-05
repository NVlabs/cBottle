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

import math
from enum import auto, Enum
from dataclasses import dataclass


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


@dataclass
class AutoregressionRuntimeConfig:
    """
    Wrap all autoregression-related variables and provide helpers for derived parameters.
    This stays strictly focused on autoregressive autoregression settings.

    duration: number of days of horizon you want to cover.
    """

    enabled: bool
    duration: int  # in days (0 or negative => no autoregression)
    num_conditioning_frames: int  # N_overlap
    model_time_length: int  # T
    time_step: int  # hours between frames (Δt)

    def __post_init__(self):
        if not (0 <= self.num_conditioning_frames < self.model_time_length):
            raise ValueError(
                f"num_conditioning_frames must be in [0, {self.model_time_length}), "
                f"got {self.num_conditioning_frames}"
            )

    def _compute_num_autoregressions(self) -> int:
        """Internal logic for computing autoregression count."""
        if not self.enabled:
            return 1

        T = self.model_time_length
        N_overlap = self.num_conditioning_frames
        delta = T - N_overlap

        horizon_hours = self.duration * 24

        frames_needed = max(
            T,
            math.ceil(horizon_hours / self.time_step) + 1,
        )

        if frames_needed <= T:
            return 1

        k_minus_1 = math.ceil((frames_needed - T) / delta)
        return 1 + max(0, k_minus_1)

    def _compute_dataset_time_length(self) -> int:
        """Internal logic for computing dataset time length."""
        T = self.model_time_length
        delta = T - self.num_conditioning_frames
        return T + (self.num_autoregressions - 1) * delta

    def _compute_start_end_pairs(self) -> int:
        """Internal logic for computing start and end indices of each autoregression step"""
        pairs = []
        delta = self.model_time_length - self.num_conditioning_frames
        for index_autoregression in range(self.num_autoregressions):
            pairs.append(
                [
                    index_autoregression * delta,
                    index_autoregression * delta + self.model_time_length,
                ]
            )
        return pairs

    # -------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------
    @property
    def num_autoregressions(self) -> int:
        """Number of autoregression windows K."""
        return self._compute_num_autoregressions()

    @property
    def dataset_time_length(self) -> int:
        """Total number of frames required by the dataset."""
        return self._compute_dataset_time_length()

    @property
    def start_end_pairs(self) -> int:
        """Number of autoregression windows K."""
        return self._compute_start_end_pairs()
