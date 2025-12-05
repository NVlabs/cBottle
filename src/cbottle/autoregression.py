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
from functools import cached_property
from typing import Tuple


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
    Configuration for autoregressive rollouts.
    duration: number of days of horizon to cover.
    num_conditioning_frames: number of frames to use as condition during rollout
    model_time_length: temporal length of model's input
    time_step: temporal interval in hours
    """

    enabled: bool
    duration: int
    num_conditioning_frames: int
    model_time_length: int
    time_step: int

    def __post_init__(self):
        if not (0 <= self.num_conditioning_frames < self.model_time_length):
            raise ValueError(
                f"num_conditioning_frames must be in [0, {self.model_time_length}), "
                f"got {self.num_conditioning_frames}"
            )

    @cached_property
    def num_autoregressions(self) -> int:
        """Number of autoregression windows K."""
        if not self.enabled or self.duration <= 0:
            return 1

        T = self.model_time_length
        delta = T - self.num_conditioning_frames
        horizon_hours = self.duration * 24

        frames_needed = max(
            T,
            math.ceil(horizon_hours / self.time_step) + 1,
        )

        if frames_needed <= T:
            return 1

        k_minus_1 = math.ceil((frames_needed - T) / delta)
        return 1 + max(0, k_minus_1)

    @cached_property
    def dataset_time_length(self) -> int:
        """Total frames needed by the dataset."""
        T = self.model_time_length
        delta = T - self.num_conditioning_frames
        return T + (self.num_autoregressions - 1) * delta

    @cached_property
    def start_end_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """(start, end) frame indices for each autoregression step."""
        pairs = []
        delta = self.model_time_length - self.num_conditioning_frames
        for k in range(self.num_autoregressions):
            start = k * delta
            end = start + self.model_time_length
            pairs.append((start, end))
        return tuple(pairs)
