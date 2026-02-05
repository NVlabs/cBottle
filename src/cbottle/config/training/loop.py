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
from typing import Optional
import dataclasses


@dataclasses.dataclass
class TrainingLoopBase:
    run_dir: str = "."  # Output directory.
    seed: int = 0  # Global random seed.
    batch_size: int = 512  # Total batch size for one training iteration.
    batch_gpu: Optional[int] = None  # Limit batch size per GPU, None = no limit.
    ema_halflife_kimg: int = (
        500  # Half-life of the exponential moving average (EMA) of model weights.
    )
    ema_rampup_ratio: float = 0.05  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_img: int = 10_000_000  # Learning rate ramp-up duration.
    loss_scaling: float = 1.0  # Loss scaling factor for reducing FP16 under/overflows.
    gradient_clip_max_norm: Optional[float] = None
    total_ticks: int = 10
    steps_per_tick: int = 1024
    snapshot_ticks: int = 50  # How often to save network snapshots, None = disable.
    state_dump_ticks: int = 500  # How often to dump training state, None = disable.
    cudnn_benchmark: bool = True  # Enable torch.backends.cudnn.benchmark?
    bf16: bool = False  # Use bfloat16 for training.
    compile: bool = False  # torch.compile the entire network
