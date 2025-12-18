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
import dataclasses
import json


@dataclasses.dataclass
class ModelConfigV1:
    architecture: str = "unet_hpx64"
    model_channels: int = 128
    label_dim: int = 0
    out_channels: int = 1
    condition_channels: int = 0
    time_length: int = 1
    label_dropout: float = 0.0
    level: int = 10
    # by default use the buggy implementation of the calendar embedding, since
    # this is needed for backwards compatibility w/ existing checkpoints
    calendar_include_legacy_bug: bool = True
    # for backwards compatibility with old checkpoints
    noise_dependent_dropout_config: dict | None = None
    # if true enable the HPX16 classifier -- used for hurricane guidance
    enable_classifier: bool = False
    # arguments for SongUnetHPX1024
    position_embed_channels: int = 20
    img_resolution: int = 128
    # EDM Preconditioning
    sigma_data: float = 1.0

    def dumps(self):
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def loads(cls, s):
        d = json.loads(s)
        return cls(**d)
