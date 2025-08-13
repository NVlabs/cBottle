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

from cbottle.models import networks
from cbottle.config.models import ModelConfigV1


def get_model(
    config: ModelConfigV1,
    *,
    use_apex_groupnorm: bool | None = None,
    padding_backend=None,
    in_place_operations: bool = True,
) -> torch.nn.Module:
    if config.architecture == "unet_hpx64":
        precond_cls = networks.EDMPrecond
        if config.time_length > 1:
            architecture = networks.SongUNetHPX64Video(
                config.out_channels + config.condition_channels,
                config.out_channels,
                label_dim=config.label_dim,
                calendar_embed_channels=8,
                model_channels=config.model_channels,
                time_length=config.time_length,
                label_dropout=config.label_dropout,
                calendar_include_legacy_bug=config.calendar_include_legacy_bug,
                use_apex_groupnorm=use_apex_groupnorm,
                padding_backend=padding_backend,
                in_place_operations=in_place_operations,
            )
        else:
            architecture = networks.SongUNetHPX64(
                config.out_channels + config.condition_channels,
                config.out_channels,
                label_dim=config.label_dim,
                calendar_embed_channels=8,
                model_channels=config.model_channels,
                calendar_include_legacy_bug=config.calendar_include_legacy_bug,
                enable_classifier=config.enable_classifier,
                use_apex_groupnorm=use_apex_groupnorm,
                padding_backend=padding_backend,
                in_place_operations=in_place_operations,
            )

    elif config.architecture == "unet_hpx1024_patch":
        precond_cls = networks.EDMPrecondLegacy
        architecture = networks.SongUnetHPXPatch(
            in_channels=config.condition_channels
            + config.out_channels
            + config.position_embed_channels,
            out_channels=config.out_channels,
            img_resolution=config.img_resolution,
            model_channels=config.model_channels,
            pos_embed_channels=config.position_embed_channels,
            label_dim=config.label_dim,
            level=config.level,
            use_apex_groupnorm=use_apex_groupnorm,
            padding_backend=padding_backend,
            in_place_operations=in_place_operations,
        )
    else:
        raise NotImplementedError(config.architecture)

    return precond_cls(
        model=architecture,
        domain=architecture.domain,
        img_channels=config.out_channels,
        time_length=config.time_length,
        label_dim=config.label_dim,
    )
