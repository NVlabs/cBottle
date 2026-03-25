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

"""Distillation helpers for cBottle models.

This module adapts FastGen's distillation methods for cBottle's Super-Resolution Model.
The main additions over the base FastGen models are:

- ``cBottleFastGenNet`` wraps the cBottle network into FastGen's
  ``FastGenNetwork`` interface, adding SuperPatching2D unfold/fold and
  window smoothing to support super-patch distillation training.
- ``CMModel``, ``SCMModel``, and ``DMD2Model`` override ``build_model``
  to construct the network via ``cBottleFastGenNet`` instead of FastGen's
  default path.
- ``DistillLoss`` bridges cBottle's SR training interface to FastGen's ``single_train_step`` API.

See ``https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/README.md`` for the base implementations.
"""

from functools import partial
import torch
from typing import Optional
from copy import deepcopy

from physicsnemo.nn.module import UNetBlock
from cbottle.patchify import SuperPatching2D

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.methods.consistency_model.CM import CMModel as CMBaseModel
from fastgen.methods.consistency_model.sCM import SCMModel as SCMBaseModel
from fastgen.methods.distribution_matching.dmd2 import DMD2Model as DMD2BaseModel
from fastgen.networks.discriminators import Discriminator_EDM as BaseDiscriminator_EDM

from omegaconf import DictConfig


def change_block(module, attr, value):
    if isinstance(module, UNetBlock):
        assert hasattr(module, attr), f"Attribute {attr} not found in module"
        setattr(module, attr, value)


# TODO: add unit tests for this class
class DistillLoss:
    def __init__(
        self,
        net,
    ):
        self.net = net

    def compute_loss(
        self,
        net,
        img_clean,
        img_lr,
        pos_embed,
        labels,
        augment_labels,
        iteration=None,
        eval_on_regular_patches=False,
    ):
        assert not any(p.requires_grad for p in [img_clean, img_lr])
        data = {
            "real": img_clean,
            "condition": (
                labels,
                img_lr,
                pos_embed,
                augment_labels,
                eval_on_regular_patches,
            ),
            "neg_condition": None,
        }

        loss_map, output = net.single_train_step(data, iteration)
        return loss_map["total_loss"], loss_map, output

    def __call__(
        self,
        net,
        img_clean,
        img_lr,
        pos_embed,
        iteration=None,
        # whether to evaluate on regular patches during super-patch training
        # required to evaluate on pre-saved regular patch to better compare performance
        eval_on_regular_patches=False,
    ):
        labels = None
        augment_labels = None
        return self.compute_loss(
            net=net,
            img_clean=img_clean,
            img_lr=img_lr,
            pos_embed=pos_embed,
            labels=labels,
            augment_labels=augment_labels,
            iteration=iteration,
            eval_on_regular_patches=eval_on_regular_patches,
        )


class cBottleFastGenNet(FastGenNetwork):
    """
    A wrapper around the FastGenNetwork in FastGen, which enables distilling cBottle models with various methods in FastGen framework.
    Supports super-patching training and window smoothing.

    See `fastgen.networks.network.FastGenNetwork` for more details.
    """

    def __init__(
        self,
        net,
        block_kwargs=None,
        patching=None,
        window=None,
        net_pred_type="x0",
        schedule_type="edm",
        **kwargs,
    ):
        super().__init__(
            net_pred_type=net_pred_type, schedule_type=schedule_type, **kwargs
        )
        self.net = net
        self.logvar_linear = torch.nn.Linear(self.net.model.map_noise.num_channels, 1)
        if block_kwargs is not None:
            for attr, value in block_kwargs.items():
                self.apply(partial(change_block, attr=attr, value=value))

        # patching
        if patching is not None and not isinstance(patching, SuperPatching2D):
            raise ValueError("patching must be a 'SuperPatching2D' object.")
        self.patching = patching
        self.window = window

    def forward(
        self,
        y_t,
        t,
        condition=None,
        return_features_early=False,
        feature_indices=None,
        return_logvar=False,
        fwd_pred_type: Optional[str] = None,
    ):
        labels, img_lr, pos_embed, augment_labels, eval_on_regular_patches = condition
        # squeeze all dims after the first one and expand to batchsize
        t = t.squeeze(list(range(1, t.ndim))).expand(y_t.shape[0])
        assert t.shape == (y_t.shape[0],)

        # Preconditioning weights for input
        y_t_in, t_in = y_t, t

        self.net.model.feature_indices = feature_indices
        self.net.model.features = []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert (
                fwd_pred_type in NET_PRED_TYPES
            ), f"{fwd_pred_type} is not supported as fwd_pred_type"

        # superpatch unfolding
        if (self.patching is not None) and (not eval_on_regular_patches):
            y_t = self.patching.apply(y_t)
            img_lr = self.patching.apply(img_lr)
            pos_embed = self.patching.apply(pos_embed)
            t = t.repeat(self.patching.num_patches_y * self.patching.num_patches_x)

        out = self.net(
            y_t,  # noised y
            t,  # sigma
            class_labels=labels,
            condition=img_lr,
            position_embedding=pos_embed,
            augment_labels=augment_labels,
        )

        # superpatch folding
        if (self.patching is not None) and (not eval_on_regular_patches):
            out = self.patching.fuse(out, window=self.window)

        out = self.noise_scheduler.convert_model_output(
            y_t_in,
            out,
            t_in,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        if feature_indices is not None and len(feature_indices) > 0:
            features = self.net.model.features
            # reset features
            self.net.model.features = None
            assert len(features) == len(
                feature_indices
            ), f"{len(features)} != {len(feature_indices)}"
            if return_features_early:
                return features
            # score and features; score, features
            out = [out, features]

        if return_logvar:
            emb_timestep = self.net.model.map_noise(t.flatten())
            logvar = self.logvar_linear(emb_timestep)
            return out, logvar
        return out


def build(config: DictConfig, use_ema: bool = False):
    # Patching
    patching = None
    if "patch_shape" in config:
        patching = SuperPatching2D(
            img_shape=config.input_shape[-2:],
            patch_shape=config.patch_shape,
            overlap_pix=config.overlap_pix,
        )
    window = None
    if "window" in config:
        window = config.window

    # Instantiate the generator network
    net = cBottleFastGenNet(
        net=config.net,
        patching=patching,
        window=window,
        train_p_mean=config.sample_t_cfg.train_p_mean,
        train_p_std=config.sample_t_cfg.train_p_std,
        min_t=config.sample_t_cfg.min_t,
        max_t=config.sample_t_cfg.max_t,
        net_pred_type="x0",
        schedule_type="edm",
        block_kwargs=config.get("block_kwargs"),
    )

    net.train().requires_grad_(True)

    # initialize EMA network
    ema = None
    if use_ema:
        ema = deepcopy(net)
        ema.eval().requires_grad_(False)

    return net, ema


class CMModel(CMBaseModel):
    """Consistency Model for cBottle distillation.

    A wrapper around the FastGen CM model in FastGen framework.
    See `fastgen.methods.consistency_model.CM.CMModel` for more details.

    References:
        - Song et al., 2023: https://arxiv.org/abs/2303.01469
        - Geng et al., 2024: https://arxiv.org/abs/2406.14548
    """

    def build_model(self):
        self.net, self.ema = build(self.config, use_ema=self.use_ema)

        # instantiate the teacher and consistency network
        if self.config.loss_config.use_cd:
            self.teacher = deepcopy(self.net)
            self.teacher.eval().requires_grad_(False)


class SCMModel(SCMBaseModel):
    """Continuous-time Consistency Model with TrigFlow for cBottle distillation.

    A wrapper around the FastGen sCM model in FastGen framework.
    See `fastgen.methods.consistency_model.sCM.SCMModel` for more details.

    References:
        - Lu & Song, 2024: https://arxiv.org/abs/2410.11081
    """

    def build_model(self):
        self.net, self.ema = build(self.config, use_ema=self.use_ema)

        # instantiate the teacher and consistency network
        if self.config.loss_config.use_cd:
            self.teacher = deepcopy(self.net)
            self.teacher.eval().requires_grad_(False)
        else:
            # TODO(jberner): remove this once we do not require a teacher anymore
            self.teacher = torch.nn.Identity()


class Discriminator_EDM(BaseDiscriminator_EDM):
    """EDM Discriminator for cBottle distillation.

    A wrapper around the FastGen EDM discriminator in FastGen framework.
    See `fastgen.networks.discriminators.Discriminator_EDM` for more details.
    """

    def __init__(
        self,
        feature_indices=None,
        all_res=[32, 16, 8],
        in_channels=256,
    ):
        torch.nn.Module.__init__(self)
        if feature_indices is None:
            feature_indices = {len(all_res) - 1}  # use the middle bottleneck feature
        self.feature_indices = {
            i for i in feature_indices if i < len(all_res)
        }  # make sure feature indices are valid
        self.in_res = [all_res[i] for i in sorted(feature_indices)]
        if not isinstance(in_channels, (list, tuple)):
            in_channels = [in_channels] * len(self.feature_indices)
        self.in_channels = [in_channels[i] for i in sorted(self.feature_indices)]

        self.discriminator_heads = torch.nn.ModuleList()
        for res, in_channels in zip(self.in_res, self.in_channels):
            layers = []
            while res > 8:
                # reduce the resolution by half, until 8x8
                layers.extend(
                    [
                        torch.nn.Conv2d(
                            kernel_size=4,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            stride=2,
                            padding=1,
                        ),
                        torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
                        torch.nn.SiLU(),
                    ]
                )
                res //= 2

            layers.extend(
                [
                    torch.nn.Conv2d(
                        kernel_size=4,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=2,
                        padding=1,
                    ),
                    # 8x8 -> 4x4
                    torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(
                        kernel_size=4,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=4,
                        padding=0,
                    ),
                    # 4x4 -> 1x1
                    torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(
                        kernel_size=1,
                        in_channels=in_channels,
                        out_channels=1,
                        stride=1,
                        padding=0,
                    ),
                    # 1x1 -> 1x1
                ]
            )

            # append the layers for current resolution to the discriminator head
            self.discriminator_heads.append(torch.nn.Sequential(*layers))


class DMD2Model(DMD2BaseModel):
    """VSD + GAN for cBottle distillation.

    A wrapper around the FastGen DMD2 model in FastGen framework.
    See `fastgen.methods.distribution_matching.dmd2.DMD2Model` for more details.

    References:
        - 	Yin et al., 2024: https://arxiv.org/abs/2405.14867
    """

    def build_model(self):
        self.net, self.ema = build(self.config, use_ema=self.use_ema)

        # instantiate the teacher and consistency network
        self.teacher = deepcopy(self.net)
        self.teacher.eval().requires_grad_(False)

        # instantiate the fake_score
        self.fake_score = deepcopy(self.net)

        if self.config.gan_loss_weight_gen > 0:
            # instantiate the discriminator in DMD2 ({0, 1, 2} are all features)
            self.discriminator = Discriminator_EDM(
                feature_indices={0, 1, 2},
                all_res=[64, 32, 16],
                in_channels=[64, 128, 128],
            )
