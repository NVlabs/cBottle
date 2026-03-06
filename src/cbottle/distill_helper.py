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

from functools import partial
from einops import rearrange
import torch
from typing import Optional, Tuple
from copy import deepcopy
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from physicsnemo.models.diffusion import layers
from physicsnemo.utils.patching import BasePatching2D

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.methods.consistency_model.CM import CMModel as CMBaseModel
from fastgen.methods.consistency_model.sCM import SCMModel as SCMBaseModel
from fastgen.methods.distribution_matching.dmd2 import DMD2Model as DMD2BaseModel
from fastgen.utils import lr_scheduler
from fastgen.networks.discriminators import Discriminator_EDM as BaseDiscriminator_EDM

from scipy.signal import windows
import numpy as np


PRECISION_MAP = {
    "fp32": "float32",
    "torch.float32": "float32",
    "fp16": "float16",
    "torch.float16": "float16",
    "amp-fp16": "float16",
    "amp-bf16": "bfloat16",
}


def change_block(module, attr, value):
    if isinstance(module, layers.UNetBlock):
        assert hasattr(module, attr), f"Attribute {attr} not found in module"
        setattr(module, attr, value)


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
        is_superpatch_eval=False,
    ):
        assert not any(p.requires_grad for p in [img_clean, img_lr])
        data = {
            "real": img_clean,
            "condition": (
                labels,
                img_lr,
                pos_embed,
                augment_labels,
                is_superpatch_eval,
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
        is_superpatch_eval=False,
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
            is_superpatch_eval=is_superpatch_eval,
        )


class FastGridPatching2D(BasePatching2D):
    def __init__(
        self,
        img_shape: Tuple[int, int],
        patch_shape: Tuple[int, int],
        overlap_pix: int = 0,
    ):
        super().__init__(img_shape, patch_shape)
        self.overlap_pix = overlap_pix
        self.patch_shape_y = self.patch_shape[0]
        self.patch_shape_x = self.patch_shape[1]
        self.img_shape_y = self.img_shape[0]
        self.img_shape_x = self.img_shape[1]

        self.num_patches_y, remainder_y = divmod(
            self.img_shape_y - self.overlap_pix, self.patch_shape_y - self.overlap_pix
        )
        self.num_patches_x, remainder_x = divmod(
            self.img_shape_x - self.overlap_pix, self.patch_shape_x - self.overlap_pix
        )
        assert remainder_x == 0 and remainder_y == 0

        # Initialize cache for overlap count
        self.overlap_count = None

    def unfold(self, x):
        # Cast to float
        dtype = x.dtype
        if dtype == torch.int32:
            x = x.view(torch.float32)
        elif dtype == torch.int64:
            x = x.view(torch.float64)

        x = torch.nn.functional.unfold(
            input=x,
            kernel_size=(self.patch_shape_y, self.patch_shape_x),
            stride=(
                self.patch_shape_y - self.overlap_pix,
                self.patch_shape_x - self.overlap_pix,
            ),
        )

        # cast back
        if dtype in [torch.int32, torch.int64]:
            x = x.view(dtype)

        return x

    def fold(self, x):
        # Cast to float
        dtype = x.dtype
        if dtype == torch.int32:
            x = x.view(torch.float32)
        elif dtype == torch.int64:
            x = x.view(torch.float64)

        x = torch.nn.functional.fold(
            input=x,
            output_size=(self.img_shape_y, self.img_shape_x),
            kernel_size=(self.patch_shape_y, self.patch_shape_x),
            stride=(
                self.patch_shape_y - self.overlap_pix,
                self.patch_shape_x - self.overlap_pix,
            ),
        )

        # cast back
        if dtype in [torch.int32, torch.int64]:
            x = x.view(dtype)

        return x

    def apply(self, input, additional_input=None):
        unfold = self.unfold(input)
        unfold = rearrange(
            unfold,
            "b (c p_h p_w) (nb_p_h nb_p_w) -> (nb_p_w nb_p_h b) c p_h p_w",
            p_h=self.patch_shape_y,
            p_w=self.patch_shape_x,
            nb_p_h=self.num_patches_y,
            nb_p_w=self.num_patches_x,
        )
        if additional_input is not None:
            additional_input = torch.nn.functional.interpolate(
                input=additional_input, size=self.patch_shape, mode="bilinear"
            )
            num_super_patches, rem = divmod(input.shape[0], additional_input.shape[0])
            assert rem == 0, (
                f"{additional_input.shape[0]} must be a factor of {input.shape[0]}"
            )
            repeats = self.num_patches_y * self.num_patches_x * num_super_patches
            # repeat each patch in the batch patch_num times
            # TODO(jberner): (1) check that this is equal to interleave and rearrange (2) this assumes a specific patching on input
            additional_input = additional_input.repeat(repeats, 1, 1, 1)
            unfold = torch.cat((unfold, additional_input), dim=1)

        return unfold

    def get_overlap_count(self, device, dtype):
        # compute overlap count
        ones = torch.ones(
            (1, 1, self.img_shape_y, self.img_shape_x), device=device, dtype=dtype
        )
        overlap_count = self.unfold(ones)
        return self.fold(overlap_count)

    def fuse(self, input, batch_size=None, window=None):
        if window is not None:
            if window.shape[0] == 1:
                window = window.tile((input.shape[0], input.shape[1], 1, 1))

            x = rearrange(
                input * window,
                "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
                p_h=self.patch_shape_y,
                p_w=self.patch_shape_x,
                nb_p_h=self.num_patches_y,
                nb_p_w=self.num_patches_x,
            )
            weights = rearrange(
                window,
                "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
                p_h=self.patch_shape_y,
                p_w=self.patch_shape_x,
                nb_p_h=self.num_patches_y,
                nb_p_w=self.num_patches_x,
            )

            # Stitch patches together (by summing over overlapping patches)
            folded = self.fold(x)
            weights = self.fold(weights)
            return folded / weights
        else:
            # Reshape input to make it 3D to apply fold
            x = rearrange(
                input,
                "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
                p_h=self.patch_shape_y,
                p_w=self.patch_shape_x,
                nb_p_h=self.num_patches_y,
                nb_p_w=self.num_patches_x,
            )
            # Stitch patches together (by summing over overlapping patches)
            folded = self.fold(x)

            if self.overlap_count is None:
                self.overlap_count = self.get_overlap_count(
                    device=folded.device, dtype=folded.dtype
                )
            if not (
                self.overlap_count.dtype == folded.dtype
                and self.overlap_count.device == folded.device
            ):
                self.overlap_count = self.overlap_count.to(folded)

            # Normalize by overlap count
            return folded / self.overlap_count


class FastGenNet(FastGenNetwork):
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
        if patching is not None and not isinstance(patching, FastGridPatching2D):
            raise ValueError("patching must be a 'FastGridPatching2D' object.")
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
        labels, img_lr, pos_embed, augment_labels, is_superpatch_eval = condition
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
            assert fwd_pred_type in NET_PRED_TYPES, (
                f"{fwd_pred_type} is not supported as fwd_pred_type"
            )

        # superpatch unfolding
        if (self.patching is not None) and (not is_superpatch_eval):
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
        if (self.patching is not None) and (not is_superpatch_eval):
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
            assert len(features) == len(feature_indices), (
                f"{len(features)} != {len(feature_indices)}"
            )
            if return_features_early:
                return features
            # score and features; score, features
            out = [out, features]

        if return_logvar:
            emb_timestep = self.net.model.map_noise(t.flatten())
            logvar = self.logvar_linear(emb_timestep)
            return out, logvar
        return out


def build(config, use_ema=False):
    # Patching
    patching = None
    if "patch_shape" in config:
        patching = FastGridPatching2D(
            img_shape=config.input_shape[-2:],
            patch_shape=config.patch_shape,
            overlap_pix=config.overlap_pix,
        )
    window = None
    if "window" in config:
        window = config.window

    # Instantiate the generator network
    net = FastGenNet(
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
    def build_model(self):
        self.net, self.ema = build(self.config, use_ema=self.use_ema)

        # instantiate the teacher and consistency network
        if self.config.loss_config.use_cd:
            self.teacher = deepcopy(self.net)
            self.teacher.eval().requires_grad_(False)


class SCMModel(SCMBaseModel):
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
    def __init__(
        self,
        feature_indices=None,
        all_res=[32, 16, 8],
        in_channels=256,
    ):
        nn.Module.__init__(self)
        if feature_indices is None:
            feature_indices = {len(all_res) - 1}  # use the middle bottleneck feature
        self.feature_indices = {
            i for i in feature_indices if i < len(all_res)
        }  # make sure feature indices are valid
        self.in_res = [all_res[i] for i in sorted(feature_indices)]
        if not isinstance(in_channels, (list, tuple)):
            in_channels = [in_channels] * len(self.feature_indices)
        self.in_channels = [in_channels[i] for i in sorted(self.feature_indices)]

        self.discriminator_heads = nn.ModuleList()
        for res, in_channels in zip(self.in_res, self.in_channels):
            layers = []
            while res > 8:
                # reduce the resolution by half, until 8x8
                layers.extend(
                    [
                        nn.Conv2d(
                            kernel_size=4,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(num_groups=32, num_channels=in_channels),
                        nn.SiLU(),
                    ]
                )
                res //= 2

            layers.extend(
                [
                    nn.Conv2d(
                        kernel_size=4,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=2,
                        padding=1,
                    ),
                    # 8x8 -> 4x4
                    nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        kernel_size=4,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=4,
                        padding=0,
                    ),
                    # 4x4 -> 1x1
                    nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    nn.SiLU(),
                    nn.Conv2d(
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
            self.discriminator_heads.append(nn.Sequential(*layers))


class DMD2Model(DMD2BaseModel):
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


MODEL_MAP = {
    "cm": CMModel,
    "scm": SCMModel,
    "dmd2": DMD2Model,
}


def get_distill_model(teacher_model, model_cfg, distill_cfg, device):
    # set teacher model and device
    model_cfg.net = teacher_model
    model_cfg.device = device
    model = MODEL_MAP[distill_cfg.mode](model_cfg)
    return model


def get_scheduler(name, cfg, optimizer):
    if name == "modulus_default":
        return None
    schedule = getattr(lr_scheduler, name)(**cfg)
    return LambdaLR(
        optimizer,
        lr_lambda=schedule,
    )


def get_window_function(patch_size, window_alpha, type="KBD", **kwargs):
    functions = {
        "uniform": torch.ones,
        "hann": lambda ps: windows.hann(ps, sym=True),
        "hamming": lambda ps: windows.hamming(ps, sym=True),
        "general_hamming": lambda ps: windows.general_hamming(
            ps, window_alpha, sym=True
        ),
        "kaiser": lambda ps: windows.kaiser(ps, beta=window_alpha * np.pi, sym=True),
        "tukey": lambda ps: windows.tukey(ps, alpha=window_alpha, sym=True),
        "gaussian": lambda ps: windows.gaussian(
            ps, std=window_alpha * ps / 2, sym=True
        ),
        "KBD": lambda ps: windows.kaiser_bessel_derived(ps, window_alpha * np.pi),
    }
    if type not in functions.keys():
        raise ValueError(
            f"Unknown window function type {type}. Supported types are {list(functions.keys())}"
        )

    window = torch.tensor(functions[type](patch_size), **kwargs)
    window = window.unsqueeze(0) * window.unsqueeze(1)
    return window
