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

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from typing import Type, Literal, Optional
import inspect

import einops
import numpy as np
import torch
import torch.utils.checkpoint
from torch.nn.functional import silu

from cbottle.domain import Domain, HealPixDomain, PatchedHealpixDomain
from cbottle.models.embedding import (
    CalendarEmbedding,
    PositionalEmbedding,
    FourierEmbedding,
)
from earth2grid.healpix import pad as healpix_pad, Grid, HEALPIX_PAD_XY
from earth2grid.healpix import PixelOrder
# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.up:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )
        if self.down:
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=f_pad,
            )
        if w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class Conv2dHealpix(torch.nn.Module):
    """Same as Conv2D but works with healpix gridded data"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def preprocess(self, x, padding):
        F = 12
        B, C, T = x.shape[:3]
        nx = x.shape[-1]
        nside = int(np.sqrt(nx / 12))
        assert nside * nside * F == nx
        x = einops.rearrange(x, "b c t (f x y) -> (b c t) f x y", y=nside, x=nside, f=F)
        # healpix pad and earth2-grid have different convetions for the orientation of the grid
        # See explore/healpix_numbering.py for more information
        # TODO generalize earth2grid to support XY layouts with different orientations
        torch.cuda.nvtx.range_push("healpix_pad")
        x = healpix_pad(x, padding)
        torch.cuda.nvtx.range_pop()
        return einops.rearrange(x, "(b c t) f x y -> (b t f) c x y", b=B, t=T, f=F, c=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [b, c, t, x] shaped tensor
        """
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        F = 12
        B, C, T = x.shape[:3]

        def postprocess(x):
            return einops.rearrange(x, "(b t f) c x y -> b c t (f x y)", b=B, t=T, f=F)

        if self.up:
            x = torch.nn.functional.conv_transpose2d(
                self.preprocess(x, f_pad),
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=0,
            )
        elif self.down:
            x = torch.nn.functional.conv2d(
                self.preprocess(x, f_pad),
                f.tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=0,
            )
        else:
            x = self.preprocess(x, 0)

        if w is not None:
            x = torch.nn.functional.conv2d(
                self.preprocess(postprocess(x), w_pad), w, padding=0
            )

        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))

        return postprocess(x)


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.
class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.
class Attention(torch.nn.Module):
    def __init__(
        self, *, out_channels, eps, init_zero, init_attn, init, num_heads
    ) -> None:
        super().__init__()
        self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
        self.qkv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 3,
            kernel=1,
            **(init_attn if init_attn is not None else init),
        )
        self.proj = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero
        )
        self.num_heads = num_heads

    def forward(self, x):
        q, k, v = (
            self.qkv(self.norm2(x))
            .reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1)
            .unbind(2)
        )
        w = AttentionOp.apply(q, k)
        a = torch.matmul(v, w.mT)
        x = self.proj(a.reshape(*x.shape)).add_(x)
        return x


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        *,
        factory,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
        temporal_attention: bool = False,
        time_length: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not (attention or temporal_attention)
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.temporal_attention = temporal_attention
        self.attention = attention
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = factory.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = factory.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = factory.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.attention:
            self.attn = factory.Attention(
                out_channels=out_channels,
                eps=eps,
                init_zero=init_zero,
                init_attn=init_attn,
                init=init,
                num_heads=self.num_heads,
            )
        else:
            self.attn = None

        if self.temporal_attention:
            if time_length is None:
                raise ValueError(
                    "time_length must be specified if temporal_attention is not None"
                )
            self.time_length = time_length
            self.temporal_attention = TemporalAttention(
                out_channels=out_channels,
                seq_length=time_length,
                eps=1e-5,
                num_heads=self.num_heads,
            )

    def forward(self, x, emb):
        # TODO add a flag controlling this
        return torch.utils.checkpoint.checkpoint(
            self.run_forward, x, emb, use_reentrant=False
        )

    def run_forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.attn:
            x = self.attn(x)
            x = x * self.skip_scale

        if self.temporal_attention:
            x = self.skip_scale * (self.temporal_attention(x) + x)

        return x


class OriginalFactory:
    def __init__(self, img_size: tuple[int, int]):
        self.img_size = img_size

    def Conv2d(self, **kwargs):
        return Conv2d(**kwargs)

    def Attention(self, **kwargs):
        return Attention(**kwargs)

    def UNetBlock(self, **kwargs):
        return UNetBlock(factory=self, **kwargs)


class UnetBlockSpace(torch.nn.Module):
    def __init__(self, block, img_size: tuple[int, int]):
        super().__init__()
        self.block = block
        self.h, self.w = img_size

    def forward(self, x, emb):
        b, _, t, _ = x.shape
        x = einops.rearrange(x, "b c t (h w) -> (b t) c h w", h=self.h, w=self.w)
        y = self.block(x, emb)
        return einops.rearrange(y, "(b t) c h w -> b c t (h w)", b=b, t=t)

    @property
    def out_channels(self):
        return self.block.out_channels


class ApplySpace(torch.nn.Module):
    """Apply an image module (b, c, x, y) along final dimension of a tensor
    shaped (b, c, t, x * y)
    """

    def __init__(self, module, img_size: tuple[int, int]):
        super().__init__()
        self.module = module
        self.h, self.w = img_size

    def forward(self, x):
        b, _, t, _ = x.shape
        x = einops.rearrange(x, "b c t (h w) -> (b t) c h w", h=self.h, w=self.w)
        y = self.module(x)
        return einops.rearrange(y, "(b t) c h w -> b c t (h w)", b=b, t=t)

    @property
    def out_channels(self):
        return self.module.out_channels


class SpatialAttention(torch.nn.Module):
    def __init__(self, *, out_channels, eps, init_zero, init_attn, init, num_heads):
        super().__init__()
        self.attention = Attention(
            out_channels=out_channels,
            eps=eps,
            init_zero=init_zero,
            init_attn=init_attn,
            init=init,
            num_heads=num_heads,
        )

    def forward(self, x):
        B, _, T, X = x.shape
        x = einops.rearrange(x, "b c t (h w) -> (b t) c h w", h=1, w=X)
        y = self.attention(x)
        return einops.rearrange(y, "(b t) c h w -> b c t (h w)", b=B, t=T)


class TemporalAttention(torch.nn.Module):
    def __init__(
        self, *, out_channels: int, seq_length: int, eps, num_heads: int
    ) -> None:
        super().__init__()
        self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
        self.qkv = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels * 3, kernel_size=1
        )
        self.proj = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
        torch.nn.init.normal_(self.proj.weight.data).mul_(1e-5)
        self.positional_embedding = torch.nn.Parameter(
            torch.empty(num_heads, 2 * seq_length)
        )
        torch.nn.init.normal_(self.positional_embedding.data).mul_(1e-5)
        self.num_heads = num_heads
        self.seq_length = seq_length

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        qkv = self.qkv(self.norm2(x))
        q, k, v = einops.rearrange(
            qkv, "b (n heads c) t x -> n (b x) heads t c", n=3, heads=self.num_heads
        )
        # q - q time dim
        # k - k time dim
        attn = torch.einsum("b h q c, b h k c -> b h q k", q, k / np.sqrt(k.shape[-1]))

        i = torch.arange(self.seq_length)
        pairwise_distance = i.unsqueeze(1) - i + self.seq_length - 1
        relative_embedding = self.positional_embedding[:, pairwise_distance]

        w = attn + relative_embedding
        w = w.softmax(-1)
        out = torch.einsum("bhqk,bhkc->b h c q", w, v)
        out = einops.rearrange(out, "(b x) h c t -> b (h c) t x", x=x.shape[-1])
        return self.proj(out).to(dtype)


class SpatialFactory(OriginalFactory):
    def __init__(self, img_size: tuple[int, int]):
        self.img_size = img_size

    def Conv2d(self, up=False, down=False, **kwargs):
        if down:
            img_size = (k * 2 for k in self.img_size)
        elif up:
            img_size = (k // 2 for k in self.img_size)
        else:
            img_size = self.img_size

        return ApplySpace(super().Conv2d(up=up, down=down, **kwargs), img_size)

    def Attention(self, **kwargs):
        return ApplySpace(super().Attention(**kwargs), self.img_size)


class HealPixFactory(OriginalFactory):
    def __init__(self, img_size: tuple[int, int]):
        self.img_size = img_size

    def Conv2d(self, *args, **kwargs):
        return Conv2dHealpix(*args, **kwargs)

    def Attention(self, **kwargs):
        return SpatialAttention(**kwargs)


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch


class SongUNet(torch.nn.Module):
    def __init__(
        self,
        domain: Domain,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++, 'None' for Regression
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        mixing_type: Literal["xt", "spatial", "healpix"] = "xt",
        temporal_attention_resolutions: Optional[list[int]] = None,
        time_length: int = 1,
        add_spatial_embedding: bool = False,
        calendar_embed_channels: int = 0,  # embedding dimension for year fraction and second fraction
        decoder_start_with_temporal_attention: bool = False,
        upsample_temporal_attention: bool = False,
        channels_per_head: int = -1,  # uses all heads if -1, otherwise uses this many per head
        patched: bool = False,
        pos_embed_channels: int = 128,
    ):
        super().__init__()
        assert embedding_type in ["fourier", "positional", "zero"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]
        self.add_spatial_embedding = add_spatial_embedding
        self.domain = domain
        img_resolution = domain.img_resolution
        factory_cls: Type[OriginalFactory] = {
            "xt": OriginalFactory,
            "spatial": SpatialFactory,
            "healpix": HealPixFactory,
        }[mixing_type]
        self.input_shape = (in_channels, time_length, domain.numel())
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            time_length=time_length,
            emb_channels=emb_channels,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )
        if channels_per_head == -1:
            block_kwargs.update(num_heads=1)
        else:
            block_kwargs.update(channels_per_head=channels_per_head)

        if calendar_embed_channels:
            # needs healpix
            self.embed_calendar = CalendarEmbedding(
                torch.from_numpy(self.grid.lon).float(), calendar_embed_channels
            )
            in_channels += self.embed_calendar.out_channels
        else:
            self.embed_calendar = None

        self.emb_channels = emb_channels
        self.embedding_type = embedding_type
        self.noise_channels = noise_channels
        self.patched = patched
        self.pos_embed_channels = pos_embed_channels

        if self.add_spatial_embedding:
            if self.patched:
                self.pos_embed = self.init_pos_embed_sinusoid()
            else:
                # shape is [..., C, T, X]
                self.pos_embed = torch.nn.Parameter(
                    torch.zeros(model_channels, 1, domain.numel())
                )
                torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.temporal_attention_resolutions = temporal_attention_resolutions or set()

        if embedding_type != "zero":
            # Mapping.
            self.map_noise = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )

        self.map_label = (
            Linear(in_features=label_dim, out_features=noise_channels, **init)
            if label_dim
            else None
        )
        self.map_augment = (
            Linear(
                in_features=augment_dim, out_features=noise_channels, bias=False, **init
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            img_size = (res, res)
            factory = factory_cls(img_size)
            if level == 0:
                cin = cout
                cout = model_channels
                if not self.patched:
                    self.enc[f"{res}x{res}_conv_embed"] = factory.Conv2d(
                        in_channels=cin, out_channels=cout, kernel=3, **init
                    )
                else:
                    self.enc[f"{res}x{res}_conv"] = factory.Conv2d(
                        in_channels=cin, out_channels=cout, kernel=3, **init
                    )
            else:
                self.enc[f"{res}x{res}_down"] = factory.UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = factory.Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = factory.Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = factory.Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = factory.UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=attn,
                    temporal_attention=res in self.temporal_attention_resolutions,
                    **block_kwargs,
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            factory = factory_cls((res, res))
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = factory.UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    attention=True,
                    temporal_attention=decoder_start_with_temporal_attention,
                    **block_kwargs,
                )
                self.dec[f"{res}x{res}_in1"] = factory.UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = factory.UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    up=True,
                    temporal_attention=upsample_temporal_attention,
                    **block_kwargs,
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                temporal_attn = (
                    idx == num_blocks and res in self.temporal_attention_resolutions
                )

                self.dec[f"{res}x{res}_block{idx}"] = factory.UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=attn,
                    temporal_attention=temporal_attn,
                    **block_kwargs,
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = factory.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = factory.Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    @property
    def grid(self):
        return self.domain._grid

    def _get_arg(self, x, day_of_year, second_of_day):
        inputs = [x]
        if self.embed_calendar:
            inputs.append(self.embed_calendar(day_of_year, second_of_day))

        return torch.cat(inputs, dim=1)

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        augment_labels=None,
        day_of_year=None,
        second_of_day=None,
        position_embedding=None,  # local positional embedding with shape [B, C_embd, N, N], matching the shape of x except for the channel dimension
    ):
        torch.cuda.nvtx.range_push("SongUNet:forward")
        # if patched, concat local position_embedding to input
        if self.patched and self.add_spatial_embedding:
            x = torch.cat((x, position_embedding), dim=1)
        x = self._get_arg(x, day_of_year, second_of_day)
        if self.embedding_type == "zero":
            emb = torch.zeros((noise_labels.shape[0], self.noise_channels)).cuda()
        else:
            emb = self.map_noise(noise_labels)
            emb = (
                emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
            )  # swap sin/cos

        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)

        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # position embedding
        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif name.endswith("_conv_embed"):
                x = block(x)
                if self.add_spatial_embedding:
                    x = x + self.pos_embed.to(dtype=x.dtype)
                skips.append(x)
            else:
                sig = inspect.signature(block.forward)
                nparams = len(sig.parameters)
                x = block(x, emb) if nparams == 2 else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                x = block(x, emb)
        torch.cuda.nvtx.range_pop()
        return aux

    def init_pos_embed_sinusoid(self):
        with torch.no_grad():
            num_freq = self.pos_embed_channels // 4
            grid = self.grid
            lat = grid.lat
            lon = grid.lon
            lon = np.deg2rad(lon)
            lat = np.deg2rad(90 - lat)
            freq_bands = 2.0 ** np.linspace(0.0, num_freq, num=num_freq)
            grid_list = []
            for freq in freq_bands:
                for p_fn in [np.sin, np.cos]:
                    grid_list.append(p_fn(lat * freq))
                    grid_list.append(p_fn(lon * freq))
            grid = torch.from_numpy(np.stack(grid_list, axis=0)).float()
            grid = torch.nn.Parameter(grid)
        return grid


def get_model(
    model_type,
    domain,
    img_channels,
    condition_channels,
    label_dim,
    time_length,
    augment_dim=0,
    **kwargs,
) -> torch.nn.Module:
    return globals()[model_type](
        domain=domain,
        in_channels=img_channels + condition_channels,
        out_channels=img_channels,
        label_dim=label_dim,
        augment_dim=augment_dim,
        time_length=time_length,
        **kwargs,
    )


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).
class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        *,
        domain: Domain,
        img_channels,  # Number of color channels.
        model: torch.nn.Module,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        time_length: int = 1,
        condition_channels: int | None = None,
    ):
        super().__init__()
        self.time_length = time_length
        self.domain = domain
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model
        self.register_buffer("device_buffer", torch.empty(0))
        self.condition_channels = condition_channels

    @property
    def device(self):
        return self.device_buffer.device

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        condition=None,
        force_fp32=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if condition is None:
            arg = c_in * x
        else:
            condition = condition.to(torch.float32).to(x.device)
            condition = torch.nan_to_num(condition, nan=0.0)
            arg = torch.cat([c_in * x, condition], dim=1)

        F_x = self.model(
            arg.to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs
        )
        # assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def SongUNetHPX64(
    in_channels: int,
    out_channels: int,
    level=6,
    calendar_embed_channels: int = 0,
    **kwargs,
) -> SongUNet:
    """Base Unet for HPX64 resolution"""
    domain = HealPixDomain(Grid(level=level, pixel_order=HEALPIX_PAD_XY))
    config = {
        "add_spatial_embedding": True,
        "attn_resolutions": [8],
        "augment_dim": 0,
        "channel_mult": [1, 2, 2, 2],
        "channel_mult_noise": 1,
        "decoder_type": "standard",
        "dropout": 0.0,
        "embedding_type": "positional",
        "encoder_type": "standard",
        "mixing_type": "healpix",
        "model_channels": 128,
        "resample_filter": [1, 1],
    }
    config.update(kwargs)
    return SongUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        domain=domain,
        calendar_embed_channels=calendar_embed_channels,
        **config,
    )


def SongUNetHPX64Video(
    in_channels: int,
    out_channels: int,
    level=6,
    calendar_embed_channels: int = 0,
    time_length: int = 12,
    **kwargs,
) -> SongUNet:
    """Base Unet for HPX64 resolution"""
    domain = HealPixDomain(Grid(level=level, pixel_order=HEALPIX_PAD_XY))
    config = {
        "add_spatial_embedding": True,
        "attn_resolutions": [8],
        "temporal_attention_resolutions": [8, 16, 32, 64],
        "decoder_start_with_temporal_attention": True,
        "upsample_temporal_attention": True,
        "channels_per_head": 64,
        "time_length": time_length,
        "augment_dim": 0,
        "channel_mult": [1, 2, 2, 2],
        "channel_mult_noise": 1,
        "decoder_type": "standard",
        "dropout": 0.0,
        "embedding_type": "positional",
        "encoder_type": "standard",
        "mixing_type": "healpix",
        "model_channels": 128,
        "resample_filter": [1, 1],
    }
    config.update(kwargs)
    return SongUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        domain=domain,
        calendar_embed_channels=calendar_embed_channels,
        **config,
    )


def SongUNetHPX16(
    in_channels: int,
    out_channels: int,
    calendar_embed_channels: int = 0,
    **kwargs,
) -> SongUNet:
    """Base Unet for HPX64 resolution"""
    domain = HealPixDomain(Grid(level=4, pixel_order=HEALPIX_PAD_XY))
    # res is 16, 8, 4, 2
    config = {
        "add_spatial_embedding": True,
        "attn_resolutions": [8, 4, 2],
        "augment_dim": 0,
        "channel_mult": [1, 2, 4],
        "channel_mult_noise": 1,
        "decoder_type": "standard",
        "dropout": 0.0,
        "embedding_type": "positional",
        "encoder_type": "standard",
        "mixing_type": "healpix",
        "model_channels": 128,
        "resample_filter": [1, 1],
    }
    config.update(kwargs)
    return SongUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        domain=domain,
        calendar_embed_channels=calendar_embed_channels,
        **config,
    )


def SongUNetHPX256(in_channels: int, out_channels: int, **kwargs) -> SongUNet:
    """Unet for HPX256 resolution"""
    domain = HealPixDomain(Grid(level=8, pixel_order=HEALPIX_PAD_XY))
    config = {
        "add_spatial_embedding": True,
        "attn_resolutions": [16, 32, 64],
        "channel_mult": [1, 1, 2, 2, 2],
        "channel_mult_noise": 1,
        "decoder_type": "standard",
        "dropout": 0.0,
        "embedding_type": "positional",
        "encoder_type": "standard",
        "mixing_type": "healpix",
        "model_channels": 128,
        "resample_filter": [1, 1],
    }
    config.update(kwargs)
    return SongUNet(
        in_channels=in_channels, out_channels=out_channels, domain=domain, **config
    )


def SongUNetHPX1024(
    in_channels: int, out_channels: int, img_resolution: int, level: int, **kwargs
) -> SongUNet:
    """Unet for patched HPX1024 resolution"""
    domain = PatchedHealpixDomain(
        Grid(level=level, pixel_order=PixelOrder.NEST), patch_size=img_resolution
    )
    config = {
        "add_spatial_embedding": True,
        "mixing_type": "xt",
        "patched": True,
        "attn_resolutions": [28],
        "channel_mult": [1, 2, 2, 2, 2],
        "channel_mult_noise": 1,
        "dropout": 0.0,
        "embedding_type": "positional",
        "encoder_type": "standard",
        "decoder_type": "standard",
        "resample_filter": [1, 1],
        "label_dim": 0,
    }
    config.update(kwargs)
    return SongUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        domain=domain,
        **config,
    )
