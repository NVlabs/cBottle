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
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tempfile
import torch
from earth2grid.healpix import HEALPIX_PAD_XY, Grid, PaddingBackends

import cbottle.datasets.healpix_artificial
import cbottle.models.networks
from cbottle import domain
from cbottle.models.networks import (
    CalendarEmbedding,
    Conv2dHealpix,
    HealPixDomain,
    NoCopyNCHW2NHWC,
    NoCopyNHWC2NCHW,
    SongUNet,
)


def test_conv2d_healpix():
    in_channels = 2
    out_channels = 3

    nside = 32
    F = 12
    B = 2
    T = 1
    X = nside**2 * F

    conv2d = Conv2dHealpix(
        in_channels=in_channels, out_channels=out_channels, kernel=3
    ).to(memory_format=torch.channels_last)

    x = torch.ones([B, in_channels, T, X]).to(memory_format=torch.channels_last)
    out = conv2d(x)
    assert out.shape == (B, out_channels, T, X)


def test_conv2d_healpix_batch():
    """TEsts that batch and channel dim are handled correctly"""
    in_channels = 2
    out_channels = 2

    nside = 32
    F = 12
    B = 4
    T = 1
    X = nside**2 * F

    conv2d = Conv2dHealpix(
        in_channels=in_channels, out_channels=out_channels, kernel=3
    ).to(memory_format=torch.channels_last)
    conv2d.weight = None
    conv2d.bias = None

    x = torch.randn([B, in_channels, T, X]).to(memory_format=torch.channels_last)
    out = conv2d(x)
    np.testing.assert_array_equal(out, x)
    assert out.shape == (B, out_channels, T, X)


def test_conv2d_healpix_down():
    in_channels = 2
    out_channels = 3

    nside = 32
    F = 12
    B = 2
    T = 1
    X = nside**2 * F

    conv2d = Conv2dHealpix(
        in_channels=in_channels, out_channels=out_channels, down=True, kernel=3
    ).to(memory_format=torch.channels_last)

    x = torch.ones([B, in_channels, T, X]).to(memory_format=torch.channels_last)
    out = conv2d(x)
    assert out.shape == (B, out_channels, T, X // 4)


def test_conv2d_healpix_up():
    in_channels = 2
    out_channels = 3

    nside = 32
    F = 12
    B = 2
    T = 1
    X = nside**2 * F

    conv2d = Conv2dHealpix(
        in_channels=in_channels, out_channels=out_channels, up=True, kernel=3
    ).to(memory_format=torch.channels_last)

    x = torch.ones([B, in_channels, T, X]).to(memory_format=torch.channels_last)
    out = conv2d(x)
    assert out.shape == (B, out_channels, T, 4 * X)


@pytest.mark.parametrize("enable_classifier", [True, False])
def test_song_unet(enable_classifier: bool):
    m = cbottle.models.networks.SongUNet(
        domain=domain.Ring(64),
        in_channels=2,
        out_channels=3,
        time_length=64,
        enable_classifier=enable_classifier,
    )
    noise_labels = torch.ones([1])
    a = torch.ones(1, 2, 64, 64)
    out = m(a, noise_labels, None)
    assert out.out.shape == (1, 3, 64, 64)
    assert out.logits.shape == (1, 1, 8, 8) if enable_classifier else out.logits is None


def test_song_unet_spatial_mixing():
    m = cbottle.models.networks.SongUNet(
        domain=domain.Plane(64, 64),
        in_channels=2,
        out_channels=3,
        time_length=1,
        mixing_type="spatial",
    )
    noise_labels = torch.ones([1])
    a = torch.ones(1, 2, 1, 64 * 64)
    out = m(a, noise_labels, None)
    assert out.out.shape == (1, 3, 1, 64 * 64)
    assert out.logits is None


def test_temporal_attention():
    t = 10
    c = 32

    x = torch.zeros(1, c, t, 3)
    attn = cbottle.models.networks.TemporalAttention(
        out_channels=c, seq_length=t, eps=0.1, num_heads=4
    )
    y = attn(x)
    assert y.shape == x.shape


def test_healpix_conv_preprocess(tmp_path):
    ds = cbottle.datasets.healpix_artificial.HealPIXData(level=5)
    # %%
    lon = torch.from_numpy(ds._grid.lat)
    lon = lon.view(1, 1, 1, -1)

    z = torch.cos(5 * torch.deg2rad(lon)).to(memory_format=torch.channels_last)

    conv = Conv2dHealpix(1, 1, kernel=3).to(memory_format=torch.channels_last)
    y = conv.preprocess(z, 1)
    dx = torch.diff(y, dim=-1)
    dy = torch.diff(y, dim=-2)

    smoothness = torch.max(torch.abs(dx)) + torch.max(torch.abs(dy))
    # tune this tolerance by introducing an error in the HealPIXData and observing `m`
    tolerance = 0.3
    if smoothness > tolerance:
        plt.imshow(y[5, 0])
        plt.savefig(tmp_path / "out.png")
        path = tmp_path / "out.png"
        assert False, f"{smoothness} too big. See image at {path}."


def test_CalendarEmbedding():
    g = 8
    npix = 32
    n = 4
    t = 2

    lon = torch.arange(npix)
    calendar = CalendarEmbedding(lon, g)
    doy = torch.ones([n, t])
    second = torch.ones([n, t])

    out = calendar(doy, second)
    assert out.shape == (n, 4 * g, t, npix)


def test_SongUnetCalendarEmbeddings():
    domain = HealPixDomain(Grid(level=4, pixel_order=HEALPIX_PAD_XY))
    net = SongUNet(
        add_spatial_embedding=True,
        in_channels=3,
        out_channels=3,
        domain=domain,
        model_channels=4,
        calendar_embed_channels=1,
        mixing_type="healpix",
        padding_backend=PaddingBackends.cuda,
    )
    device = "cuda"
    net.to(device).to(memory_format=torch.channels_last)

    noise_labels = torch.ones([1], device=device)
    n, t = 1, 1
    img = torch.ones(n, 3, t, net.domain.numel(), device=device).to(
        memory_format=torch.channels_last
    )
    doy = torch.ones([n, t], device=device)
    second = torch.ones([n, t], device=device)

    out = net(
        img, noise_labels, class_labels=None, day_of_year=doy, second_of_day=second
    )
    assert out.out.shape == (n, 3, t, net.domain.numel())


def test_NoCopyTransposes(device=torch.device("cuda")):
    # Channels last NCHW tensor
    x = torch.rand(5, 3, 224, 224).to(memory_format=torch.channels_last, device=device)
    # Convert to explicit NHWC
    x1 = NoCopyNCHW2NHWC(x)
    assert x1.shape == torch.Size([5, 224, 224, 3])  # Check explicit shape
    assert x1.is_contiguous(memory_format=torch.contiguous_format)  # Check contiguous
    if device.type == "cuda":
        # Ensure GPU pointers are exactly the same
        assert (
            x.__cuda_array_interface__["data"][0]
            == x1.__cuda_array_interface__["data"][0]
        )

    # Convert back to NCHW
    x2 = NoCopyNHWC2NCHW(x1)
    assert x2.shape == torch.Size([5, 3, 224, 224])  # Check explicit shape
    assert x2.is_contiguous(memory_format=torch.channels_last)  # Check contiguous
    if device.type == "cuda":
        # Ensure GPU tensors are exactly the same
        assert x.__cuda_array_interface__ == x2.__cuda_array_interface__


@pytest.mark.parametrize("enable_classifier", [True, False])
def test_song_unet_unused_params(enable_classifier: bool, device=torch.device("cuda")):
    m = cbottle.models.networks.SongUNet(
        domain=domain.Ring(64),
        in_channels=2,
        out_channels=3,
        time_length=64,
        enable_classifier=enable_classifier,
    ).to(device)
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-4, momentum=0.9)
    noise_labels = torch.ones([1], device=device)
    a = torch.ones(1, 2, 64, 64, device=device)
    out = m(a, noise_labels, None)
    assert out.out.shape == (1, 3, 64, 64)
    assert out.logits.shape == (1, 1, 8, 8) if enable_classifier else out.logits is None

    loss = out.out.mean()
    if enable_classifier:
        loss += out.logits.mean()
    loss.backward()
    optimizer.step()
    for name, param in m.named_parameters():
        # Ensure no unused parameters
        assert param.grad is not None


@pytest.mark.skipif(not cbottle.models._is_apex_available, reason="Apex unavailable")
def test_group_norm_apex_checkpoint_compatibility(device=torch.device("cuda")):
    gn1 = cbottle.models.networks.group_norm_factory(
        num_channels=128,
        use_apex_gn=True,
    ).to(device)
    with tempfile.TemporaryFile() as f:
        torch.save(gn1.state_dict(), f)

        gn2 = cbottle.models.networks.group_norm_factory(
            num_channels=128,
            use_apex_gn=False,
        ).to(device)

        f.seek(0)
        gn2.load_state_dict(torch.load(f, weights_only=True, map_location=device))
