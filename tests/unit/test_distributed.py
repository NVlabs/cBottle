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
import torch.distributed as dist
from cbottle.models.distributed import (
    shard_x,
    shard_t,
    compute_t_split,
)
from cbottle.distributed import init


def test_sharding_routines_even():
    """Test sharding with even T split"""
    if not torch.distributed.is_initialized():
        init()

    group_size = 2
    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)

    b, c, t, x = 1, 3, 2, 8

    tensor = torch.arange(b * c * t * x).view(b, t, x, c).cuda()
    out = shard_t(tensor, group)

    assert out.shape == (b, t // 2, x * 2, c)

    # back
    roundtrip = shard_x(out, group)
    assert torch.all(roundtrip == tensor)


def test_sharding_routines_uneven():
    """Test sharding with uneven T split"""
    if not torch.distributed.is_initialized():
        init()

    group_size = 2
    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)
    rank = dist.get_rank(group)

    b, c, x = 2, 4, 8
    t_full = 3  # Uneven: 3 / 2 -> [2, 1]
    t_splits = compute_t_split(t_full, group_size)
    t_local = t_splits[rank]

    full_tensor = torch.arange(b * t_full * x * c).view(b, t_full, x, c).cuda().float()

    t_start = sum(t_splits[:rank])
    t_end = t_start + t_local
    local_tensor = full_tensor[:, t_start:t_end, :, :].contiguous()

    assert local_tensor.shape == (b, t_local, x, c)

    x_sharded = shard_x(local_tensor, group, t_full=t_full)
    x_local = x // group_size
    assert x_sharded.shape == (
        b,
        t_full,
        x_local,
        c,
    ), f"Expected {(b, t_full, x_local, c)}, got {x_sharded.shape}"

    roundtrip = shard_t(x_sharded, group, t_full=t_full)
    assert (
        roundtrip.shape == local_tensor.shape
    ), f"Expected {local_tensor.shape}, got {roundtrip.shape}"

    assert torch.allclose(roundtrip, local_tensor), "Roundtrip failed: data mismatch"


def test_sharding_unet():
    """Test model parallel sharding for UNet with EDMPrecond wrapper."""
    if not torch.distributed.is_initialized():
        init()

    group_size = 2
    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [world_size // group_size, group_size])
    group = mesh.get_group(1)

    level = 6  # HPX64
    t_full = 2 * group_size  # Total time across all ranks
    t_local = t_full // group_size  # Time per rank
    b, c = 1, 3
    x = 12 * 4**level  # HPX64 has 12 * 4^6 pixels

    # Create EDMPrecond wrapping SongUNetHPX64Video
    from cbottle.models import networks

    unet = networks.SongUNetHPX64Video(
        in_channels=c,
        out_channels=c,
        level=level,
        calendar_embed_channels=8,
        model_channels=32,  # Small for testing
        time_length=t_full,
    )

    model = networks.EDMPrecond(
        model=unet,
        domain=unet.domain,
        img_channels=c,
        time_length=t_full,
        label_dim=0,
    )

    # Register hooks to ensure temporal attention sees x-sharded data
    # After shard_x_bctx: full T, sharded X
    def ensure_x_sharded(mod, inputs):
        (z,) = inputs  # [b, c, t, x] format for UNet
        assert z.shape[2] == t_full, f"Expected t={t_full}, got {z.shape[2]}"
        assert (
            z.shape[3] == x // group_size
        ), f"Expected x={x // group_size}, got {z.shape[3]}"

    for module in model.modules():
        if isinstance(module, networks.TemporalAttention):
            module.register_forward_pre_hook(ensure_x_sharded)

    model.cuda()
    model.set_parallel_group(group, t_full=t_full)

    # Input tensor: [b, c, t_local, x] - each rank has a slice of time
    tensor = torch.randn(b, c, t_local, x).cuda()
    sigma = torch.ones(b, 1).cuda()
    class_labels = None

    out = model(
        tensor,
        sigma=sigma,
        class_labels=class_labels,
        day_of_year=torch.zeros([b, t_local]).cuda(),
        second_of_day=torch.zeros([b, t_local]).cuda(),
    )

    assert (
        out.out.shape == tensor.shape
    ), f"Expected {tensor.shape}, got {out.out.shape}"
