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
"""
Model parallel distributed tests.

Run with: torchrun --nproc_per_node=4 -m pytest tests/unit/test_distributed.py -W ignore::UserWarning
"""

import torch
import torch.distributed as dist

from cbottle.models.distributed import shard_x, shard_t, compute_t_split
from cbottle.distributed import init as dist_init

# Shared test config
LEVEL, T_FULL, B, C = 6, 25, 1, 45
X = 12 * 4**LEVEL
LABEL_DIM = 1024


def ensure_dist_init():
    if not torch.distributed.is_initialized():
        dist_init()


def create_model(t_full=T_FULL, label_dim=LABEL_DIM):
    from cbottle.models import networks

    unet = networks.SongUNetHPX64Video(
        in_channels=C,
        out_channels=C,
        level=LEVEL,
        calendar_embed_channels=8,
        model_channels=64,
        time_length=t_full,
        label_dim=label_dim,
    )
    return networks.EDMPrecond(
        model=unet,
        domain=unet.domain,
        img_channels=C,
        time_length=t_full,
        label_dim=label_dim,
    )


# ============================================================================
# Sharding primitive tests
# ============================================================================


def test_sharding_routines_even():
    """Test shard_t/shard_x roundtrip with even T split."""
    ensure_dist_init()
    world_size = dist.get_world_size()
    mesh = dist.init_device_mesh("cuda", [1, world_size])
    group = mesh.get_group(1)

    t_local = 2
    t_full = t_local * world_size  # shard_t input needs full T
    x_local = 8
    x_full = x_local * world_size

    # Input: (B, t_full, x_local, C) -> Output: (B, t_local, x_full, C)
    tensor = (
        torch.arange(B * t_full * x_local * C)
        .view(B, t_full, x_local, C)
        .cuda()
        .float()
    )
    out = shard_t(tensor, group)
    assert out.shape == (B, t_local, x_full, C)

    roundtrip = shard_x(out, group)
    assert torch.allclose(roundtrip, tensor), "Roundtrip failed"


def test_sharding_routines_uneven():
    """Test shard_t/shard_x roundtrip with uneven T split (uses global T_FULL)."""
    ensure_dist_init()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    mesh = dist.init_device_mesh("cuda", [1, world_size])
    group = mesh.get_group(1)

    x_full = world_size * 4
    x_local = x_full // world_size
    t_splits = compute_t_split(T_FULL, world_size)
    t_local, t_start = t_splits[rank], sum(t_splits[:rank])

    full_tensor = (
        torch.arange(B * T_FULL * x_full * C).float().view(B, T_FULL, x_full, C).cuda()
    )
    local_tensor = full_tensor[:, t_start : t_start + t_local, :, :].contiguous()

    x_sharded = shard_x(local_tensor, group, t_full=T_FULL)
    assert x_sharded.shape == (B, T_FULL, x_local, C)

    roundtrip = shard_t(x_sharded, group, t_full=T_FULL)
    assert torch.allclose(roundtrip, local_tensor), "Roundtrip failed"


def test_sharding_unet():
    """Test UNet forward pass with model parallel sharding (uneven T=25)."""
    ensure_dist_init()
    from cbottle.models import networks

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    mesh = dist.init_device_mesh("cuda", [1, world_size])
    mp_group = mesh.get_group(1)
    t_splits = compute_t_split(T_FULL, world_size)
    t_local = t_splits[rank]

    model = create_model(T_FULL)

    def ensure_full_t(mod, inputs):
        assert inputs[0].shape[2] == T_FULL

    for m in model.modules():
        if isinstance(m, networks.TemporalAttention):
            m.register_forward_pre_hook(ensure_full_t)

    model.cuda()
    model.set_parallel_group(mp_group, t_full=T_FULL)

    tensor = torch.randn(B, C, t_local, X).cuda()
    out = model(
        tensor,
        sigma=torch.ones(B, 1).cuda(),
        class_labels=None,
        day_of_year=torch.zeros([B, t_local]).cuda(),
        second_of_day=torch.zeros([B, t_local]).cuda(),
    )
    assert out.out.shape == tensor.shape


def test_mp_vs_no_mp():
    """Verify MP produces identical output, loss, and gradients to non-MP."""
    ensure_dist_init()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    seed = 12345
    torch.manual_seed(seed)
    model = create_model().cuda().train()

    # Create inputs on rank 0, broadcast to all
    if rank == 0:
        full_input = torch.randn(B, C, T_FULL, X).cuda()
        class_labels = torch.randn(B, LABEL_DIM).cuda()
        day_of_year = torch.rand([B, T_FULL]).cuda()
        second_of_day = torch.rand([B, T_FULL]).cuda()
    else:
        full_input = torch.empty(B, C, T_FULL, X).cuda()
        class_labels = torch.empty(B, LABEL_DIM).cuda()
        day_of_year = torch.empty([B, T_FULL]).cuda()
        second_of_day = torch.empty([B, T_FULL]).cuda()
    dist.broadcast(full_input, src=0)
    dist.broadcast(class_labels, src=0)
    dist.broadcast(day_of_year, src=0)
    dist.broadcast(second_of_day, src=0)
    sigma = torch.ones(B, 1).cuda()

    # Non-MP reference on rank 0 (no parallel group set yet)
    torch.manual_seed(seed)
    if rank == 0:
        out = model(
            full_input,
            sigma=sigma,
            class_labels=class_labels,
            day_of_year=day_of_year,
            second_of_day=second_of_day,
        )
        ref_output = out.out.clone()
        ref_loss = out.out.sum() / (B * T_FULL)
        ref_loss.backward()
        ref_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )
    else:
        ref_output = torch.empty(B, C, T_FULL, X).cuda()
        ref_loss = torch.tensor(0.0).cuda()
        ref_grad_norm = torch.tensor(0.0).cuda()

    dist.broadcast(ref_output, src=0)
    dist.broadcast(ref_loss, src=0)
    dist.broadcast(ref_grad_norm, src=0)
    model.zero_grad()

    # Now set parallel group and run MP
    mesh = dist.init_device_mesh("cuda", [1, world_size])
    mp_group = mesh.get_group(1)
    model.set_parallel_group(mp_group, t_full=T_FULL)

    t_splits = compute_t_split(T_FULL, world_size)
    t_local = t_splits[rank]
    t_start = sum(t_splits[:rank])
    local_input = full_input[:, :, t_start : t_start + t_local, :].contiguous()
    local_day = day_of_year[:, t_start : t_start + t_local].contiguous()
    local_second = second_of_day[:, t_start : t_start + t_local].contiguous()

    torch.manual_seed(
        seed
    )  # make sure any random state is identical before model forward
    out_mp = model(
        local_input,
        sigma=sigma,
        class_labels=class_labels,
        day_of_year=local_day,
        second_of_day=local_second,
    )

    # 1. Compare output
    gathered = [torch.empty(B, C, t_splits[r], X).cuda() for r in range(world_size)]
    dist.all_gather(gathered, out_mp.out.contiguous(), group=mp_group)
    mp_output = torch.cat(gathered, dim=2)
    out_diff = (mp_output - ref_output).abs().max().item()
    assert torch.allclose(
        mp_output, ref_output, rtol=1e-4, atol=1e-5
    ), f"Output diff: {out_diff:.2e}"

    # 2. Compare loss (mimics training: scale by world_size then DDP averages)
    mp_loss_for_compare = out_mp.out.sum().detach() * world_size / (B * T_FULL)
    dist.all_reduce(mp_loss_for_compare, op=dist.ReduceOp.AVG, group=mp_group)
    max_loss_diff = (mp_loss_for_compare - ref_loss).abs().item()
    assert torch.allclose(
        mp_loss_for_compare, ref_loss, rtol=1e-4, atol=1e-5
    ), f"Loss diff: {max_loss_diff:.3e}"

    # 3. Compare gradients
    mp_loss = out_mp.out.sum() * world_size / (B * T_FULL)
    mp_loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=mp_group)
    mp_grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float("inf")
    )
    grad_diff = (mp_grad_norm - ref_grad_norm).abs().max().item()
    assert torch.allclose(
        mp_grad_norm, ref_grad_norm, rtol=1e-4, atol=1e-5
    ), f"Grad norm diff: {grad_diff:.2e}"

    if rank == 0:
        print(f"  1. Output max diff: {out_diff:.2e}")
        print(f"  2. Loss diff: {max_loss_diff:.2e}")
        print(f"  3. Grad norm diff: {grad_diff:.2e}")
