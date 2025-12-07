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
Distributed utilities for working with [b t x c] shaped data

there are two states
    - t-sharded (the input)
    - x-sharded (used for temporal attention)
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.distributed as dist
import einops
from torch.distributed.nn.functional import all_to_all_single

# Debug flag - set to True to enable gradient flow debugging
DEBUG_GRADIENTS = False


def _debug_tensor(tensor, name):
    """Print debug info about tensor's gradient status."""
    if not DEBUG_GRADIENTS:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(
        f"[rank {rank}] {name}: shape={tensor.shape}, requires_grad={tensor.requires_grad}, grad_fn={tensor.grad_fn}"
    )


def unwrap_net(net):
    """Unwrap DDP/compiled network to access model attributes.

    Handles: DDP(compiled(net)) -> net
    """
    # Unwrap DDP
    if hasattr(net, "module"):
        net = net.module
    # Unwrap torch.compile (OptimizedModule has _orig_mod)
    if hasattr(net, "_orig_mod"):
        net = net._orig_mod
    return net


@dataclass
class DistributedConfig:
    """Configuration for distributed training with data and model parallelism.

    Attributes:
        data_rank: Rank for data parallelism (dataset sharding)
        data_world_size: World size for data parallelism
        model_rank: Rank for model parallelism (time dimension sharding)
        model_world_size: World size for model parallelism
    """

    data_rank: int
    data_world_size: int
    model_rank: int
    model_world_size: int


def compute_t_split(t_full: int, n: int) -> list[int]:
    """Compute how T is distributed across n ranks (first ranks get +1 if uneven)."""
    base = t_full // n
    remainder = t_full % n
    return [base + 1 if r < remainder else base for r in range(n)]


def shard_x(tensor, group, t_full: Optional[int] = None):
    """Unshard t and shard x across ranks. Supports uneven T.

    Args:
        tensor: (b, t_local, n * x_local, c) where t_local is this rank's portion of T
        group: model parallel group
        t_full: total T across all ranks. If None, assumes even split.

    Returns:
        (b, t_full, x_local, c)
    """
    n = dist.get_world_size(group)
    b, t_local, x_full, c = tensor.shape

    assert x_full % n == 0, f"x ({x_full}) must be divisible by world_size ({n})"
    x_local = x_full // n

    if t_full is None:
        t_full = t_local * n
    t_splits = compute_t_split(t_full, n)

    # Fast path: even split
    if all(t == t_splits[0] for t in t_splits):
        tensor = einops.rearrange(tensor, "b t (n x) c -> n b t x c", n=n)
        tensor = tensor.contiguous()
        output = torch.empty_like(tensor)
        output = all_to_all_single(output, tensor, group=group)
        result = einops.rearrange(output, "n b t x c -> b (n t) x c")
        return result

    # Uneven path: move T to the front so split sizes are just T counts
    flat_in = einops.rearrange(tensor, "b t (n x) c -> (n t) b x c", n=n).contiguous()

    # Split sizes are just T counts (each "row" is [b, x_local, c])
    input_split_sizes = [t_local] * n  # send t_local rows to each dest
    output_split_sizes = t_splits  # receive t_splits[r] rows from rank r

    flat_out = tensor.new_empty((t_full, b, x_local, c))

    flat_out = all_to_all_single(
        flat_out,
        flat_in,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )

    result = einops.rearrange(flat_out, "t b x c -> b t x c")
    return result


def shard_t(tensor, group, t_full: Optional[int] = None):
    """Unshard x and shard t across ranks. Supports uneven T.

    Args:
        tensor: (b, t_full, x_local, c) with full T, sharded x
        group: model parallel group
        t_full: total T. If None, inferred from tensor.

    Returns:
        (b, t_local, x_full, c) where t_local is this rank's portion
    """
    n = dist.get_world_size(group)
    rank = dist.get_rank(group)
    b, t_tensor, x_local, c = tensor.shape

    if t_full is None:
        t_full = t_tensor

    t_splits = compute_t_split(t_full, n)
    t_local = t_splits[rank]

    # Fast path: even split
    if all(t == t_splits[0] for t in t_splits):
        tensor = einops.rearrange(tensor, "b (n t) x c -> n b t x c", n=n)
        tensor = tensor.contiguous()
        output = torch.empty_like(tensor)
        output = all_to_all_single(output, tensor, group=group)
        result = einops.rearrange(output, "n b t x c -> b t (n x) c")
        return result

    # Uneven path: move T to the front so split sizes are just T counts
    flat_in = einops.rearrange(tensor, "b t x c -> t b x c")

    input_split_sizes = t_splits  # send t_splits[r] rows to rank r
    output_split_sizes = [t_local] * n  # receive t_local rows from each src

    flat_out = tensor.new_empty((n * t_local, b, x_local, c))

    flat_out = all_to_all_single(
        flat_out,
        flat_in,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )

    result = einops.rearrange(flat_out, "(n t) b x c -> b t (n x) c", n=n, t=t_local)
    return result


# UNet variants (channels-first: [b, c, t, x])
def shard_x_bctx(tensor, group, t_full: Optional[int] = None):
    """Unshard t and shard x for [b, c, t, x] tensors (UNet format)."""
    tensor = tensor.permute(0, 2, 3, 1)  # [b,c,t,x] -> [b,t,x,c]
    tensor = shard_x(tensor, group, t_full=t_full)
    return tensor.permute(0, 3, 1, 2)  # [b,t,x,c] -> [b,c,t,x]


def shard_t_bctx(tensor, group, t_full: Optional[int] = None):
    """Unshard x and shard t for [b, c, t, x] tensors (UNet format)."""
    tensor = tensor.permute(0, 2, 3, 1)  # [b,c,t,x] -> [b,t,x,c]
    tensor = shard_t(tensor, group, t_full=t_full)
    return tensor.permute(0, 3, 1, 2)  # [b,t,x,c] -> [b,c,t,x]


def make_mp_randn_like(t_bounds, t_dim=2):
    """Create a randn_like function that handles MP sharding.

    Generates noise for full T dimension then slices to local T,
    ensuring all MP ranks consume the same amount of random numbers.

    Args:
        t_bounds: (t_start, t_end, t_full) or None for non-MP
        t_dim: which dimension is the time dimension (default 2 for [B,C,T,X])

    Returns:
        A randn_like function with .t_bounds attribute attached
    """
    if t_bounds is None:

        def randn_like(tensor):
            return torch.randn_like(tensor)

        randn_like.t_bounds = None
        return randn_like

    t_start, t_end, t_full = t_bounds

    def randn_like(tensor):
        full_shape = list(tensor.shape)
        full_shape[t_dim] = t_full
        # Create a dummy tensor with full shape to use randn_like
        full_ref = tensor.new_empty(full_shape)
        full_noise = torch.randn_like(full_ref)
        slices = [slice(None)] * len(full_shape)
        slices[t_dim] = slice(t_start, t_end)
        return full_noise[tuple(slices)]

    randn_like.t_bounds = t_bounds
    return randn_like
