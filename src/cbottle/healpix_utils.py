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
import numpy as np
from earth2grid import healpix
import earth2grid
import random
import torch
import einops
from functools import lru_cache


def hpxpad2ring(x):
    return healpix.reorder(x, healpix.HEALPIX_PAD_XY, healpix.PixelOrder.RING)


def reorder_local_patch_to_xy(map):
    n = int(np.sqrt(map.shape[-1]))
    i_xy = np.arange(n * n)
    i_nest = healpix.xy2nest(n, i_xy)
    data_as_xy = map[..., i_nest]
    return data_as_xy


def get_global_index(nside, patch_size, patch_num):
    """Get the global NEST index corresponding to a patch

    Returns [patch_size, patch_size] array of ints
    """
    assert nside % patch_size == 0
    assert np.all(patch_num < get_num_patches(nside, patch_size))
    i_local_xy = np.arange(patch_size * patch_size)
    i_local_nest = healpix.xy2nest(patch_size, i_local_xy)
    i_global_nest = i_local_nest + patch_num * patch_size**2
    return i_global_nest.reshape([patch_size, patch_size])


def get_num_patches(nside, patch_size):
    assert nside % patch_size == 0
    patch_nside = nside // patch_size
    return patch_nside**2 * 12


def average_pool(map):
    """Coarsen map in NEST convention"""
    npix = map.shape[-1]
    shape = map.shape[:-1]
    return map.reshape(shape + (npix // 4, 4)).mean(-1)


@lru_cache(maxsize=4)
def _get_regridder_cached(level: int, device_str: str):
    """
    Build once and cache a regridder for a given healpix level and device.
    """
    src_grid = earth2grid.healpix.Grid(level=level, pixel_order=healpix.PixelOrder.NEST)
    dest_grid = earth2grid.healpix.Grid(level=level, pixel_order=healpix.HEALPIX_PAD_XY)
    regridder = earth2grid.get_regridder(src_grid, dest_grid).float()
    return regridder.to(torch.device(device_str))


def to_faces(map):
    """Give a padded view of the data in a NEST convention map

    (*, npix) -> (*, f, nside, nside)

    """
    hpx_level = healpix.npix2level(map.shape[-1])
    regridder = _get_regridder_cached(hpx_level, str(map.device))
    xy_map = regridder(map)
    nside = 2**hpx_level
    xy_map = xy_map.view(xy_map.shape[:-1] + (12, nside, nside))
    return xy_map


def to_patches(
    nest_tensors,
    *,
    patch_size,
    pre_padded_tensors=None,
    padding=None,
    stride=None,
    batch_size: int = 1,
    shuffle: bool = True,
):
    stride = stride or patch_size // 2
    padding = padding or patch_size // 2
    padded_tensors = [healpix.pad(to_faces(a), padding=padding) for a in nest_tensors]
    del nest_tensors
    torch.cuda.empty_cache()

    if pre_padded_tensors is not None:
        for tensor in pre_padded_tensors:
            padded_tensors.append(tensor)

    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)

    def unfold_and_batch(x):
        c = 0
        f = 1
        x = x.transpose(c, f)

        if x.shape[1] > 20:
            # To reduce GPU mem usage, allocate and patch on cpu.
            # However, this incurs some gpu->cpu data transfer overhead in creating uf
            temp = unfold(x[0:1])
            uf = torch.zeros((12, temp.shape[1], temp.shape[2]))
            uf[0:1] = temp
            for face in range(1, 12):
                uf[face : face + 1] = unfold(x[face : face + 1])
            patches = einops.rearrange(
                uf,
                "f (c x y) l -> (f l) c x y",
                c=x.shape[1],
                x=patch_size,
                y=patch_size,
            )
        else:
            uf = unfold(x)
            patches = einops.rearrange(
                uf,
                "f (c x y) l -> (f l) c x y",
                c=x.shape[1],
                x=patch_size,
                y=patch_size,
            )
        return torch.split(patches, batch_size, dim=0)

    patches = [unfold_and_batch(a) for a in padded_tensors]

    del padded_tensors
    torch.cuda.empty_cache()
    nbatches = len(patches[0])

    index = list(range(nbatches))
    if shuffle:
        random.shuffle(index)

    for i in index:
        is_last = i == index[-1]
        yield [p[i] for p in patches] + [is_last]
