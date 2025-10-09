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
import torch
import einops
import earth2grid
import math


def patch_index_from_bounding_box(order, box, patch_size, overlap_size, device="cuda"):
    nside = 2**order
    src_grid = earth2grid.healpix.Grid(
        level=order, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    lat = src_grid.lat
    lon = src_grid.lon
    lat_south, lon_west, lat_north, lon_east = box

    # Normalize longitudes to [0, 360)
    lon = lon % 360
    lon_west = lon_west % 360
    lon_east = lon_east % 360

    # Latitude mask
    lat_mask = (lat >= lat_south) & (lat <= lat_north)

    # Longitude mask (handle dateline crossing)
    if lon_west <= lon_east:
        lon_mask = (lon >= lon_west) & (lon <= lon_east)
    else:
        # Box crosses the dateline
        lon_mask = (lon >= lon_west) | (lon <= lon_east)

    mask = torch.from_numpy(lat_mask & lon_mask)

    xy_grid = earth2grid.healpix.Grid(
        level=order, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
    )

    # TODO: image_patching behaves differently when the input mask is on CPU vs. CUDA. Investigate and ensure consistent behavior across devices.
    mask_patch = image_patching(
        mask[None, None], src_grid, xy_grid, nside, patch_size, overlap_size
    )[0]

    # Find which batch entries have any non-zero values
    inbox_patches = (mask_patch != 0).reshape(mask_patch.shape[0], -1).any(dim=1)

    # Get indices of batches with non-zero content
    inbox_patch_index = inbox_patches.nonzero(as_tuple=True)[0]

    return inbox_patch_index


def image_patching(img, src_grid, xy_grid, nside, patch_size, overlap_size):
    # Reorder image to [N, 12, nside, nside]
    img = src_grid.reorder(xy_grid.pixel_order, img)
    stride = patch_size - overlap_size
    # Pad image
    img_reshaped = einops.rearrange(
        img, "n c (f x y) -> (n c) f x y", f=12, x=nside, y=nside
    )

    padded = earth2grid.healpix.pad(img_reshaped, padding=overlap_size)
    padded = einops.rearrange(
        padded,
        "(n c) f x y -> n f c x y",
        n=img.shape[0],
    )

    padded_patch_size = padded.shape[-1]
    patches = padded.unfold(4, patch_size, stride).unfold(3, patch_size, stride)
    cx = patches.shape[3]
    cy = patches.shape[4]

    # Merge all batch dimensions
    patches = einops.rearrange(
        patches,
        "n f c cx cy x y -> (n f cx cy) c x y",
        x=patch_size,
        y=patch_size,
    ).permute((0, 1, 3, 2))
    return patches, padded_patch_size, cx, cy, stride


def apply_on_patches(
    denoise,
    patch_size,
    overlap_size,
    x_hat,
    x_lr,
    t_hat,
    class_labels,
    batch_size=64,
    pbar=None,
    global_lr=None,
    inbox_patch_index=None,
    window=None,
    device="cuda",
):
    """
    Args:
        denoise: used like this `out = denoise(patches, sigma)
        x_hat: Latent map, NEST convention
        x_lr: condition, NEST convention
    """
    order = int(np.log2(np.sqrt(x_lr.shape[-1] // 12)))
    nside = 2**order
    src_grid = earth2grid.healpix.Grid(
        level=order, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    xy_grid = earth2grid.healpix.Grid(
        level=order, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
    )

    # Reorder image to [N, 12, nside, nside]
    x_hat_patch, padded_patch_size, cx, cy, stride = image_patching(
        x_hat, src_grid, xy_grid, nside, patch_size, overlap_size
    )
    x_lr_patch = image_patching(
        x_lr, src_grid, xy_grid, nside, patch_size, overlap_size
    )[0]
    augment_labels = None

    if global_lr is not None:
        x_lr_patch = torch.cat(
            (x_lr_patch, global_lr.repeat(x_lr_patch.shape[0], 1, 1, 1)), dim=1
        )

    pos_embd_patch = image_patching(
        denoise.model.pos_embed[None,],
        src_grid,
        xy_grid,
        nside,
        patch_size,
        overlap_size,
    )[0]

    if patch_size is None:
        batch_size = x_hat_patch.shape[0]
        x_hat_patch = einops.rearrange(x_hat_patch, "(n f) c x y -> n c f (x y)", f=12)
        x_lr_patch = einops.rearrange(x_lr_patch, "(n f) c x y -> n c f (x y)", f=12)
        pos_embd_patch = einops.rearrange(
            pos_embd_patch, "(n f) c x y -> n c f (x y)", f=12
        )

    # divide the patchified maps into batches
    out = x_lr_patch[:, : int(x_lr_patch.shape[1] / 2)].clone().to(device)
    batch_index = torch.arange(x_hat_patch.shape[0])
    if inbox_patch_index is not None:
        batch_index = torch.tensor(inbox_patch_index)
    num_batch = math.ceil(len(batch_index) / batch_size)

    # denoise
    for batch in range(num_batch):
        patch_indices_in_batch = batch_index[
            batch * batch_size : (batch + 1) * batch_size
        ].to(device)
        out[patch_indices_in_batch] = denoise(
            x_hat_patch[patch_indices_in_batch].to(device),
            t_hat,
            class_labels=class_labels,
            condition=x_lr_patch[patch_indices_in_batch].to(device),
            position_embedding=pos_embd_patch[patch_indices_in_batch].to(device),
            augment_labels=augment_labels,
        ).to(torch.float64)
    if pbar is not None:
        pbar.update()

    # Un-merge batch dim of output
    if patch_size and (window is not None):
        if window.shape[0] == 1:
            window = window.tile((out.shape[0], out.shape[1], 1, 1))

        out = einops.rearrange(
            out * window,
            "(n f cx cy) c x y -> n (f c x y) (cx cy)",
            x=patch_size,
            y=patch_size,
            f=12,
            cx=cx,
            cy=cy,
        )
        weights = einops.rearrange(
            window,
            "(n f cx cy) c x y -> n (f c x y) (cx cy)",
            x=patch_size,
            y=patch_size,
            f=12,
            cx=cx,
            cy=cy,
        )

        # Compute average of overlapping patches
        weights = torch.nn.functional.fold(
            weights,
            (padded_patch_size, padded_patch_size),
            (patch_size, patch_size),
            stride=stride,
        )
        out = torch.nn.functional.fold(
            out,
            (padded_patch_size, padded_patch_size),
            (patch_size, patch_size),
            stride=stride,
        )
        out = out / weights

        # Reshape again and discard padding
        out = einops.rearrange(
            out,
            "n (f c) x y -> n f c x y",
            f=12,
        )
        weights = einops.rearrange(
            weights,
            "n (f c) x y -> n f c x y",
            f=12,
        )
        out = out[
            ...,
            overlap_size : nside + overlap_size,
            overlap_size : nside + overlap_size,
        ]
        weights = weights[
            ...,
            overlap_size : nside + overlap_size,
            overlap_size : nside + overlap_size,
        ]

        out_xy = einops.rearrange(
            out,
            "n f c x y -> n c (f x y)",
            x=nside,
            y=nside,
            f=12,
        )

    elif patch_size:
        out = einops.rearrange(
            out,
            "(n f cx cy) c x y -> n (f c x y) (cx cy)",
            x=patch_size,
            y=patch_size,
            f=12,
            cx=cx,
            cy=cy,
        )

        # Compute average of overlapping patches
        weights = torch.nn.functional.fold(
            torch.ones_like(out),
            (padded_patch_size, padded_patch_size),
            (patch_size, patch_size),
            stride=stride,
        )
        out = torch.nn.functional.fold(
            out,
            (padded_patch_size, padded_patch_size),
            (patch_size, patch_size),
            stride=stride,
        )
        out = out / weights

        # Reshape again and discard padding
        out = einops.rearrange(
            out,
            "n (f c) x y -> n f c x y",
            f=12,
        )
        out = out[
            ...,
            overlap_size : nside + overlap_size,
            overlap_size : nside + overlap_size,
        ]

        out_xy = einops.rearrange(
            out,
            "n f c x y -> n c (f x y)",
            x=nside,
            y=nside,
            f=12,
        )
    else:
        out_xy = einops.rearrange(
            out,
            "n c f (x y) -> n c (f x y)",
            x=nside,
        )
    return xy_grid.reorder(src_grid.pixel_order, out_xy)
