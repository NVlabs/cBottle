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
import random
import pytest
import torch

from cbottle.training.super_resolution import BatchedPatchIterator, Mockdataset
from cbottle import healpix_utils


class _InnerModel(torch.nn.Module):
    def __init__(self, channels: int, level: int):
        super().__init__()
        nside = 2**level
        pe = torch.randn(channels * 12 * nside * nside).view(
            channels, 12 * nside * nside
        )
        self.pos_embed = torch.nn.Parameter(pe.cuda(), requires_grad=False)
        self.model = self


class DummyNet(torch.nn.Module):
    """Dummy network that fulfils `net.module.model.pos_embed` contract."""

    def __init__(self, channels: int = 12, level: int = 10):
        super().__init__()
        self.module = _InnerModel(channels, level)


def _slice_padded_tensor(
    x: torch.Tensor,
    faces: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    img_res: int,
) -> torch.Tensor:
    # x shape: (C, 12, N+2p, N+2p)
    # faces,rows,cols have shape (B,)
    slices = [
        x[:, f, r : r + img_res, c : c + img_res]
        for f, r, c in zip(faces.tolist(), rows.tolist(), cols.tolist())
    ]
    return torch.stack(slices, dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_extract_positional_embeddings():
    dataset = Mockdataset()
    sr_level = 10
    channels = 12
    lr_level = 6
    img_res = 128
    padding = img_res // 2

    net = DummyNet(channels=channels, level=sr_level)
    patch_iterator = BatchedPatchIterator(
        net,
        training_dataset_grid=dataset.grid,
        lr_level=lr_level,
        img_resolution=img_res,
        padding=padding,
    )

    nside_padded = 2**sr_level + 2 * padding
    padded_pe = (
        torch.arange(channels * 12 * nside_padded * nside_padded)
        .view(channels, 12, nside_padded, nside_padded)
        .float()
        .cuda()
    )

    # Pick two arbitrary patch locations
    batch_size = 2
    face = torch.tensor([0, 5])
    row = torch.tensor([1, 3])
    col = torch.tensor([2, 4])

    # Assemble patch_coord_map: each entry holds id = face*N*N + row*N + col
    ids = face * nside_padded * nside_padded + row * nside_padded + col
    patch_coord_map = (
        ids.view(batch_size, 1, 1, 1).repeat(1, 1, img_res, img_res).long().cuda()
    )

    result = patch_iterator.extract_positional_embeddings(patch_coord_map, padded_pe)
    expected = _slice_padded_tensor(padded_pe, face, row, col, img_res)

    assert result.shape == (batch_size, channels, img_res, img_res)
    assert torch.equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_patch_iterator():
    dataset = Mockdataset()
    sr_level = 10
    channels = 12
    lr_level = 6
    img_res = 128
    padding = img_res // 2
    n_batches = 10
    batch_size = 8
    shuffle = True

    net = DummyNet(channels=channels, level=sr_level)
    patch_iterator = BatchedPatchIterator(
        net,
        training_dataset_grid=dataset.grid,
        lr_level=lr_level,
        img_resolution=img_res,
        padding=padding,
        shuffle=shuffle,
    )
    batch = dataset[0]

    random.seed(42)
    results = []
    for i, (lpe, ltarget, llr) in enumerate(patch_iterator(batch, batch_size)):
        results.append((lpe, ltarget, llr))
        if i == n_batches - 1:
            break

    random.seed(42)
    reference = []
    target = batch["target"].cuda()[:, 0]
    lr, global_lr = patch_iterator.compute_low_res_conditioning(target)
    for i, (lpe, ltarget, llr, _) in enumerate(
        healpix_utils.to_patches(
            [net.module.model.pos_embed, target, lr],
            patch_size=img_res,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    ):
        llr = torch.cat(
            (llr, global_lr.repeat(llr.shape[0], 1, 1, 1)),
            dim=1,
        )
        reference.append((lpe, ltarget, llr))
        if i == n_batches - 1:
            break

    for i, (lpe, ltarget, llr) in enumerate(results):
        assert lpe.shape == (batch_size, channels, img_res, img_res)
        assert ltarget.shape == (batch_size, channels, img_res, img_res)
        assert llr.shape == (batch_size, 2 * channels, img_res, img_res)

        lpe_ref, ltarget_ref, llr_ref = reference[i]
        assert torch.allclose(ltarget, ltarget_ref)
        assert torch.allclose(llr, llr_ref)
        assert torch.allclose(lpe, lpe_ref)
