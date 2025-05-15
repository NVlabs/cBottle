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
from cbottle.datasets.amip_sst_loader import AmipSSTLoader
from cbottle.datasets import dataset_3d
from cbottle.datasets.dataset_3d import INDEX
from cbottle.storage import StorageConfigError
from cbottle.training.video.frame_masker import FrameMasker

import datetime
import pytest

nchannel = len(INDEX)


def test_AmipSSTDataset():
    try:
        ds = AmipSSTLoader()
    except (FileNotFoundError, PermissionError):
        pytest.skip()

    out = ds.interp(datetime.datetime(2000, 1, 1))
    assert out.shape == (180, 360)


def test_dataset_v6():
    try:
        ds = dataset_3d.get_dataset(split="train")
    except StorageConfigError:
        pytest.skip()

    data = next(iter(ds))
    npix = 12 * 64**2
    assert data["target"].shape == (nchannel, 1, npix)
    assert data["condition"].shape[-1] == npix

    # rsut, rlut, rsds are the only missing channels for era5
    y = data["target"]
    missing = y.isnan().any(dim=(1, 2))
    missing_vars = INDEX[missing.numpy()].get_level_values("variable")
    assert set(missing_vars) == {"rlut", "rsut", "rsds"}


def test_dataset_v6_icon():
    try:
        ds = dataset_3d.get_dataset(split="train", dataset="icon", sst_input=False)
    except StorageConfigError:
        pytest.skip()

    next(iter(ds))


@pytest.mark.parametrize("dataset_name", ["era5", "icon"])
def test_dataset_v6_video(dataset_name):
    frame_masker = FrameMasker(keep_frames=[])
    time_length = 12
    time_step = 6  # hours
    if dataset_name == "icon":
        chunk_size = 56
    else:
        chunk_size = 96
    try:
        ds = dataset_3d.get_dataset(
            split="train",
            dataset=dataset_name,
            chunk_size=chunk_size,
            time_length=time_length,
            time_step=time_step,
            frame_masker=frame_masker,
        )
    except StorageConfigError:
        pytest.skip()

    data = next(iter(ds))
    npix = 12 * 64**2
    assert data["target"].shape == (nchannel, time_length, npix)
    assert data["condition"].shape[-1] == npix

    # verify time_step
    seconds_per_day = 24 * 3600
    for i in range(time_length - 1):
        t1 = int(data["second_of_day"][i].item())
        t2 = int(data["second_of_day"][i + 1].item())
        diff = (t2 - t1) % seconds_per_day

        assert diff == time_step * 3600
