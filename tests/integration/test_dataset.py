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
from cbottle.datasets.amip_sst_loader import AmipSSTLoader, AImip_SSTLoader
from cbottle.datasets import ibtracs
from cbottle.datasets import dataset_3d
from cbottle.storage import StorageConfigError
from cbottle.training.video.frame_masker import FrameMasker
import numpy as np
import asyncio

import datetime
import pytest


def test_AmipSSTDataset():
    try:
        ds = AmipSSTLoader()
    except (FileNotFoundError, PermissionError):
        pytest.skip()

    out = ds.interp(datetime.datetime(2000, 1, 1))
    assert out.shape == (180, 360)


def test_AImipSSTDataset():
    try:
        ds = AImip_SSTLoader()
    except (FileNotFoundError, PermissionError):
        pytest.skip()

    out = ds.interp(datetime.datetime(2000, 1, 1))
    assert out.shape == (721, 1440)


def test_dataset_era5_aimip():
    try:
        ds = dataset_3d.get_dataset(split="train", dataset="era5-aimip")
    except StorageConfigError:
        pytest.skip()
    # Don't fetch data to keep tests fast
    assert ds is not None


def test_dataset_v6():
    try:
        ds = dataset_3d.get_dataset(split="train")
    except StorageConfigError:
        pytest.skip()

    data = next(iter(ds))
    npix = 12 * 64**2
    index = dataset_3d._get_index()
    nchannel = len(index)
    assert data["target"].shape == (nchannel, 1, npix)
    assert data["condition"].shape[-1] == npix

    # rsut, rlut, rsds are the only missing channels for era5
    y = data["target"]
    missing = y.isnan().any(dim=(1, 2))
    missing_vars = index[missing.numpy()].get_level_values("variable")
    assert set(missing_vars) == {"rlut", "rsut", "rsds"}


@pytest.mark.parametrize("name", list(dataset_3d.VARIABLE_CONFIGS))
def test_dataset_v6_icon(name: str):
    variable_config = dataset_3d.VARIABLE_CONFIGS[name]
    try:
        ds = dataset_3d.get_dataset(
            split="train",
            dataset="icon",
            sst_input=False,
            variable_config=variable_config,
        )
    except StorageConfigError:
        pytest.skip()

    sample = next(iter(ds))
    nchannel = len(dataset_3d.get_batch_info(variable_config).channels)
    assert sample["target"].shape[0] == nchannel


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
    index = dataset_3d._get_index()
    nchannel = len(index)
    assert data["target"].shape == (nchannel, time_length, npix)
    assert data["condition"].shape[-1] == npix

    # verify time_step
    seconds_per_day = 24 * 3600
    for i in range(time_length - 1):
        t1 = int(data["second_of_day"][i].item())
        t2 = int(data["second_of_day"][i + 1].item())
        diff = (t2 - t1) % seconds_per_day

        assert diff == time_step * 3600


def test_ibtracs():
    loader = ibtracs.IBTracs()
    times = np.array([np.datetime64("2000-08-01T00:00:00")])
    output = asyncio.run(loader.sel_time(times))
    value = output.pop(("ibtracs_labels", -1))
    assert value.shape == (1, 1, 1, 768)
    assert not output
