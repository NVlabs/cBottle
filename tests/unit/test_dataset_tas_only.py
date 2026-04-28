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
import datetime

import cftime
import numpy as np
import torch

from cbottle.datasets import dataset_3d


def test_get_loaders_tas_only_with_sst(monkeypatch):
    calls = {"tas": None, "sst_offset": None}

    class DummyNetCDFTasLoader:
        def __init__(self, directory: str, variable_name: str = "tas"):
            calls["tas"] = (directory, variable_name)

    class DummyAmipSSTLoader:
        def __init__(self, grid, sst_offset: float = 0.0):
            calls["sst_offset"] = sst_offset
            self.grid = grid

    class DummyGrid:
        pass

    monkeypatch.setattr(dataset_3d, "NetCDFTasLoader", DummyNetCDFTasLoader)
    monkeypatch.setattr(dataset_3d, "AmipSSTLoader", DummyAmipSSTLoader)
    monkeypatch.setattr(
        dataset_3d.earth2grid.healpix, "Grid", lambda *args, **kwargs: DummyGrid()
    )

    loaders = dataset_3d._get_loaders(
        "tas_only",
        sst_input=True,
        sst_offset=1.5,
        variable_config=dataset_3d.VARIABLE_CONFIGS["tas_only"],
    )

    assert len(loaders) == 2
    assert isinstance(loaders[0], DummyNetCDFTasLoader)
    assert isinstance(loaders[1], DummyAmipSSTLoader)
    assert calls["tas"] == (dataset_3d.config.TAS_ONLY_NETCDF_DIR, "tas")
    assert calls["sst_offset"] == 1.5


def test_encode_netcdf_tas_applies_zero_spatial_mask():
    npix = 12 * 4**dataset_3d.HPX_LEVEL
    variable_config = dataset_3d.VARIABLE_CONFIGS["tas_only"]
    data = {
        ("tas", dataset_3d.NO_LEVEL): np.ones(npix, dtype=np.float32),
        ("tosbcs", dataset_3d.NO_LEVEL): np.ones(npix, dtype=np.float32) * 300.0,
    }
    time = cftime.DatetimeGregorian(2001, 1, 1, 0, 0, 0)
    spatial_mask = torch.zeros(npix, dtype=torch.float32)

    out = dataset_3d._encode_netcdf_tas(
        time=time,
        data=data,
        label=dataset_3d.LABELS.index("era5"),
        mean=dataset_3d.get_mean(variable_config),
        scale=dataset_3d.get_std(variable_config),
        variable_config=variable_config,
        spatial_mask=spatial_mask,
    )

    assert out["target"].shape == (1, 1, npix)
    assert torch.all(out["target"] == 0)
    assert torch.equal(out["spatial_mask"], spatial_mask)
    day_start = time.replace(hour=0, minute=0, second=0)
    year_start = day_start.replace(month=1, day=1)
    expected_second_of_day = (time - day_start) / datetime.timedelta(seconds=1)
    expected_day_of_year = (time - year_start) / datetime.timedelta(seconds=86400)
    assert out["second_of_day"].item() == expected_second_of_day
    assert out["day_of_year"].item() == expected_day_of_year
