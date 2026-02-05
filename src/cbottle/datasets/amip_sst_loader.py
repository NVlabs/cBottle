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
import os
import shutil
import sys
import tempfile
import urllib.parse
import earth2grid
import requests
import torch
import xarray

from cbottle.config import environment as config
from cbottle.storage import get_storage_options
from pathlib import Path


AMIP_SST_FILENAME = (
    "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc"
)
AMIP_SST_URL = (
    "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/input4MIPs/CMIP6Plus/CMIP/"
    f"PCMDI/PCMDI-AMIP-1-1-9/ocean/mon/tosbcs/gn/v20230512/{AMIP_SST_FILENAME}"
)

AIMIP_SST_FILENAME = "ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc"
AIMIP_SST_URL = (
    f"https://zenodo.org/records/17065758/files/{AIMIP_SST_FILENAME}?download=1"
)


def _download_sst(output_path: Path):
    """Download the AMIP SST data from NGC

    This is much faster than the THREDDS server above
    """

    print(f"Downloading SST data to {output_path} ...", file=sys.stderr)

    url = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/earth-2/cbottle/1.2/files?redirect=true&path=amip_midmonth_sst.nc"
    output_file = "amip_midmonth_sst.nc"
    with tempfile.TemporaryDirectory() as dir_:
        tmp_out = os.path.join(dir_, output_file)

        response = requests.get(url, allow_redirects=True)
        with open(tmp_out, "wb") as file:
            file.write(response.content)
            shutil.move(tmp_out, output_path)
    print("Successfully downloaded SST data", file=sys.stderr)


def _download_aimip_sst(output_path: Path):
    print(f"Downloading AIMIP SST data to {output_path} ...", file=sys.stderr)
    ds = xarray.open_dataset(AIMIP_SST_URL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = tempfile.mktemp(
        dir=output_path.parent.as_posix(), prefix=output_path.name
    )
    ds.to_netcdf(tmp_out)  # Save to netCDF
    shutil.move(tmp_out, output_path)
    print("Successfully downloaded AIMIP SST data", file=sys.stderr)


def _is_file(path: str) -> str:
    url = urllib.parse.urlparse(path)
    return url.scheme in ["file", ""]


class AmipSSTLoader:
    """AMIP-II SST Forcing dataset

    This is derived from the observed SSTs but is adjusted so that the monthly
    average of linearly interpolated values equals the observed monthly mean. This is
    achieved by solving a linear system enforcing the constraint, with the true obs as the RHS
    This procedure is explained at length here: https://pcmdi.llnl.gov/mips/amip/details/index.html


    # Data Access:

    The data can be downloaded from the Earth System Grid Federation (https://aims2.llnl.gov/)

    The filename is input4MIPs.CMIP6Plus.CMIP.PCMDI.PCMDI-AMIP-1-1-9.ocean.mon.tosbcs.gn

    The unadjusted observed monthly means are input4MIPs.CMIP6Plus.CMIP.PCMDI.PCMDI-AMIP-1-1-9.ocean.mon.tos.gn

    """

    units = "Kelvin"
    path = config.AMIP_MID_MONTH_SST

    def __init__(self, target_grid=None):
        self.ensure_downloaded()

        self.ds = xarray.open_dataset(
            self.path,
            engine="h5netcdf",
            storage_options=get_storage_options(config.AMIP_MID_MONTH_SST_PROFILE),
        ).load()

        self.times = self.ds.indexes["time"]

        if target_grid is not None:
            lon_center = self.ds.lon.values
            # need to workaround bug where earth2grid fails to interpolate in circular manner
            # if lon[0] > 0
            # hack: rotate both src and target grids by the same amount so that src_lon[0] == 0
            # See https://github.com/NVlabs/earth2grid/issues/21
            src_lon = lon_center - lon_center[0]
            target_lon = (target_grid.lon - lon_center[0]) % 360

            grid = earth2grid.latlon.LatLonGrid(self.ds.lat.values, src_lon)
            self._regridder = grid.get_bilinear_regridder_to(
                target_grid.lat, lon=target_lon
            )

    @classmethod
    def ensure_downloaded(cls):
        path = Path(cls.path)

        if path.exists():
            print(f"SST data already exists at {path}", file=sys.stderr)
            return

        if _is_file(cls.path):
            _download_sst(Path(cls.path))

    async def sel_time(self, times):
        data = self.interp(times)
        return {("tosbcs", -1): self.regrid(data)}

    def interp(self, time: datetime.datetime):
        """Linearly interpolate between the available points"""
        return torch.from_numpy(
            self.ds["tosbcs"].interp(time=time, method="linear").values + 273.15
        )

    def regrid(self, arr):
        return self._regridder(arr)


class AImip_SSTLoader(AmipSSTLoader):
    """AIMIP SST Forcing dataset"""

    path = config.AIMIP_1ST_MONTH_SST

    @classmethod
    def ensure_downloaded(cls):
        path = Path(cls.path)

        if path.exists():
            print(f"SST data already exists at {path}", file=sys.stderr)
            return

        if _is_file(cls.path):
            _download_aimip_sst(Path(cls.path))

    def __init__(self, target_grid=None):
        self.ensure_downloaded()

        self.ds = xarray.open_dataset(
            self.path,
            engine="h5netcdf",
            storage_options=get_storage_options(config.AIMIP_1ST_MONTH_SST_PROFILE),
        ).load()

        self.times = self.ds.indexes["time"]

        if target_grid is not None:
            lon_center = self.ds.longitude.values
            # need to workaround bug where earth2grid fails to interpolate in circular manner
            # if lon[0] > 0
            # hack: rotate both src and target grids by the same amount so that src_lon[0] == 0
            # See https://github.com/NVlabs/earth2grid/issues/21
            src_lon = lon_center - lon_center[0]
            target_lon = (target_grid.lon - lon_center[0]) % 360

            grid = earth2grid.latlon.LatLonGrid(self.ds.latitude.values, src_lon)
            self._regridder = grid.get_bilinear_regridder_to(
                target_grid.lat, lon=target_lon
            )

    def interp(self, time: datetime.datetime):
        """Linearly interpolate between the available points"""
        return torch.from_numpy(
            self.ds["sea_surface_temperature"].interp(time=time, method="linear").values
        )
