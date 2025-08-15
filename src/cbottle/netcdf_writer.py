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
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import cftime
from cbottle.inference import Coords
import netCDF4 as nc
import earth2grid
import numpy as np
import torch

logger = logging.getLogger(__name__)

NetCDFFormat = Literal[
    "NETCDF4",
    "NETCDF4_CLASSIC",
    "NETCDF3_CLASSIC",
    "NETCDF3_64BIT_OFFSET",
    "NETCDF3_64BIT_DATA",
]


@dataclass
class NetCDFConfig:
    """Configuration for netCDF file creation"""

    hpx_level: int
    time_units: str = "seconds since 1900-1-1 0:0:0"
    calendar: str = "proleptic_gregorian"
    format: NetCDFFormat = "NETCDF3_64BIT_DATA"
    attrs: Optional[Dict[str, Any]] = None


class NetCDFWriter:
    """Helper class for writing data to netCDF files in the standard format"""

    pixel_order = earth2grid.healpix.PixelOrder.RING

    def __init__(
        self,
        output_path: str,
        config: NetCDFConfig,
        channels: list[str],
        rank: int | None = None,
        add_video_variables: bool = False,
    ):
        self.output_path = output_path
        self.config = config
        self.nc_path = os.path.join(output_path, f"{rank}.nc")
        self.channels = channels
        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Initialize netCDF file if it doesn't exist
        if not os.path.exists(self.nc_path):
            self._init_netcdf()
            self._add_field_variables()
            if add_video_variables:
                self._add_video_variables()

        # Initialize time index based on existing file
        with nc.Dataset(self.nc_path, "r") as ds:
            self.time_index = len(ds["time"])

    def _init_netcdf(self):
        """Initialize the netCDF file with the correct structure"""
        logger.info(f"Initializing netCDF file at {self.nc_path}")
        with nc.Dataset(self.nc_path, "w", format=self.config.format) as ds:
            ds.history = shlex.join(sys.argv)

            # Add any additional attributes
            if self.config.attrs:
                for k, v in self.config.attrs.items():
                    setattr(ds, k, v)

            # Create dimensions
            npix = 12 * (2**self.config.hpx_level) ** 2
            ds.createDimension("pix", npix)
            ds.createDimension("time")

            # Create grid metadata
            ds.createDimension("crs", 1)
            crs = ds.createVariable("crs", dimensions=("crs"), datatype="f")
            crs.grid_mapping_name = "healpix"
            crs.healpix_nside = 2**self.config.hpx_level
            crs.healpix_order = "ring"

            # Create time variable
            ds.createVariable("time", dimensions=("time",), datatype="i8")
            ds["time"].units = self.config.time_units
            ds["time"].calendar = self.config.calendar

    def _add_field_variables(self):
        """Add variables for each channel to the netCDF file"""
        with nc.Dataset(self.nc_path, "r+") as ds:
            npix = 12 * (2**self.config.hpx_level) ** 2
            for field in self.channels:
                v = ds.createVariable(
                    field,
                    dimensions=("time", "pix"),
                    datatype="f",
                    chunksizes=(1, npix),
                    fill_value=np.nan,
                )
                v.grid_mapping = "crs"

    def _add_video_variables(self):
        """Add video-specific variables to the netCDF file"""
        with nc.Dataset(self.nc_path, "r+") as ds:
            v = ds.createVariable(
                "frame_source_flag",
                dimensions=("time",),
                datatype="i1",
                fill_value=-1,
            )
            v.description = "Frame source flag (0=ground truth; 1=generated; 2=generated with GT conditioning)"
            v.flag_values = np.array([0, 1, 2], dtype="i1")
            v.flag_meanings = "ground_truth generated conditional_generation"

            v = ds.createVariable(
                "lead_time",
                dimensions=("time",),
                datatype="i4",
                fill_value=-1,
            )
            v.description = "Lead time in hours"
            v.time_units = "hours"

    def write_target(
        self,
        target: torch.Tensor,
        coords: Coords,
        timestamps: torch.Tensor,
        **kwargs,
    ):
        out = coords.grid.reorder(self.pixel_order, target)
        output = {c: out[:, k] for k, c in enumerate(coords.batch_info.channels)}
        self.write_batch(output, timestamps, **kwargs)

    def write_batch(
        self,
        fields: Dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        scalars: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Write a batch of data to the netCDF file

        Args:
            fields: Dictionary mapping field names to tensors of shape (B, T, npix)
            timestamps: Tensor of timestamps in seconds since 1900-1-1 - shape (B,) or (B, T)
            scalars: Dictionary mapping field names to tensors of shape (B, T)
        """
        with nc.Dataset(self.nc_path, "r+") as ds:
            if timestamps.ndim == 1:  # (B,) -> (B, 1)
                timestamps = timestamps.unsqueeze(-1)
            elif timestamps.ndim != 2:
                raise ValueError(
                    f"Timestamps must have shape (B,) or (B, T), got {timestamps.shape}"
                )

            batch_size = timestamps.shape[0]
            time_length = timestamps.shape[1]

            # Write time data
            for j in range(batch_size):
                for t in range(time_length):
                    curr_timestamp = timestamps[j, t]
                    curr_idx = self.time_index + j * time_length + t
                    time = cftime.num2date(
                        curr_timestamp,
                        units=self.config.time_units,
                        calendar=self.config.calendar,
                    )
                    logger.info(f"Writing time {time} at index {curr_idx}")
                    ds["time"][curr_idx] = curr_timestamp

            # Write field data
            for field in fields:
                arr = fields[field].cpu()
                for j in range(batch_size):
                    for t in range(time_length):
                        curr_idx = self.time_index + j * time_length + t
                        ds[field][curr_idx] = arr[j, t, :].numpy()

            # Write scalar data
            if scalars is not None:
                for key, value in scalars.items():
                    arr = value.cpu()
                    for j in range(batch_size):
                        for t in range(time_length):
                            curr_idx = self.time_index + j * time_length + t
                            ds[key][curr_idx] = arr[j, t]

            self.time_index += batch_size * time_length
