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
import asyncio
import glob
import logging
from collections import defaultdict

import earth2grid
import numpy as np
import pandas as pd
import torch
import xarray as xr

NO_LEVEL = -1
HPX_LEVEL = 6

logger = logging.getLogger(__name__)


class NetCDFTasLoader:
    """Load tas from a directory of netCDF files produced by cbottle inference.

    Files are assumed to be HEALPix level 6, RING order (as written by
    ``NetCDFWriter``). Data is reordered to NEST on output so it matches the
    convention of the existing zarr loaders used in ``dataset_3d``.

    All file handles are kept open and the reorder matrix is precomputed so that
    ``sel_time`` only does numpy indexing + a sparse matmul.
    """

    def __init__(self, directory: str, variable_name: str = "tas"):
        self.variable_name = variable_name
        nc_files = sorted(glob.glob(f"{directory}/*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found in {directory}")

        logger.info("Scanning %d netCDF files in %s", len(nc_files), directory)

        all_times = []
        self._file_index: list[tuple[str, int]] = []
        self._open_datasets: dict[str, xr.Dataset] = {}

        for path in nc_files:
            ds = xr.open_dataset(path, decode_times=True)
            self._open_datasets[path] = ds
            file_times = pd.DatetimeIndex(ds["time"].values)
            for local_idx, t in enumerate(file_times):
                all_times.append(t)
                self._file_index.append((path, local_idx))

        self.times = pd.DatetimeIndex(all_times).sort_values()
        sort_order = pd.DatetimeIndex(all_times).argsort()
        self._file_index = [self._file_index[i] for i in sort_order]
        self._time_to_pos = {t: i for i, t in enumerate(self.times)}

        ring_grid = earth2grid.healpix.Grid(
            HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.RING
        )
        nest_grid = earth2grid.healpix.Grid(
            HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        self._ring_to_nest = ring_grid.get_bilinear_regridder_to(
            nest_grid.lat, nest_grid.lon
        ).float()

    def _load_slice(self, times) -> dict[tuple[str, int], np.ndarray]:
        requests: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for out_idx, t in enumerate(times):
            pos = self._time_to_pos.get(t)
            if pos is None:
                pos = self.times.get_indexer([t], method="nearest")[0]
                if pos == -1:
                    raise KeyError(f"Time {t} not found in loader")
            path, local_idx = self._file_index[pos]
            requests[path].append((out_idx, local_idx))

        npix = 12 * 4**HPX_LEVEL
        out = np.empty((len(times), npix), dtype=np.float32)

        for path, indices in requests.items():
            ds = self._open_datasets[path]
            local_idxs = [li for _, li in indices]
            out_idxs = [oi for oi, _ in indices]
            chunk = (
                ds[self.variable_name].isel(time=local_idxs).values.astype(np.float32)
            )
            chunk_nest = self._ring_to_nest(torch.from_numpy(chunk).float()).numpy()
            out[out_idxs] = chunk_nest

        return {(self.variable_name, NO_LEVEL): out}

    async def sel_time(self, times) -> dict[tuple[str, int], np.ndarray]:
        return await asyncio.to_thread(self._load_slice, times)
