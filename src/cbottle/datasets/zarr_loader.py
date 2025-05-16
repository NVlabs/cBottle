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
import zarr
from zarr.core.sync import sync
import xarray as xr
import pandas as pd
import cftime
import asyncio
import numpy as np
import urllib.parse
import logging


NO_LEVEL = -1


def _is_local(path):
    url = urllib.parse.urlparse(path)
    return url.scheme == ""


async def _getitem(array, index):
    i, *index = index
    # adjust to slice indexer since there is a bug in zarr v3 for fancy indexing
    match i:
        case np.ndarray():
            indexer = slice(i.min(), i.max() + 1)
            # if indexer
            ok = i - i.min()
        case _:
            indexer = i
            ok = slice(None)

    chunk = await array.getitem((indexer, *index))
    return chunk[ok]


class ZarrLoader:
    """Load 2d and 3d data from a zarr dataset"""

    def __init__(
        self,
        *,
        path: str,
        variables_3d,
        variables_2d,
        levels,
        level_coord_name: str = "",
        storage_options=None,
        time_sel_method: str | None = None,
    ):
        """
        Args:
            time_sel_method: passed to pd.Index.get_indexer(method=)
        """
        self.time_sel_method = time_sel_method
        self.variables_2d = variables_2d
        self.variables_3d = variables_3d
        self.levels = levels

        if _is_local(path):
            storage_options = None

        logging.info(f"opening {path}")

        self.group = sync(
            zarr.api.asynchronous.open_group(
                path, storage_options=storage_options, use_consolidated=True, mode="r"
            )
        )

        self.inds = None
        if self.variables_3d:
            self.inds = sync(self._get_vertical_indices(level_coord_name, levels))

        self._arrays = {}
        time_num, self.units, self.calendar = sync(self._get_time())
        if np.issubdtype(time_num.dtype, np.datetime64):
            self.times = pd.DatetimeIndex(time_num)
        else:
            self.times = xr.CFTimeIndex(
                cftime.num2date(time_num, units=self.units, calendar=self.calendar)
            )

    async def sel_time(self, times) -> dict[tuple[str, int], np.ndarray]:
        """

        Returns:
            dict of output data:
                keys are like (name, level), level == -1 for 2d variables

        """
        index_in_loader = self.times.get_indexer(times, method=self.time_sel_method)
        if (index_in_loader == -1).any():
            raise KeyError("Index not found.")
        arr = await self._get(index_in_loader)
        return arr

    async def _get_time(self):
        time = await self.group.get("time")
        time_data = await time.getitem(slice(None))
        return time_data, time.attrs.get("units"), time.attrs.get("calendar")

    async def _get_vertical_indices(self, coord_name, levels):
        levels_var = await self.group.get(coord_name)
        levels_arr = await levels_var.getitem(slice(None))
        return pd.Index(levels_arr).get_indexer(levels)

    async def _get_array(self, name):
        if name not in self._arrays:
            self._arrays[name] = await self.group.get(name)
        return self._arrays[name]

    async def _get(self, t) -> dict[tuple[str, int | None], np.ndarray]:
        tasks = []
        keys = []

        for name in self.variables_3d:
            arr = await self._get_array(name)
            if arr is None:
                raise KeyError(name)
            for level, k in zip(self.levels, self.inds):
                key = (name, level)
                value = _getitem(arr, (t, k))
                tasks.append(value)
                keys.append(key)

        for name in self.variables_2d:
            arr = await self._get_array(name)
            if arr is None:
                raise KeyError(name)
            key = (name, NO_LEVEL)
            value = _getitem(arr, (t,))
            tasks.append(value)
            keys.append(key)

        arrays = await asyncio.gather(*tasks)
        return dict(zip(keys, arrays))
