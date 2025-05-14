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
from zarr.core.sync import sync
import xarray as xr
import torch
import math
import pandas as pd
import cftime
import numpy as np
import asyncio
from typing import Callable


async def _sel_time(loaders, time):
    arrays = await asyncio.gather(*[loader.sel_time(time) for loader in loaders])

    data = {}
    for d in arrays:
        data.update(d)

    return data


def _split(x, rank, world_size, drop_extra=True):
    samples_per_rank = len(x) / world_size
    rem = len(x) % world_size
    if drop_extra and (rem > 0):
        x = x[:-rem]

    samples_per_rank = math.ceil(len(x) / world_size)
    start = rank * samples_per_rank
    return x[start : start + samples_per_rank]


class TimeMergedDataset(torch.utils.data.IterableDataset):
    """Merge several loader objects in time and apply transforms.

    This is used to join several datasets along time, and grab data in a chunked manner.

    ``time_loaders`` is a list of objects with this interface::

        class Loader:

            async def sel_time(self, times) -> dict[str, np.ndarray]:
                pass

    ``chunk_size`` should ideally be larger than the chunking of each dataset.

    ``transform`` is a function that prepares the raw loaded data for the model::

        def transform(
            times: list[pd.Timestamp],
            data: list[dict[str, np.ndarray]]
        ) -> dict[str, Any]

    When `time_length = 1` and `frame_step = 1`, this collapses to the image case.
    """

    def __init__(
        self,
        times,
        # for performance times should be in sequence
        *,
        time_loaders,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        chunk_size: int = 48,
        transform: Callable,
        infinite: bool = True,
        time_length: int = 1,
        frame_step: int = 1,
        window_stride: int = 1,
    ):
        if len(times) < world_size:
            raise ValueError(f"Not enough times provided. Received {len(times)=}.")

        if time_length == 1 and frame_step != 1:
            raise ValueError("Frame_step must be 1 for image setting")

        frames_per_window = (time_length - 1) * frame_step + 1
        if chunk_size < frames_per_window:
            raise ValueError(
                f"Chunk size {chunk_size} is too small to fit a window of length "
                f"{time_length} with step {frame_step} (needs {frames_per_window} frames)"
            )

        self._loaders = time_loaders
        self.rank = rank
        self.world_size = world_size
        self.times = times
        self.set_times(times)  # Shard times across ranks

        if len(self._times) < chunk_size:
            raise ValueError(
                f"Sharded times too small for chunk size. Need {chunk_size} "
                f"frames but only got {len(self._times)}"
            )

        self.shuffle = shuffle
        self.transform = transform
        self.chunk_size = chunk_size
        self.infinite = infinite

        self.time_length = time_length
        self.frame_step = frame_step
        self.window_stride = window_stride

        self._generator = None

        max_valid_idx = len(times) - self.chunk_size
        self.max_valid_chunk_idx = max_valid_idx // self.chunk_size

        self.overlap = frames_per_window - 1

    def set_times(self, times):
        self._times = _split(times, self.rank, self.world_size)

    def _load_chunk(self, chunk: int):
        return sync(_sel_time(self._loaders, self._times_for_chunk(chunk)))

    def _times_for_chunk(self, chunk: int) -> pd.DatetimeIndex | xr.CFTimeIndex:
        return self._times[
            chunk * self.chunk_size : (chunk + 1) * self.chunk_size + self.overlap
        ]

    def __iter__(self):
        if self.infinite:
            while True:
                yield from self._iter()
        else:
            yield from self._iter()

    def __len__(self):
        return len(self._times)

    def sel_time(self, time):
        return self.get(self.times.get_loc(time))

    def get(self, idx):
        # if idx < 0 or idx >= self.valid_length:
        #     raise IndexError(
        #         f"Index {idx} out of bounds for dataset of length {self.valid_length}"
        #     )

        frame_idxs = range(
            idx, idx + self.time_length * self.frame_step, self.frame_step
        )

        window_times = self.times[list(frame_idxs)]
        window_data = sync(_sel_time(self._loaders, window_times))

        frames = []
        timestamps = []
        for i, time in enumerate(window_times):
            arr_i = {k: v[i] for k, v in window_data.items()}
            timestamp = pd.Timestamp(*cftime.to_tuple(time))

            frames.append(arr_i)
            timestamps.append(timestamp)

        window_tensor = self.transform(timestamps, frames)

        return window_tensor


    def _generator_shuffle(self, arr, worker_info=None):
        if self._generator is None:
            if worker_info:
                seed = worker_info.seed
            else:
                seed = np.random.randint(0, 2**31) + self.rank

            self._generator = np.random.default_rng(seed=(seed % 2**32))
        self._generator.shuffle(arr)

    def _iter(self):
        num_chunks = math.ceil(len(self._times) / self.chunk_size)
        chunk_idxs = np.arange(num_chunks)

        info = torch.utils.data.get_worker_info()
        num_workers = 1 if info is None else info.num_workers
        worker_id = 0 if info is None else info.id

        if self.shuffle:
            self._generator_shuffle(chunk_idxs, info)

        # Shard chunks across the data workers
        chunk_idxs = _split(chunk_idxs, worker_id, num_workers, drop_extra=False)

        for chunk_idx in chunk_idxs:
            if chunk_idx > self.max_valid_chunk_idx:
                continue

            arr = self._load_chunk(chunk_idx)
            times_for_chunk = self._times_for_chunk(chunk_idx)

            max_window_start = (
                len(times_for_chunk) - (self.time_length - 1) * self.frame_step
            )

            window_starts = np.arange(0, max_window_start, self.window_stride)
            if self.shuffle:
                self._generator_shuffle(window_starts, info)

            for start_idx in window_starts:
                frame_idxs = range(
                    start_idx,
                    start_idx + self.time_length * self.frame_step,
                    self.frame_step,
                )

                frames = []
                timestamps = []
                for idx in frame_idxs:
                    time = times_for_chunk[idx]
                    arr_i = {k: v[idx] for k, v in arr.items()}
                    timestamp = pd.Timestamp(*cftime.to_tuple(time))
                    frames.append(arr_i)
                    timestamps.append(timestamp)

                window_tensor = self.transform(timestamps, frames)

                yield window_tensor


class TimeMergedMapStyle(torch.utils.data.Dataset):
    """
    Map-style version of TimeMergedIterable that simply reads required data
    (without explicit chunking) for each data window without any worker sharding.

    Designed primarily as dataset for validation/inference.
    """

    def __init__(
        self,
        times,
        *,
        time_loaders,
        time_length: int = 1,
        frame_step: int = 1,
        transform: Callable,
    ):
        if time_length == 1 and frame_step != 1:
            raise ValueError("Frame_step must be 1 for image setting")

        self.times = times
        self.transform = transform
        self.time_length = time_length
        self.frame_step = frame_step
        self._loaders = time_loaders

        # can only use idxs that can create a full window
        frames_per_window = (self.time_length - 1) * self.frame_step + 1
        self.valid_length = len(times) - frames_per_window + 1
        if self.valid_length <= 0:
            raise ValueError(
                f"Dataset too small for window length. Need {frames_per_window} "
                f"frames but only got {len(times)}"
            )

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.valid_length:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {self.valid_length}"
            )

        frame_idxs = range(
            idx, idx + self.time_length * self.frame_step, self.frame_step
        )

        window_times = self.times[list(frame_idxs)]
        window_data = sync(_sel_time(self._loaders, window_times))

        frames = []
        timestamps = []
        for i, time in enumerate(window_times):
            arr_i = {k: v[i] for k, v in window_data.items()}
            timestamp = pd.Timestamp(*cftime.to_tuple(time))

            frames.append(arr_i)
            timestamps.append(timestamp)

        window_tensor = self.transform(timestamps, frames)

        return window_tensor
