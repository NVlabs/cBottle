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
"""

Mermaid chart of the use of MergedTimeDataset in this module::

    sequenceDiagram
        participant MergedTimeLoader
        participant Loaders

        participant encode_task

        Note over MergedTimeLoader: Retrieve data from loaders
        MergedTimeLoader->>Loaders: Load atmospheric data
        MergedTimeLoader->>Loaders: Load SST data (if sst_input=True)
        MergedTimeLoader->>Loaders: Load land fraction (ICON only)


        Note over MergedTimeLoader: Process data
        MergedTimeLoader->>encode_task: Call encode_task on sample
        encode_task->>encode_task: Collect fields into array
        encode_task->>encode_task: Apply mean/scale normalization
        encode_task->>encode_task: Encode SST (if needed)

        Note over encode_task: Pixel order in NEST at this point
        encode_task->>encode_task: Reorder target data (NEST -> HPX PAD)
        encode_task->>encode_task: Reorder condition data (NEST -> HPX PAD)

        encode_task ->> MergedTimeLoader: return data

"""

import datetime
import functools
import pathlib
from typing import Optional, Callable
import dataclasses
import cbottle.config.environment as config
import cftime
import earth2grid
import numpy as np
import pandas as pd
import torch
import zarr
from cbottle.datasets.base import BatchInfo, TimeUnit
from cbottle.datasets.dataset_2d import (
    LABELS,
    MAX_CLASSES,
    MONTHLY_SST,
    SST_LAND_FILL_VALUE,
    encode_sst,
)
from cbottle.datasets.amip_sst_loader import AmipSSTLoader
from cbottle.datasets.merged_dataset import TimeMergedDataset, TimeMergedMapStyle
from cbottle.datasets.zarr_loader import ZarrLoader
from cbottle.storage import get_storage_options

from cbottle.training.video.frame_masker import FrameMasker

NO_LEVEL = -1

HPX_LEVEL = 6
# in hpa
LEVELS = [1000, 850, 700, 500, 300, 200, 50, 10]
VARIABLES_3D = ["U", "V", "T", "Z"]
VARIABLES_2d = [
    "tcwv",
    "cllvi",
    "clivi",
    "tas",
    "uas",
    "vas",
    "rlut",
    "rsut",
    "pres_msl",
    "pr",
    "rsds",
    "sst",
    "sic",
]


@dataclasses.dataclass
class DatasetMetadata:
    name: str
    start: str
    end: str
    time_step: int  # time between successive data points in `time_unit`
    time_unit: TimeUnit

    @property
    def freq(self) -> str:
        return f"{self.time_step}{self.time_unit.value}"


DATASET_METADATA: dict[str, DatasetMetadata] = {
    "era5": DatasetMetadata(
        name="era5",
        start="1980",
        end="2019",
        time_step=1,
        time_unit=TimeUnit.HOUR,
    ),
    "icon": DatasetMetadata(
        name="icon",
        start="2020-01-20 03:00:00",
        end="2025-07-22 00:00:00",
        time_step=3,
        time_unit=TimeUnit.HOUR,
    ),
    "amip": DatasetMetadata(
        name="amip",
        start="1940-01-01",
        end="2021-12-31",
        time_step=1,
        time_unit=TimeUnit.HOUR,
    ),
}

INDEX = pd.MultiIndex.from_tuples(
    [(v, level) for v in VARIABLES_3D for level in LEVELS]
    + [(v, NO_LEVEL) for v in VARIABLES_2d],
    names=["variable", "level"],
)


def encode_channel(channel) -> str:
    name, level = channel
    if level != NO_LEVEL:
        return f"{name}{level}"
    else:
        return name


def get_stats():
    path = pathlib.Path(__file__).parent / "dataset_v6_stats.csv"
    return pd.read_csv(path).set_index(["variable", "level"])


def get_std():
    stats = get_stats()
    return stats.loc[INDEX]["std"].values


def get_mean():
    stats = get_stats()
    return stats.loc[INDEX]["mean"].values


def get_batch_info(
    time_step: int = 1, time_unit: TimeUnit = TimeUnit.HOUR
) -> BatchInfo:
    return BatchInfo(
        channels=[encode_channel(tup) for tup in INDEX.tolist()],
        scales=get_std(),
        center=get_mean(),
        time_step=time_step,
        time_unit=time_unit,
    )


def _compute_frame_step(
    metadata: DatasetMetadata, time_step: int, time_length: int
) -> int:
    if time_length == 1:
        return 1

    dataset_spacing = metadata.time_unit.to_timedelta(metadata.time_step)
    model_resolution_timedelta = datetime.timedelta(hours=time_step)
    return model_resolution_timedelta // dataset_spacing


def cftime_to_timestamp(time: cftime.DatetimeGregorian) -> float:
    return datetime.datetime(
        *cftime.to_tuple(time), tzinfo=datetime.timezone.utc
    ).timestamp()


def collect_fields(data: dict[tuple[str, int | None], np.ndarray]) -> np.ndarray:
    out = np.full(
        shape=[INDEX.size, 1, 4**HPX_LEVEL * 12], dtype=np.float32, fill_value=np.nan
    )
    for i, key in enumerate(INDEX):
        if key in data:
            out[i, 0] = data[key]
    return out


def combine_and_mask_frames(
    frames: list[dict], frame_masker: FrameMasker
) -> dict[str, torch.Tensor | float]:
    """
    For video usage, combines a list of unbatched frame dictionaries into a single dictionary
    by concatenating tensors along the time dimension. Then applies masking to the combined frames.

    Input shapes:
    - Field tensors (target, condition): (C, 1, X) -> (C, T, X)
    - Scalar values (second_of_day, day_of_year): [1] -> [T]
    - Labels and timestamps: kept from first frame
    """
    out = {}

    out["target"] = torch.cat([f["target"] for f in frames], dim=1)
    out["condition"] = torch.cat([f["condition"] for f in frames], dim=1)

    out["second_of_day"] = torch.cat([f["second_of_day"] for f in frames], dim=0)
    out["day_of_year"] = torch.cat([f["day_of_year"] for f in frames], dim=0)

    # Current network uses a single label regardless of time length
    # and all frames are assumed to have the same dataset label
    out["labels"] = frames[0]["labels"]
    out["timestamp"] = frames[0]["timestamp"]

    return frame_masker(out)


def _transform(
    times: list[cftime.DatetimeGregorian],
    data_list: list[dict],
    *,
    encode_frame,
    frame_masker,
) -> dict:
    frames = [encode_frame(time=t, data=d) for t, d in zip(times, data_list)]

    if len(frames) == 1:
        return frames[0]

    if frame_masker is None:
        raise ValueError("Frame masker must be provided in video mode")

    return combine_and_mask_frames(frames, frame_masker)


def encode_task(
    label: int,
    mean: np.ndarray,
    scale: np.ndarray,
    time: cftime.DatetimeGregorian,
    data: dict[tuple[str, int | None], np.ndarray],
    *,
    sst_input: bool,
    is_land: np.ndarray | None = None,
):
    """

    Args:
        arr: (c, x) in NEST order. in standard units
        sst: (c, x) in NEST order. in deg K

    Returns:
        output dict, condition and target in HEALPIX_PAD_XY order

    """
    labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=MAX_CLASSES)

    mean = mean[:, np.newaxis, np.newaxis]
    scale = scale[:, np.newaxis, np.newaxis]

    # TODO make this work for ICON
    if label == 1:  # if ERA5
        _convert_era5_to_standard(data)
    elif label == 0:
        _convert_icon_to_standard(data, is_land)

    arr = collect_fields(data)
    arr = (arr - mean) / scale

    if sst_input:
        sst = data[(MONTHLY_SST, NO_LEVEL)]
        cond = encode_sst(sst, is_land=is_land)
    else:
        cond = np.ones([0, arr.shape[-2], arr.shape[-1]])

    def reorder(x):
        x = torch.as_tensor(x)
        return earth2grid.healpix.reorder(
            x, earth2grid.healpix.PixelOrder.NEST, earth2grid.healpix.HEALPIX_PAD_XY
        ).to(torch.float32)

    day_start = time.replace(hour=0, minute=0, second=0)
    year_start = day_start.replace(month=1, day=1)
    second_of_day = (time - day_start) / datetime.timedelta(seconds=1)
    day_of_year = (time - year_start) / datetime.timedelta(seconds=86400)
    out = {
        "target": reorder(arr),
        "labels": labels,
        "condition": reorder(cond),
        "second_of_day": torch.tensor([second_of_day]),
        "day_of_year": torch.tensor([day_of_year]),
    }
    out["timestamp"] = cftime_to_timestamp(time)

    return out


def _convert_icon_to_standard(data, is_land):
    data[(MONTHLY_SST, NO_LEVEL)] = np.where(
        is_land, SST_LAND_FILL_VALUE, data[("ts_monmean", NO_LEVEL)]
    )
    data[("sst", NO_LEVEL)] = np.where(
        is_land, SST_LAND_FILL_VALUE, data[("ts", NO_LEVEL)]
    )

    # convert pressure levels to hPa
    for (name, level), value in list(data.items()):
        if level != NO_LEVEL:
            data[(name, level // 100)] = value


def _get_dataset_wrapper(
    times,
    loaders,
    transform: Callable,
    *,
    rank: int = 0,
    world_size: int = 1,
    infinite: bool,
    shuffle: bool = True,
    chunk_size: int = 8,
    frame_step: int = 1,  # Spacing of consecutive frames in the dataset
    time_length: int = 1,
    map_style: bool = False,
) -> TimeMergedDataset | TimeMergedMapStyle:
    if map_style:
        # Used for video validation/inference
        wrapper = TimeMergedMapStyle(
            times,
            time_loaders=loaders,
            frame_step=frame_step,
            time_length=time_length,
            transform=transform,
        )
    else:
        wrapper = TimeMergedDataset(
            times,
            time_loaders=loaders,
            transform=transform,
            rank=rank,
            world_size=world_size,
            infinite=infinite,
            shuffle=shuffle,
            chunk_size=chunk_size,
            frame_step=frame_step,
            time_length=time_length,
        )
    return wrapper


def _get_dataset_icon(
    *,
    split: str = "",
    rank: int = 0,
    world_size: int = 1,
    sst_input: bool = True,
    infinite: bool,
    shuffle: bool = True,
    chunk_size: int = 8,
    time_step: int = 1,
    time_length: int = 1,
    frame_masker: Optional[Callable] = None,
):
    # min and max times
    # (Pdb) p loaders[1].times[[0,-1]]
    # CFTimeIndex([2020-01-20 00:30:00, 2025-07-22 00:00:00],
    #            dtype='object',
    #            length=2,
    #            calendar='proleptic_gregorian',
    #            freq=None)
    # (Pdb) p loaders[0].times[[0,-1]]
    # DatetimeIndex(['2020-01-20 03:00:00', '2025-07-22 00:00:00'], dtype='datetime64[s]', freq=None)
    # TODO add linear interpolation to DataWrapper with 30 min data
    metadata = DATASET_METADATA["icon"]
    valid_times = pd.date_range(metadata.start, metadata.end, freq=metadata.freq)
    loaders = [
        ZarrLoader(
            path=config.V6_ICON_ZARR,
            storage_options=get_storage_options(config.V6_ICON_ZARR_PROFILE),
            variables_3d=["T", "U", "V", "Z"],
            variables_2d=["tcwv"],
            level_coord_name="level",
            # convert levels to Pa
            levels=[lev * 100 for lev in LEVELS],
        ),
        ZarrLoader(
            path=config.RAW_DATA_URL_6,
            storage_options=get_storage_options(config.RAW_DATA_PROFILE),
            levels=[],
            variables_3d=[],
            variables_2d=[
                "tas",
                "uas",
                "vas",
                "cllvi",
                "clivi",
                "rlut",
                "rsut",
                "pres_msl",
                "pr",
                "rsds",
                "ts",
                "sic",
            ],
        ),
        ZarrLoader(
            path=config.SST_MONMEAN_DATA_URL_6,
            storage_options=get_storage_options(config.SST_MONMEAN_DATA_PROFILE),
            levels=[],
            variables_3d=[],
            variables_2d=["ts_monmean"],
            time_sel_method="nearest",
        ),
    ]

    train_times = valid_times[valid_times < "2024-03-06 15:00:00"]
    test_times = valid_times[valid_times >= "2024-03-06 15:00:00"]
    times = {"train": train_times, "test": test_times, "": valid_times}[split]

    if times.size == 0:
        raise RuntimeError("No times are selected.")

    # open land data
    land_data = zarr.open_group(
        config.LAND_DATA_URL_6,
        storage_options=get_storage_options(config.LAND_DATA_PROFILE),
    )
    land_fraction = land_data["land_fraction"][:]

    label = LABELS.index("icon")

    encode_frame = functools.partial(
        encode_task,
        label,
        get_mean(),
        get_std(),
        sst_input=sst_input,
        is_land=land_fraction > 0,
    )

    transform = functools.partial(
        _transform,
        encode_frame=encode_frame,
        frame_masker=frame_masker,
    )

    frame_step = _compute_frame_step(metadata, time_step, time_length)

    return _get_dataset_wrapper(
        times,
        loaders,
        transform,
        rank=rank,
        world_size=world_size,
        infinite=infinite,
        shuffle=shuffle,
        chunk_size=chunk_size,
        frame_step=frame_step,
        time_length=time_length,
        map_style=time_length > 1 and split != "train",
    )


def _convert_era5_to_standard(data):
    sstk = data[("sstk", NO_LEVEL)]
    ci = data[("ci", NO_LEVEL)]

    if not np.ma.isMaskedArray(sstk):
        sstk = np.ma.masked_invalid(sstk)

    if not np.ma.isMaskedArray(ci):
        ci = np.ma.masked_invalid(ci)

    data[("sstk", NO_LEVEL)] = sstk.filled(SST_LAND_FILL_VALUE)
    data[("ci", NO_LEVEL)] = ci.filled(0)
    # era5 precip is in liquid water equivalent accumulated over 1 hour (m)
    # icon is in mass flux units (kg / s / m^2)
    # unit conversion: tp / 3600 * density water = tp / 3600 * 1000
    water_density = 1000
    seconds_per_hour = 3600
    data[("tp", NO_LEVEL)] = data[("tp", NO_LEVEL)] * water_density / seconds_per_hour

    fields_out_map = {
        # mapping of ecmwf name to icon name
        "tclw": "cllvi",
        "tciw": "clivi",
        "2t": "tas",
        "10u": "uas",
        "10v": "vas",
        "msl": "pres_msl",
        "tp": "pr",
        "sstk": "sst",
        "ci": "sic",
        "tcwv": "prw",
        "u": "U",
        "v": "V",
        "t": "T",
        "z": "Z",
        "tosbcs": MONTHLY_SST,
    }
    for (name, level), value in list(data.items()):
        if name in fields_out_map:
            data[(fields_out_map[name], level)] = value


def _get_dataset_era5(
    *,
    split: str = "",
    rank: int = 0,
    world_size: int = 1,
    sst_input: bool = True,
    infinite: bool,
    shuffle: bool = True,
    chunk_size: int = 48,
    time_step: int = 1,
    time_length: int = 1,
    frame_masker: Optional[Callable] = None,
):
    target_data_loader = ZarrLoader(
        path=config.V6_ERA5_ZARR,
        storage_options=get_storage_options(config.V6_ERA5_ZARR_PROFILE),
        variables_3d=["u", "v", "t", "z"],
        variables_2d=[
            "sstk",
            "ci",
            "msl",
            "tp",
            "10u",
            "10v",
            "2t",
            "tclw",
            "tciw",
            "tcwv",
        ],
        level_coord_name="levels",
        levels=LEVELS,
    )

    loaders = [target_data_loader]

    if sst_input:
        grid = earth2grid.healpix.Grid(
            HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        loaders.append(
            AmipSSTLoader(
                grid,
            )
        )

    metadata = DATASET_METADATA["era5"]
    valid_times = pd.date_range(metadata.start, metadata.end, freq=metadata.freq)

    train_times = valid_times[valid_times < "2018"]
    test_times = valid_times[valid_times.year == 2018]
    times = {"train": train_times, "test": test_times, "": valid_times}[split]

    if times.size == 0:
        raise RuntimeError("No times are selected.")

    label = LABELS.index("era5")

    encode_frame = functools.partial(
        encode_task, label, get_mean(), get_std(), sst_input=sst_input
    )

    transform = functools.partial(
        _transform,
        encode_frame=encode_frame,
        frame_masker=frame_masker,
    )

    frame_step = _compute_frame_step(metadata, time_step, time_length)

    return _get_dataset_wrapper(
        times,
        loaders,
        transform,
        rank=rank,
        world_size=world_size,
        infinite=infinite,
        shuffle=shuffle,
        chunk_size=chunk_size,
        frame_step=frame_step,
        time_length=time_length,
        map_style=time_length > 1 and split != "train",
    )


def _encode_amip(
    time: cftime.DatetimeGregorian,
    data: dict[tuple[str, int | None], np.ndarray],
    *,
    label: int,
):
    """

    Args:
        arr: (c, x) in NEST order. in standard units
        sst: (c, x) in NEST order. in deg K

    Returns:
        output dict, condition and target in HEALPIX_PAD_XY order

    """
    labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=MAX_CLASSES)

    sst = data[("tosbcs", NO_LEVEL)]
    cond = encode_sst(sst)

    def reorder(x):
        x = torch.as_tensor(x)
        return earth2grid.healpix.reorder(
            x, earth2grid.healpix.PixelOrder.NEST, earth2grid.healpix.HEALPIX_PAD_XY
        )

    day_start = time.replace(hour=0, minute=0, second=0)
    year_start = day_start.replace(month=1, day=1)
    second_of_day = (time - day_start) / datetime.timedelta(seconds=1)
    day_of_year = (time - year_start) / datetime.timedelta(seconds=86400)

    nan_channels = [
        INDEX.get_loc((channel, -1)) for channel in ["rlut", "rsut", "rsds"]
    ]
    target = np.zeros((INDEX.size, 1, 4**HPX_LEVEL * 12), dtype=np.float32)
    target[nan_channels, ...] = np.nan

    out = {
        "target": torch.tensor(target),
        "labels": labels,
        "condition": reorder(cond),
        "second_of_day": torch.tensor([second_of_day]),
        "day_of_year": torch.tensor([day_of_year]),
    }
    out["timestamp"] = cftime_to_timestamp(time)

    return out


def get_amip_dataset(
    rank: int = 0,
    world_size: int = 1,
    infinite: bool = False,
    *,
    split,
    sst_input,
    shuffle,
    chunk_size: int = 48,
    time_step: int = 1,
    time_length: int = 1,
    frame_masker: Optional[Callable] = None,
):
    if not sst_input:
        raise ValueError("AMIP inference only works with SST input.")

    if shuffle:
        raise NotImplementedError("Shuffling not implemented for AMIP dataset.")

    metadata = DATASET_METADATA["amip"]
    times = pd.date_range(metadata.start, metadata.end, freq=metadata.freq)

    grid = earth2grid.healpix.Grid(
        HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    loaders = [
        AmipSSTLoader(
            grid,
        )
    ]
    encode_frame = functools.partial(_encode_amip, label=LABELS.index("era5"))

    transform = functools.partial(
        _transform,
        encode_frame=encode_frame,
        frame_masker=frame_masker,
    )

    frame_step = _compute_frame_step(metadata, time_step, time_length)

    # TODO AMIP infernece only works with era5

    return _get_dataset_wrapper(
        times,
        loaders,
        transform,
        rank=rank,
        world_size=world_size,
        infinite=infinite,  # todo check
        shuffle=False,
        chunk_size=chunk_size,
        frame_step=frame_step,
        time_length=time_length,
        map_style=time_length > 1,
    )


def get_dataset(
    *,
    split: str = "",
    dataset: str = "era5",
    rank: int = 0,
    world_size: int = 1,
    sst_input: bool = True,
    infinite: bool = False,
    shuffle: bool = True,
    chunk_size: int = 8,
    time_step: int = 1,  # in hours
    time_length: int = 1,
    frame_masker: Optional[FrameMasker] = None,
) -> TimeMergedDataset | TimeMergedMapStyle:
    dataset_func = {
        "icon": _get_dataset_icon,
        "era5": _get_dataset_era5,
        "amip": get_amip_dataset,
    }[dataset]

    ds = dataset_func(
        split=split,
        rank=rank,
        world_size=world_size,
        sst_input=sst_input,
        infinite=infinite,
        shuffle=shuffle,
        chunk_size=chunk_size,
        time_step=time_step,
        time_length=time_length,
        frame_masker=frame_masker,
    )

    ds.batch_info = get_batch_info(time_step=time_step, time_unit=TimeUnit.HOUR)
    ds.calendar = "standard"
    ds.time_units = "seconds since 1970-1-1 0:0:0"
    return ds
