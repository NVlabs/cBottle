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
from typing import Optional
import dataclasses
import cftime
import earth2grid
import numpy as np
import pandas as pd
import torch
from cbottle.datasets import catalog
from cbottle.datasets.base import BatchInfo, TimeUnit
from cbottle.datasets.dataset_2d import (
    LABELS,
    MAX_CLASSES,
    MONTHLY_SST,
    SST_LAND_FILL_VALUE,
    encode_sst,
)
from cbottle.datasets.ibtracs import IBTracs
from cbottle.datasets.amip_sst_loader import AmipSSTLoader
from cbottle.datasets.merged_dataset import TimeMergedDataset, TimeMergedMapStyle
from cbottle.datasets.zarr_loader import ZarrLoader
from cbottle.training.video.frame_masker import FrameMasker

NO_LEVEL = -1

HPX_LEVEL = 6
# in hpa


@dataclasses.dataclass(frozen=True)
class VariableConfig:
    variables_2d: list[str]
    variables_3d: list[str]
    levels: list[int]


VARIABLE_CONFIGS = {}
VARIABLE_CONFIGS["default"] = VariableConfig(
    levels=[1000, 850, 700, 500, 300, 200, 50, 10],
    variables_3d=["U", "V", "T", "Z"],
    variables_2d=[
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
    ],
)
VARIABLE_CONFIGS["q"] = VariableConfig(
    levels=[1000, 850, 700, 500, 300, 200, 50, 10],
    variables_3d=["U", "V", "T", "Z", "Q"],
    variables_2d=[
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
    ],
)
_default_config = VARIABLE_CONFIGS["default"]


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
        end="2022",
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


def _get_index(config: VariableConfig = _default_config):
    return pd.MultiIndex.from_tuples(
        [(v, level) for v in config.variables_3d for level in config.levels]
        + [(v, NO_LEVEL) for v in config.variables_2d],
        names=["variable", "level"],
    )


def _encode_channel(channel) -> str:
    name, level = channel
    if level != NO_LEVEL:
        return f"{name}{level}"
    else:
        return name


def _get_stats():
    path = pathlib.Path(__file__).parent / "dataset_v6_stats.csv"
    return pd.read_csv(path).set_index(["variable", "level"])


def get_std(config: VariableConfig = _default_config):
    stats = _get_stats()
    return stats.loc[_get_index(config)]["std"].values


def get_mean(config: VariableConfig = _default_config):
    stats = _get_stats()
    return stats.loc[_get_index(config)]["mean"].values


def get_batch_info(
    config: VariableConfig = _default_config,
    time_step: int = 1,
    time_unit: TimeUnit = TimeUnit.HOUR,
) -> BatchInfo:
    return BatchInfo(
        channels=[_encode_channel(tup) for tup in _get_index(config).tolist()],
        scales=get_std(config),
        center=get_mean(config),
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


def _cftime_to_timestamp(time: cftime.DatetimeGregorian) -> float:
    return datetime.datetime(
        *cftime.to_tuple(time), tzinfo=datetime.timezone.utc
    ).timestamp()


def _collect_fields(
    index, data: dict[tuple[str, int | None], np.ndarray]
) -> np.ndarray:
    out = np.full(
        shape=[index.size, 1, 4**HPX_LEVEL * 12], dtype=np.float32, fill_value=np.nan
    )
    for i, key in enumerate(index):
        if key in data:
            out[i, 0] = data[key]
    return out


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

    out = {}

    out["target"] = torch.cat([f["target"] for f in frames], dim=1)
    out["condition"] = torch.cat([f["condition"] for f in frames], dim=1)

    out["second_of_day"] = torch.cat([f["second_of_day"] for f in frames], dim=0)
    out["day_of_year"] = torch.cat([f["day_of_year"] for f in frames], dim=0)

    # Current network uses a single label regardless of time length
    # and all frames are assumed to have the same dataset label
    out["labels"] = frames[0]["labels"]
    out["timestamp"] = frames[0]["timestamp"]

    if frame_masker is not None:
        out = frame_masker(out)
    return out


def _encode_task(
    label: int,
    mean: np.ndarray,
    scale: np.ndarray,
    time: cftime.DatetimeGregorian,
    data: dict[tuple[str, int | None], np.ndarray],
    *,
    sst_input: bool,
    config: VariableConfig,
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

    arr = _collect_fields(_get_index(config), data)
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
    out["timestamp"] = _cftime_to_timestamp(time)

    # Add TC labels if available in the data dictionary
    if IBTracs.KEY in data:
        # Reorder to match model's pixel ordering (HEALPIX_PAD_XY)
        out["classifier_labels"] = reorder(data[IBTracs.KEY])

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
        "q": "Q",
        "tosbcs": MONTHLY_SST,
    }
    for (name, level), value in list(data.items()):
        if name in fields_out_map:
            data[(fields_out_map[name], level)] = value


def _encode_amip(
    time: cftime.DatetimeGregorian,
    data: dict[tuple[str, int | None], np.ndarray],
    *,
    label: int,
    variable_config: VariableConfig,
):
    """

    Args:
        arr: (c, x) in NEST order. in standard units
        sst: (c, x) in NEST order. in deg K

    Returns:
        output dict, condition and target in HEALPIX_PAD_XY order

    """
    index = _get_index(variable_config)
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
        index.get_loc((channel, -1)) for channel in ["rlut", "rsut", "rsds"]
    ]
    target = np.zeros((index.size, 1, 4**HPX_LEVEL * 12), dtype=np.float32)
    target[nan_channels, ...] = np.nan

    out = {
        "target": torch.tensor(target),
        "labels": labels,
        "condition": reorder(cond),
        "second_of_day": torch.tensor([second_of_day]),
        "day_of_year": torch.tensor([day_of_year]),
    }
    out["timestamp"] = _cftime_to_timestamp(time)
    # Add TC labels if available in the data dictionary
    if IBTracs.KEY in data:
        # Reorder to match model's pixel ordering (HEALPIX_PAD_XY)
        out["classifier_labels"] = reorder(data[IBTracs.KEY])

    return out


def _loader_from_catalog(dataset: catalog._Zarr, **kwargs) -> ZarrLoader:
    return ZarrLoader(
        path=dataset.path, storage_options=dataset.storage_options, **kwargs
    )


def _get_loaders(
    dataset: str,
    *,
    sst_input: bool = True,
    ibtracs_input: bool = False,
    variable_config: VariableConfig = _default_config,
):
    """Get the appropriate loaders for a given dataset.

    Args:
        dataset: The dataset name ("icon", "era5", or "amip")
        sst_input: Whether to include SST input
        ibtracs_input: Whether to include IBTrACS input
        variable_config: Variable configuration for the dataset

    Returns:
        List of loaders for the specified dataset
    """
    use_q = "Q" in variable_config.variables_3d

    if dataset == "icon":
        loaders = [
            _loader_from_catalog(
                catalog.icon_plevel(),
                variables_3d=["T", "U", "V", "Z"] + (["Q"] if use_q else []),
                variables_2d=["tcwv"],
                level_coord_name="level",
                # convert levels to Pa
                levels=[lev * 100 for lev in variable_config.levels],
            ),
            _loader_from_catalog(
                catalog.icon(level=6, freq="PT30M"),
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
            _loader_from_catalog(
                catalog.icon_sst_monmean(),
                levels=[],
                variables_3d=[],
                variables_2d=["ts_monmean"],
                time_sel_method="nearest",
            ),
        ]

    elif dataset == "era5":
        target_data_loader = _loader_from_catalog(
            catalog.era5_hpx6(),
            variables_3d=["u", "v", "t", "z"] + (["q"] if use_q else []),
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
            levels=variable_config.levels,
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
        if ibtracs_input:
            loaders.append(IBTracs())

    elif dataset == "amip":
        if not sst_input:
            raise ValueError("AMIP inference only works with SST input.")

        grid = earth2grid.healpix.Grid(
            HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        loaders = [
            AmipSSTLoader(
                grid,
            )
        ]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return loaders


def _get_frame_encoder(
    dataset: str,
    *,
    sst_input: bool = True,
    variable_config: VariableConfig = _default_config,
):
    """Get the appropriate transform function for a given dataset.

    Args:
        dataset: The dataset name ("icon", "era5", or "amip")
        sst_input: Whether to include SST input
        frame_masker: Optional frame masker for video processing
        variable_config: Variable configuration for the dataset

    Returns:
        Transform function for the specified dataset
    """
    if dataset == "icon":
        label = LABELS.index("icon")

        # open land data
        land_data = catalog.icon_land(level=6, freq="P1D").to_zarr()
        land_fraction = land_data["land_fraction"][:]
        encode_frame = functools.partial(
            _encode_task,
            label,
            get_mean(variable_config),
            get_std(variable_config),
            sst_input=sst_input,
            is_land=land_fraction > 0,
            config=variable_config,
        )

    elif dataset == "era5":
        label = LABELS.index("era5")
        encode_frame = functools.partial(
            _encode_task,
            label,
            get_mean(variable_config),
            get_std(variable_config),
            sst_input=sst_input,
            config=variable_config,
        )

    elif dataset == "amip":
        if not sst_input:
            raise ValueError("AMIP inference only works with SST input.")

        encode_frame = functools.partial(
            _encode_amip, label=LABELS.index("era5"), variable_config=variable_config
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return encode_frame


def _get_splits(dataset: str):
    # Get metadata for the dataset
    metadata = DATASET_METADATA[dataset]
    valid_times = pd.date_range(metadata.start, metadata.end, freq=metadata.freq)

    # Handle time splitting based on dataset
    if dataset == "icon":
        train_times = valid_times[valid_times < "2024-03-06 15:00:00"]
        test_times = valid_times[valid_times >= "2024-03-06 15:00:00"]
    elif dataset == "era5":
        train_times = valid_times[valid_times < "2018"]
        test_times = valid_times[valid_times.year == 2018]
    elif dataset == "amip":
        # AMIP doesn't have train/test split, use all times
        train_times = valid_times
        test_times = valid_times
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return {"train": train_times, "test": test_times, "": valid_times}


def get_dataset(
    *,
    split: str = "",
    dataset: str = "era5",
    rank: int = 0,
    world_size: int = 1,
    sst_input: bool = True,
    infinite: bool = False,
    ibtracs_input: bool = False,
    shuffle: bool = True,
    chunk_size: int = 8,
    time_step: int = 1,  # in hours
    time_length: int = 1,
    frame_masker: Optional[FrameMasker] = None,
    variable_config: VariableConfig = _default_config,
    map_style: bool = False,
) -> TimeMergedDataset | TimeMergedMapStyle:
    # Get the appropriate loaders for the dataset
    loaders = _get_loaders(
        dataset,
        sst_input=sst_input,
        ibtracs_input=ibtracs_input,
        variable_config=variable_config,
    )
    times = _get_splits(dataset)[split]
    if times.size == 0:
        raise RuntimeError("No times are selected.")

    # Compute frame step
    frame_step = _compute_frame_step(DATASET_METADATA[dataset], time_step, time_length)

    # Handle special cases for AMIP dataset
    if dataset == "amip":
        # if shuffle:
        #     raise NotImplementedError("Shuffling not implemented for AMIP dataset.")
        # shuffle = False
        # map_style = time_length > 1
        pass
    else:
        map_style = map_style or (time_length > 1 and split != "train")

    transform = functools.partial(
        _transform,
        encode_frame=_get_frame_encoder(
            dataset,
            sst_input=sst_input,
            variable_config=variable_config,
        ),
        frame_masker=frame_masker,
    )

    # Create and return the dataset
    if map_style:
        # Used for video validation/inference
        ds = TimeMergedMapStyle(
            times,
            time_loaders=loaders,
            frame_step=frame_step,
            time_length=time_length,
            transform=transform,
        )
    else:
        ds = TimeMergedDataset(
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

    ds.batch_info = get_batch_info(
        config=variable_config, time_step=time_step, time_unit=TimeUnit.HOUR
    )
    ds.calendar = "standard"
    ds.time_units = "seconds since 1970-1-1 0:0:0"
    return ds


def guess_variable_config(channels: list[str]) -> str:
    for v in VARIABLE_CONFIGS:
        if get_batch_info(VARIABLE_CONFIGS[v]).channels == channels:
            return v
    raise ValueError()
