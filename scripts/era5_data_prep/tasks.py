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
import logging
import os
import time
import subprocess
import tempfile
import dotenv

import joblib
import numpy as np
import pandas as pd
import create_index
import cbottle.storage
import xarray
import zarr
import zarr.errors
from celery import Celery
from flower.utils.broker import Broker
from zarr.core.sync import sync
import config

memory = joblib.Memory(".tmp")

# Load environment variables
dotenv.load_dotenv()

# Variables to process
sl_variables = [
    "tclw",
    "tciw",
    "2t",
    "10u",
    "10v",
    "msl",
    "tp",
    "sstk",
    "ci",
    # new variables
    "2d",
    "tcwv",
    # additional variables from config.py
    "100u",
    "100v",
]

pl_variables = [
    "z",
    "t",
    "u",
    "v",
    "q",
]

app = Celery("tasks", broker=config.CELERY_BROKER)


def update_index(root, profile):
    fs = cbottle.storage.get_filesystem(profile)
    files = fs.glob(root + "/*.nc")
    index = create_index.index_from_files(files)
    index.to_csv("index.csv")
    fs.put_file("index.csv", os.path.join(root, "index.csv"))


def load_index(root, profile) -> pd.DataFrame:
    fs = cbottle.storage.get_filesystem(profile)
    with fs.open(os.path.join(root, "index.csv")) as f:
        df = pd.read_csv(f, index_col=0)

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def _initialize_zarr(
    pl_variables,
    sl_variables,
    hpx_level: int,
    zarr_path,
    **kwargs,
):
    """Initialize a zarr for variables contained at `root`"""
    npix = 12 * 4**hpx_level

    # declare all the pieces of the array in memory
    # so we can consolidate it quickly
    g = zarr.open_group(zarr_path, mode="a", **kwargs)

    for v in sl_variables:
        try:
            arr = g.create_array(
                v,
                shape=(len(config.TIME_RANGE), npix),
                chunks=(config.TIME_CHUNK, npix),
                dtype=np.float32,
                dimension_names=("time", "cells"),
            )
            print(f"created {v}")
            arr.attrs["_ARRAY_DIMENSIONS"] = ("time", "cells")
        except zarr.errors.ContainsArrayError:
            print(f"{v} exists")

    for v in pl_variables:
        try:
            g.create_array(
                v,
                shape=(len(config.TIME_RANGE), config.LEVELS, npix),
                chunks=(config.TIME_CHUNK, 1, npix),
                shards=(config.TIME_CHUNK, config.LEVELS, npix),
                dtype=np.float32,
                dimension_names=("time", "levels", "cells"),
            )
            print(f"created {v}")
        except zarr.errors.ContainsArrayError:
            print(f"{v} exists")

    dt = (config.TIME_RANGE - config.START_TIME) // pd.Timedelta(1, "min")

    try:
        array = g.create_array(
            "time",
            shape=len(dt),
            dtype=np.int64,
            chunks=(len(dt),),
            dimension_names=("time",),
        )
        array[:] = np.asarray(dt)
        array.attrs.update(dict(units=config.TIME_UNITS, calendar="standard"))
        print("created time array")
    except zarr.errors.ContainsArrayError:
        pass
    zarr.consolidate_metadata(g.store)

    _initialize_levels_var()
    _consolidate_metadata()


def _consolidate_metadata():
    g = get_output_group("r+")
    zarr.consolidate_metadata(g.store)


def _initialize_levels_var():
    index = load_index(config.OUTPUT_BUCKET, config.OUTPUT_PROFILE)
    file = index.set_index("name").loc["u"].file.iloc[0]
    fs = cbottle.storage.get_filesystem(config.ZARR_PROFILE)
    with fs.open(file, "rb") as f:
        ds = xarray.open_dataset(f, engine="scipy")
        levels = ds["level"].load()

    group = get_output_group(mode="a")
    try:
        arr = group.create_array(
            "levels",
            shape=levels.shape,
            dtype=levels.dtype,
            chunks=levels.shape,
            dimension_names=["levels"],
        )
        arr.attrs.update(levels.attrs)
        arr[:] = levels.values
    except zarr.errors.ContainsArrayError:
        pass


def initialize_zarr():
    _initialize_zarr(
        sl_variables=sl_variables,
        pl_variables=pl_variables,
        hpx_level=6,
        zarr_path=config.ZARR_PATH,
        storage_options=cbottle.storage.get_storage_options(config.ZARR_PROFILE),
    )


def get_output_group(mode):
    return zarr.open_group(
        config.ZARR_PATH,
        mode=mode,
        storage_options=cbottle.storage.get_storage_options(config.ZARR_PROFILE),
    )


@app.task(acks_late=True)
def process(file):
    logging.info(f"Processing {file}")
    fs = cbottle.storage.get_filesystem(config.ZARR_PROFILE)
    with tempfile.TemporaryDirectory() as dir_:
        tmppath = os.path.join(dir_, "file.nc")
        fs.get(file, tmppath)
        ds = xarray.open_dataset(tmppath).load()

    print(f"loaded {file}")
    index = config.TIME_RANGE.get_indexer(ds.time)
    group = get_output_group(mode="r+")
    for v in ds:
        if "time" in ds[v].dims:
            name = ds[v].attrs.get("short_name", "")
            if name:
                group[name][index.min() : index.max() + 1] = ds[v].values
    logging.info(f"done with {file}")


def get_curation_tasks():
    index = load_index(config.OUTPUT_BUCKET, config.OUTPUT_PROFILE)
    index = index.set_index("name").loc[sl_variables + pl_variables]
    return index.file


# persistent task log
# This is stored in ZARR_PATH/.logs/
# Each worker should check this to avoid repeating tasks
def mark_complete(tasks: list[str]):
    """Needs to be run on a single process"""
    fs = cbottle.storage.get_filesystem(config.ZARR_PROFILE)
    task_log = os.path.join(config.ZARR_PATH, ".logs", str(time.time()))
    with fs.open(task_log, "w") as f:
        ntasks = len(tasks)
        print(f"writing {ntasks=} to {task_log}")
        f.write("\n".join(tasks))


async def list_all(array):
    return [int(f) async for f in array.store.list_dir(f"{array.store_path.path}/c")]


def times_present(array):
    return sync(list_all(array))


@memory.cache
def get_times_for_variables():
    g = get_output_group("r")
    out = {}
    for v in g:
        if v == "time":
            continue
        if v == "levels":
            continue
        o = times_present(g[v])
        o = np.array(o)
        if o.size == 0:
            out[v] = None
            continue

        chunkarr = np.zeros((g[v].shape[0] // g[v].chunks[0] + 1,), dtype=np.bool_)
        chunkarr[o] = True
        nt = len(config.TIME_RANGE)
        inds = np.r_[:nt]
        array = chunkarr[inds // g[v].chunks[0]]
        out[v] = config.TIME_RANGE[array]
    return out


def get_uncompleted_curation_tasks():
    files = get_curation_tasks()
    index = create_index.index_from_files(files)
    index["interval"] = pd.IntervalIndex.from_arrays(
        index.start_date, index.end_date, closed="both"
    )
    index = index.set_index(["name", "interval"])
    times = get_times_for_variables()

    todo_files = set()
    for v in times:
        time = times[v]
        if time is None:
            todo_files |= set(index.loc[v].file)
            continue
        file_index = index.loc[v].index.get_indexer(time)  # (nt, )
        # number of timesteps in output data that came from each input file
        file_bin_counts = np.bincount(
            file_index[file_index != -1], minlength=index.loc[v].shape[0]
        )

        todo_files |= set(index.loc[v][file_bin_counts == 0].file)

    return todo_files


# Queue management
def queue_size():
    async def get_size():
        broker = Broker(config.CELERY_BROKER)
        queues = await broker.queues(["celery"])
        return queues[0].get("messages", 0)

    return asyncio.run(get_size())


def monitor():
    import tqdm

    n = queue_size()
    with tqdm.tqdm(total=n, smoothing=0.0) as pb:
        while True:
            time.sleep(1)
            m = queue_size()
            pb.update(n - m)
            n = queue_size()


@app.task(acks_late=True)
def regrid_file(file_path):
    """Regrid a single ERA5 file to HPX grid and upload to S3.

    Args:
        file_path: Path to the input file in S3

    Returns:
        str: Path to the regridded file in S3
    """
    logging.info(f"Regridding {file_path}")
    dest_fs = cbottle.storage.get_filesystem(config.ZARR_PROFILE)
    src_fs = cbottle.storage.get_filesystem(config.ERA5_PROFILE)

    with tempfile.TemporaryDirectory() as d:
        # Download file
        local_path = os.path.join(d, os.path.basename(file_path))
        src_fs.get(file_path, local_path)

        # Regrid to HP256
        out_256 = os.path.join(d, "256.nc")
        subprocess.check_call(["cdo", "remapbil,hp256", local_path, out_256])

        # Get variables with grid mapping
        variables = ["-selvar"]
        with xarray.open_dataset(out_256) as ds:
            for v in ds.variables:
                if "grid_mapping" in ds[v].attrs:
                    variables.append(v)

        vararg = ",".join(variables)

        # Regrid to HP64
        transformed_file = f"t.{os.path.basename(file_path)}"
        out_64 = os.path.join(d, transformed_file)
        subprocess.check_call(
            [
                "cdo",
                "--format",
                "nc",
                "-L",
                "hpdegrade,nside=64",
                vararg,
                out_256,
                out_64,
            ]
        )

        # Upload to S3
        output_path = os.path.join(config.OUTPUT_BUCKET, os.path.basename(file_path))
        dest_fs.put(out_64, output_path)

        logging.info(f"Regridded file uploaded to: {output_path}")
        return output_path
