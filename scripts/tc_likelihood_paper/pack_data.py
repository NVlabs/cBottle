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
"""Packs the data behind the figures into a new directory ``DEST`` with the
layout of the ``run_gpu.sh`` output, so that ``OUT=DEST ./make_figures.sh``
redraws the figures without the samples or ERA5."""

import argparse
import glob
import os
import shutil

import pandas as pd
import xarray as xr
from importance_sampling import load_likelihoods
from plot_is_comparison import MEAN_MAP, load_maps

RUNS = [
    "guidance_0.0",
    "guidance_0.1",
    "fig3_sweep/guidance_0.05",
    "fig3_sweep/guidance_0.2",
    "fig3_sweep/guidance_0.5",
    "fig3_sweep/guidance_1.0",
    "tier2_windows/full",
    "tier2_windows/full_g0.01",
    "tier1_cost_calibration/guided",
]
MAPS = ["control_tcs", "era5_tcs"]
FIG2_WINDOWS = ["fig2_windows/15_20", "fig2_windows/15_200", "fig2_windows/full"]
COPIES = [
    "tier4_classifier/skill_vs_sigma.csv",
    "tier4_classifier/reliability_vs_sigma.csv",
    "derived/physical_metrics_guidance_0.0.csv",
    "derived/physical_metrics_guidance_0.1.csv",
    "derived/era5_hpx3_metrics.csv",
    *(f"{window}/traces" for window in FIG2_WINDOWS),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    os.makedirs(args.dest)

    def target(path):
        path = os.path.join(args.dest, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def write_netcdf(ds, path):
        encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
        ds.to_netcdf(target(path), encoding=encoding)

    sweep = glob.glob("tier1_cost_sweep/*/", root_dir=args.out)
    for run in RUNS + sorted(sweep):
        load_likelihoods(os.path.join(args.out, run)).to_csv(
            target(os.path.join(run, "likelihoods.csv")), index=False
        )
    for name in MAPS:
        data = load_maps(os.path.join(args.out, name))
        times = pd.DatetimeIndex(data["time"].values)
        attrs = {
            "description": "Time mean of the HPX3 cBottle TC classifier probability",
            "n_times": len(times),
            "first_time": times.min().isoformat(),
            "last_time": times.max().isoformat(),
        }
        mean = xr.Dataset({"tc_probability": data.mean(dim="time")}, attrs=attrs)
        write_netcdf(mean, os.path.join(name, MEAN_MAP))
    for window in FIG2_WINDOWS:
        with xr.open_dataset(os.path.join(args.out, window, "0.nc")) as ds:
            uas = ds[["uas"]].load()
        uas.attrs = {}
        write_netcdf(uas, os.path.join(window, "0.nc"))
    for path in COPIES:
        src = os.path.join(args.out, path)
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, target(path))


if __name__ == "__main__":
    main()
