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
"""Fig. 4 and Fig. A2: exceedance frequencies from ``importance_sampling.py`` and
time-mean ERA5 and control TC probability maps."""

import argparse
import glob
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from earth2grid import healpix
from importance_sampling import MIN_YEAR
from matplotlib.colors import LogNorm

MEAN_MAP = "tc_probability_mean.nc"
SERIES = {
    "guided": (-1.5, "^", "#b2182b", "Guided (unweighted)", 50),
    "is": (-0.5, "o", "#ef8a62", "Importance Sampling", 36),
    "control": (0.5, "s", "#67a9cf", "Control", 22),
    "era5": (1.5, "d", "#2166ac", "ERA5", 8),
}
COLUMNS = {
    "guided": ("is", "probability_unweighted"),
    "is": ("is", "probability"),
    "control": ("control", "probability"),
    "era5": ("is", "era5_probability"),
}


def plot_healpix_map(ax, data, crs, norm, nlat=256, nlon=512):
    hpx = healpix.Grid(healpix.npix2level(data.shape[-1]))
    lons, lats = np.meshgrid(np.linspace(-180, 180, nlon), np.linspace(-90, 90, nlat))
    coords = crs.transform_points(ccrs.PlateCarree(), lons, lats)
    valid = np.isfinite(coords[..., 0])
    data = torch.as_tensor(data)
    out = torch.full(lats.shape, torch.nan, dtype=data.dtype)
    regrid = hpx.get_bilinear_regridder_to(lats[valid], lons[valid])
    regrid.to(data)
    out[torch.as_tensor(valid)] = regrid(data)
    im = ax.pcolormesh(
        coords[..., 0],
        coords[..., 1],
        out.numpy(),
        cmap="YlOrRd",
        norm=norm,
        transform=crs,
        rasterized=True,
    )
    ax.coastlines(color="k")
    return im


def load_maps(directory: str) -> xr.DataArray:
    files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    data = xr.concat([xr.open_dataset(f)["tc_probability"] for f in files], dim="time")
    return data.sel(time=data["time"].dt.year >= MIN_YEAR)


def time_mean_map(directory: str) -> np.ndarray:
    packed = os.path.join(directory, MEAN_MAP)
    if os.path.exists(packed):
        return xr.open_dataset(packed)["tc_probability"].values
    return load_maps(directory).mean(dim="time").values


def read_estimates(csv_dir: str) -> tuple[list[str], dict[str, pd.DataFrame]]:
    tables = {k: pd.read_csv(f"{csv_dir}/{k}.csv") for k in ("is", "control")}
    est = {
        key: pd.DataFrame({"p": tables[src][col], "std": tables[src][f"{col}_std"]})
        for key, (src, col) in COLUMNS.items()
    }
    est["era5"]["p"] = est["era5"]["p"].where(est["era5"]["p"] > 0)
    return [f"{t:g}" for t in tables["is"]["threshold"]], est


def plot_estimates(ax, csv_dir, xlabel) -> None:
    thresholds, est = read_estimates(csv_dir)
    x = np.arange(len(thresholds))
    for key, (shift, fmt, color, label, _) in SERIES.items():
        ax.errorbar(
            x + shift * 0.18,
            est[key]["p"],
            yerr=est[key]["std"],
            fmt=fmt,
            color=color,
            markersize=8,
            capsize=5,
            capthick=2,
            elinewidth=2,
            label=label,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    for i in x:
        shown = ["control", "is", "guided"]
        if est["era5"]["p"][i] > 0:
            shown.insert(0, "era5")
        top = max(est[k]["p"][i] + est[k]["std"][i] for k in shown)
        for key in shown:
            p, std = est[key]["p"][i], est[key]["std"][i]
            ax.annotate(
                f"{100 * p:.2f}±{100 * std:.2f}%",
                xy=(i, top),
                xytext=(0, SERIES[key][4]),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=SERIES[key][2],
                fontweight="medium",
            )

    ax.set_xlabel(xlabel, fontweight="medium")
    ax.set_ylabel("Frequency of exceeding threshold", fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_yscale("log")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.95)
    lows = [(e["p"] - e["std"]).min() for e in est.values()]
    highs = [(e["p"] + e["std"]).max() for e in est.values()]
    ax.set_ylim(min(v for v in lows if v > 0) * 0.3, np.nanmax(highs) * 7)


def panel_label(ax, label, x, y):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--xlabel", default="TC detection threshold")
    parser.add_argument("--era5-maps", required=True)
    parser.add_argument("--control-maps", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.25, wspace=0.15)
    ax_a = fig.add_subplot(gs[:, 0])
    plot_estimates(ax_a, args.csv_dir, args.xlabel)
    panel_label(ax_a, "a)", -0.05, 1.05)

    maps = {
        "ERA5 TC probability": time_mean_map(args.era5_maps),
        "cBottle EMA control TC probability": time_mean_map(args.control_maps),
    }
    norm = LogNorm(vmin=1e-4, vmax=max(np.nanmax(m) for m in maps.values()))
    crs = ccrs.Robinson()
    for row, (label, data) in enumerate(maps.items()):
        ax = fig.add_subplot(gs[row, 1], projection=crs)
        im = plot_healpix_map(ax, data, crs, norm)
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.08)
        cb.set_label(label, fontsize=11)
        cb.ax.tick_params(labelsize=9)
        panel_label(ax, "bc"[row] + ")", -0.02, 1.02)

    fig.savefig(f"{args.output}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
