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
"""Fig. A3: TC classifier probability versus HPX3 max wind and min MSLP for
control, guided and ERA5, from ``compute_physical_metrics.py``."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde

MIN_BIN = 15
SERIES = [
    ("Control", "#7eb6d9", "o", 3),
    ("Guided 0.1", "#c44e52", "s", 4),
    ("ERA5", "#2166ac", "D", 5),
]
YMIN, YMAX = 3e-7, 1.4


def binned_mean(x, y, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= MIN_BIN:
            rows.append(
                (0.5 * (lo + hi), y[m].mean(), y[m].std(ddof=1) / np.sqrt(m.sum()))
            )
    return np.array(rows).T


def plot_panel(fig, spec, series, xlabel, x_edges, xlim, invert_x, title):
    inner = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
        width_ratios=[4.2, 1.0],
        height_ratios=[1.0, 4.2],
        hspace=0.04,
        wspace=0.04,
    )
    ax_x = fig.add_subplot(inner[0, 0])
    ax = fig.add_subplot(inner[1, 0], sharex=ax_x)
    ax_y = fig.add_subplot(inner[1, 1], sharey=ax)
    y_grid = np.logspace(np.log10(YMIN), 0, 256)
    x_grid = np.linspace(xlim[0], xlim[1], 256)

    for name, color, marker, z in SERIES:
        x, y = series[name]
        ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x, y = x[ok], y[ok]
        ax.scatter(
            x,
            y,
            s=6,
            alpha=0.14,
            color=color,
            edgecolors="none",
            zorder=z,
            rasterized=True,
        )
        cx, my, se = binned_mean(x, y, x_edges)
        ax.errorbar(
            cx,
            my,
            yerr=se,
            fmt=marker + "-",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            markersize=4.5,
            linewidth=1.7,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=z + 10,
            label=name,
        )
        density_x = gaussian_kde(x)(x_grid)
        ax_x.fill_between(x_grid, density_x, color=color, alpha=0.28, linewidth=0)
        ax_x.plot(x_grid, density_x, color=color, linewidth=1.2)
        density_y = gaussian_kde(np.log10(y))(np.log10(y_grid))
        ax_y.fill_betweenx(y_grid, density_y, color=color, alpha=0.28, linewidth=0)
        ax_y.plot(density_y, y_grid, color=color, linewidth=1.2)

    ax.set_yscale("log")
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("TC classifier probability")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left" if invert_x else "lower right", framealpha=0.92)

    ax_x.set_title(title, fontsize=12)
    ax_x.tick_params(labelbottom=False)
    ax_x.set_yticks([])
    ax_x.set_ylabel("dens.")
    ax_y.tick_params(labelleft=False)
    ax_y.set_xticks([])
    ax_y.set_xlabel("dens.")
    for side in ("left", "top", "right"):
        ax_x.spines[side].set_visible(False)
    for side in ("bottom", "top", "right"):
        ax_y.spines[side].set_visible(False)
    ax_y.set_yscale("log")
    ax_y.set_ylim(YMIN, YMAX)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def read(name):
        return pd.read_csv(f"{args.metrics_dir}/{name}", parse_dates=["timestamp"])

    ctrl = read("physical_metrics_guidance_0.0.csv")
    era5 = read("era5_hpx3_metrics.csv").merge(
        ctrl[["timestamp", "era5_tc_probability"]], on="timestamp"
    )
    era5 = era5[
        (era5["ws_hpx3_max"] > 1e-6)
        & (era5["mslp_hpx3_min"] > 1e4)
        & era5["era5_tc_probability"].notna()
    ]
    frames = {
        "Control": (ctrl, "tc_probability"),
        "Guided 0.1": (read("physical_metrics_guidance_0.1.csv"), "tc_probability"),
        "ERA5": (era5, "era5_tc_probability"),
    }
    wind = {
        k: (df["ws_hpx3_max"].to_numpy(), df[p].to_numpy())
        for k, (df, p) in frames.items()
    }
    mslp = {
        k: (df["mslp_hpx3_min"].to_numpy() / 100.0, df[p].to_numpy())
        for k, (df, p) in frames.items()
    }

    wmax = max(s[0].max() for s in wind.values())
    pmin = min(s[0].min() for s in mslp.values())
    pmax = max(s[0].max() for s in mslp.values())
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9})
    fig = plt.figure(figsize=(13.2, 6.2))
    outer = GridSpec(1, 2, figure=fig, wspace=0.22)
    plot_panel(
        fig,
        outer[0],
        wind,
        "Max 10 m wind in HPX3 pixel (m/s)",
        np.arange(2.0, np.ceil(wmax) + 1.0, 1.0),
        (0, wmax + 1),
        invert_x=False,
        title="Wind",
    )
    plot_panel(
        fig,
        outer[1],
        mslp,
        "Min MSLP in HPX3 pixel (hPa)",
        np.arange(np.floor(pmin), np.ceil(pmax) + 2.0, 2.0),
        (pmin - 2, pmax + 2),
        invert_x=True,
        title="MSLP",
    )
    fig.savefig(f"{args.output}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
