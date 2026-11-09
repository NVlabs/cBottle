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
"""Fig. 2: guidance divergence versus sigma and Caribbean 10 m zonal wind for
three guidance windows, from the ``fig2`` runs of ``run_gpu.sh``."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import xarray as xr

from cbottle.visualizations import visualize

WINDOWS = {
    "15_20": "interval 15-20",
    "15_200": "interval 15-200",
    "full": "interval full",
}


def load_window(run: str):
    with xr.open_dataset(os.path.join(run, "0.nc")) as ds:
        uas = ds["uas"].isel(time=0).values
    with open(os.path.join(run, "traces", "trace_rank0_batch0_backward.json")) as f:
        rows = sorted(json.load(f)["divergence_data"], key=lambda r: r["sigma"])
    return uas, [r["sigma"] for r in rows], [r["divergence"] for r in rows]


def panel_label(ax, label, x):
    ax.text(
        x,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 14})
    n = len(WINDOWS)
    fig = plt.figure(figsize=(15, 8.2))
    margin_left, margin_right, spacing = 0.07, 0.02, 0.04
    plot_width = (1 - margin_left - margin_right - spacing * (n - 1)) / n
    maps = []
    for col, (name, title) in enumerate(WINDOWS.items()):
        uas, sigma, divergence = load_window(os.path.join(args.base_dir, name))
        left = margin_left + col * (plot_width + spacing)
        ax = fig.add_axes([left, 0.74, plot_width, 0.20])
        ax.plot(sigma, divergence, "b-", marker="o", linewidth=2, ms=4)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel(r"$\sigma$")
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel("Guidance Divergence")
        panel_label(ax, "abc"[col] + ")", -0.12)

        vis = visualize(
            uas,
            region="carib",
            nest=False,
            pos=(1, n, col + 1),
            vmin=-25,
            vmax=25,
            cmap="RdBu_r",
            add_colorbar=False,
            border=False,
            coastlines_color="black",
        )
        vis.ax.set_title("")
        panel_label(vis.ax, "def"[col] + ")", -0.08)
        vis.im.set_rasterized(True)
        maps.append(vis)

    fig.subplots_adjust(top=0.72, bottom=0.16, left=0.05, right=0.98, wspace=0.18)
    cb = fig.colorbar(
        maps[0].im,
        ax=[vis.ax for vis in maps],
        orientation="horizontal",
        pad=0.06,
        fraction=0.05,
        aspect=40,
    )
    cb.set_label("uas (m/s)")
    fig.savefig(f"{args.output}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{args.output}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
