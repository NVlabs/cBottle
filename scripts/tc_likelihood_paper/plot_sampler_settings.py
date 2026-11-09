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
"""Fig. A1: bias, weight spread and GPU cost of the ``tier1`` sampler settings
relative to the densest one. Writes ``OUTPUT.csv`` and the figure."""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from importance_sampling import load_likelihoods, log_weights

PAPER = "s36_e25_k3"
REFERENCE = "s36_e25_k6"
EVENT_THRESHOLD = 0.5
BIAS_TOLERANCE = 0.3
N_BOOT = 2000
GRID_COLOR = {
    (18, 0): "#a1d99b",
    (18, 5): "#41ab5d",
    (18, 25): "#006d2c",
    (36, 0): "#9ecae1",
    (36, 5): "#4292c6",
    (36, 25): "#08519c",
}
PROBE_MARKER = {1: "o", 3: "s", 6: "D"}
PAPER_LABEL = "paper: 36+25, $k=3$"
REF_MATH = r"36+25,\,k=6"
HINTS = {"x": r"$\leftarrow$ lower is better", "y": r"$\downarrow$ lower is better"}
ANCHORS = {
    "upper right": (0.97, 0.97, "right", "top"),
    "lower left": (0.03, 0.03, "left", "bottom"),
    "lower right": (0.97, 0.03, "right", "bottom"),
}


def load_sweep(sweep_dir: str, guided_dir: str) -> dict[str, pd.DataFrame]:
    probs = load_likelihoods(guided_dir).set_index("timestamp")[["tc_probability"]]
    settings = {}
    for directory in sorted(glob.glob(os.path.join(sweep_dir, "*/"))):
        df = load_likelihoods(directory).join(probs, on="timestamp")
        df["log_w"] = log_weights(df)
        settings[os.path.basename(directory.rstrip("/"))] = df.set_index("timestamp")
    return settings


def summarize(settings: dict[str, pd.DataFrame], generation_seconds: float):
    reference = settings[REFERENCE]
    rows = []
    for name, df in settings.items():
        common = reference.index.intersection(df.index)
        log_w = df.loc[common, "log_w"].to_numpy()
        log_w_ref = reference.loc[common, "log_w"].to_numpy()
        event = (reference.loc[common, "tc_probability"] > EVENT_THRESHOLD).to_numpy()
        delta = log_w - log_w_ref
        p, p_ref = np.exp(log_w) * event, np.exp(log_w_ref) * event
        idx = np.random.default_rng(0).integers(0, len(common), (N_BOOT, len(common)))
        spread = delta[idx].std(ddof=1, axis=1)
        ratio = np.log(p[idx].mean(axis=1) / p_ref[idx].mean(axis=1))
        base, extra, probes = name.split("_")[:3]
        rows.append(
            {
                "setting": name,
                "base_steps": int(base[1:]),
                "extra_steps": int(extra[1:]),
                "probes": int(probes[1:]),
                "stepper": "heun" if name.endswith("_heun") else "euler",
                "n": len(df),
                "rescore_seconds": df["gpu_seconds"].mean(),
                "sample_seconds": df["gpu_seconds"].mean() + generation_seconds,
                "log_o_sigma": delta.std(ddof=1),
                "log_o_sigma_lo": np.percentile(spread, 2.5),
                "log_o_sigma_hi": np.percentile(spread, 97.5),
                "log_o_bias": delta.mean(),
                "log_phat_ratio": np.log(p.mean() / p_ref.mean()),
                "log_phat_ratio_lo": np.percentile(ratio, 2.5),
                "log_phat_ratio_hi": np.percentile(ratio, 97.5),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_seconds").reset_index(drop=True)


def mark_paper(ax, x: float, y: float) -> None:
    ax.plot(
        x,
        y,
        marker="o",
        ms=15,
        mfc="none",
        mec="black",
        mew=1.6,
        ls="",
        zorder=5,
        label=PAPER_LABEL,
    )


def mark_target(ax) -> None:
    ax.axhline(0.0, color="0.4", ls="--", lw=0.8, zorder=1)
    ax.text(
        0.02,
        0.0,
        r"target (36+25, $k=6$)",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.25",
        zorder=2,
        clip_on=False,
    )


def add_better_hint(ax, axis: str, loc: str) -> None:
    x, y, ha, va = ANCHORS[loc]
    ax.text(
        x,
        y,
        HINTS[axis],
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.5,
        color="0.3",
        zorder=6,
    )


def style_point(row: pd.Series) -> tuple[str, str]:
    color = GRID_COLOR.get((row["base_steps"], row["extra_steps"]), "0.55")
    return color, "*" if row["stepper"] == "heun" else PROBE_MARKER[row["probes"]]


def errorbar(ax, x, y, lo, hi, row, capsize=2, ms=8):
    color, marker = style_point(row)
    ax.errorbar(
        x,
        y,
        yerr=[[y - lo], [hi - y]],
        fmt=marker,
        color=color,
        ecolor=color,
        capsize=capsize,
        ms=11 if marker == "*" else ms,
        zorder=3,
    )


def draw_phat_cost(ax, table: pd.DataFrame) -> None:
    clip = 3.2
    ax.axhspan(-BIAS_TOLERANCE, BIAS_TOLERANCE, color="0.88", zorder=0, lw=0)
    mark_target(ax)
    for _, row in table.iterrows():
        if row["setting"] == REFERENCE:
            color, marker = style_point(row)
            ax.plot(
                row["sample_seconds"], 0.0, marker=marker, color=color, ms=8, zorder=4
            )
            continue
        y = row["log_phat_ratio"]
        y_plot, lo, hi = np.clip(
            [y, row["log_phat_ratio_lo"], row["log_phat_ratio_hi"]], -clip, clip
        )
        errorbar(ax, row["sample_seconds"], y_plot, lo, hi, row, capsize=3)
        if abs(y) > clip:
            ax.annotate(
                f"{y:+.1f}",
                (row["sample_seconds"], y_plot),
                textcoords="offset points",
                xytext=(0, 6 if y > 0 else -12),
                ha="center",
                fontsize=7,
                color=style_point(row)[0],
            )
    paper_row = table.set_index("setting").loc[PAPER]
    mark_paper(ax, paper_row["sample_seconds"], paper_row["log_phat_ratio"])
    ax.set_xscale("log")
    ax.set_ylim(-clip - 0.15, clip + 0.55)
    ax.set_xlabel("GPU-seconds per sample")
    add_better_hint(ax, "x", "lower right")
    ax.set_ylabel(
        "Bias of IS probability estimate\n"
        rf"($\log\,\hat p/\hat p_{{{REF_MATH}}}$ at classifier $p>0.5$)",
        fontsize=9,
    )

    def handle(**kwargs):
        return plt.Line2D([0], [0], ls="", **kwargs)

    handles = [
        handle(marker="o", color=GRID_COLOR[g], ms=7, label=f"{g[0]}+{g[1]}")
        for g in sorted(GRID_COLOR)
    ]
    handles += [
        handle(marker=m, color="0.3", ms=7, label=f"$k={k}$")
        for k, m in PROBE_MARKER.items()
    ]
    if (table["stepper"] == "heun").any():
        handles.append(handle(marker="*", color="0.3", ms=11, label="Heun, $k=3$"))
    handles.append(
        handle(marker="o", color="black", mfc="none", mew=1.6, ms=9, label=PAPER_LABEL)
    )
    ax.legend(handles=handles, fontsize=6.5, loc="lower left", ncol=2)


def plot_sampler_settings(table: pd.DataFrame, output: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
    ax_phat, ax_extra, ax_cost, ax_k = axes.ravel()
    ax_phat.sharex(ax_cost)
    ax_cost.sharey(ax_k)
    spread = table.set_index("setting")
    paper_row = spread.loc[PAPER]
    spread_label = (
        "Variability of IS weights\n"
        rf"(across-sample std. of $\log o - \log o_{{{REF_MATH}}}$)"
    )

    draw_phat_cost(ax_phat, table)

    jitter = {1: -0.55, 3: 0.0, 6: 0.55}
    for _, row in table.iterrows():
        x = (
            row["extra_steps"]
            + jitter[row["probes"]]
            + 0.35 * (row["stepper"] == "heun")
        )
        sem = row["log_o_sigma"] / np.sqrt(row["n"])
        bias = row["log_o_bias"]
        errorbar(ax_extra, x, bias, bias - sem, bias + sem, row)
    mark_paper(
        ax_extra,
        paper_row["extra_steps"] + jitter[paper_row["probes"]],
        paper_row["log_o_bias"],
    )
    mark_target(ax_extra)
    ax_extra.set_xticks([0, 5, 25])
    ax_extra.set_xlabel("Extra steps inside the guidance window")
    ax_extra.set_ylabel(
        "Bias of IS weights\n" rf"(mean $\log o - \log o_{{{REF_MATH}}}$)",
        fontsize=9,
    )

    others = table[table["setting"] != REFERENCE]
    for _, row in others.iterrows():
        sigma = row["log_o_sigma"]
        lo, hi = row["log_o_sigma_lo"], row["log_o_sigma_hi"]
        errorbar(ax_cost, row["sample_seconds"], sigma, lo, hi, row, ms=7)
    mark_paper(ax_cost, paper_row["sample_seconds"], paper_row["log_o_sigma"])
    ax_cost.set_xscale("log")
    ax_cost.set_yscale("log")
    ax_cost.set_xlabel("GPU-seconds per sample")
    add_better_hint(ax_cost, "x", "lower left")
    ax_cost.set_ylabel(spread_label, fontsize=9)
    add_better_hint(ax_cost, "y", "upper right")
    for ax in (ax_phat, ax_cost):
        ax.set_xticks([100, 200, 400, 600])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(labelbottom=True)

    for grid, color in sorted(GRID_COLOR.items()):
        names = [f"s{grid[0]}_e{grid[1]}_k{k}" for k in PROBE_MARKER]
        rows = spread.loc[[name for name in names if name != REFERENCE]]
        for _, row in rows.iterrows():
            sigma = row["log_o_sigma"]
            lo, hi = row["log_o_sigma_lo"], row["log_o_sigma_hi"]
            errorbar(ax_k, row["probes"], sigma, lo, hi, row)
        ax_k.plot(rows["probes"], rows["log_o_sigma"], color=color, lw=1.0, zorder=2)
    heun = spread[spread["stepper"] == "heun"]
    for _, row in heun.iterrows():
        ax_k.plot(
            row["probes"],
            row["log_o_sigma"],
            marker="*",
            color=GRID_COLOR[(36, 25)],
            ms=11,
            zorder=4,
        )
    mark_paper(ax_k, 3, paper_row["log_o_sigma"])
    ax_k.set_xscale("log")
    ax_k.set_xticks(list(PROBE_MARKER))
    ax_k.set_xticklabels(list(PROBE_MARKER))
    ax_k.xaxis.set_minor_locator(mticker.NullLocator())
    ax_k.tick_params(labelleft=True)
    ax_k.set_xlabel("Hutchinson probes $k$")
    ax_k.set_ylabel(spread_label, fontsize=9)
    add_better_hint(ax_k, "y", "upper right")
    ax_cost.set_ylim(
        others["log_o_sigma"].min() / 1.5, others["log_o_sigma"].max() * 1.35
    )
    seconds = table["sample_seconds"]
    ax_cost.set_xlim(seconds.min() / 1.15, seconds.max() * 1.15)

    for idx, ax in enumerate(axes.ravel()):
        ax.grid(alpha=0.25, which="both")
        ax.text(
            -0.12,
            1.05,
            f"{'abcd'[idx]})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    ax_phat.tick_params(labelbottom=True)
    ax_k.tick_params(labelleft=True)
    fig.savefig(f"{output}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output}.pdf", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--guided-dir", required=True)
    parser.add_argument("--cost-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    generation_seconds = load_likelihoods(args.cost_dir)["gpu_seconds"].mean()
    table = summarize(load_sweep(args.sweep_dir, args.guided_dir), generation_seconds)
    table.to_csv(f"{args.output}.csv", index=False)
    plot_sampler_settings(table, args.output)


if __name__ == "__main__":
    main()
