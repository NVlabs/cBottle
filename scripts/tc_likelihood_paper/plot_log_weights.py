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
"""Fig. 3: TC probability and log-weight components versus guidance scale, for
``--run SCALE=DIR`` restricted to the Fig. 3 period."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from importance_sampling import PERIOD, load_likelihoods

TITLES = {
    "tc_prob": "TC probability",
    "total": r"$log(o(x))$",
    "logp_diff": r"$\Delta_{latent}$",
    "score_diff": r"$\Delta_{denoiser}$",
    "guidance_div": r"$\Delta_{guidance}$",
}


def components(run_dir: str) -> dict[str, np.ndarray]:
    df = load_likelihoods(run_dir)
    df = df[df["timestamp"].between(*PERIOD)]
    logp_diff = (
        df["gaussian_logp_backward_without_guidance"]
        - df["gaussian_logp_backward_with_guidance"]
    )
    score_diff = (
        df["score_likelihood_backward_without_guidance"]
        - df["score_likelihood_backward_with_guidance"]
    )
    guidance_div = -df["guidance_likelihood_backward_with_guidance"]
    out = {
        "tc_prob": df["tc_probability"],
        "total": logp_diff + score_diff + guidance_div,
        "logp_diff": logp_diff,
        "score_diff": score_diff,
        "guidance_div": guidance_div,
    }
    return {k: v[np.isfinite(v)].to_numpy() for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = dict(spec.split("=", 1) for spec in args.run)
    scales = sorted(runs, key=float)
    data = {s: components(runs[s]) for s in scales}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for idx, (ax, (key, title)) in enumerate(zip(axes, TITLES.items())):
        ax.violinplot(
            [data[s][key] for s in scales],
            positions=range(len(scales)),
            showmeans=True,
            showmedians=False,
            showextrema=True,
        )
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels(scales, fontsize=9)
        ax.set_xlabel("Guidance scale", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.text(
            -0.1,
            1.05,
            f"{'abcde'[idx]})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        if key == "tc_prob":
            ax.set_ylim(-0.02, 1.02)
        else:
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.output}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
