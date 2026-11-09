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
"""Fig. A4: classifier skill and calibration versus sigma, from the output of
``eval_classifier_vs_sigma.py``."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIGMA_COLOR = {0.2: "#08519c", 15.0: "#4292c6", 80.0: "#9ecae1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    skill = pd.read_csv(f"{args.input_dir}/skill_vs_sigma.csv")
    rel = pd.read_csv(f"{args.input_dir}/reliability_vs_sigma.csv")
    fig, (ax_auc, ax_rel) = plt.subplots(1, 2, figsize=(9, 3.8))

    ax_auc.axvspan(
        15, 20, color="0.88", lw=0, zorder=0, label="guidance window [15, 20]"
    )
    ax_auc.plot(skill["sigma"], skill["auc_trop"], "o-", color="#08519c", ms=6)
    ax_auc.set_xscale("log")
    ax_auc.set_xlabel(r"noise level $\sigma$")
    ax_auc.set_ylabel(r"ROC AUC, tropical pixels ($|\mathrm{lat}|\leq 40°$)")
    ax_auc.legend(loc="lower left", fontsize=8)
    ax_auc.text(
        0.98,
        0.98,
        "↑ higher is better",
        transform=ax_auc.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.4",
    )

    ax_rel.plot(
        [0, 1], [0, 1], color="0.4", lw=0.8, ls="--", label="perfect calibration"
    )
    for sigma, color in SIGMA_COLOR.items():
        r = rel.loc[np.isclose(rel["sigma"], sigma) & (rel["n"] > 0)]
        ax_rel.plot(
            r["mean_p"],
            r["freq"],
            "o-",
            color=color,
            ms=5,
            label=rf"$\sigma={sigma:g}$",
        )
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_aspect("equal")
    ax_rel.set_xlabel("predicted TC probability")
    ax_rel.set_ylabel("observed TC frequency")
    ax_rel.legend(loc="upper left", fontsize=8)

    for ax, label in ((ax_auc, "a)"), (ax_rel, "b)")):
        ax.grid(alpha=0.25, which="both")
        ax.text(
            -0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold"
        )
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=200)
    fig.savefig(f"{args.output}.pdf", dpi=200)


if __name__ == "__main__":
    main()
