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
"""Importance-sampling estimates of TC exceedance frequencies near Miami.

Writes ``is.csv`` and ``control.csv``, one row per threshold, for the classifier
TC probability (``fig4/``, Fig. 4a) and the HPX3 max 10 m wind (``wind_hpx3/``,
Fig. A2).
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

CLIPVALUE = 5.0
MIN_YEAR = 1980
PERIOD = pd.Timestamp("2004-09-01"), pd.Timestamp("2009-09-26 18:00")
FIG4_THRESHOLDS = (0.2, 0.5, 0.9, 0.98)
WIND_THRESHOLDS = (12.0, 15.0, 18.0, 20.0, 22.0)


def rank_csvs(run_dir: str) -> dict[int, str]:
    paths = glob.glob(os.path.join(run_dir, "likelihoods_rank*.csv"))
    return dict(sorted((int(re.search(r"(\d+)\.csv$", p)[1]), p) for p in paths))


def load_likelihoods(run_dir: str) -> pd.DataFrame:
    merged = os.path.join(run_dir, "likelihoods.csv")
    paths = [merged] if os.path.exists(merged) else rank_csvs(run_dir).values()
    df = pd.concat(map(pd.read_csv, paths), ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df.drop_duplicates("timestamp").reset_index(drop=True)


def run_bounds(times: pd.DatetimeIndex):
    breaks = np.flatnonzero(np.diff(times) > pd.Timedelta(days=1)) + 1
    return zip(np.r_[0, breaks], np.r_[breaks, len(times)])


def contiguous_runs(times) -> list[pd.DatetimeIndex]:
    times = pd.DatetimeIndex(times).unique().sort_values()
    return [times[start:end] for start, end in run_bounds(times)]


def log_weights(df: pd.DataFrame) -> pd.Series:
    return (
        (
            df["gaussian_logp_backward_without_guidance"]
            + df["score_likelihood_backward_without_guidance"]
        )
        - (
            df["gaussian_logp_backward_with_guidance"]
            + df["score_likelihood_backward_with_guidance"]
        )
        - df["guidance_likelihood_backward_with_guidance"]
    )


def clip_weights(log_w) -> tuple[np.ndarray, np.ndarray]:
    log_w = np.asarray(log_w, dtype=float)
    keep = log_w < CLIPVALUE
    return np.exp(log_w[keep]), keep


def estimate(weights: np.ndarray, indicator: np.ndarray) -> dict:
    n = len(weights)
    wi = weights * indicator
    p = float(np.mean(wi))
    pu = float(np.mean(indicator))
    return {
        "probability": p,
        "probability_std": float(np.sqrt((np.mean(wi**2) - p**2) / n)),
        "probability_unweighted": pu,
        "probability_unweighted_std": float(np.sqrt(pu * (1 - pu) / n)),
        "total_samples": n,
        "event_samples": int(np.sum(indicator)),
    }


def block_bootstrap_std(values, timestamps: pd.Series, block=16, n_boot=1000):
    order = np.argsort(timestamps.to_numpy())
    values = np.asarray(values)[order]
    times = pd.DatetimeIndex(timestamps.to_numpy()[order])
    blocks = [
        values[i : min(i + block, end)]
        for start, end in run_bounds(times)
        for i in range(start, end, block)
    ]
    rng = np.random.RandomState(42)
    means = [
        np.concatenate(
            [blocks[c] for c in rng.randint(0, len(blocks), len(blocks))]
        ).mean()
        for _ in range(n_boot)
    ]
    return float(np.std(means))


def write_estimates(out_dir, thresholds, w, guided, control, era5, era5_times):
    os.makedirs(out_dir, exist_ok=True)
    rows = {"is": [], "control": []}
    for thr in thresholds:
        ind = (era5 > thr).astype(float)
        rows["is"].append(
            {
                "threshold": thr,
                **estimate(w, guided > thr),
                "era5_probability": float(ind.mean()),
                "era5_probability_std": block_bootstrap_std(ind, era5_times),
            }
        )
        ctrl = estimate(np.ones(len(control)), control > thr)
        rows["control"].append({"threshold": thr, **ctrl})
    for name, r in rows.items():
        pd.DataFrame(r).to_csv(f"{out_dir}/{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ema-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    def metrics(name):
        return pd.read_csv(f"{args.metrics_dir}/{name}", parse_dates=["timestamp"])

    lik = load_likelihoods(f"{args.ema_dir}/guidance_0.1")
    lik["log_w"] = log_weights(lik)
    control = load_likelihoods(f"{args.ema_dir}/guidance_0.0")
    control = control[control["timestamp"].dt.year >= MIN_YEAR]
    guided = lik[lik["timestamp"].dt.year >= MIN_YEAR]
    w, keep = clip_weights(guided["log_w"])
    guided = guided[keep]
    write_estimates(
        f"{args.output_dir}/fig4",
        FIG4_THRESHOLDS,
        w,
        guided["tc_probability"].to_numpy(),
        control["tc_probability"].to_numpy(),
        guided["era5_tc_probability"].to_numpy(),
        guided["timestamp"],
    )

    guided = metrics("physical_metrics_guidance_0.1.csv").merge(
        lik[["timestamp", "log_w"]], on="timestamp", how="left"
    )
    w, keep = clip_weights(guided["log_w"])
    era5 = metrics("era5_hpx3_metrics.csv").query("ws_hpx3_max > 1e-6")
    write_estimates(
        f"{args.output_dir}/wind_hpx3",
        WIND_THRESHOLDS,
        w,
        guided["ws_hpx3_max"].to_numpy()[keep],
        metrics("physical_metrics_guidance_0.0.csv")["ws_hpx3_max"].to_numpy(),
        era5["ws_hpx3_max"].to_numpy(),
        era5["timestamp"],
    )


if __name__ == "__main__":
    main()
