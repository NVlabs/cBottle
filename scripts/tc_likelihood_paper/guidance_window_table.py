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
"""Guidance-window table: importance-sampling estimates per guidance window."""

import argparse

import numpy as np
import pandas as pd
from importance_sampling import (
    CLIPVALUE,
    PERIOD,
    clip_weights,
    load_likelihoods,
    log_weights,
)

THRESHOLD = 0.5


def summarize(name: str, df: pd.DataFrame, control: bool) -> dict:
    log_o = pd.Series(0.0, index=df.index) if control else log_weights(df)
    w, keep = clip_weights(log_o)
    return {
        "window": name,
        "n": len(df),
        "mean_tc": df["tc_probability"].mean(),
        "mean_log_o": log_o.mean(),
        "std_log_o": log_o.std(ddof=1),
        "frac_clip": (log_o >= CLIPVALUE).mean(),
        f"p_{THRESHOLD}": np.mean(
            w * (df["tc_probability"].to_numpy()[keep] > THRESHOLD)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ema-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def period(df):
        return df[df["timestamp"].between(*PERIOD)]

    runs = {
        "control": period(load_likelihoods(f"{args.ema_dir}/guidance_0.0")),
        "15_20": period(load_likelihoods(f"{args.ema_dir}/guidance_0.1")),
        "full": load_likelihoods(f"{args.ema_dir}/tier2_windows/full"),
        "full_g0.01": load_likelihoods(f"{args.ema_dir}/tier2_windows/full_g0.01"),
    }
    rows = [summarize(name, df, name == "control") for name, df in runs.items()]
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
