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
"""Classifier AUC and calibration versus sigma on September 2018 ERA5 with
IBTrACS labels. Writes ``skill_vs_sigma.csv`` and ``reliability_vs_sigma.csv``."""

import argparse

import numpy as np
import pandas as pd
import torch
from earth2grid import healpix
from torch.utils.data import DataLoader

import cbottle.inference
from cbottle.datasets.dataset_3d import get_dataset

SIGMAS = (0.002, 0.02, 0.2, 1, 5, 10, 15, 18, 20, 30, 50, 80, 120, 200)
TIMES = pd.date_range("2018-09-01", "2018-09-30 18:00", freq="6h")
N_BINS = 10


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)[np.argsort(-s, kind="mergesort")]
    tpr = np.concatenate([[0.0], np.cumsum(y) / y.sum(), [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(~y) / (~y).sum(), [1.0]])
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def reliability_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bins = np.digitize(p, edges[1:-1])
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = bins == i
        rows.append((i, lo, hi, mask.sum(), p[mask].mean(), y[mask].mean()))
    return pd.DataFrame(rows, columns=["bin", "p_lo", "p_hi", "n", "mean_p", "freq"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    classifier = cbottle.inference.load(
        "cbottle-3d-moe-tc", root=args.checkpoint
    ).separate_classifier
    device = next(classifier.parameters()).device
    dataset = get_dataset(
        dataset="era5",
        split="",
        ibtracs_input=True,
        shuffle=False,
        infinite=False,
        rank=0,
        world_size=1,
    )
    dataset.set_times(TIMES)
    batches = [
        {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        for batch in DataLoader(dataset, batch_size=4)
    ]
    grid = healpix.Grid(level=3, pixel_order=healpix.HEALPIX_PAD_XY)
    tropical = np.abs(np.asarray(grid.lat)) <= 40.0
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    skill, reliability = [], []
    for sigma in SIGMAS:
        logits, labels = [], []
        for batch in batches:
            x0 = batch["target"]
            mask = ~torch.isnan(x0).any(dim=(2, 3), keepdim=True)
            noise = torch.randn(x0.shape, device=device, generator=rng)
            x = torch.where(mask, x0 + sigma * noise, 0.0)
            with torch.no_grad():
                out = classifier(
                    x,
                    torch.full((len(x), 1, 1, 1), sigma, device=device, dtype=x.dtype),
                    class_labels=batch["labels"],
                    condition=batch["condition"],
                    second_of_day=batch["second_of_day"],
                    day_of_year=batch["day_of_year"],
                )
            logits.append(out.logits.float().cpu())
            labels.append(batch["classifier_labels"].float().cpu())
        logit = torch.cat(logits).numpy().reshape(-1)
        y = (torch.cat(labels).numpy().reshape(-1) > 0.5).astype(np.float64)
        trop = np.tile(tropical, len(logit) // tropical.size)
        skill.append({"sigma": sigma, "auc_trop": roc_auc(y[trop], logit[trop])})
        rel = reliability_table(y, 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))
        reliability.append(rel.assign(sigma=sigma))

    pd.DataFrame(skill).to_csv(f"{args.output_dir}/skill_vs_sigma.csv", index=False)
    pd.concat(reliability).to_csv(
        f"{args.output_dir}/reliability_vs_sigma.csv", index=False
    )


if __name__ == "__main__":
    main()
