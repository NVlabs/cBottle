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
import torch
import random
from typing import Optional
from dataclasses import asdict
from cbottle.config.training.masking import MaskingConfig
from cbottle.models.distributed import compute_t_split


class FrameMasker:
    """
    Applies masking strategies to batched or unbatched video sequences.

    Assumes input batch contains keys:
        - `target`:    Tensor of shape (..., C1, T, X)
        - `condition`: Tensor of shape (..., C2, T, X)

    Modifies and returns a batch dictionary with:
        - `condition`: Tensor of shape (..., C1 + C2 + 1, T, X)
        - `mask`:      Tensor of shape (..., 1, T, X). 0 on dropped frames (to predict),
                        1 on frames to condition on
        - Other original keys are retained unchanged

    Masking strategies supported:
        - `random`:        Randomly drops frames independently with a min threshold
        - `blockwise`:     Drops a contiguous block of frames (past or future)
        - `interpolation`: Drops interior frames, keeps endpoints
        - `full_dropout`:  Masks all frames (fully unconditional)

    If keep_frames is provided, it overrides masking_config. If both are None, it will
    mask all frames.

    For distributed model parallelism, set model_rank, model_world_size, and t_full.
    The masker will compute a global mask and slice to the local time range.
    """

    def __init__(
        self,
        masking_config: Optional[MaskingConfig] = None,
        keep_frames: Optional[list[int]] = None,
        model_rank: int = 0,
        model_world_size: int = 1,
        t_full: Optional[int] = None,
    ):
        if keep_frames is None and masking_config is None:
            keep_frames = []

        self.config = masking_config
        self.keep_frames = keep_frames
        self.model_rank = model_rank
        self.model_world_size = model_world_size
        self.t_full = t_full

    def _compute_global_mask_indices(self, t_full: int, strategy_name: str):
        """Compute which global frame indices should be masked (set to 0).

        Args:
            t_full: Total number of frames
            strategy_name: The masking strategy to use

        Returns a list of global indices to mask.

        Note: Uses global random module, which is seeded at loop level using
        data_rank to ensure all MP ranks get identical mask decisions.
        """
        if strategy_name == "random":
            attempts = 0
            while attempts < 10:
                bernoulli_mask = [
                    random.random() < self.config.random_mask_prob
                    for _ in range(t_full)
                ]
                indices_to_mask = [i for i, m in enumerate(bernoulli_mask) if m]
                num_masked = len(indices_to_mask)
                if num_masked >= max(1, int(t_full * 0.5)) and num_masked < t_full:
                    break
                attempts += 1

            if attempts == 10:
                indices_to_mask = random.sample(
                    range(t_full), int(t_full * self.config.random_mask_prob)
                )
            return indices_to_mask

        elif strategy_name == "blockwise":
            num_to_mask = round(t_full * self.config.block_mask_fraction)
            num_to_mask -= int(random.random() < 0.2)
            mask_past = (
                random.random() < 0.5 if self.config.block_predict_past else False
            )
            if mask_past:
                return list(range(num_to_mask))
            else:
                return list(range(t_full - num_to_mask, t_full))

        elif strategy_name == "interpolation":
            num_to_mask = max(
                1, round(t_full * self.config.interpolation_mask_fraction)
            )
            if num_to_mask > t_full - 2:
                num_to_mask = t_full - 2
            return random.sample(range(1, t_full - 1), num_to_mask)

        elif strategy_name == "full_dropout":
            return list(range(t_full))

        elif strategy_name == "specific_frames":
            return [i for i in range(t_full) if i not in self.keep_frames]

        return []

    def __call__(self, batch):
        batch = {**batch}
        target = batch["target"]
        has_batch_dim = target.ndim == 4

        if not has_batch_dim:
            target = target.unsqueeze(0)
        B, C, T_local, X = target.shape

        # Determine full T and local T range
        if self.model_world_size > 1 and self.t_full is not None:
            t_full = self.t_full
            t_splits = compute_t_split(t_full, self.model_world_size)
            t_start = sum(t_splits[: self.model_rank])
        else:
            t_full = T_local
            t_start = 0

        if self.keep_frames is not None:
            strategy_name = "specific_frames"
        else:
            strategy_weights = asdict(self.config.strategy_weights)
            strategy_name = random.choices(
                population=list(strategy_weights.keys()),
                weights=list(strategy_weights.values()),
                k=1,
            )[0]

        global_indices_to_mask = self._compute_global_mask_indices(
            t_full, strategy_name
        )

        # Convert global indices to local indices
        local_indices_to_mask = [
            i - t_start
            for i in global_indices_to_mask
            if t_start <= i < t_start + T_local
        ]

        # Build local mask
        mask = torch.ones((B, 1, T_local, X), dtype=torch.bool, device=target.device)
        if local_indices_to_mask:
            mask[:, :, local_indices_to_mask, :] = 0

        if not has_batch_dim:
            mask = mask.squeeze(0)

        # Add the masked targets and the mask itself as conditioning
        batch["condition"] = torch.cat(
            [batch["target"] * mask, batch["condition"]], dim=-3
        )
        batch["condition"] = torch.cat([batch["condition"], mask], dim=-3)

        batch["mask"] = mask
        return batch
