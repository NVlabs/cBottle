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
    """

    def __init__(
        self,
        masking_config: Optional[MaskingConfig] = None,
        keep_frames: Optional[list[int]] = None,
    ):
        if keep_frames is None and masking_config is None:
            keep_frames = []

        self.config = masking_config
        self.keep_frames = keep_frames

    def __call__(self, batch, condition_to_insert=None):
        batch = {**batch}
        if condition_to_insert is None:
            target = batch["target"]
        else:
            target = condition_to_insert
        has_batch_dim = target.ndim == 4

        if not has_batch_dim:
            target = target.unsqueeze(0)
        B, C, T, X = target.shape
        mask = torch.ones((B, 1, T, X), dtype=torch.bool, device=target.device)

        # Select masking strategy
        if self.keep_frames is not None:
            strategy_name = "specific_frames"
        else:
            strategy_weights = asdict(self.config.strategy_weights)
            strategy_name = random.choices(
                population=list(strategy_weights.keys()),
                weights=list(strategy_weights.values()),
                k=1,
            )[0]

        if strategy_name == "random":
            # For each frame, "flip a coin" whether to mask or not
            attempts = 0
            while attempts < 10:
                bernoulli_mask = torch.tensor(
                    [random.random() < self.config.random_mask_prob for _ in range(T)],
                    dtype=torch.bool,
                    device=target.device,
                )
                indices_to_mask = torch.nonzero(bernoulli_mask).squeeze()
                num_masked = indices_to_mask.numel()
                if num_masked >= max(1, int(T * 0.5)) and num_masked < T:
                    break
                attempts += 1

            if attempts == 10:
                indices_to_mask = random.sample(
                    range(T), int(T * self.config.random_mask_prob)
                )

            mask[:, :, indices_to_mask, :] = 0

        elif strategy_name == "blockwise":
            # Mask either the first or last num_to_mask contiguous frames
            num_to_mask = round(T * self.config.block_mask_fraction)
            num_to_mask -= int(random.random() < 0.2)

            mask_past = (
                random.random() < 0.5 if self.config.block_predict_past else False
            )
            if mask_past:
                mask[:, :, :num_to_mask, :] = 0
            else:
                mask[:, :, T - num_to_mask :, :] = 0

        elif strategy_name == "interpolation":
            # Mask interior frames, keep endpoints
            num_to_mask = max(1, round(T * self.config.interpolation_mask_fraction))
            if num_to_mask > T - 2:
                num_to_mask = T - 2
            indices_to_mask = random.sample(range(1, T - 1), num_to_mask)
            mask[:, :, indices_to_mask, :] = 0

        elif strategy_name == "full_dropout":
            # Mask all frames (unconditional)
            mask[:, :, :, :] = 0

        elif strategy_name == "specific_frames":
            # Keep only the specified frames (for inference)
            mask[:, :, :, :] = 0
            valid_indices = [i for i in self.keep_frames if 0 <= i < T]
            if len(valid_indices) != len(self.keep_frames):
                raise ValueError(f"Invalid keep_frames: {self.keep_frames}")

            mask[:, :, valid_indices, :] = 1

        if not has_batch_dim:
            mask = mask.squeeze(0)

        # Add the masked targets and the mask itself as conditioning
        batch["condition"] = torch.cat(
            [batch["target"] * mask, batch["condition"]], dim=-3
        )
        batch["condition"] = torch.cat([batch["condition"], mask], dim=-3)

        batch["mask"] = mask
        return batch
