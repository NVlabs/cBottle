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
import random
import torch


class RoundRobinLoader(torch.utils.data.IterableDataset):
    """Round-robin interleaving of multiple map-style dataloaders.

    This allows converting map-style datasets to iterable-style. It loads data
    from the dataloaders in a round-robin manner, removing exhausted dataloaders
    from rotation until ALL dataloaders are exhausted.

    Args:
        dataloaders: List of DataLoader instances to interleave
        shuffle_order: If True, shuffle the worker visitation order periodically
        shuffle_every: Reshuffle worker order every N rounds (one round = one batch from each worker)
        seed: Random seed for shuffling (use data_rank for different orders per DP group)
    """

    def __init__(
        self,
        dataloaders: list[torch.utils.data.DataLoader],
        shuffle_order: bool = False,
        shuffle_every: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.dataloaders = dataloaders
        self.shuffle_order = shuffle_order
        self.shuffle_every = shuffle_every
        self.seed = seed
        self._epoch = 0
        self._batch_count = 0

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        iterators = [iter(dl) for dl in self.dataloaders]
        active_indices = list(range(len(self.dataloaders)))
        batch_in_epoch = 0
        round_count = 0

        # RNG for shuffling worker order
        rng = random.Random(self.seed + self._epoch * 100003)

        print(
            f"[RoundRobin] Starting epoch {self._epoch} with {len(active_indices)} loaders, "
            f"shuffle_order={self.shuffle_order}, shuffle_every={self.shuffle_every}"
        )

        while active_indices:
            if self.shuffle_order and round_count % self.shuffle_every == 0:
                rng.shuffle(active_indices)
            round_count += 1

            for idx in list(active_indices):
                try:
                    batch_in_epoch += 1
                    self._batch_count += 1
                    yield next(iterators[idx])
                except StopIteration:
                    # Remove exhausted dataloader from rotation
                    print(
                        f"[RoundRobin] Loader {idx} exhausted after {batch_in_epoch} batches "
                        f"({len(active_indices)-1} loaders remain)"
                    )
                    active_indices.remove(idx)

        print(
            f"[RoundRobin] Epoch {self._epoch} complete: {batch_in_epoch} batches, "
            f"{round_count} rounds, {self._batch_count} total batches"
        )
        self._epoch += 1
