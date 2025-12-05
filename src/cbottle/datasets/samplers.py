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
import math
import random

import torch.utils.data

import cbottle.distributed as dist


def subsample(dataset, min_samples):
    samples = min_samples % dist.get_world_size() + min_samples
    golden_ratio = 1.618033988749
    n = len(dataset)
    sampler = [int((i * n * golden_ratio) % n) for i in range(samples)]
    sampler = sorted(sampler)
    return sampler


def distributed_split(tasks, drop_last=True):
    n = len(tasks)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunk = math.ceil(len(tasks) / world_size)
    start = rank * chunk
    stop = n if drop_last and (rank == world_size - 1) else start + chunk
    return [t for i, t in enumerate(tasks) if start <= i < stop]


class InfiniteSequentialSampler(torch.utils.data.Sampler):
    """An infinite sampler that iterates sequentially through a dataset
    reshuffling every ``shuffle_every`` iterations
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
        shuffle_every: int = 48,
    ):
        self.shuffle = shuffle
        self.shuffle_every = shuffle_every
        self.n = len(dataset)
        self.rank = dist.get_rank()
        self.replicas = dist.get_world_size()

    def __iter__(self):
        i = random.randint(0, self.n - 1)
        k = 0
        while True:
            if (self.shuffle_every > 0) and (k % self.shuffle_every == 0):
                i = random.randint(0, self.n - 1)

            yield i

            i = (i + 1) % self.n
            k += 1


class InfiniteChunkedIterable(torch.utils.data.IterableDataset):
    """
    Infinitely yields batches of contiguous samples from the dataset, reshuffling every
    ``chunk_size // batch_size`` batches. As each worker runs __iter__ in its own process,
    workers are assigned independent chunks. Data is yielded from workers round-robin, so
    chunks will be interleaved across iterations.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        chunk_size: int = 48,
        batch_size: int = 4,
    ):
        """
        Args:
            base_dataset: A map-style dataset (e.g. HealpixDatasetV5).
            chunk_size: Number of consecutive samples in each chunk.
            batch_size: Size of the mini-batches yielded to the main loop.
        """
        super().__init__()
        self.dataset = base_dataset
        self.n = len(base_dataset)
        self.chunk_size = chunk_size
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            start_idx = random.randint(0, self.n - 1)
            indices = [(start_idx + j) % self.n for j in range(self.chunk_size)]

            for i in range(0, len(indices), self.batch_size):
                batch = [self.dataset[idx] for idx in indices[i : i + self.batch_size]]
                yield torch.utils.data.default_collate(batch)  # batch the list of dicts
