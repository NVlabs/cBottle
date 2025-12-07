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
import os
import random

import torch
import torch.utils.data

import cbottle.distributed as dist


def subsample(dataset, min_samples, world_size: int = None):
    if world_size is None:
        world_size = dist.get_world_size()
    samples = min_samples % world_size + min_samples
    golden_ratio = 1.618033988749
    n = len(dataset)
    sampler = [int((i * n * golden_ratio) % n) for i in range(samples)]
    sampler = sorted(sampler)
    return sampler


def distributed_split(tasks, drop_last=True, rank: int = None, world_size: int = None):
    n = len(tasks)
    if rank is None:
        rank = dist.get_rank()
    if world_size is None:
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


class ChunkedDistributedSampler(torch.utils.data.Sampler):
    """A chunked random sampler. This allows accessing the dataset sequentially
    within chunks, for better performance w/ chunked datasets that have caching
    implemented.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        chunk_size: int = 1,
        rank=0,
        num_replicas=1,
        shuffle=False,
        shuffle_within_chunk=False,
        drop_last=True,
        seed=42,
        sampler_fn=None,
        subsampled_chunk_size=None,
    ):
        """
        Args:
            base_dataset: A map-style dataset (e.g. HealpixDatasetV5).
            chunk_size: Number of consecutive samples in each chunk.
            batch_size: Size of the mini-batches yielded to the main loop.
            shuffle: Whether to shuffle order of chunks.
            shuffle_within_chunk: Whether to shuffle indices within each chunk.
            seed: random seed for the sampler, will be broadcasted from rank 0 to all other ranks
        """
        super().__init__()
        self.n = len(dataset)
        nchunks = self.n // chunk_size
        chunks = list(range(nchunks))

        if torch.distributed.is_initialized():
            seed = torch.tensor(seed).cuda()
            torch.distributed.broadcast(seed, src=0)
            seed = seed.item()
        self._chunk_sampler = (
            sampler_fn(chunks)
            if sampler_fn is not None
            else torch.utils.data.DistributedSampler(
                chunks,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        )
        self.chunk_size = chunk_size
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seed = seed
        self.rank = rank
        self.epoch = 0
        self.index_within_chunk = 0
        self._chunk_iter = iter(self._chunk_sampler)
        self._current_chunk_indices = None
        self.subsampled_chunk_size = (
            subsampled_chunk_size if subsampled_chunk_size is not None else chunk_size
        )

        if self.shuffle_within_chunk:
            self.rng = random.Random(seed + rank)

        # Logging state
        self._chunk_count = 0  # Chunks processed in current epoch
        self._total_chunks = 0  # Total chunks processed across all epochs
        self._log_every_n_chunks = (
            50  # Log every N chunks (set to 1 for full verbosity)
        )

        # print(f"[Sampler r={rank} pid={os.getpid()}] Init: n={self.n}, nchunks={nchunks}, "
        #       f"chunk_size={chunk_size}, chunks_per_replica~={nchunks // num_replicas}. Shuffle within chunk: {shuffle_within_chunk}")

    def set_epoch(self, epoch):
        old_epoch = self.epoch
        try:
            self._chunk_sampler.set_epoch(epoch)
        except AttributeError:
            pass
        self.epoch = epoch
        self.index_within_chunk = 0
        self._chunk_iter = iter(self._chunk_sampler)

        if self.shuffle_within_chunk:
            self.rng = random.Random(self.seed + self.rank + epoch * 100003)

        print(
            f"[Sampler r={self.rank} pid={os.getpid()}] set_epoch: {old_epoch} -> {epoch} "
            f"(processed {self._chunk_count} chunks in prev epoch, {self._total_chunks} total)"
        )
        self._chunk_count = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        if self.index_within_chunk == 0:
            try:
                self.active_chunk = next(self._chunk_iter)
            except StopIteration:
                self.set_epoch(self.epoch + 1)
                raise StopIteration()

            chunk_start = self.active_chunk * self.chunk_size
            self._current_chunk_indices = list(
                range(chunk_start, chunk_start + self.chunk_size)
            )

            # original_indices = self._current_chunk_indices[:10]

            if self.shuffle_within_chunk:
                self.rng.shuffle(self._current_chunk_indices)

            self._current_chunk_indices = self._current_chunk_indices[
                : self.subsampled_chunk_size
            ]

            self._chunk_count += 1
            self._total_chunks += 1

            # shuffled_indices = self._current_chunk_indices[:10]

        i = self._current_chunk_indices[self.index_within_chunk]
        self.index_within_chunk = (self.index_within_chunk + 1) % self.chunk_size
        return i


class RestartableDistributedSampler(torch.utils.data.Sampler):
    """A stateful distributed sampler that automatically loops over the dataset."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        rank=0,
        num_replicas=1,
        shuffle=True,
        drop_last=True,
        seed=42,
    ):
        super().__init__()
        self.iteration = 0
        self.epoch = 0
        self.len = len(dataset)
        self.seed = seed
        self.permutation = None
        self.rank = rank
        self.num_replicas = num_replicas

    def __len__(self):
        return self.len // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iteration = 0
        rng = torch.Generator().manual_seed(self.seed + self.epoch + self.rank)
        permutation = torch.randperm(self.len, generator=rng)

        rem = self.len % self.num_replicas
        if rem > 0:
            permutation = permutation[:-rem]
        self.permutation = permutation[self.rank :: self.num_replicas]

    def restart(self, epoch, iteration, seed=None):
        self.seed = seed or self.seed
        self.set_epoch(epoch)
        self.iteration = iteration

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= len(self):
            self.set_epoch(self.epoch + 1)
            raise StopIteration()

        idx = self.permutation[self.iteration]
        self.iteration += 1
        return idx
