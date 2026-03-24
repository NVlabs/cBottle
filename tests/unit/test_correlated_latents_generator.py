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
from cbottle.correlated_latents_generator import CorrelatedLatentGenerator

DEVICE = torch.device("cpu")
SEED = 42
C, T, S = 2, 1, 16  # channels, time_length, spatial_dims


def _make_gen(rank=0):
    return CorrelatedLatentGenerator(
        device=DEVICE,
        correlation_half_life=8.0,
        seed=SEED,
        rank=rank,
    )


class TestRankSeedsAreUnique:
    """Different ranks (= different time_step offsets) should get different seeds."""

    def test_many_ranks_all_unique(self):
        samples_per_rank = 1055
        seeds = set()
        for rank in range(256):
            gen = _make_gen(rank=rank)
            gen._initialize_state(C, T, S, time_step=rank * samples_per_rank)
            seeds.add(gen.rng.initial_seed())
        assert len(seeds) == 256


class TestSeedReproducibility:
    """Same inputs must always produce the same seeds."""

    def test_deterministic(self):
        gen1 = _make_gen()
        gen2 = _make_gen()

        gen1._initialize_state(C, T, S, time_step=500)
        gen2._initialize_state(C, T, S, time_step=500)

        assert gen1.rng.initial_seed() == gen2.rng.initial_seed()
