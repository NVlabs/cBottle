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
import pytest
import os
from cbottle.training import super_resolution


def test_train(tmp_path):
    """Run like this

    torchrun --nproc_per_node 1 -m pytest -vv -rs tests/integration/test_train_multidiffusion.py -s

    """
    if "WORLD_SIZE" not in os.environ:
        pytest.skip(
            "This test must be run in a parallel context. "
            "Try running pytest with `torchrun ... -m pytest` or mpirun."
        )

    super_resolution.train(
        tmp_path.as_posix(),
        num_steps=1,
        log_freq=1,
        test_fast=True,
        dataloader_num_workers=0,
    )
    # this should load the checkpoint
    super_resolution.train(
        tmp_path.as_posix(),
        num_steps=1,
        log_freq=1,
        test_fast=True,
        dataloader_num_workers=0,
    )
