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
import torch
import pandas as pd
import numpy as np
from unittest.mock import Mock

from cbottle.video_rollout import VideoRollout
from cbottle.datasets.base import BatchInfo, TimeUnit


@pytest.fixture
def setup():
    """Create mock model and dataset."""
    batch_info = BatchInfo(
        channels=["a", "b", "c"],
        time_step=3,
        time_unit=TimeUnit.HOUR,
    )
    nchannels = len(batch_info.channels)

    model = Mock()
    npix = 100
    model.time_length = 4
    model.coords.batch_info = batch_info
    model.coords.npix = npix
    model.net.parameters.return_value = iter([torch.nn.Parameter(torch.randn(1))])

    def mock_sample(batch, return_untransformed=False, **kwargs):
        target = batch["target"]
        coords = Mock()
        coords.batch_info = batch_info
        coords.npix = npix
        return (
            (target, coords, target.clone())
            if return_untransformed
            else (target, coords)
        )

    model.sample = mock_sample

    times = pd.date_range("2020-01-01", periods=240, freq="1h")

    class FakeDataset:
        def __init__(self):
            self.times = times
            self.batch_info = batch_info

        def __getitem__(self, idx):
            return {
                "target": torch.ones(1, nchannels, model.time_length, npix) * (idx + 1),
                "timestamp": torch.tensor([self.times[idx].value / 1e9]),
                "condition": torch.zeros(1, 1, model.time_length, npix),
            }

    dataset = FakeDataset()

    return model, dataset, nchannels, npix, batch_info.time_step


def test_video_rollout(setup):
    model, dataset, nchannels, npix, time_step = setup
    time_length = model.time_length
    rollout = VideoRollout(model, dataset)
    num_conditioning_frames = 1

    rollout.seed_generation(dataset.times[0])
    step1 = rollout.step_forward([])
    step2 = rollout.step_forward(list(range(num_conditioning_frames)))
    step3 = rollout.step_forward(list(range(num_conditioning_frames + 1)))

    # Seed: all frames are new
    assert step1.is_seed
    assert step1.frames.shape == (1, nchannels, time_length, npix)
    assert step1.new_frames.shape == (1, nchannels, time_length, npix)

    assert not step2.is_seed
    assert step2.frames.shape == (1, nchannels, time_length, npix)
    assert step2.new_frames.shape == (1, nchannels, time_length - 1, npix)
    assert step3.new_frames.shape == (1, nchannels, time_length - 2, npix)

    # Cumulative tracking
    assert step1.cumulative_frames_written == 0
    assert step2.cumulative_frames_written == time_length
    assert step3.cumulative_frames_written == time_length + (time_length - 1)

    # Timestamps: contiguous 6-hour steps
    times1 = step1.new_timestamps[0].cpu().numpy()
    times2 = step2.new_timestamps[0].cpu().numpy()
    times3 = step3.new_timestamps[0].cpu().numpy()

    assert len(times1) == time_length
    assert len(times2) == time_length - 1
    assert len(times3) == time_length - 2

    all_times = np.concatenate([times1, times2, times3])
    time_diffs = np.diff(all_times)
    expected_diff = time_step * 3600
    assert np.allclose(
        time_diffs, expected_diff, atol=200.0
    )  # extra tolerance for floating point precision
