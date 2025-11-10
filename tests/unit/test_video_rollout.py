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


def test_video_rollout_forward(setup):
    model, dataset, nchannels, npix, time_step = setup
    time_length = model.time_length
    rollout = VideoRollout(model, dataset)

    state, diags1 = rollout.initialize(dataset.times[0])

    assert diags1.is_initialization
    assert diags1.frames.shape == (1, nchannels, time_length, npix)
    assert diags1.new_frames.shape == (1, nchannels, time_length, npix)
    assert len(diags1.frames_to_write_indices) == time_length

    assert state.rollout_start_time == state.window_start_time == dataset.times[0]
    assert state.current_frames.shape == (1, nchannels, time_length, npix)

    # Step forward with 1 conditioning frame (moves forward by 3 frames = 9 hours)
    state, diags2 = rollout.step_forward(state, num_conditioning_frames=1)

    assert not diags2.is_initialization
    assert diags2.frames.shape == (1, nchannels, time_length, npix)
    assert diags2.new_frames.shape == (1, nchannels, time_length - 1, npix)
    assert state.window_start_time == dataset.times[0] + pd.Timedelta(hours=9)
    assert len(diags2.frames_to_write_indices) == time_length - 1

    # Step forward with 2 conditioning frames (moves forward by 2 frames = 6 hours)
    state, diags3 = rollout.step_forward(state, num_conditioning_frames=2)

    assert not diags3.is_initialization
    assert diags3.new_frames.shape == (1, nchannels, time_length - 2, npix)
    assert state.window_start_time == dataset.times[0] + pd.Timedelta(hours=15)
    assert len(diags3.frames_to_write_indices) == time_length - 2

    # Check timestamps are contiguous
    times1, times2, times3 = (
        d.new_timestamps[0].cpu().numpy() for d in (diags1, diags2, diags3)
    )

    assert len(times1) == time_length
    assert len(times2) == time_length - 1
    assert len(times3) == time_length - 2

    all_times = np.concatenate([times1, times2, times3])
    time_diffs = np.diff(all_times)
    expected_diff = time_step * 3600
    assert np.allclose(time_diffs, expected_diff, atol=200.0)

    # Check lead times
    lead1, lead2, lead3 = (
        d.new_lead_time_hours[0].cpu().numpy() for d in (diags1, diags2, diags3)
    )

    assert np.allclose(lead1, [0, 3, 6, 9])
    assert np.allclose(lead2, [12, 15, 18])  # only new frames are counted
    assert np.allclose(lead3, [21, 24])


def test_video_rollout_backward(setup):
    """Test backward rollout produces negative lead times."""
    model, dataset, nchannels, npix, time_step = setup
    time_length = model.time_length
    rollout = VideoRollout(model, dataset)

    start_index = 20
    state, diags1 = rollout.initialize(dataset.times[start_index])
    assert (
        state.rollout_start_time
        == state.window_start_time
        == dataset.times[start_index]
    )

    # Step backward (moves back by 3 frames = 9 hours)
    state, diags2 = rollout.step_backward(state, num_conditioning_frames=1)

    assert not diags2.is_initialization
    assert state.rollout_start_time == dataset.times[start_index]
    assert state.window_start_time == dataset.times[start_index] - pd.Timedelta(hours=9)
    assert len(diags2.frames_to_write_indices) == time_length - 1

    lead2 = diags2.new_lead_time_hours[0].cpu().numpy()
    assert np.allclose(lead2, [-9, -6, -3])  # frames -3,-2,-1


def test_video_rollout_with_conditioning(setup):
    """Test rollout with initial conditioning frames."""
    model, dataset, nchannels, npix, time_step = setup
    time_length = model.time_length
    rollout = VideoRollout(model, dataset)

    # Initialize with conditioning on frames 0 and 1
    batch = dataset[0]
    target = batch["target"]
    frames = {0: target[:, :, 0, :], 1: target[:, :, 1, :]}

    _, diags = rollout.initialize(dataset.times[0], frames=frames)

    assert diags.is_initialization
    assert len(diags.frames_to_write_indices) == time_length

    assert diags.frame_source[0] == 0
    assert diags.frame_source[1] == 0
    assert diags.frame_source[2] == 1
    assert diags.frame_source[3] == 1
