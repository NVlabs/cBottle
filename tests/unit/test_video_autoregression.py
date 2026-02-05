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
import pandas as pd
from unittest.mock import Mock

from cbottle.inference import VideoAutoregression
from cbottle.datasets.base import BatchInfo, TimeUnit


def make_model_and_dataset():
    """Create a minimal mock model and dataset for autoregression tests."""
    batch_info = BatchInfo(
        channels=["a", "b", "c"],
        time_step=3,
        time_unit=TimeUnit.HOUR,
    )
    nchannels = len(batch_info.channels)
    npix = 100

    model = Mock()
    model.time_length = 4
    model.coords = Mock()
    model.coords.batch_info = batch_info
    model.coords.npix = npix
    model.net = Mock()
    model.net.parameters.return_value = iter([torch.nn.Parameter(torch.randn(1))])

    def mock_sample(batch, **kwargs):
        target = batch["target"]
        coords = Mock()
        coords.batch_info = batch_info
        coords.npix = npix
        return target, coords, target.clone()

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

    return model, FakeDataset()


def test_video_autoregression_forward():
    model, dataset = make_model_and_dataset()
    batch_info = model.coords.batch_info
    nchannels = len(batch_info.channels)
    npix = model.coords.npix
    time_length = model.time_length
    time_step_hours = batch_info.time_step

    autoreg = VideoAutoregression(
        model=model,
        evaluation=model.sample,
        dataset=dataset,
    )

    # Initialize at the first time
    state, diags1 = autoreg.initialize(dataset.times[0])

    assert state.window_start_time == dataset.times[0]
    assert state.autoregression_start_time == dataset.times[0]
    assert diags1.frames.shape == (1, nchannels, time_length, npix)
    assert state.current_frames.shape == (1, nchannels, time_length, npix)
    assert diags1.conditioning_indices == []

    # Step forward with 1 conditioning frame (moves forward by (T-1) * Δt)
    state, diags2 = autoreg.step_forward(state, num_conditioning_frames=1)
    expected_delta_1 = (time_length - 1) * time_step_hours
    assert state.window_start_time == dataset.times[0] + pd.Timedelta(
        hours=expected_delta_1
    )
    assert diags2.frames.shape == (1, nchannels, time_length, npix)

    # Step forward with 2 conditioning frames (moves forward by (T-2) * Δt)
    state, diags3 = autoreg.step_forward(state, num_conditioning_frames=2)
    expected_delta_2 = (time_length - 2) * time_step_hours
    total_delta = expected_delta_1 + expected_delta_2
    assert state.window_start_time == dataset.times[0] + pd.Timedelta(hours=total_delta)
    assert diags3.frames.shape == (1, nchannels, time_length, npix)


def test_video_autoregression_backward():
    """Backward autoregression should move the window back in time while keeping the start time fixed."""
    model, dataset = make_model_and_dataset()
    batch_info = model.coords.batch_info
    nchannels = len(batch_info.channels)
    npix = model.coords.npix
    time_length = model.time_length
    time_step_hours = batch_info.time_step

    autoreg = VideoAutoregression(
        model=model,
        evaluation=model.sample,
        dataset=dataset,
    )

    start_index = 20
    start_time = dataset.times[start_index]

    state, diags1 = autoreg.initialize(start_time)

    assert state.autoregression_start_time == start_time
    assert state.window_start_time == start_time
    assert diags1.frames.shape == (1, nchannels, time_length, npix)

    # Step backward with 1 conditioning frame (moves back by (T-1) * Δt)
    state, diags2 = autoreg.step_backward(state, num_conditioning_frames=1)
    expected_delta = (time_length - 1) * time_step_hours
    assert state.autoregression_start_time == start_time
    assert state.window_start_time == start_time - pd.Timedelta(hours=expected_delta)
    assert diags2.frames.shape == (1, nchannels, time_length, npix)


def test_video_autoregression_with_conditioning():
    """Initialization should honor the requested conditioning frame indices."""
    model, dataset = make_model_and_dataset()
    batch_info = model.coords.batch_info
    nchannels = len(batch_info.channels)
    npix = model.coords.npix

    autoreg = VideoAutoregression(
        model=model,
        evaluation=model.sample,
        dataset=dataset,
    )

    batch = dataset[0]
    target = batch["target"]

    # Condition on frames 0 and 1
    frames = {0: target[:, :, 0, :], 1: target[:, :, 1, :]}

    state, diags = autoreg.initialize(dataset.times[0], frames=frames)

    assert state.current_frames.shape == (1, nchannels, model.time_length, npix)
    # conditioning_indices is just the data we passed in (no extra magic)
    assert sorted(diags.conditioning_indices) == [0, 1]
