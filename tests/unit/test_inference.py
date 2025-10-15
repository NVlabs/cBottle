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
from cbottle.inference import (
    CBottle3d,
    SuperResolutionModel,
    DistilledSuperResolutionModel,
    Coords,
)
from cbottle import models
from cbottle.datasets.base import BatchInfo
from cbottle.datasets.dataset_2d import MAX_CLASSES as LABEL_DIM
from cbottle.models.networks import Output
from cbottle.patchify import apply_on_patches


def create_cbottle3d(separate_classifier=None, sampler_fn="heun", channels_last=True):
    # Create a CBottle3d object with a simple network
    net = models.get_model(
        models.ModelConfigV1(model_channels=8, out_channels=3, label_dim=LABEL_DIM)
    )
    net.batch_info = BatchInfo(
        channels=["rlut", "rsut", "rsds"],
        scales=[1.0, 1.0, 1.0],
        center=[0.0, 0.0, 0.0],
    )
    net.cuda()
    return CBottle3d(
        net,
        sigma_min=0.02,
        sigma_max=200.0,
        num_steps=2,
        sampler_fn=sampler_fn,
        channels_last=channels_last,
        separate_classifier=separate_classifier,
    )


def create_super_resolution_model():
    # Create a SuperResolutionModel object with a simple network
    batch_info = BatchInfo(
        channels=["rlut", "rsut", "rsds"],
        scales=[1.0, 1.0, 1.0],
        center=[0.0, 0.0, 0.0],
    )
    out_channels = len(batch_info.scales)
    local_lr_channels = out_channels
    global_lr_channels = out_channels
    net = models.get_model(
        models.ModelConfigV1(
            "unet_hpx1024_patch",
            model_channels=8,
            out_channels=3,
            condition_channels=local_lr_channels + global_lr_channels,
            label_dim=LABEL_DIM,
        )
    )
    return SuperResolutionModel(
        net,
        batch_info,
        hpx_level=10,
        hpx_lr_level=6,
        patch_size=128,
        overlap_size=32,
        num_steps=2,
        sigma_max=800,
        device="cuda",
    )


def create_distilled_super_resolution_model():
    # Create a SuperResolutionModel object with a simple network
    batch_info = BatchInfo(
        channels=["rlut", "rsut", "rsds"],
        scales=[1.0, 1.0, 1.0],
        center=[0.0, 0.0, 0.0],
    )
    out_channels = len(batch_info.scales)
    local_lr_channels = out_channels
    global_lr_channels = out_channels
    net = models.get_model(
        models.ModelConfigV1(
            "unet_hpx1024_patch",
            model_channels=8,
            out_channels=3,
            condition_channels=local_lr_channels + global_lr_channels,
            label_dim=LABEL_DIM,
        )
    )
    return DistilledSuperResolutionModel(
        net,
        batch_info,
        hpx_level=10,
        hpx_lr_level=6,
        patch_size=128,
        overlap_size=32,
        sigma_max=800,
        window_function="KBD",
        window_alpha=1,
        device="cuda",
    )


def create_input_data(target_shape):
    """Helper function to create input data for tests."""
    b, c, t, x = target_shape
    return {
        "target": torch.randn(*target_shape).cuda(),
        "labels": torch.nn.functional.one_hot(
            torch.tensor([1]), num_classes=LABEL_DIM
        ).cuda(),
        "condition": torch.zeros(b, 0, t, x).cuda(),
        "second_of_day": torch.tensor([[43200]]).cuda(),
        "day_of_year": torch.tensor([[180]]).cuda(),
    }


def test_cbottle3d_infill():
    mock_cbottle3d = create_cbottle3d()
    # Test the infill method
    batch = create_input_data((1, 3, 1, 12 * 64 * 64))
    batch["target"][:, 0] = float("nan")
    output, coords = mock_cbottle3d.infill(batch)
    assert output is not None
    assert coords is not None


def test_cbottle3d_translate():
    mock_cbottle3d = create_cbottle3d()
    # Test the translate method
    batch = create_input_data((1, 3, 1, 12 * 64 * 64))
    output, coords = mock_cbottle3d.translate(batch, "icon")
    assert output is not None
    assert coords is not None


def test_super_resolution_model_call():
    model = create_super_resolution_model()
    # Test the __call__ method
    low_res_tensor = torch.randn(1, 3, 1, 12 * 64**2).cuda()
    coords = model.batch_info
    extents = (0, 5, 0, 5)
    coords = Coords(model.batch_info, model.low_res_grid)
    output, hr_coords = model(low_res_tensor, coords, extents)
    assert output is not None
    assert hr_coords is not None


def test_distilled_super_resolution_model_call():
    model = create_distilled_super_resolution_model()
    # Test the __call__ method
    low_res_tensor = torch.randn(1, 3, 1, 12 * 64**2).cuda()
    coords = model.batch_info
    extents = (0, 5, 0, 5)
    coords = Coords(model.batch_info, model.low_res_grid)
    output, hr_coords = model(low_res_tensor, coords, extents)
    assert output is not None and output.shape == (1, 3, 1, 12 * 1024**2)
    assert hr_coords is not None


class MockClassifier:
    def __init__(self):
        self.is_called = False

    def __call__(self, x_hat, *args, **kwargs):
        self.is_called = True
        # Use x_hat in the computation so gradients can flow through
        logits = (
            torch.ones(1, 1, 1, 12 * 8**2, requires_grad=True).cuda() * x_hat.mean()
        )
        return Output(out=None, logits=logits)


separate_classifier = MockClassifier()


@pytest.mark.parametrize("sampler_fn", ["heun", "euler"])
@pytest.mark.parametrize("channels_last", [True, False])
def test_cbottle3d_sample(sampler_fn, channels_last):
    mock_cbottle3d = create_cbottle3d(
        separate_classifier, sampler_fn=sampler_fn, channels_last=channels_last
    )
    # Test the sample method
    batch = create_input_data((1, 3, 1, 12 * 64 * 64))
    output, coords = mock_cbottle3d.sample(
        batch, guidance_pixels=torch.tensor([0]).cuda()
    )
    assert output is not None
    assert coords is not None
    assert separate_classifier.is_called


def test_apply_on_patches():
    model = create_super_resolution_model()
    x_hat = torch.randn(1, 3, 12 * 1024**2).cuda()
    x_lr = torch.randn(1, 3, 12 * 1024**2).cuda().to(torch.float64)
    t_hat = torch.tensor(0.3).cuda()
    global_lr = torch.rand(1, 3, 128, 128).cuda()

    out = apply_on_patches(
        denoise=model.net,
        patch_size=128,
        overlap_size=32,
        x_hat=x_hat,
        x_lr=x_lr,
        t_hat=t_hat,
        class_labels=None,
        batch_size=128,
        global_lr=global_lr,
        device="cuda",
    )

    assert out is not None and out.shape == (1, 3, 12 * 1024**2)


def test_apply_on_patches_window():
    model = create_distilled_super_resolution_model()
    x_hat = torch.randn(1, 3, 12 * 1024**2).cuda()
    x_lr = torch.randn(1, 3, 12 * 1024**2).cuda().to(torch.float64)
    t_hat = torch.tensor(0.3).cuda()
    global_lr = torch.rand(1, 3, 128, 128).cuda()

    window = model._get_window_function(
        patch_size=128,
        window_alpha=1,
        type="KBD",
        dtype=torch.float32,
        device="cuda",
    )
    window = window.reshape((1, 1, window.shape[0], window.shape[1]))

    out = apply_on_patches(
        denoise=model.net,
        patch_size=128,
        overlap_size=32,
        x_hat=x_hat,
        x_lr=x_lr,
        t_hat=t_hat,
        class_labels=None,
        batch_size=128,
        global_lr=global_lr,
        window=window,
        device="cuda",
    )

    assert out is not None and out.shape == (1, 3, 12 * 1024**2)
