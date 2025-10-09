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


def create_cbottle3d(separate_classifier=None):
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
        device="cuda",
    )


def create_input_data(target_shape):
    """Helper function to create input data for tests."""
    b, c, t, x = target_shape
    return {
        "target": torch.randn(*target_shape).cuda(),
        "labels": torch.nn.functional.one_hot(torch.tensor([1]), num_classes=LABEL_DIM).cuda(),
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


def test_cbottle3d_sample():
    mock_cbottle3d = create_cbottle3d(separate_classifier)
    # Test the sample method
    batch = create_input_data((1, 3, 1, 12 * 64 * 64))
    output, coords = mock_cbottle3d.sample(
        batch, guidance_pixels=torch.tensor([0]).cuda()
    )
    assert output is not None
    assert coords is not None
    assert separate_classifier.is_called
