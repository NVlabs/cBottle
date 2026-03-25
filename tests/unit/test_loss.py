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
from cbottle.models import networks
import cbottle.loss
import torch
from cbottle._distill_helper import DistillLoss


def test_edm_loss_with_classifier():
    n_im = 100
    n = 25
    image_shape = (1, 4, 1, n_im)
    logits_shape = (1, 1, 1, n)

    def mock_model(x, t) -> networks.Output:
        logits = torch.ones(logits_shape)
        return networks.Output(x, logits)

    loss_fn = cbottle.loss.EDMLoss()

    x = torch.zeros(image_shape)
    classifier_labels = torch.zeros(logits_shape)
    loss = loss_fn(mock_model, x, classifier_labels)
    assert loss.classification.sum() > 0
    assert loss.total.sum() > 0


def test_edm_loss_without_classifier():
    n_im = 100
    image_shape = (1, 4, 1, n_im)

    def mock_model(x, t) -> networks.Output:
        return networks.Output(0.9 * x)

    loss_fn = cbottle.loss.EDMLoss()

    x = torch.zeros(image_shape)
    loss = loss_fn(mock_model, x)
    assert loss.classification is None
    assert loss.total.sum() > 0


def test_distill_loss():
    img_clean = torch.zeros(1, 4, 8, 8)
    img_lr = torch.zeros(1, 4, 4, 4)
    pos_embed = torch.zeros(1, 4, 8, 8)

    def mock_net_single_train_step(data, iteration):
        loss_map = {"total_loss": torch.tensor(1.0)}
        output = {"pred": torch.zeros_like(data["real"])}
        return loss_map, output

    class MockNet:
        def __init__(self):
            self.single_train_step = mock_net_single_train_step

    loss_fn = DistillLoss(net=MockNet())
    total_loss, loss_map, output = loss_fn(
        net=MockNet(),
        img_clean=img_clean,
        img_lr=img_lr,
        pos_embed=pos_embed,
    )
    assert total_loss.sum() > 0
