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
"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from typing import List, Optional


def edm_sampler_steps(
    num_steps=18,
    sigma_min=0.002,
    sigma_max: float = 80.0,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
):
    outputs = []
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = t_cur + gamma * t_cur

        outputs.append(t_hat)
        outputs.append(t_next)
    return outputs


def edm_sampler(
    net,
    latents,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def edm_sampler_from_sigma(
    net,
    latents,
    randn_like=torch.randn_like,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    reverse=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(t_steps)

    if not reverse:
        t_steps = torch.cat(
            [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

    if reverse:
        t_steps = torch.flip(t_steps, [0])

    # Main sampling loop.
    x_next = latents.to(torch.float64)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def edm_sampler_from_sigma_euler(
    net,
    latents,
    randn_like=torch.randn_like,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    reverse=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(t_steps)

    if not reverse:
        t_steps = torch.cat(
            [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

    if reverse:
        t_steps = torch.flip(t_steps, [0])

    # Main sampling loop.
    x_next = latents.to(torch.float64)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next


def few_step_sampler(
    net,
    latents,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_mid: Optional[List[float]] = None,
):
    # Adjust noise levels based on what's supported by the network.
    # Time step discretization.
    sigma_mid = [] if sigma_mid is None else sigma_mid
    # t_0 = T, t_N = 0
    # Max noise level (adjust based on what's supported by the network)
    sigma_max = min(sigma_max, net.sigma_max)

    t_steps = torch.tensor(
        [sigma_max] + list(sigma_mid), dtype=latents.dtype, device=latents.device
    )
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    assert torch.all(t_steps[1:] <= t_steps[:-1])

    # Sampling steps
    latents = latents * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        latents = net(latents, t_cur).to(torch.float64)

        if t_next > 0:
            latents = latents + t_next * torch.randn_like(latents)

    return latents


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )
