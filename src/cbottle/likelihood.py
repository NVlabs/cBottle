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
from scipy.integrate import solve_ivp
from scipy.stats import norm


def log_prob(
    D,
    image,
    mask=None,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    divergence_samples: int = 32,
    rtol=0.05,
    randn_like=torch.randn_like,
):
    """Compute likelihood following Appendix D.2 of Song, et. al. (2020).

    Probability flow ODE for edm-style model is given by

        dx/dt = (x - D(x,t)) / t = f(x, t)

    The likelihood is given by

        log(p(x_0)) = log(p(x_T)) + int_0^T div(f(x,t))

    The divergence is approximated by

        div(f(x, t)) = E(e, grad f(x, t) e)
        e = N(0, I)

    Args:
        D: a denoiser function D(x, t)
        image:
        divergence_samples: number of samples to use in estimating div(f). This
            incurs now function evals, so use a relatively large default.

    Notes:


        Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole,
        B. (2020). Score-based generative modeling through stochastic differential
        equations. In arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2011.13456
    """
    # large divf confuses the timestepper so divide by a numerical factor
    numerical_factor = image[0].numel()
    batch_size = image.size(0)

    def f(t, y):
        y = torch.tensor(y).to(image.device)
        x = y[batch_size:].view(image.shape)
        if mask is not None:
            x = torch.where(mask, x, 0.0)

        x.requires_grad_(True)
        dx = (x - D(x, t)) / t

        if mask is not None:
            dx = torch.where(mask, dx, 0.0)

        divf = 0
        for i in range(divergence_samples):
            x.grad = None
            e = randn_like(image)
            gf = torch.sum(dx.clone() * e)
            gf.backward(retain_graph=True)
            dims = list(range(1, image.ndim))
            divf += (
                torch.sum(x.grad * e, dim=dims) / divergence_samples / numerical_factor
            )

        d = torch.cat([divf, dx.contiguous().flatten()]).detach().cpu().numpy()
        return d

    if mask is not None:
        image = torch.where(mask, image, 0.0)

    x0 = image.contiguous().flatten()
    v = torch.zeros(batch_size).to(x0)

    y0 = torch.cat([v, x0]).cpu().numpy()
    solution = solve_ivp(f, t_span=[sigma_min, sigma_max], y0=y0, rtol=rtol)
    xT = solution.y[batch_size:, -1]
    divf_int = torch.from_numpy(solution.y[:batch_size, -1] * numerical_factor).to(
        image.device
    )
    T = solution.t[-1]

    # compute log prob of final distribution
    logpT = (
        torch.from_numpy(norm.logpdf(xT, scale=T, loc=0))
        .to(image.device)
        .view(image.shape)
    )

    if mask is not None:
        logpT = torch.where(mask, logpT, 0)
    logpT = logpT.sum(dim=list(range(1, logpT.ndim)))
    log_prob = logpT + divf_int
    return log_prob, solution
