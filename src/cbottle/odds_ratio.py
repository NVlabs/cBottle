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
"""
Classifier-guided log-odds-ratio helpers for :class:`CBottle3d`.

Provides the divergence tracker, classifier guidance strategy, and custom
EDM schedule used by :meth:`CBottle3d.calculate_odds_ratio`. The module
computes odds-ratio *ingredients* (per-phase Hutchinson divergence integrals,
Gaussian reference logps), not a standalone log-likelihood; for the latter
use :mod:`cbottle.likelihood`.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


__all__ = [
    "DivergenceTracker",
    "OddsRatioResult",
    "classifier_guidance",
    "calculate_divergence_integral",
    "calculate_gaussian_logp",
    "default_sigma_schedule",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OddsRatioResult:
    """Return type of :meth:`CBottle3d.calculate_odds_ratio`.

    Forward-phase fields are always populated. Backward-phase fields are
    ``None`` when ``run_backward=False``. The :attr:`log_odds_ratio` property
    combines the backward fields into ``log p_unguided / p_guided`` at the
    target sample.
    """

    # Forward phase
    forward_guidance_div_integral: float
    forward_score_div_integral: float
    forward_latents: torch.Tensor
    initial_log_prob: float | None

    # Backward-with-guidance phase
    backward_guidance_div_integral: float | None = None
    backward_score_div_integral: float | None = None
    backward_gaussian_logp: float | None = None
    backward_latents: torch.Tensor | None = None

    # Backward-without-guidance phase
    backward_no_guidance_guidance_div_integral: float | None = None
    backward_no_guidance_score_div_integral: float | None = None
    backward_no_guidance_gaussian_logp: float | None = None
    backward_no_guidance_latents: torch.Tensor | None = None

    @property
    def log_odds_ratio(self) -> float:
        """``log p_unguided / p_guided`` at the target sample.

        Requires the two backward phases (``run_backward=True``).
        """
        if (
            self.backward_gaussian_logp is None
            or self.backward_no_guidance_gaussian_logp is None
            or self.backward_score_div_integral is None
            or self.backward_no_guidance_score_div_integral is None
            or self.backward_guidance_div_integral is None
        ):
            raise ValueError(
                "log_odds_ratio requires both backward phases; "
                "call calculate_odds_ratio with run_backward=True."
            )
        return (
            (self.backward_no_guidance_gaussian_logp - self.backward_gaussian_logp)
            + (
                self.backward_no_guidance_score_div_integral
                - self.backward_score_div_integral
            )
            - self.backward_guidance_div_integral
        )


# ---------------------------------------------------------------------------
# Divergence computation (Hutchinson trace estimator)
# ---------------------------------------------------------------------------


def _hutchinson_divergence(
    x_hat: torch.Tensor,
    vector: torch.Tensor,
    t_hat: torch.Tensor,
    num_samples: int = 3,
) -> float:
    """Estimate ``div_x (vector)`` via the Hutchinson trace trick.

    Uses ``num_samples`` Gaussian probes with deterministic seeding (a function
    of ``t_hat``) for reproducibility across calls at the same sigma. Only
    ``batch_size = 1`` is supported -- larger batches OOM through the
    second-order autograd path in practice, so the caller (typically
    :meth:`CBottle3d._calculate_odds_ratio_full`) enforces this upstream.

    Args:
        x_hat: Input tensor with ``shape[0] == 1`` and ``requires_grad=True``.
        vector: Vector field of the same shape as ``x_hat``.
        t_hat: Current noise level; used only for seeding.
        num_samples: Monte Carlo probes to average over.
    """
    if torch.norm(vector) == 0:
        return 0.0

    divergence_val = torch.zeros(x_hat.shape[0], device=x_hat.device)
    dims = list(range(1, x_hat.ndim))

    for i in range(num_samples):
        seed = int(t_hat * 1000) + i
        generator = torch.Generator(device=x_hat.device).manual_seed(seed)

        e = torch.randn(
            x_hat.shape,
            device=x_hat.device,
            dtype=x_hat.dtype,
            generator=generator,
        )
        gf = torch.sum(vector * e)
        gf.backward(retain_graph=True)

        if x_hat.grad is not None:
            div_sample = torch.sum(x_hat.grad * e, dim=dims)
            divergence_val += div_sample / num_samples
            x_hat.grad = None
            gf = None
        else:
            break

    return float(divergence_val.item())


# ---------------------------------------------------------------------------
# Classifier-based guidance gradient
# ---------------------------------------------------------------------------


def classifier_guidance(
    guidance_data: torch.Tensor, logits: torch.Tensor, x_hat: torch.Tensor
) -> torch.Tensor:
    """Raw (unscaled) BCE-classifier guidance gradient ``-dL/dx_hat``.

    The caller (typically :class:`DivergenceTracker`) applies the sigma
    schedule and ``guidance_scale``. Returns a zero tensor if the guidance
    mask is empty.
    """
    guidance_mask = ~torch.isnan(guidance_data)
    valid_logits = logits[guidance_mask]
    valid_targets = guidance_data[guidance_mask]

    if valid_logits.numel() == 0:
        return torch.zeros_like(x_hat)

    x_hat.requires_grad_(True)
    x_hat.grad = None

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        valid_logits, valid_targets, reduction="sum"
    )
    classifier_grad = torch.autograd.grad(loss, x_hat, create_graph=True)[0]
    # Negative gradient because we minimise the classifier loss.
    return -classifier_grad


# ---------------------------------------------------------------------------
# Divergence tracker (used as ``guidance_fn`` hook)
# ---------------------------------------------------------------------------


def default_sigma_schedule(
    t_hat, guidance_on: float = 0.0, guidance_off: float = float("inf")
):
    """Default classifier-guidance sigma schedule: linear-in-sigma in [on, off]."""
    return t_hat if guidance_on < t_hat < guidance_off else 0.0


class DivergenceTracker:
    """Callable ``guidance_fn`` that records per-step divergence traces.

    Installed via ``_sample_with_latents(..., guidance_fn=tracker)``. Pass
    ``guidance_fn=None`` for no-guidance mode: the hook still fires so score
    divergence is tracked, but the returned guidance contribution is zero.
    The ``"forward"`` phase also records ``initial_log_prob`` at step 0.
    """

    def __init__(
        self,
        guidance_fn: Callable | None = None,
        *,
        guidance_scale: float = 1.0,
        sigma_max: float = 200.0,
        divergence_samples: int = 1,
        phase: str = "forward",
        compute_score_div: bool = True,
        compute_guidance_div: bool = True,
        sigma_schedule: Callable[[torch.Tensor], float] = default_sigma_schedule,
    ):
        self.guidance_fn = guidance_fn
        self.guidance_scale = guidance_scale
        self.sigma_max = sigma_max
        self.divergence_samples = divergence_samples
        self.phase = phase
        self.compute_score_div = compute_score_div
        self.compute_guidance_div = compute_guidance_div
        self.sigma_schedule = sigma_schedule

        self.data: list[dict] = []
        self.initial_log_prob: float | None = None

    # -- guidance hook ----------------------------------------------------
    def __call__(
        self,
        guidance_data: torch.Tensor | None,
        logits: torch.Tensor | None,
        x_hat: torch.Tensor,
        denoised: torch.Tensor,
        t_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Record per-step divergence trace and return scaled guidance.

        Shapes (with ``B`` = batch, ``C`` = ``net.img_channels``, ``T`` =
        ``net.time_length``, ``X`` = number of HPX pixels at the model
        resolution, ``X_c`` = number of pixels at the classifier resolution):

        - ``guidance_data``: ``(B, 1, 1, X_c)`` (``NaN`` outside guidance
          mask) or ``None`` in no-guidance mode.
        - ``logits``: ``(B, 1, 1, X_c)`` from the classifier head, or
          ``None`` in no-guidance mode.
        - ``x_hat``: ``(B, C, T, X)``, ``requires_grad=True``.
        - ``denoised``: ``(B, C, T, X)``.
        - ``t_hat``: scalar tensor (current sigma).

        Returns ``(B, C, T, X)`` -- the sigma-schedule-scaled guidance vector,
        detached from the backward graph.
        """
        if x_hat.grad is not None:
            x_hat.grad = None

        # Record the initial (step 0) log p(x | N(0, sigma_max^2)) as a scalar
        # on any forward variant ("forward" or "forward_no_guidance") so
        # downstream consumers can combine with the backward logps.
        if self.initial_log_prob is None and self.phase.startswith("forward"):
            self.initial_log_prob = calculate_gaussian_logp(
                x_hat.detach(), self.sigma_max
            )

        sigma_weight = self.sigma_schedule(t_hat)

        # Compute RAW guidance. If guidance_fn is None run in no-guidance
        # mode: zeros are contributed and guidance divergence is skipped
        # (score divergence is still tracked below).
        if self.guidance_fn is not None:
            raw_guidance = self.guidance_fn(guidance_data, logits, x_hat)
            scaled_guidance = sigma_weight * raw_guidance
            scaled_norm = float(torch.norm(raw_guidance).item() * sigma_weight)
        else:
            scaled_guidance = torch.zeros_like(x_hat)
            scaled_norm = 0.0

        # Score
        score = (x_hat - denoised) / t_hat
        score_norm_val = float(torch.norm(score).item())

        # Divergences
        divergence_val = 0.0
        score_div_val = 0.0

        if (
            self.compute_guidance_div
            and self.guidance_fn is not None
            and torch.norm(scaled_guidance) > 0
            and sigma_weight != 0
        ):
            # _sample_with_latents adds ``guidance_scale * d_guide`` to ``d``.
            # ``scaled_guidance`` is already pre-multiplied by the sigma
            # schedule, so dividing by t_hat gives the per-step contribution
            # to the drift whose divergence we want.
            full_guidance = self.guidance_scale / t_hat * scaled_guidance
            divergence_val = -_hutchinson_divergence(
                x_hat, full_guidance, t_hat, self.divergence_samples
            )

        if self.compute_score_div:
            score_div_val = _hutchinson_divergence(
                x_hat, score.clone(), t_hat, self.divergence_samples
            )

        sigma_val = float(t_hat) if hasattr(t_hat, "item") else float(t_hat)
        self.data.append(
            {
                "sigma": sigma_val,
                "score_norm": score_norm_val,
                "classifier_norm": scaled_norm,
                "divergence": float(divergence_val),
                "score_divergence": float(score_div_val),
                "phase": self.phase,
            }
        )

        # Detach before returning: the divergence has already been evaluated,
        # so the autograd graph on ``scaled_guidance`` is no longer needed.
        scaled_guidance_detached = scaled_guidance.detach()
        if x_hat.grad is not None:
            x_hat.grad = None
        return scaled_guidance_detached


# ---------------------------------------------------------------------------
# Likelihood reductions
# ---------------------------------------------------------------------------


def calculate_divergence_integral(
    divergence_data: Sequence[dict],
    divergence_key: str = "divergence",
    phase: str = "forward",
) -> float:
    """Trapezoidal integration of a per-step divergence trace over sigma.

    For forward sampling the integration goes high -> low sigma, so the sign
    is flipped; for backward we integrate low -> high and keep the sign.
    """
    if not divergence_data:
        return 0.0

    sigma = [data["sigma"] for data in divergence_data]
    divergence = [data[divergence_key] for data in divergence_data]
    divergence = [0.0 if np.isnan(d) else d for d in divergence]

    try:
        if phase == "backward":
            return np.trapezoid(divergence, sigma)
        return -np.trapezoid(divergence, sigma)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Error integrating {divergence_key}: {e}")
        return 0.0


def calculate_gaussian_logp(sample: torch.Tensor, sigma: float) -> float:
    """Log probability of ``sample`` under ``N(0, sigma^2 I)``.

    Torch-native (no scipy dependency).
    """
    d = sample.numel()
    log_2pi = math.log(2.0 * math.pi)
    log_sigma_sq = 2.0 * math.log(sigma)
    norm_sq = torch.sum(sample**2).item()
    logp = -0.5 * (d * log_2pi + d * log_sigma_sq + norm_sq / (sigma**2))
    return float(logp)


# ---------------------------------------------------------------------------
# Custom EDM sampler / schedule (used by CBottle3d.calculate_odds_ratio)
# ---------------------------------------------------------------------------


def _create_custom_time_steps(
    net,
    device,
    *,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 200.0,
    rho: float = 7.0,
    extra_steps_intervals: Sequence[tuple[float, float, int]] = (),
) -> torch.Tensor:
    """Build the EDM sigma schedule used by the likelihood sampler.

    Mirrors the schedule used by the original discrete likelihood sampler:
    standard EDM polynomial discretisation, optionally densified inside one
    or more ``(start_sigma, end_sigma, num_extra)`` intervals.
    """
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = net.round_sigma(t_steps)

    if extra_steps_intervals:
        extra_steps: list[float] = []
        for start_sigma, end_sigma, num_extra in extra_steps_intervals:
            extra_indices = torch.linspace(0, 1, num_extra, device=device)
            extra_sigmas = start_sigma + extra_indices * (end_sigma - start_sigma)
            extra_sigmas = net.round_sigma(extra_sigmas)
            extra_steps.extend(extra_sigmas.tolist())

        all_steps = torch.cat(
            [t_steps, torch.tensor(extra_steps, dtype=torch.float64, device=device)]
        )
        t_steps = torch.sort(all_steps, descending=True)[0]

    return t_steps


def _edm_sampler_with_custom_steps(
    net,
    latents,
    t_steps,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 0,
    randn_like=torch.randn_like,
    progress_wrapper: Callable[[Iterable], Iterable] | None = None,
    # The following are accepted and ignored to satisfy the calling convention
    # used by ``_sample_with_latents``; the schedule is fully specified by
    # ``t_steps`` so they are redundant here. Explicit rather than **kwargs to
    # prevent silent typo-swallowing.
    sigma_min=None,
    sigma_max=None,
    num_steps=None,
    time_stepper=None,
):
    """Deterministic EDM sampler honoring an externally provided sigma schedule.
    See Karras et al. 2022 for comparing this sampler (Euler) to EDM (2nd order/Heun).
    If ``progress_wrapper`` is set, the per-step iterable is passed through it
    once before iteration -- typically a tqdm wrapper for live progress, but
    callers can substitute any iterable-to-iterable transform.
    """
    x_next = latents.to(torch.float64)
    steps: Iterable[tuple[torch.Tensor, torch.Tensor]] = list(
        zip(t_steps[:-1], t_steps[1:])
    )
    if progress_wrapper is not None:
        steps = progress_wrapper(steps)

    for t_cur, t_next in steps:
        x_cur = x_next
        gamma = (
            min(S_churn / len(t_steps), np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max
            else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        denoised = net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next


def _make_reverse_aware_sampler(
    reverse: bool,
    start_latents: torch.Tensor | None = None,
    *,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 200.0,
    rho: float = 7.0,
    extra_steps_intervals: Sequence[tuple[float, float, int]] = (),
    progress_wrapper: Callable[[Iterable], Iterable] | None = None,
):
    """Build a sampler closure compatible with ``_sample_with_latents``.

    For ``reverse=False`` the schedule runs high -> low sigma and the raw
    integration endpoint ``sigma=0`` is appended; for ``reverse=True`` the
    schedule is flipped (low -> high sigma) with the tail restored to the
    *raw* ``sigma_max`` (before round_sigma) so the downstream
    ``log p(x | N(0, sigma_max^2))`` uses exactly the caller's sigma_max.
    """
    # Alias the outer schedule params so the inner closure can name its own
    # (shadowed) kwargs with the same names as in the ``_sample_with_latents``
    # calling convention without accidentally referring to them.
    _num_steps = num_steps
    _sigma_min = sigma_min
    _sigma_max = sigma_max
    _rho = rho
    _extra_steps_intervals = extra_steps_intervals
    _progress_wrapper = progress_wrapper

    def _sampler(
        net,
        latents,
        *,
        randn_like=torch.randn_like,
        # Accepted and ignored (see _edm_sampler_with_custom_steps above).
        sigma_min=None,
        sigma_max=None,
        num_steps=None,
        time_stepper=None,
    ):
        device = latents.device
        t_steps = _create_custom_time_steps(
            net,
            device,
            num_steps=_num_steps,
            sigma_min=_sigma_min,
            sigma_max=_sigma_max,
            rho=_rho,
            extra_steps_intervals=_extra_steps_intervals,
        )

        if reverse and start_latents is not None:
            latents = start_latents.clone()

        if not reverse:
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        else:
            t_steps = torch.flip(t_steps, [0]).clone()
            t_steps[-1] = _sigma_max

        return _edm_sampler_with_custom_steps(
            net,
            latents,
            t_steps,
            randn_like=randn_like,
            progress_wrapper=_progress_wrapper,
        )

    return _sampler
