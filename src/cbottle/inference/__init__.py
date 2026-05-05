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
Inference utilities for cBottle models.

This module provides high-level interfaces for running inference with trained cBottle models.
"""

import os

import numpy as np
import torch
import earth2grid
from earth2grid import healpix
from tqdm.auto import tqdm
from typing import Callable, Literal, Sequence
import dataclasses
import logging
from scipy.signal.windows import kaiser_bessel_derived

import cbottle.denoiser_factories
from cbottle import checkpointing, patchify
from cbottle.diffusion_samplers import (
    edm_sampler,
    edm_sampler_from_sigma,
    edm_sampler_steps,
    few_step_sampler,
    StackedRandomGenerator,
)
from cbottle.datasets import base
from cbottle.odds_ratio import (
    DivergenceTracker,
    OddsRatioResult,
    _make_reverse_aware_sampler,
    calculate_divergence_integral,
    calculate_gaussian_logp,
    classifier_guidance,
    default_sigma_schedule,
)

from cbottle.datasets.dataset_2d import HealpixDatasetV5, LABELS
from cbottle.config import environment
from ._video_autoregression import (
    VideoAutoregression,
    VideoAutoregressionState,
    AutoregressionDiagnostics,
)

__all__ = [
    "CBottle3d",
    "Coords",
    "VideoAutoregression",
    "VideoAutoregressionState",
    "AutoregressionDiagnostics",
    "SuperResolutionModel",
    "DistilledSuperResolutionModel",
    "MixtureOfExpertsDenoiser",
]


@dataclasses.dataclass
class Coords:
    batch_info: base.BatchInfo
    grid: earth2grid.base.Grid  # applied to final dimension


# Build labels for infilling (use "icon" as default)
def _build_labels(labels, denoiser_when_nan: str):
    out_labels = torch.zeros_like(labels)
    if denoiser_when_nan == "icon":
        out_labels[:, HealpixDatasetV5.LABEL] = 1
    return out_labels


class CBottle3d:
    """
    A callable object that provides both infilling and ERA5-to-ICON translation using diffusion models.

    This class combines the functionality of infilling missing values (NaNs) and translating
    ERA5 data to ICON-like data using a single loaded model.
    """

    classifier_grid = earth2grid.healpix.Grid(
        3, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
    )
    output_grid = earth2grid.healpix.Grid(
        6, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )

    def __init__(
        self,
        net: "MixtureOfExpertsDenoiser",
        separate_classifier: torch.nn.Module | None = None,
        *,
        sigma_min: float = 0.02,
        sigma_max: float = 200.0,
        num_steps: int = 18,
        time_stepper: Literal["heun", "euler"] = "heun",
        channels_last: bool = True,
        torch_compile: bool = False,
        device: str | None = None,
    ):
        """
        Initialize the CBottle3d model.

        Args:
            net: The denoiser network (MixtureOfExpertsDenoiser)
            separate_classifier: Optional separate classifier for guidance
            sigma_min: Minimum noise sigma for diffusion
            sigma_max: Maximum noise sigma for diffusion
            num_steps: Number of sampling steps
            time_stepper: Which time stepper to use (heun, euler)
            channels_last: Whether to convert input and model to channels_last
            torch_compile: Whether to compile the model with torch.compile
            device: Device to move models to. If None (default), auto-detects
                "cuda" when available else "cpu".
        """
        self.net = net
        self.separate_classifier = separate_classifier
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.time_stepper = time_stepper
        self.channels_last = channels_last
        self._move_models_to_device(device)
        self._convert_model_NHWC()
        if torch_compile:
            self.torch_compile()

    def torch_compile(self):
        if isinstance(self.net, MixtureOfExpertsDenoiser):
            for i, expert in enumerate(self.net.experts):
                self.net.experts[i] = torch.compile(expert, fullgraph=True)
        else:
            self.net = torch.compile(self.net, fullgraph=True)

    def _convert_model_NHWC(self):
        if self.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)

    def _move_models_to_device(self, device: str | None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = self.net.to(device)
        if self.separate_classifier is not None:
            self.separate_classifier = self.separate_classifier.to(device)

    @classmethod
    def from_pretrained(
        cls,
        path: str | list[str],
        separate_classifier_path: str | None = None,
        sigma_thresholds: tuple[float, ...] = (),
        allow_second_order_derivatives: bool = False,
        map_location: str = "cpu",
        **kwargs,
    ) -> "CBottle3d":
        net = MixtureOfExpertsDenoiser.from_pretrained(
            path,
            sigma_thresholds,
            allow_second_order_derivatives=allow_second_order_derivatives,
        )

        separate_classifier = None
        if separate_classifier_path is not None:
            logging.info(f"Opening additional classifier at {separate_classifier_path}")
            with checkpointing.Checkpoint(separate_classifier_path) as c:
                separate_classifier = c.read_model(
                    map_location=map_location,
                    allow_second_order_derivatives=allow_second_order_derivatives,
                ).eval()

        return cls(net, separate_classifier=separate_classifier, **kwargs)

    @property
    def batch_info(self) -> base.BatchInfo:
        return self.net.batch_info

    def _post_process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Post-process the output by reordering to NEST convention and denormalizing.

        Args:
            x: Input tensor

        Returns:
            Post-processed tensor
        """
        info = self.batch_info
        # Fix grid access - get the grid from the domain properly
        grid_obj = getattr(self.net.domain, "_grid", None)
        if grid_obj is None:
            # Fallback: try to get grid from domain directly
            grid_obj = self.net.domain

        x = grid_obj.reorder(self.output_grid.pixel_order, x)
        if info.scales is not None and info.center is not None:
            scales = info.scales
            center = info.center
            x = torch.tensor(scales)[:, None, None].to(x) * x + torch.tensor(center)[
                :, None, None
            ].to(x)
        return x

    def _move_to_device(self, batch: dict) -> dict:
        """
        Move all batch tensors to the correct device.

        Args:
            batch: Input batch dictionary

        Returns:
            Batch with tensors moved to device
        """
        device = next(self.net.parameters()).device
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    @property
    def device(self):
        device = next(self.net.parameters()).device
        return device

    def infill(
        self, batch: dict, bf16: bool = True, return_untransformed: bool = False
    ) -> tuple[torch.Tensor, Coords]:
        """
        Perform infilling on batch with NaN values.
        Args:
            batch: Input batch containing data with NaN values

        Returns:
            Tuple of ((original_data, infilled_data), coords)
        """
        # Move all tensors to the correct device
        batch = self._move_to_device(batch)
        images = batch["target"]
        labels = batch["labels"]
        condition = batch["condition"]
        second_of_day = batch["second_of_day"]
        day_of_year = batch["day_of_year"]

        if self.channels_last:
            condition = condition.to(memory_format=torch.channels_last)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)

        # Post-process the original images for return
        original_processed = self._post_process(images)

        # Check if there are any NaN values
        if not torch.isnan(images).any():
            # No NaN values, return input as-is for both
            grid_obj = getattr(self.net.domain, "_grid", self.net.domain)
            return (original_processed, original_processed), Coords(
                self.batch_info, grid_obj
            )

        labels = _build_labels(labels, "icon")

        # Create infilling denoiser directly
        mask = ~torch.isnan(images)
        tsteps = edm_sampler_steps(
            sigma_max=self.sigma_max, num_steps=self.num_steps, sigma_min=self.sigma_min
        )
        tsteps = torch.unique_consecutive(torch.tensor(tsteps, device=images.device))
        tsteps = tsteps.flip(0)
        dt = torch.diff(tsteps).float()
        dW = torch.randn(
            [len(dt), *images.shape], dtype=images.dtype, device=images.device
        ) * dt.view(-1, 1, 1, 1, 1)
        W = torch.cumsum(dW, dim=0)
        y = images + W

        def D(x_hat, t_hat):
            ts = tsteps[1:]
            # TODO could use linear interp instead
            i = torch.searchsorted(ts, t_hat)
            x = torch.where(mask, y[i], x_hat)
            D_out = self.net(
                x,
                t_hat,
                labels,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
            ).out
            return D_out

        D.round_sigma = self.net.round_sigma
        D.sigma_max = self.net.sigma_max
        D.sigma_min = self.net.sigma_min

        # Generate random latents
        latents = torch.randn_like(images) * self.sigma_max

        # Run infilling sampling
        with torch.no_grad():
            with torch.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
                out = edm_sampler_from_sigma(
                    D,
                    latents,
                    randn_like=torch.randn_like,
                    sigma_max=self.sigma_max,
                    sigma_min=self.sigma_min,
                    num_steps=self.num_steps,
                )

        # Post-process the output
        if return_untransformed:
            raw = out
            processed = self._post_process(out)
            return processed, Coords(self.batch_info, self.output_grid), raw
        else:
            out = self._post_process(out)
            return out, Coords(self.batch_info, self.output_grid)

    def _encode(self, batch: dict, bf16: bool = True) -> dict:
        batch = self._move_to_device(batch)

        images, labels, condition = batch["target"], batch["labels"], batch["condition"]
        second_of_day = batch["second_of_day"].float()
        day_of_year = batch["day_of_year"].float()

        if self.channels_last:
            condition = condition.to(memory_format=torch.channels_last)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)

        mask = ~torch.isnan(images)

        y0 = images + torch.randn_like(images) * self.sigma_min

        def D(x_hat, t_hat):
            return self.net(
                x_hat.where(mask, 0),
                t_hat,
                class_labels=labels,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
            ).out

        D.sigma_min = self.sigma_min
        D.sigma_max = self.sigma_max
        D.round_sigma = lambda x: x
        with torch.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
            encoded = edm_sampler_from_sigma(
                D,
                y0,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                num_steps=self.num_steps,
                randn_like=torch.randn_like,
                reverse=True,
                S_noise=0,
            )

        # add noise for missing channels
        encoded = encoded.where(mask, torch.randn_like(encoded) * self.sigma_max)

        output = batch.copy()
        output["encoded"] = encoded
        return output

    def _decode(
        self,
        batch: dict,
        dataset: Literal["era5", "icon"],
        dataset_when_nan: str = "icon",
        bf16: bool = True,
    ) -> dict:
        batch = self._move_to_device(batch)

        images = batch["encoded"]
        condition = batch["condition"]
        second_of_day = batch["second_of_day"].float()
        day_of_year = batch["day_of_year"].float()

        if self.channels_last:
            condition = condition.to(memory_format=torch.channels_last)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)

        labels_when_nan = torch.zeros_like(batch["labels"].to(images.device))
        labels_when_nan[:, LABELS.index(dataset_when_nan)] = 1.0
        labels = torch.nn.functional.one_hot(
            torch.tensor([LABELS.index(dataset)], device=condition.device), 1024
        )

        if dataset == "era5":
            denoiser = cbottle.denoiser_factories.get_denoiser(
                net=self.net,
                images=images,
                labels=labels,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
                denoiser_type=cbottle.denoiser_factories.DenoiserType.mask_filling,
                labels_when_nan=labels_when_nan,
            )
        elif dataset == "icon":
            denoiser = cbottle.denoiser_factories.get_denoiser(
                net=self.net,
                images=images,
                labels=labels,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
                denoiser_type=cbottle.denoiser_factories.DenoiserType.standard,
                labels_when_nan=labels_when_nan,
            )
        else:
            raise ValueError(dataset)

        output = batch.copy()
        with torch.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
            output["target"] = edm_sampler_from_sigma(
                denoiser,
                images,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                randn_like=torch.randn_like,
                num_steps=self.num_steps,
                S_noise=0,
            )
        return output

    def to_icon(self, batch: dict) -> tuple[torch.Tensor, Coords]:
        """
        Translate ERA5 batch to ICON-like data.

        Args:
            batch: Input batch containing ERA5 data

        Returns:
            Tuple of ((era5_processed, icon_like), coords)
        """
        return self.translate(batch, "icon")

    def translate(
        self,
        batch: dict,
        dataset: Literal["icon", "era5"],
        bf16: bool = True,
        return_untransformed: bool = False,
    ) -> tuple[torch.Tensor, Coords]:
        # Move all tensors to the correct device
        with torch.no_grad():
            encoded = self._encode(batch, bf16=bf16)
            out = self._decode(encoded, dataset, bf16=bf16)["target"]
        if return_untransformed:
            raw = out
            processed = self._post_process(out)
            return processed, Coords(self.batch_info, self.output_grid), raw
        else:
            out = self._post_process(out)
            return out, Coords(self.batch_info, self.output_grid)

    def denormalize(self, batch: dict) -> tuple[torch.Tensor, Coords]:
        # Move all tensors to the correct device
        out = batch["target"]
        out = self._post_process(out)
        out = out.to(self.device)
        return out, Coords(self.batch_info, self.output_grid)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpost-process the output by normalizing.
        """
        info = self.batch_info
        if info.scales is not None and info.center is not None:
            scales = info.scales
            center = info.center
            x = (x - torch.tensor(center)[:, None, None].to(x)) / torch.tensor(scales)[
                :, None, None
            ].to(x)
        return x

    def _reorder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpost-process the output by reordering to HPXPAD convention.
        """
        grid_obj = getattr(self.net.domain, "_grid", None)
        if grid_obj is None:
            # Fallback: try to get grid from domain directly
            grid_obj = self.net.domain
        x = self.output_grid.reorder(grid_obj.pixel_order, x)
        return x

    @property
    def coords(self) -> Coords:
        return Coords(self.batch_info, self.output_grid)

    @property
    def icon_mask(self):
        return torch.ones(len(self.batch_info.channels), dtype=torch.bool)

    @property
    def era5_mask(self):
        masked_vars = ["rlut", "rsut", "rsds"]
        return torch.tensor([c not in masked_vars for c in self.batch_info.channels])

    @property
    def time_length(self):
        return self.net.time_length

    def sample(
        self,
        batch,
        seed: int | None = None,
        start_from_noisy_image: bool = False,
        guidance_pixels: torch.Tensor | None = None,
        guidance_scale: float = 0.03,
        bf16=True,
        spatial_mask: torch.Tensor | None = None,
        return_untransformed: bool = False,
    ):
        """
        Args:

            guidance_pixels: Either the pixel index of ``self.input_grid``` where the
                TCs are desired. 0<= guidance_pixels < 12 * nside ^2. Or the enitre HPX
                tensor already set. If None, no guidance used.
            guidance_scale: float = 0.03
            return_untransformed: If True, also returns the un-post-processed data (normalized and model specific grid)

        """
        return self._sample_with_latents(
            batch=batch,
            seed=seed,
            start_from_noisy_image=start_from_noisy_image,
            guidance_pixels=guidance_pixels,
            guidance_scale=guidance_scale,
            bf16=bf16,
            spatial_mask=spatial_mask,
            pre_generated_latents=None,
            return_untransformed=return_untransformed,
        )

    def _sample_with_latents(
        self,
        batch,
        seed: int | None = None,
        start_from_noisy_image: bool = False,
        guidance_pixels: torch.Tensor | None = None,
        guidance_scale: float = 0.03,
        bf16=True,
        spatial_mask: torch.Tensor | None = None,
        pre_generated_latents: torch.Tensor | None = None,
        return_untransformed: bool = False,
        guidance_fn: Callable[
            [
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            torch.Tensor,
        ]
        | None = None,
        sampler: Callable[..., torch.Tensor] | None = None,
    ):
        """
        Args:
            guidance_pixels: Either the pixel index of ``self.input_grid`` where
                the TCs are desired (``0 <= guidance_pixels < 12 * nside ** 2``)
                or the entire HPX tensor already set. If ``None``, no guidance.
            guidance_scale: Multiplier on the per-step guidance vector returned
                by ``guidance_fn``.
            guidance_fn: Optional per-step guidance hook replacing
                ``cbottle.denoiser_factories.get_guidance``. Called once per
                sampler step with positional args
                ``(guidance_data, logits, x_hat, denoised, t_hat)`` and must
                return a ``torch.Tensor`` of the
                same shape as ``x_hat`` representing the *unscaled* guidance
                vector. The caller adds ``guidance_scale * d_guide`` to the
                drift, so ``guidance_fn`` is responsible only for the gradient
                direction and any sigma-schedule weighting it applies
                internally.
            sampler: Optional EDM sampler replacing ``edm_sampler_from_sigma``.
                Called as ``sampler(D, xT, randn_like=..., sigma_min=...,
                sigma_max=..., num_steps=..., time_stepper=...)`` and must
                return a ``torch.Tensor`` of the same shape as ``xT``.
        """
        if batch["target"].device != self.device:
            batch = self._move_to_device(batch)
        images, labels, condition = batch["target"], batch["labels"], batch["condition"]
        second_of_day = batch["second_of_day"].float()
        day_of_year = batch["day_of_year"].float()
        batch_size = second_of_day.shape[0]

        label_ind = labels.nonzero()[:, 1]
        mask = torch.stack([self.icon_mask, self.era5_mask]).to(self.device)[
            label_ind
        ]  # n, c
        mask = mask[:, :, None, None]
        spatial_mask_gpu = (
            spatial_mask.to(self.device) if spatial_mask is not None else None
        )

        with torch.no_grad():
            device = condition.device

            if pre_generated_latents is not None:
                # Use pre-generated latents (e.g., from CorrelatedLatentGenerator)
                latents = pre_generated_latents
                if latents.device != device:
                    latents = latents.to(device)
            else:
                # Generate new latents
                if seed is None:
                    rnd = torch
                else:
                    rnd = StackedRandomGenerator(device, seeds=[seed] * batch_size)

                latents = rnd.randn(
                    (
                        batch_size,
                        self.net.img_channels,
                        self.time_length,
                        self.net.domain.numel(),
                    ),
                    device=self.device,
                )

            if self.channels_last:
                latents = latents.to(memory_format=torch.channels_last)
                condition = condition.to(memory_format=torch.channels_last)
                if images.dim() == 4:
                    images = images.to(memory_format=torch.channels_last)

            if start_from_noisy_image:
                xT = latents * self.sigma_max + images
            else:
                xT = latents * self.sigma_max

            any_nan = not (mask.all())

            # ICON fills NaNs
            labels_when_nan = torch.zeros_like(labels)
            labels_when_nan[:, 0] = 1

            guidance_data = guidance_pixels
            if guidance_pixels is not None and guidance_pixels.ndim == 1:
                guidance_data = torch.full(
                    (batch_size, 1, 1, *self.classifier_grid.shape),
                    torch.nan,
                    device=self.device,
                )
                guidance_data[:, :, :, guidance_pixels] = 1
                if self.channels_last:
                    guidance_data = guidance_data.to(memory_format=torch.channels_last)

            # Call the guidance hook whenever the caller supplied a guidance_fn
            # (even if guidance_data is None and guidance_scale is 0) OR when we
            # have classifier guidance data with positive scale (the original
            # default path). A caller supplying only guidance_fn uses it purely
            # for per-step inspection/tracking; its contribution to ``d`` is
            # suppressed by the ``guidance_scale * d_guide`` multiplier.
            _call_guidance = guidance_fn is not None or (
                guidance_data is not None and guidance_scale > 0
            )
            _classifier_logits_needed = (
                guidance_data is not None and guidance_scale > 0 and guidance_fn is None
            )

            def D(x_hat, t_hat):
                if _call_guidance:
                    x_hat.requires_grad_(True)

                x_in = x_hat.where(mask, 0)
                if spatial_mask_gpu is not None:
                    x_in = x_in * spatial_mask_gpu
                out = self.net(
                    x_in,
                    t_hat,
                    class_labels=labels,
                    condition=condition,
                    second_of_day=second_of_day,
                    day_of_year=day_of_year,
                )

                if any_nan:
                    d2 = self.net(
                        x_hat,
                        t_hat,
                        class_labels=labels_when_nan,
                        condition=condition,
                        second_of_day=second_of_day,
                        day_of_year=day_of_year,
                    ).out
                else:
                    d2 = 0.0
                d = out.out.where(mask, d2)
                if spatial_mask_gpu is not None:
                    d = d * spatial_mask_gpu
                if _call_guidance:
                    # Logits are only required when we'll be calling the default
                    # classifier get_guidance (which needs them). Custom
                    # guidance_fns can handle ``logits=None`` themselves.
                    if guidance_data is not None and guidance_scale > 0:
                        if self.separate_classifier is not None:
                            out.logits = self.separate_classifier(
                                x_hat,
                                t_hat,
                                class_labels=labels,
                                condition=condition,
                                second_of_day=second_of_day,
                                day_of_year=day_of_year,
                            ).logits
                        elif out.logits is None and _classifier_logits_needed:
                            raise ValueError(
                                "Model did not produce `logits`. Are you sure this model was trained with guidance?"
                            )
                    # If the caller supplied guidance_fn, use it; else fall back to the
                    # module-level get_guidance (still monkey-patchable for back-compat).
                    _gfn = (
                        guidance_fn
                        if guidance_fn is not None
                        else cbottle.denoiser_factories.get_guidance
                    )
                    d_guide = _gfn(guidance_data, out.logits, x_hat, d, t_hat)
                    d = d + guidance_scale * d_guide  # zero-safe when guidance_scale=0
                    if self.channels_last:
                        d = d.to(memory_format=torch.channels_last)

                return d

            if _call_guidance:
                D = torch.enable_grad(D)

            D.round_sigma = self.net.round_sigma
            D.sigma_max = self.net.sigma_max
            D.sigma_min = self.net.sigma_min

            # If the caller supplied a custom sampler, use it; else fall back to the
            # module-level edm_sampler_from_sigma (still monkey-patchable for back-compat).
            _sampler = sampler if sampler is not None else edm_sampler_from_sigma
            with torch.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
                out = _sampler(
                    D,
                    xT,
                    randn_like=torch.randn_like,
                    sigma_min=self.sigma_min,
                    sigma_max=int(
                        self.sigma_max
                    ),  # Convert to int for type compatibility
                    num_steps=self.num_steps,
                    time_stepper=self.time_stepper,
                )

            if return_untransformed:
                raw = out
                processed = self._post_process(out)
                return processed, self.coords, raw
            else:
                out = self._post_process(out)
                return out, self.coords

    def get_guidance_pixels(self, lons, lats) -> torch.Tensor:
        return self.classifier_grid.ang2pix(
            torch.as_tensor(lons), torch.as_tensor(lats)
        )

    def calculate_odds_ratio(
        self,
        batch: dict,
        guidance_pixels: torch.Tensor,
        **kwargs,
    ) -> tuple[float, torch.Tensor]:
        """Compute classifier-guided ``(log_odds_ratio, forward_latents)``.

        Thin wrapper over :meth:`_calculate_odds_ratio_full`; ``**kwargs`` are
        forwarded verbatim. The two backward phases always run -- without them
        ``log_odds_ratio`` is undefined -- so callers who want forward-only
        mode, the per-phase divergence integrals, the Gaussian logps, or the
        backward latents should call :meth:`_calculate_odds_ratio_full`
        directly.
        """
        result = self._calculate_odds_ratio_full(
            batch, guidance_pixels, run_backward=True, **kwargs
        )
        return result.log_odds_ratio, result.forward_latents

    def _calculate_odds_ratio_full(
        self,
        batch: dict[str, torch.Tensor],
        guidance_pixels: torch.Tensor,
        *,
        num_steps: int = 36,
        sigma_min: float = 0.002,
        rho: float = 7.0,
        extra_steps_intervals: "Sequence[tuple[float, float, int]]" = ((15, 20, 25),),
        guidance_scale: float = 64.0 * 20.0,
        guidance_on: float = 15.0,
        guidance_off: float = 20.0,
        divergence_samples: int = 3,
        bf16: bool = True,
        start_from_noisy_image: bool = False,
        run_backward: bool = True,
        forward_guidance: bool = True,
        compute_forward_divergences: bool = True,
        seed: int | None = None,
    ) -> OddsRatioResult:
        """Compute log-odds-ratio ingredients for ``batch`` under classifier guidance.

        Phase naming follows the script convention, NOT standard diffusion
        terminology:

        - **"forward"** = decoding direction (``reverse=False``,
          ``start_latents=None``). Standard EDM sampler: starts from Gaussian
          noise at ``sigma_max`` and integrates down to ``sigma=0`` to produce
          a clean sample (``forward_latents``).
        - **"backward"** = encoding direction (``reverse=True``,
          ``start_latents=forward_latents``). Schedule is flipped to ascend
          from ``sigma_min`` to ``sigma_max``, with the tail restored to the
          raw ``sigma_max`` so the Gaussian reference is well-defined. The
          starting latents are overridden with ``forward_latents`` from the
          forward phase, so the run encodes the forward sample back to noise
          (``backward_latents`` / ``backward_no_guidance_latents``). The
          Gaussian logp of these encoded noise tensors under
          ``N(0, sigma_max^2 I)`` provides the reference for the odds ratio.

        The ``log_odds_ratio`` property combines only the backward fields::

            log p_unguided / p_guided
            ~= (backward_no_guidance_gaussian_logp - backward_gaussian_logp)
             + (backward_no_guidance_score_div_integral - backward_score_div_integral)
             - backward_guidance_div_integral

        For an unguided true log-likelihood via the probability-flow ODE use
        :func:`cbottle.likelihood.log_prob` instead. Defaults match the TC
        paper configuration (36 steps densified to 61 in the [15,20] sigma
        window, ``guidance_scale = 64 * 20``). Note that the maximum sigma is
        always ``self.sigma_max`` (the value the model was constructed with):
        ``xT`` seeding, the schedule endpoint, and the Gaussian reference
        logp must all agree, so it is not exposed as a kwarg here.

        Args:
            batch: Data batch (same format as :meth:`sample`).
            guidance_pixels: 1-D tensor ``(N,)`` of HPX pixel indices on
                ``self.classifier_grid`` (level 3, HEALPIX_PAD_XY) where TCs
                are desired. Build via :meth:`get_guidance_pixels`.
            num_steps, sigma_min, rho, extra_steps_intervals: Parameters of
                the *custom* odds-ratio schedule built by
                :func:`cbottle.odds_ratio._create_custom_time_steps`. These
                are intentionally separate from ``self.num_steps`` /
                ``self.sigma_min`` (which govern the default EDM sampler that
                ``_sample_with_latents`` falls back on) and may be tuned
                independently to densify the guidance window.
            run_backward: If ``False``, only run the forward pass.
            forward_guidance: If ``False``, forward pass runs without the
                classifier guidance gradient.
            compute_forward_divergences: If ``False``, skip Hutchinson probes
                on the forward pass.
            seed: If set, seeds the initial Gaussian noise for the forward
                phase so the method is deterministic run-to-run. Hutchinson probes
                are always deterministically seeded per step.

        Returns:
            :class:`~cbottle.odds_ratio.OddsRatioResult` with per-phase
            divergence integrals, Gaussian reference logps, the forward
            ``initial_log_prob``, and the raw untransformed phase latents.
            Use ``result.log_odds_ratio`` to recover the combined scalar.

        Note:
            Only ``batch["target"].shape[0] == 1`` is supported. The
            Hutchinson divergence trace and Gaussian reference logp are
            both reduced as a single multivariate Gaussian over the whole
            sample tensor, and the second-order autograd path through the
            denoiser is GPU-memory-bound -- larger batches OOM in practice.
        """

        if batch["target"].shape[0] != 1:
            raise ValueError(
                f"_calculate_odds_ratio_full only supports batch_size=1; "
                f"got batch['target'].shape={tuple(batch['target'].shape)}. "
                f"Loop over batches in the caller."
            )

        sigma_schedule_fn = lambda t_hat: default_sigma_schedule(  # noqa: E731
            t_hat, guidance_on=guidance_on, guidance_off=guidance_off
        )

        def _run_phase(
            *,
            reverse: bool,
            compute_guidance: bool,
            start_latents=None,
            phase: str,
            compute_divergences: bool = True,
        ):
            logging.info(
                f"calculate_odds_ratio: running phase '{phase}' "
                f"(reverse={reverse}, guidance={compute_guidance})"
            )
            if compute_guidance:
                guidance_fn = classifier_guidance
                guidance_pixels_eff = guidance_pixels
                scale = guidance_scale
                tracker_scale = guidance_scale
            else:
                guidance_fn = None
                guidance_pixels_eff = None
                scale = 0.0
                # With no guidance_fn the scale is unused inside the tracker
                # (it only multiplies a zero vector); pass 0 for clarity
                # rather than the caller's scale.
                tracker_scale = 0.0

            tracker = DivergenceTracker(
                guidance_fn=guidance_fn,
                guidance_scale=tracker_scale,
                sigma_max=self.sigma_max,
                divergence_samples=divergence_samples,
                phase=phase,
                compute_score_div=compute_divergences,
                compute_guidance_div=compute_divergences and compute_guidance,
                sigma_schedule=sigma_schedule_fn,
            )
            sampler_func = _make_reverse_aware_sampler(
                reverse=reverse,
                start_latents=start_latents,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=self.sigma_max,
                rho=rho,
                extra_steps_intervals=extra_steps_intervals,
                progress_wrapper=lambda it, _phase=phase: tqdm(
                    it,
                    desc=f"calculate_odds_ratio[{_phase}]",
                    leave=False,
                ),
            )

            _processed, _coords, raw = self._sample_with_latents(
                batch,
                seed=seed,
                start_from_noisy_image=start_from_noisy_image,
                guidance_pixels=guidance_pixels_eff,
                guidance_scale=scale,
                bf16=bf16,
                return_untransformed=True,
                guidance_fn=tracker,
                sampler=sampler_func,
            )
            return raw, tracker

        # ---- Forward ---------------------------------------------------------
        forward_phase = "forward" if forward_guidance else "forward_no_guidance"
        forward_latents, forward_tracker = _run_phase(
            reverse=False,
            compute_guidance=forward_guidance,
            start_latents=None,
            phase=forward_phase,
            compute_divergences=compute_forward_divergences,
        )
        forward_guidance_div_integral = calculate_divergence_integral(
            forward_tracker.data, "divergence", "forward"
        )
        forward_score_div_integral = calculate_divergence_integral(
            forward_tracker.data, "score_divergence", "forward"
        )

        result = OddsRatioResult(
            forward_guidance_div_integral=float(forward_guidance_div_integral),
            forward_score_div_integral=float(forward_score_div_integral),
            initial_log_prob=forward_tracker.initial_log_prob,
            forward_latents=forward_latents,
        )

        if not run_backward:
            return result

        # ---- Backward with guidance -----------------------------------------
        backward_latents, backward_tracker = _run_phase(
            reverse=True,
            compute_guidance=True,
            start_latents=forward_latents,
            phase="backward",
        )
        backward_guidance_div_integral = calculate_divergence_integral(
            backward_tracker.data, "divergence", "backward"
        )
        backward_score_div_integral = calculate_divergence_integral(
            backward_tracker.data, "score_divergence", "backward"
        )
        backward_gaussian_logp = calculate_gaussian_logp(
            backward_latents, self.sigma_max
        )

        # ---- Backward without guidance --------------------------------------
        (
            backward_no_guidance_latents,
            backward_no_guidance_tracker,
        ) = _run_phase(
            reverse=True,
            compute_guidance=False,
            start_latents=forward_latents,
            phase="backward_no_guidance",
        )
        backward_no_guidance_guidance_div_integral = calculate_divergence_integral(
            backward_no_guidance_tracker.data, "divergence", "backward"
        )
        backward_no_guidance_score_div_integral = calculate_divergence_integral(
            backward_no_guidance_tracker.data, "score_divergence", "backward"
        )
        backward_no_guidance_gaussian_logp = calculate_gaussian_logp(
            backward_no_guidance_latents, self.sigma_max
        )

        result.backward_guidance_div_integral = float(backward_guidance_div_integral)
        result.backward_score_div_integral = float(backward_score_div_integral)
        result.backward_gaussian_logp = float(backward_gaussian_logp)
        result.backward_latents = backward_latents
        result.backward_no_guidance_guidance_div_integral = float(
            backward_no_guidance_guidance_div_integral
        )
        result.backward_no_guidance_score_div_integral = float(
            backward_no_guidance_score_div_integral
        )
        result.backward_no_guidance_gaussian_logp = float(
            backward_no_guidance_gaussian_logp
        )
        result.backward_no_guidance_latents = backward_no_guidance_latents
        return result

    def sample_for_superresolution(
        self,
        batch: dict,
        indices_where_tc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Coords]:
        out, coords = self.sample(
            batch,
            guidance_pixels=indices_where_tc,
        )
        out = self._normalize(out)
        out = self._reorder(out)

        batch["target"] = out
        out, coords = self.translate(batch, dataset="icon")

        return out, coords


class SuperResolutionModel(torch.nn.Module):
    """
    A callable object that performs super-resolution on low-resolution Healpix data.

    Takes a low-res tensor in NEST convention and returns the higher resolution output.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        batch_info: base.BatchInfo,
        hpx_level: int = 10,
        hpx_lr_level: int = 6,
        patch_size: int = 128,
        overlap_size: int = 32,
        num_steps: int = 18,
        sigma_max: int = 800,
        torch_compile: bool = False,
    ):
        """
        Initialize the super-resolution model.

        Args:
            net: A torch module with the same API as networks.EDMPrecond
            hpx_level: HPX level for high resolution output
            hpx_lr_level: HPX level for low resolution input
            patch_size: Patch size for multidiffusion
            overlap_size: Overlapping pixel number between patches
            num_steps: Sampler iteration number
            sigma_max: Noise sigma max
            torch_compile: Whether to compile the model with torch.compile
        """
        super().__init__()
        self.hpx_level = hpx_level
        self.hpx_lr_level = hpx_lr_level
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.num_steps = num_steps
        self.sigma_max = sigma_max

        self.batch_info = batch_info
        self.net = net
        if torch_compile:
            self.torch_compile()

        self.net.eval().requires_grad_(False)

        # Setup grids
        self.high_res_grid = healpix.Grid(
            level=hpx_level, pixel_order=healpix.PixelOrder.NEST
        )
        self.low_res_grid = healpix.Grid(
            level=hpx_lr_level, pixel_order=healpix.PixelOrder.NEST
        )

        # Setup regridders
        lat = np.linspace(-90, 90, 128)[:, None]
        lon = np.linspace(0, 360, 128)[None, :]
        self.regrid_to_latlon = self.low_res_grid.get_bilinear_regridder_to(lat, lon)
        self.regrid = earth2grid.get_regridder(
            self.low_res_grid, self.high_res_grid
        ).float()

    def torch_compile(self):
        self.net = torch.compile(self.net, fullgraph=True)

    @classmethod
    def from_pretrained(cls, state_path: str, map_location: str = "cpu", **kwargs):
        # Load model
        with checkpointing.Checkpoint(state_path) as checkpoint:
            net = checkpoint.read_model(map_location=map_location)
            batch_info = checkpoint.read_batch_info()
        return cls(net, batch_info, **kwargs)

    def _sample(self, denoiser, latents):
        return edm_sampler(
            denoiser, latents, num_steps=self.num_steps, sigma_max=self.sigma_max
        )

    def _apply_on_patches(
        self,
        x_hat,
        x_lr,
        t_hat,
        class_labels,
        batch_size,
        global_lr,
        inbox_patch_index,
    ):
        return patchify.apply_on_patches(
            self.net,
            patch_size=self.patch_size,
            overlap_size=self.overlap_size,
            x_hat=x_hat,
            x_lr=x_lr,
            t_hat=t_hat,
            class_labels=class_labels,
            batch_size=batch_size,
            global_lr=global_lr,
            inbox_patch_index=inbox_patch_index,
            device=x_hat.device,
        )

    def __call__(
        self,
        x: torch.Tensor,
        coords: Coords,
        extents: tuple,
    ) -> tuple[torch.Tensor, Coords]:
        """
        Perform super-resolution on a low-resolution tensor with batch processing.

        Args:
            x: Low-resolution tensor in NEST convention, shape (b, c, t, npix_lr)
               where b=batch, c=channels, t=time, npix_lr corresponds to the low-resolution grid
            coords: Low-resolution coordinates
            extents: Bounding box (lon_west, lon_east, lat_south, lat_north) to override super_resolution_box

        Returns:
            Tuple of:
            - High-resolution tensor in NEST convention, shape (b, c_out, t, npix_hr)
              where c_out corresponds to self.batch_info.channels
            - High-resolution coordinates with grid and channel information
        """
        # Convert from (lon_west, lon_east, lat_south, lat_north) to (lat_south, lon_west, lat_north, lon_east)
        if extents:
            lon_west, lon_east, lat_south, lat_north = extents
            current_super_resolution_box = (lat_south, lon_west, lat_north, lon_east)

            # Calculate inbox_patch_index for the current bounding box
            inbox_patch_index = patchify.patch_index_from_bounding_box(
                self.hpx_level,
                current_super_resolution_box,
                self.patch_size,
                self.overlap_size,
                x.device,
            )
        else:
            inbox_patch_index = None

        # Get input dimensions
        b, c, t, npix_lr = x.shape

        # Process each batch separately
        batch_results = []
        for batch_idx in range(b):
            batch_tensor = x[batch_idx]  # Shape: (c, t, npix_lr)

            # Process each time step
            time_results = []
            for time_idx in range(t):
                # Extract single time step: (c, npix_lr)
                lr_tensor = batch_tensor[:, time_idx, :]

                # Reorder channels to match super-resolution model
                # Map input channels to output channels using self.batch_info.channels
                lr_tensor = lr_tensor[
                    [
                        coords.batch_info.channels.index(ch)
                        for ch in self.batch_info.channels
                    ]
                ]

                # Apply super-resolution to single time step
                hr_tensor = self._super_resolve_single_tensor(
                    lr_tensor, inbox_patch_index
                )
                time_results.append(hr_tensor)

            # Stack time results: (c_out, t, npix_hr)
            batch_result = torch.stack(time_results, dim=1)
            batch_results.append(batch_result)

        # Stack batch results: (b, c_out, t, npix_hr)
        hr_output = torch.stack(batch_results, dim=0)

        # Create high-resolution coordinates
        hr_coords = Coords(self.batch_info, self.high_res_grid)
        return hr_output, hr_coords

    def denormalize(self, x):
        # Denormalize output
        if self.batch_info.center is not None and self.batch_info.scales is not None:
            # Convert lists to tensors if necessary
            center = self.batch_info.center
            scales = self.batch_info.scales

            center = torch.tensor(center, device=x.device)
            scales = torch.tensor(scales, device=x.device)

            # Ensure proper broadcasting
            if center.dim() == 1:
                center = center.view(-1, 1)  # (channels, 1)
            if scales.dim() == 1:
                scales = scales.view(-1, 1)  # (channels, 1)
            x = x * scales + center
        return x

    def _super_resolve_single_tensor(
        self,
        lr_tensor: torch.Tensor,
        inbox_patch_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform super-resolution on a single low-resolution tensor.
        This is the original __call__ logic adapted for single tensors.

        Args:
            lr_tensor: Low-resolution tensor in NEST convention, shape (channels, npix_lr)
            inbox_patch_index: Tensor of patch indices to process

        Returns:
            High-resolution tensor in NEST convention, shape (channels, npix_hr)
        """

        # Normalize input using batch info
        if self.batch_info.center is not None and self.batch_info.scales is not None:
            # Convert lists to tensors if necessary
            center = self.batch_info.center
            scales = self.batch_info.scales

            if isinstance(center, list):
                center = torch.tensor(center, device=lr_tensor.device)
            if isinstance(scales, list):
                scales = torch.tensor(scales, device=lr_tensor.device)

            # Ensure proper broadcasting
            if center.dim() == 1:
                center = center.view(-1, 1)  # (channels, 1)
            if scales.dim() == 1:
                scales = scales.view(-1, 1)  # (channels, 1)

            lr_normalized = (lr_tensor - center) / scales
        else:
            lr_normalized = lr_tensor

        # Get global low resolution for lat-lon context
        global_lr = self.regrid_to_latlon(lr_normalized.double())[None,]

        # Regrid to high resolution
        lr_hr = self.regrid(lr_normalized)

        # Prepare tensors for diffusion
        in_channels = lr_hr.shape[0]
        latents = torch.randn_like(lr_hr)
        # Add batch dimension
        latents = latents.reshape((in_channels, -1))[None,]
        lr_hr = lr_hr.reshape((in_channels, -1))[None,]

        with torch.no_grad():
            # Define denoiser function
            def denoiser(x, t):
                return self._apply_on_patches(
                    x_hat=x,
                    x_lr=lr_hr,
                    t_hat=t,
                    class_labels=None,
                    batch_size=128,
                    global_lr=global_lr,
                    inbox_patch_index=inbox_patch_index,
                ).to(torch.float64)

            # Set denoiser attributes
            denoiser.sigma_max = self.net.sigma_max
            denoiser.sigma_min = self.net.sigma_min
            denoiser.round_sigma = self.net.round_sigma

            # Run EDM sampler
            pred = self._sample(denoiser, latents)

            pred = self.denormalize(pred)
            # Reshape back to original format
            pred = pred.reshape((in_channels, -1))
            return pred


class DistilledSuperResolutionModel(SuperResolutionModel):
    def __init__(
        self,
        net: torch.nn.Module,
        batch_info: base.BatchInfo,
        hpx_level: int = 10,
        hpx_lr_level: int = 6,
        patch_size: int = 128,
        overlap_size: int = 32,
        num_steps: int = 18,
        sigma_max: int = 800,
        torch_compile: bool = False,
        window_function: str = "KBD",
        window_alpha: int = 1,
    ):
        super().__init__(
            net=net,
            batch_info=batch_info,
            hpx_level=hpx_level,
            hpx_lr_level=hpx_lr_level,
            patch_size=patch_size,
            overlap_size=overlap_size,
            num_steps=num_steps,
            sigma_max=sigma_max,
            torch_compile=torch_compile,
        )
        window = self._get_window_function(
            patch_size=patch_size,
            window_alpha=window_alpha,
            type=window_function,
            dtype=torch.float32,
        )
        window = window.reshape((1, 1, window.shape[0], window.shape[1]))
        self.register_buffer("window", window)

    def _get_window_function(self, patch_size, window_alpha, type="KBD", **kwargs):
        functions = {
            "uniform": torch.ones,
            "KBD": lambda ps: kaiser_bessel_derived(ps, window_alpha * np.pi),
        }

        if type not in functions.keys():
            raise ValueError(
                f"Unknown window function type {type}. Supported types are {list(functions.keys())}"
            )

        window = torch.tensor(functions[type](patch_size), **kwargs)
        window = window.unsqueeze(0) * window.unsqueeze(1)
        return window

    def _apply_on_patches(
        self,
        x_hat,
        x_lr,
        t_hat,
        class_labels,
        batch_size,
        global_lr,
        inbox_patch_index,
    ):
        return patchify.apply_on_patches(
            self.net,
            patch_size=self.patch_size,
            overlap_size=self.overlap_size,
            x_hat=x_hat,
            x_lr=x_lr,
            t_hat=t_hat,
            class_labels=class_labels,
            batch_size=batch_size,
            global_lr=global_lr,
            inbox_patch_index=inbox_patch_index,
            window=self.window,
        )

    def _sample(self, denoiser, latents):
        return few_step_sampler(
            denoiser,
            latents,
            sigma_max=self.sigma_max,
            sigma_mid=[self.sigma_max / 80 * 1.5],
        )


class MixtureOfExpertsDenoiser(torch.nn.Module):
    """
    A Mixture of Experts (MoE) denoiser that selects among multiple EDMPrecond-like models
    based on sigma thresholds.
    """

    def __init__(
        self,
        experts: list[torch.nn.Module],
        sigma_thresholds: tuple[float, ...],
        batch_info: base.BatchInfo | None = None,
    ):
        super().__init__()
        assert len(experts) >= len(sigma_thresholds)
        self.experts = torch.nn.ModuleList(experts)
        self.sigma_thresholds = sigma_thresholds
        self.batch_info = batch_info

        # Inherit attributes from the last expert
        self.round_sigma = experts[-1].round_sigma
        self.sigma_max = experts[-1].sigma_max
        self.sigma_min = experts[-1].sigma_min

    @classmethod
    def from_pretrained(
        cls,
        path: str | list[str],
        sigma_thresholds: tuple[float, ...],
        *,
        allow_second_order_derivatives: bool = False,
        map_location: str = "cpu",
    ) -> "MixtureOfExpertsDenoiser":
        match path:
            case str():
                paths = [path]
            case list():
                if not path:
                    raise ValueError("Empty list passed.")
                paths = path

        experts = []
        for path in paths:
            logging.info(f"Opening {path}")
            with checkpointing.Checkpoint(path) as c:
                model = c.read_model(
                    map_location=map_location,
                    allow_second_order_derivatives=allow_second_order_derivatives,
                ).eval()
                experts.append(model)
                batch_info = c.read_batch_info()
        return cls(experts, sigma_thresholds=sigma_thresholds, batch_info=batch_info)

    @property
    def domain(self):
        return self.experts[0].domain

    @property
    def img_channels(self):
        return self.experts[0].img_channels

    @property
    def time_length(self):
        return self.experts[0].time_length

    def forward(self, x, sigma, *args, **kwargs):
        sigma_value = sigma.item() if sigma.ndim == 0 else sigma.view(-1)[0].item()

        for i, threshold in enumerate(self.sigma_thresholds):
            if sigma_value >= threshold:
                return self.experts[i](x, sigma, *args, **kwargs)
        return self.experts[-1](x, sigma, *args, **kwargs)


def load(model: str, root="", **kwargs) -> CBottle3d:
    root = root or environment.CHECKPOINT_ROOT
    if model == "cbottle-3d-moe":
        checkpoints = "training-state-000512000.checkpoint,training-state-002048000.checkpoint,training-state-009856000.checkpoint".split(
            ","
        )
        rundir = "cBottle-3d"
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        return CBottle3d.from_pretrained(
            paths, sigma_thresholds=(100.0, 10.0), **kwargs
        )
    elif model == "cbottle-3d-moe-tc":
        rundir = "cBottle-3d"
        checkpoints = "training-state-000512000.checkpoint,training-state-002048000.checkpoint,training-state-009856000.checkpoint".split(
            ","
        )
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        classifier_path = os.path.join(
            root, "cBottle-3d-tc", "training-state-002176000.checkpoint"
        )
        return CBottle3d.from_pretrained(
            paths,
            sigma_thresholds=(100.0, 10.0),
            separate_classifier_path=classifier_path,
            allow_second_order_derivatives=True,
            **kwargs,
        )
    elif model == "cbottle-3d-moe-aimip-p1":
        checkpoints = "training-state-000512000.checkpoint,training-state-002048000.checkpoint,training-state-009856000.checkpoint".split(
            ","
        )
        rundir = "aimip_v3"
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        return CBottle3d.from_pretrained(paths, sigma_thresholds=(100.0, 10.0))

    elif model == "cbottle-3d-moe-aimip-p2":
        checkpoints = "training-state-000512000.checkpoint,training-state-002176000.checkpoint,training-state-009984000.checkpoint".split(
            ","
        )
        rundir = "aimip_v3"
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        return CBottle3d.from_pretrained(paths, sigma_thresholds=(100.0, 10.0))

    elif model == "cbottle-3d-moe-aimip-p3":
        checkpoints = "training-state-000640000.checkpoint,training-state-002048000.checkpoint,training-state-010112000.checkpoint".split()
        rundir = "aimip_v3"
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        return CBottle3d.from_pretrained(paths, sigma_thresholds=(100.0, 10.0))

    elif model == "cbottle-3d-moe-aimip-p4":
        # use this model for p5 with correlation half life 0.001
        checkpoints = "training-state-000640000.checkpoint,training-state-002176000.checkpoint,training-state-009728000.checkpoint".split(
            ","
        )
        rundir = "aimip_v3"
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        return CBottle3d.from_pretrained(paths, sigma_thresholds=(100.0, 10.0))
    elif model == "cbottle-3d-video":
        rundir = "cBottle-3d-video"
        checkpoints = "training-state-000541152.checkpoint,training-state-001028656.checkpoint,training-state-003209456.checkpoint".split(
            ","
        )
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        return CBottle3d.from_pretrained(
            paths,
            sigma_thresholds=(316.0, 10.0),
            **kwargs,
        )
    raise ValueError(model)
