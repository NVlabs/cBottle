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
from typing import Literal
import dataclasses
import logging

import cbottle.denoiser_factories
from . import checkpointing, patchify
from .diffusion_samplers import (
    edm_sampler,
    edm_sampler_from_sigma,
    edm_sampler_steps,
    StackedRandomGenerator,
)
from .datasets import base

from .datasets.dataset_2d import HealpixDatasetV5, LABELS
from cbottle.config import environment


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
    ):
        """
        Initialize the CBottle3d model.

        Args:
            net: The denoiser network (MixtureOfExpertsDenoiser)
            separate_classifier: Optional separate classifier for guidance
            sigma_min: Minimum noise sigma for diffusion
            sigma_max: Maximum noise sigma for diffusion
            num_steps: Number of sampling steps
        """
        self.net = net
        self.separate_classifier = separate_classifier
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps

    @classmethod
    def from_pretrained(
        cls,
        path: str | list[str],
        separate_classifier_path: str | None = None,
        sigma_thresholds: tuple[float, ...] = (),
        allow_second_order_derivatives: bool = False,
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
                separate_classifier = (
                    c.read_model(
                        map_location=None,
                        allow_second_order_derivatives=allow_second_order_derivatives,
                    )
                    .cuda()
                    .eval()
                )

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

        scales = info.scales
        center = info.center
        x = grid_obj.reorder(self.output_grid.pixel_order, x)
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

    def infill(self, batch: dict) -> tuple[torch.Tensor, Coords]:
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
            out = edm_sampler_from_sigma(
                D,
                latents,
                randn_like=torch.randn_like,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                num_steps=self.num_steps,
            )

        # Post-process the output
        out = self._post_process(out)

        return out, Coords(self.batch_info, self.output_grid)

    def _encode(self, batch: dict) -> dict:
        batch = self._move_to_device(batch)

        images, labels, condition = batch["target"], batch["labels"], batch["condition"]
        second_of_day = batch["second_of_day"].float()
        day_of_year = batch["day_of_year"].float()
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
    ) -> dict:
        batch = self._move_to_device(batch)

        images = batch["target"]
        condition = batch["condition"]
        second_of_day = batch["second_of_day"].float()
        day_of_year = batch["day_of_year"].float()

        labels_when_nan = torch.zeros_like(batch["labels"].cuda())
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
        self, batch: dict, dataset: Literal["icon", "era5"]
    ) -> tuple[torch.Tensor, Coords]:
        # Move all tensors to the correct device
        encoded = self._encode(batch)
        with torch.no_grad():
            out = self._decode(encoded, dataset)["target"]
        out = self._post_process(out)
        return out, Coords(self.batch_info, self.output_grid)

    def denormalize(self, batch: dict) -> tuple[torch.Tensor, Coords]:
        # Move all tensors to the correct device
        out = batch["target"]
        out = self._post_process(out)
        out = out.to(self.device)
        return out, Coords(self.batch_info, self.output_grid)

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
    ):
        """

        Args:

            guidance_pixels: pixels of ``self.input_grid``` where the TCs are desired. 0<= guidance_pixels < 12 * nside ^2.
            guidance_scale: float = 0.03,

        """
        batch = self._move_to_device(batch)
        images, labels, condition = batch["target"], batch["labels"], batch["condition"]
        second_of_day = batch["second_of_day"].cuda().float()
        day_of_year = batch["day_of_year"].cuda().float()
        batch_size = second_of_day.shape[0]

        label_ind = labels.nonzero()[:, 1]
        mask = torch.stack([self.icon_mask, self.era5_mask]).cuda()[label_ind]  # n, c
        mask = mask[:, :, None, None]

        with torch.no_grad():
            device = condition.device

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
                device=device,
            )

            if start_from_noisy_image:
                xT = latents * self.sigma_max + images
            else:
                xT = latents * self.sigma_max

            any_nan = not (mask.all())

            # ICON fills NaNs
            labels_when_nan = torch.zeros_like(labels)
            labels_when_nan[:, 0] = 1

            if guidance_pixels is not None:
                guidance_data = torch.full(
                    (batch_size, 1, 1, *self.classifier_grid.shape),
                    torch.nan,
                    device=device,
                )
                guidance_data[:, :, :, guidance_pixels] = 1
            else:
                guidance_data = None

            def D(x_hat, t_hat):
                if guidance_data is not None:
                    x_hat.requires_grad_(True)

                out = self.net(
                    x_hat.where(mask, 0),
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
                    elif out.logits is None:
                        raise ValueError(
                            "Model did not produce `logits`. Are you sure this model was trained with guidance?"
                        )
                    else:
                        # use the logits from the main model
                        pass

                    d_guide = cbottle.denoiser_factories.get_guidance(
                        guidance_data, out.logits, x_hat, d, t_hat
                    )
                    d = d + guidance_scale * d_guide

                return d

            if guidance_data is not None:
                D = torch.enable_grad(D)

            D.round_sigma = self.net.round_sigma
            D.sigma_max = self.net.sigma_max
            D.sigma_min = self.net.sigma_min

            with torch.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
                out = edm_sampler_from_sigma(
                    D,
                    xT,
                    randn_like=torch.randn_like,
                    sigma_min=self.sigma_min,
                    sigma_max=int(
                        self.sigma_max
                    ),  # Convert to int for type compatibility
                )

            out = self._post_process(out)

            return out, self.coords

    def get_guidance_pixels(self, lons, lats) -> torch.Tensor:
        return self.classifier_grid.ang2pix(
            torch.as_tensor(lons), torch.as_tensor(lats)
        )


class SuperResolutionModel:
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
        device: str = "cuda",
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
            device: Device to run inference on
        """
        self.hpx_level = hpx_level
        self.hpx_lr_level = hpx_lr_level
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.device = device

        self.batch_info = batch_info
        self.net = net

        self.net.eval().requires_grad_(False).to(device)

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
        self.regrid_to_latlon = self.low_res_grid.get_bilinear_regridder_to(
            lat, lon
        ).to(device)
        self.regrid = earth2grid.get_regridder(self.low_res_grid, self.high_res_grid)
        self.regrid.to(device).float()

    @classmethod
    def from_pretrained(cls, state_path: str, **kwargs):
        # Load model
        with checkpointing.Checkpoint(state_path) as checkpoint:
            net = checkpoint.read_model()
            batch_info = checkpoint.read_batch_info()
        return cls(net, batch_info, **kwargs)

    def __call__(
        self, x: torch.Tensor, coords: Coords, extents: tuple
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
                self.device,
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

            center = torch.tensor(center, device="cpu")
            scales = torch.tensor(scales, device="cpu")

            # Ensure proper broadcasting
            if center.dim() == 1:
                center = center.view(-1, 1)  # (channels, 1)
            if scales.dim() == 1:
                scales = scales.view(-1, 1)  # (channels, 1)
            x = x.cpu() * scales + center
        else:
            x = x.cpu()
        return x

    def _super_resolve_single_tensor(
        self, lr_tensor: torch.Tensor, inbox_patch_index: torch.Tensor
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
        # Ensure input is on the correct device
        lr_tensor = lr_tensor.to(self.device)

        # Normalize input using batch info
        if self.batch_info.center is not None and self.batch_info.scales is not None:
            # Convert lists to tensors if necessary
            center = self.batch_info.center
            scales = self.batch_info.scales

            if isinstance(center, list):
                center = torch.tensor(center, device=self.device)
            if isinstance(scales, list):
                scales = torch.tensor(scales, device=self.device)

            # Ensure proper broadcasting
            if center.dim() == 1:
                center = center.view(-1, 1)  # (channels, 1)
            if scales.dim() == 1:
                scales = scales.view(-1, 1)  # (channels, 1)

            lr_normalized = (lr_tensor - center) / scales
        else:
            lr_normalized = lr_tensor

        # Get global low resolution for lat-lon context
        global_lr = self.regrid_to_latlon(lr_normalized.double())[None,].to(self.device)

        # Regrid to high resolution
        lr_hr = self.regrid(lr_normalized)

        # Prepare tensors for diffusion
        in_channels = lr_hr.shape[0]
        latents = torch.randn_like(lr_hr)
        latents = latents.reshape((in_channels, -1))
        lr_hr = lr_hr.reshape((in_channels, -1))

        # Add batch dimension
        latents = latents[None,].to(self.device)
        lr_hr = lr_hr[None,].to(self.device)
        with torch.no_grad():
            # Define denoiser function
            def denoiser(x, t):
                return (
                    patchify.apply_on_patches(
                        self.net,
                        patch_size=self.patch_size,
                        overlap_size=self.overlap_size,
                        x_hat=x,
                        x_lr=lr_hr,
                        t_hat=t,
                        class_labels=None,
                        batch_size=128,
                        global_lr=global_lr,
                        inbox_patch_index=inbox_patch_index,
                        device=self.device,
                    )
                    .to(torch.float64)
                    .to(self.device)
                )

            # Set denoiser attributes
            denoiser.sigma_max = self.net.sigma_max
            denoiser.sigma_min = self.net.sigma_min
            denoiser.round_sigma = self.net.round_sigma

            # Run EDM sampler
            pred = edm_sampler(
                denoiser,
                latents,
                num_steps=self.num_steps,
                sigma_max=self.sigma_max,
            )
            pred = self.denormalize(pred)
            # Reshape back to original format
            pred = pred.reshape((in_channels, -1))
            return pred


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
                model = (
                    c.read_model(
                        map_location=None,
                        allow_second_order_derivatives=allow_second_order_derivatives,
                    )
                    .cuda()
                    .eval()
                )
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


def load(model: str, root="") -> CBottle3d:
    root = root or environment.CHECKPOINT_ROOT
    if model == "cbottle-3d-moe":
        checkpoints = "training-state-000512000.checkpoint,training-state-002048000.checkpoint,training-state-009856000.checkpoint".split(
            ","
        )
        rundir = "v6data-mChan192"
        paths = [os.path.join(root, rundir, c) for c in checkpoints]
        return CBottle3d.from_pretrained(paths, sigma_thresholds=(100.0, 10.0))
    elif model == "cbottle-3d":
        path = os.path.join(
            root, "v6data-mChan192", "training-state-009856000.checkpoint"
        )
        return CBottle3d.from_pretrained(path)
    elif model == "cbottle-3d-tc":
        path = os.path.join(
            root,
            "test_classifier_after_main_merge2",
            "training-state-010240000.checkpoint",
        )
        return CBottle3d.from_pretrained(path)
    elif model == "cbottle-3d-moe-tc":
        checkpoints = "training-state-000512000.checkpoint,training-state-002048000.checkpoint,training-state-009856000.checkpoint".split(
            ","
        )
        paths = [os.path.join(root, c) for c in checkpoints]
        classifier_path = os.path.join(root, "training-state-002176000.checkpoint")
        return CBottle3d.from_pretrained(
            paths,
            sigma_thresholds=(100.0, 10.0),
            separate_classifier_path=classifier_path,
            allow_second_order_derivatives=True,
        )
    raise ValueError(model)
