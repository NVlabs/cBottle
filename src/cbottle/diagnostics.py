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
from cbottle.datasets.base import SpatioTemporalDataset, BatchInfo
from earth2grid import healpix
import cbottle.diffusion_samplers as diffusion_samplers
import torch
import matplotlib.pyplot as plt
import cartopy.crs
import cbottle.loss
import numpy as np


def create_regular_grid_in_projection(projection, nx, ny):
    """
    Create a regular grid of lat-lon coordinates in a given Cartopy projection.

    Parameters:
    projection (cartopy.crs.Projection): The desired Cartopy projection
    resolution (float): The grid resolution in projection units

    Returns:
    tuple: Two 2D arrays, one for latitudes and one for longitudes
    """
    # Get the projection's limits
    x_min, x_max, y_min, y_max = projection.x_limits + projection.y_limits

    # Create a regular grid in the projection coordinates
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)

    # Transform the gridded coordinates back to lat-lon
    geodetic = cartopy.crs.Geodetic()
    transformed = geodetic.transform_points(projection, xx, yy)

    lons = transformed[..., 0]
    lats = transformed[..., 1]

    # Filter out invalid points (those outside the projection's valid domain)
    valid = np.logical_and(np.isfinite(lons), np.isfinite(lats))
    lons[~valid] = np.nan
    lats[~valid] = np.nan

    return lats, lons, xx, yy


def visualize(x, clabel=None, pos=111, title=None, **kw):
    # TODO replace with cbottle.visualization
    hpx = healpix.Grid(healpix.npix2level(x.shape[-1]))
    crs = cartopy.crs.Robinson()
    lat, lon, xx, yy = create_regular_grid_in_projection(crs, 256, 512)
    mask = ~np.isnan(lat)
    latm = lat[mask]
    lonm = lon[mask]
    regrid = hpx.get_bilinear_regridder_to(latm, lonm)
    regrid.to(x)
    out = torch.zeros_like(torch.tensor(lat)).to(x)
    x = torch.as_tensor(x)
    out[mask] = regrid(x)
    out[~mask] = torch.nan
    ax = plt.subplot(pos, projection=crs)
    im = ax.pcolormesh(xx, yy, out.cpu(), transform=crs, **kw)
    ax.coastlines()
    cb = plt.colorbar(im, orientation="horizontal")
    ax.set_title(title)
    if clabel:
        cb.set_label(clabel)


def diagnostics(
    loader: SpatioTemporalDataset,
    net,
    writer,
    cur_nimg,
    plot_channels=["tcwv", "t850", "z500", "r925", "t2m"],
):
    batch = next(iter(loader))
    images = sample_images(net, batch, loader.batch_info)
    log_images(writer, cur_nimg, images=images, plot_channels=plot_channels)


def sample_regression(net, batch, batch_info: BatchInfo):
    """
    example of output format
    {"generated": {"t850": [n, npix]}}}
    """
    images, labels, condition = batch
    with torch.no_grad():
        labels = labels.cuda()
        condition = condition.cuda()
        images = images.cuda()
        hpx = net.domain._grid

        sigma = cbottle.loss.RegressLoss.SIGMA * torch.ones(
            [images.shape[0], 1, 1, 1], device=images.device
        )
        out = net(torch.zeros_like(images), sigma, labels, condition=condition)
        gen = batch_info.denormalize(out)
        hpx = net.domain._grid

        def prepare(x):
            ring_order = hpx.reorder(healpix.PixelOrder.RING, x)
            return {batch_info.channels[c]: ring_order[:, c] for c in range(x.shape[1])}

        return dict(predicted=prepare(gen))


def call_regression(net, class_labels, condition):
    sigma = cbottle.loss.RegressLoss.SIGMA * torch.ones(
        [condition.shape[0], 1, 1, 1], device=condition.device
    )
    latent = torch.zeros(
        [condition.shape[0], net.img_channels, net.time_length, net.domain.numel()],
        device=condition.device,
    )
    out = net(latent, sigma, class_labels, condition=condition)
    return out


def curry_denoiser(net, *args, **kwargs):
    def D(x, t):
        out = net(x, t, *args, **kwargs)  # Ignore classifier output
        return out.out

    D.sigma_min = net.sigma_min
    D.sigma_max = net.sigma_max
    D.round_sigma = net.round_sigma
    return D


def sample_from_condition(
    net,
    batch,
    *,
    batch_info: BatchInfo,
    regression: bool = False,
    sigma_max: float,
    sigma_min: float,
    seeds=None,
    t_bounds: tuple[
        int, int
    ] = None,  # (start, end) for MP, uses net.time_length for full
):
    """
    example of output format
    {"generated": {"t850": [n, npix]}}}
    """
    with torch.no_grad():
        hpx = net.domain._grid

        batch = batch.copy()

        condition = batch.pop("condition")
        # pop this if it exists
        batch.pop("classifier_labels", None)
        # TODO this logic is wrong
        batch.pop("target")
        labels = batch.pop("labels")
        batch.pop("mask", None)

        if regression:
            out = call_regression(
                net, class_labels=labels, condition=condition, **batch
            )
        else:
            from cbottle.diffusion_samplers import StackedRandomGenerator

            device = condition.device

            if seeds is None:
                rnd = torch
            else:
                rnd = StackedRandomGenerator(device, seeds=seeds)

            # Generate full latents and slice to keep all mp ranks with same rng state
            t_full = net.time_length
            t_start, t_end = t_bounds if t_bounds is not None else (0, t_full)

            full_latents = rnd.randn(
                (labels.shape[0], net.img_channels, t_full, net.domain.numel()),
                device=device,
            )
            latents = full_latents[:, :, t_start:t_end, :]

            def randn_like(x):
                full_noise = torch.randn(
                    x.shape[0],
                    x.shape[1],
                    t_full,
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                )
                return full_noise[:, :, t_start:t_end, :]

            out = diffusion_samplers.edm_sampler(
                curry_denoiser(net, class_labels=labels, condition=condition, **batch),
                latents,
                randn_like=randn_like,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
            )

        gen = batch_info.denormalize(out)

        def prepare(x):
            ring_order = hpx.reorder(healpix.PixelOrder.RING, x)
            return {batch_info.channels[c]: ring_order[:, c] for c in range(x.shape[1])}

        return prepare(gen)


def sample_images(net, batch, batch_info: BatchInfo, seeds=None):
    """
    example of output format
    {"generated": {"t850": [n, npix]}}}
    """
    images, labels, condition = batch
    with torch.no_grad():
        labels = labels.cuda()
        condition = condition.cuda()
        images = images.cuda()
        hpx = net.domain._grid

        from cbottle.diffusion_samplers import StackedRandomGenerator

        device = condition.device

        if seeds is None:
            rnd = torch
        else:
            rnd = StackedRandomGenerator(device, seeds=seeds)

        latents = rnd.randn(
            (images.shape[0], net.img_channels, net.time_length, net.domain.numel()),
            device=device,
        )

        out = diffusion_samplers.edm_sampler(
            net,
            latents,
            class_labels=labels,
            randn_like=torch.randn_like,
            condition=condition,
            sigma_max=500,
        )

        sigma = torch.tensor(10.0, device=device)

        D = net(images + sigma * latents, net.round_sigma(sigma), labels, condition)

        gen = batch_info.denormalize(out)
        truth = batch_info.denormalize(images)
        denoised = batch_info.denormalize(D)

        hpx = net.domain._grid

        def prepare(x):
            ring_order = hpx.reorder(healpix.PixelOrder.RING, x)
            return {batch_info.channels[c]: ring_order[:, c] for c in range(x.shape[1])}

        return dict(
            generated=prepare(gen),
            truth=prepare(truth),
            denoised=prepare(denoised),
        )


def log_images(writer, global_step: int, images, plot_channels=None):
    for source in images:
        for channel in images[source]:
            if plot_channels is not None and channel not in plot_channels:
                continue

            array = images[source][channel]  # (n, t, x)
            visualize(array[0, 0])
            fig = plt.gcf()
            writer.add_figure(f"images/{channel}/{source}", fig, global_step)
