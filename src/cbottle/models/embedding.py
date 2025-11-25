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
import math


class PositionalEmbedding(torch.nn.Module):
    """Timestep embedding used in the DDPM++ and ADM architectures.


    f = (1/M)^(i / N)
    [cos(f_i x), sin(f_i x)] for i =0,...,N - 1

    sup wavelength = sup  2 pi / f = 2 pi 1 / inf f = 2 pi / (1 / M) = 2 pi M

    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    """Timestep embedding used in the NCSN++ architecture."""

    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * math.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FrequencyEmbedding(torch.nn.Module):
    """Periodic Embedding.

    Useful for inputs defined on the circle [0, 2pi)
    """

    def __init__(self, num_channels):
        super().__init__()
        self.register_buffer(
            "freqs", torch.arange(1, num_channels + 1), persistent=False
        )

    def forward(self, x):
        freqs = self.freqs[None, :, None, None]
        x = x[:, None, :, :]
        x = x * (2 * math.pi * freqs).to(x.dtype)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CalendarEmbedding(torch.nn.Module):
    """Time embedding assuming 365.25 day years

    Args:
        day_of_year: (n, t)
        second_of_day: (n, t)
    Returns:
        (n, embed_channels * 4, t, x)

    """

    def __init__(self, lon, embed_channels: int, include_legacy_bug: bool = False):
        """
        Args:
            include_legacy_bug: Provided for backwards compatibility
                with existing checkpoints. If True, use the incorrect formula
                for local_time (hour - lon) instead of the correct formula (hour + lon)
        """
        super().__init__()
        self.register_buffer("lon", lon, persistent=False)
        self.embed_channels = embed_channels
        self.embed_second = FrequencyEmbedding(embed_channels)
        self.embed_day = FrequencyEmbedding(embed_channels)
        self.out_channels = embed_channels * 4
        self.include_legacy_bug = include_legacy_bug

    def forward(self, day_of_year, second_of_day):
        if second_of_day.shape != day_of_year.shape:
            raise ValueError()

        if self.include_legacy_bug:
            local_time = (second_of_day.unsqueeze(2) - self.lon * 86400 // 360) % 86400
        else:
            local_time = (second_of_day.unsqueeze(2) + self.lon * 86400 // 360) % 86400

        a = self.embed_second(local_time / 86400)
        doy = day_of_year.unsqueeze(2)
        b = self.embed_day((doy / 365.25) % 1)
        a, b = torch.broadcast_tensors(a, b)
        return torch.concat([a, b], dim=1)  # (n c x)


class CalendarEmbeddingOnlyDay(torch.nn.Module):
    """Time embedding assuming 365.25 day years. Only uses day_of_year for use cases with daily resolution, second_of_day arg preserved for compatibility.

    Args:
        day_of_year: (n, t)
        second_of_day: (n, t)
    Returns:
        (n, embed_channels * 2, t, x)

    """

    def __init__(self, lon, embed_channels: int, include_legacy_bug: bool = False):
        """
        Args:
            include_legacy_bug: Provided for backwards compatibility
                with existing checkpoints. If True, use the incorrect formula
                for local_time (hour - lon) instead of the correct formula (hour + lon)

            only day_of_year used, other args preserved for compatibility with SongUNet.
        """
        super().__init__()
        self.register_buffer("lon", lon, persistent=False)
        self.embed_channels = embed_channels
        self.embed_day = FrequencyEmbedding(embed_channels)
        self.out_channels = embed_channels * 2

    def forward(self, day_of_year, second_of_day):
        doy = day_of_year.unsqueeze(2)
        b = self.embed_day((doy / 365.25) % 1)
        # Ensure spatial broadcast to match X = self.lon.numel()
        if b.dim() == 3:  # (N, C, T) -> (N, C, T, 1)
            b = b.unsqueeze(-1)
        if b.shape[-1] == 1:
            b = b.expand(-1, -1, -1, self.lon.numel())
        return b
