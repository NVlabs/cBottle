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
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Correlated latent generator for distributed inference with AR(1) process.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CorrelatedLatentGenerator:
    """
    AR(1) correlated latent generator for distributed inference.

    This class generates correlated latents using an AR(1) process that maintains
    temporal correlation across time steps, even in distributed inference scenarios
    where different processes handle different time windows.

    The AR(1) process is defined as:
    x_t = φ * x_{t-1} + sqrt(1-φ²) * ε_t

    Where φ controls the correlation strength and ε_t is white noise.
    """

    def __init__(
        self,
        device: torch.device,
        correlation_half_life: float = 8.0,
        seed: Optional[int] = None,
        rank: int = 0,
    ):
        """
        Initialize the correlated latent generator.

        Args:
            device: Device to generate latents on
            correlation_half_life: Half-life for AR(1) correlation (time steps)
            seed: Random seed for reproducibility (shared across all ranks)
            rank: Process rank in distributed setup
        """
        self.device = device
        # For AR(1) process: half_life = -ln(2) / ln(φ), so φ = 2^(-1/half_life)
        self.phi = torch.tensor(2.0 ** (-1 / correlation_half_life), device=device)
        self.rank = rank

        if seed is not None:
            # Use the SAME seed for all ranks to ensure shared random sequence
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(seed)
        else:
            self.rng = None

        self.current_state: Optional[torch.Tensor] = None
        self.time_step = 0

        logger.info(
            f"Rank {rank}: Initialized CorrelatedLatentGenerator with φ={self.phi:.4f} (shared seed={seed})"
        )

    def generate_latents(
        self,
        batch_size: int,
        channels: int,
        time_length: int,
        spatial_dims: int,
        time_offset: int = 0,
    ) -> torch.Tensor:
        """
        Generate correlated latents for a batch.

        Args:
            batch_size: Number of samples in batch
            channels: Number of channels
            time_length: Time dimension (usually 1)
            spatial_dims: Spatial dimensions
            time_offset: Global time offset for this batch

        Returns:
            Correlated latents tensor of shape (batch_size, channels, time_length, spatial_dims)
        """

        # Initialize state if needed
        if self.current_state is None or time_offset != self.time_step:
            self._initialize_state(channels, time_length, spatial_dims, time_offset)

        latents = torch.zeros(
            (batch_size, channels, time_length, spatial_dims), device=self.device
        )

        for b in range(batch_size):
            current_time_step = time_offset + b

            # Generate noise
            if self.rng is not None:
                noise = torch.randn(
                    (channels, time_length, spatial_dims),
                    device=self.device,
                    generator=self.rng,
                )
            else:
                noise = torch.randn(
                    (channels, time_length, spatial_dims), device=self.device
                )

            # AR(1) process: x_t = φ * x_{t-1} + sqrt(1-φ²) * ε_t
            self.current_state = self.current_state * self.phi + noise * torch.sqrt(
                1 - self.phi**2
            )

            latents[b] = self.current_state
            self.time_step = current_time_step + 1

        return latents

    def _initialize_state(
        self,
        channels: int,
        time_length: int,
        spatial_dims: int,
        time_step: int,
    ):
        """
        Initialize AR(1) state, handling time jumps.

        For time jumps, we sample from the stationary distribution N(0,1).
        We use a deterministic seed based on the original seed and time_step
        to ensure reproducibility while avoiding expensive RNG advancement.

        Note: This means there's no temporal correlation ACROSS time jumps
        (e.g., between different ranks' starting points), but this is fine
        since each rank processes independent time blocks.
        """
        if time_step == 0 or self.rng is None:
            # Generate initial state with original RNG
            if self.rng is not None:
                self.current_state = torch.randn(
                    (channels, time_length, spatial_dims),
                    device=self.device,
                    generator=self.rng,
                )
            else:
                self.current_state = torch.randn(
                    (channels, time_length, spatial_dims), device=self.device
                )
        else:
            # For time jumps, use a deterministic seed based on original seed + time_step
            jump_generator = torch.Generator(device=self.device)
            # Use original seed from self.rng + time_step for deterministic initialization
            original_seed = self.rng.initial_seed()
            jump_seed = (original_seed + time_step) % (
                2**63 - 1
            )  # Keep within valid range
            jump_generator.manual_seed(jump_seed)

            # Sample from stationary distribution
            self.current_state = torch.randn(
                (channels, time_length, spatial_dims),
                device=self.device,
                generator=jump_generator,
            )

            logger.debug(
                f"Rank {self.rank}: Initialized state at time step {time_step} "
                f"using seed {jump_seed}"
            )

        self.time_step = time_step

    def reset(self):
        """Reset the generator state."""
        self.current_state = None
        self.time_step = 0
        logger.debug(f"Rank {self.rank}: Reset generator state")

    def get_correlation_info(self) -> dict:
        """
        Get information about the current correlation settings.

        Returns:
            Dictionary with correlation parameters
        """
        return {
            "phi": self.phi.item(),
            "correlation_half_life": -torch.log(torch.tensor(2.0)).item()
            / torch.log(self.phi).item(),
            "rank": self.rank,
            "current_time_step": self.time_step,
            "has_state": self.current_state is not None,
        }
