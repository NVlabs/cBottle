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
# TODO figure out how to make it deterministic - setting the random seed doesn't work as expected
import logging
import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import xarray as xr
import dask.array as da
import dask.delayed
import torch
from xpublish import Rest
import earth2grid
from cbottle.checkpointing import Checkpoint
from cbottle.datasets import dataset_3d
from cbottle.denoiser_factories import get_denoiser, DenoiserType
from cbottle.diffusion_samplers import edm_sampler_from_sigma, StackedRandomGenerator

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    state_path: str
    hpx_level: int = 6
    sigma_max: float = 80.0
    denoiser_type: DenoiserType = DenoiserType.standard
    batch_size: int = 4
    seed: Optional[int] = 0
    bf16: bool = False

class LazyHealpixInference:
    """Lazy xarray interface for HealpixNet inference"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Load model
        with Checkpoint(config.state_path) as checkpoint:
            self.net = checkpoint.read_model()
        
        self.net.eval()
        self.net.requires_grad_(False)
        self.net.float()
        self.net.cuda()
        
        # Get dataset info
        self.dataset = dataset_3d.get_dataset(
            split="",
            dataset="amip",  # Default to ERA5
            sst_input=True,
            infinite=False,
            shuffle=False,
        )
        
        # Initialize HEALPix grid
        self.npix = 12 * (2**config.hpx_level) ** 2
        self.hpx = earth2grid.healpix.Grid(config.hpx_level)


    @dask.delayed
    def _run_inference(self, times: List[np.datetime64]) -> xr.Dataset:
        """Run inference for given timestamps"""
        logging.info(f"Running inference for {len(times)} times")
        
        device = next(self.net.parameters()).device
        batch_size = self.config.batch_size
        
        # Process in batches
        results = []
        for i in range(0, len(times), batch_size):
            batch_times = times[i:i + batch_size]
            
            # Get conditions from dataset
            batch_data = [self.dataset.sel_time(t) for t in batch_times]
            
            # Prepare inputs
            condition = torch.stack([d["condition"] for d in batch_data]).to(device)
            labels = torch.stack([d["labels"] for d in batch_data]).to(device)
            second_of_day = torch.stack([d["second_of_day"] for d in batch_data]).to(device)
            day_of_year = torch.stack([d["day_of_year"] for d in batch_data]).to(device)

            # just used for the mask
            images = torch.stack([d["target"] for d in batch_data]).to(device)

            
            # Setup random generator
            if self.config.seed is None:
                rnd = torch
            else:
                seeds = [self.config.seed + int(t.astype('datetime64[s]').astype('int')) for t in batch_times]
                rnd = StackedRandomGenerator(device, seeds=seeds)
            
            # Generate initial noise
            latents = rnd.randn(
                (len(batch_times), self.net.img_channels, self.net.time_length, self.net.domain.numel()),
                device=device
            )
            
            xT = latents * self.config.sigma_max
            
            # Get denoiser
            D = get_denoiser(
                net=self.net,
                images=images,
                labels=labels,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
                denoiser_type=self.config.denoiser_type,
                sigma_max=self.config.sigma_max,
            )
            
            # Run inference
            logging.info(f"Running inference for {i} to {i + batch_size}")
            with torch.autocast("cuda", enabled=self.config.bf16, dtype=torch.bfloat16):
                with torch.no_grad():
                    out = edm_sampler_from_sigma(
                        D,
                        xT,
                        randn_like=rnd.randn_like,
                        sigma_max=int(self.config.sigma_max),
                    )
            
            # Denormalize and convert to numpy
            out = self.dataset.batch_info.denormalize(out)
            results.append(out.cpu().float().numpy())
            
        # Combine results
        all_results = np.concatenate(results, axis=0)
        out = all_results.squeeze(2) # squeeze the window dim
        logging.info(f"Inference done")
        return out

    def create_dataset(self, start_time: np.datetime64, end_time: np.datetime64, freq: str = "1H") -> xr.Dataset:
        """Create a lazy xarray dataset for the given time range"""
        
        # Generate time coordinates
        times = np.arange(start_time, end_time, np.timedelta64(1, freq[-1]) * int(freq[:-1]))
        
        # Create lazy dask arrays for each variable
        chunks = {"time": self.config.batch_size, "pixel": self.npix}

        
        lazy_arrays = [
             da.from_delayed( self._run_inference(times[i:i+self.config.batch_size]),
                shape=(self.config.batch_size, len(self.dataset.batch_info.channels), self.npix),
                dtype=np.float32
             )
                 for i in range(0, len(times), self.config.batch_size)
        ]
        array = da.concatenate(lazy_arrays, axis=0)
        
        # Create dataset
        ds = xr.Dataset(
            {"data": (["time", "channel", "pixel"], array)},
            # coords={
            #     "time": times,
            #     "channel": self.dataset.batch_info.channels,
            #     "pixel": np.arange(self.npix),
            # }
        )
        # Add grid mapping
        ds.attrs["grid_mapping"] = "healpix"
        ds["healpix"] = xr.DataArray(
            data=np.array([2**self.config.hpx_level]),
            attrs={
                "grid_mapping_name": "healpix",
                "nside": 2**self.config.hpx_level,
                "ordering": "ring",
            }
        )
        
        return ds

def main():
    """Main function to start the xarray server"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    parser.add_argument("--start-time", type=str, required=True, help="Start time (YYYY-MM-DD)")
    parser.add_argument("--end-time", type=str, required=True, help="End time (YYYY-MM-DD)")
    parser.add_argument("--freq", default="1h", help="Time frequency (e.g. 1H, 3H)")
    parser.add_argument("--hpx-level", type=int, default=6, help="HEALPix level")
    parser.add_argument("--sigma-max", type=float, default=80.0, help="Maximum sigma value")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create inference config
    config = InferenceConfig(
        state_path=args.state_path,
        hpx_level=args.hpx_level,
        sigma_max=args.sigma_max,
        batch_size=args.batch_size,
        seed=args.seed,
        bf16=args.bf16,
    )
    
    # Create lazy dataset
    inference = LazyHealpixInference(config)
    ds = inference.create_dataset(
        np.datetime64(args.start_time),
        np.datetime64(args.end_time),
        args.freq
    )
    
    # Create REST server
    dask.config.set(scheduler="single-threaded")
    rest = Rest({"inference": ds})
    
    # Start server
    rest.serve(host=args.host, port=args.port)

if __name__ == "__main__":
    main() 