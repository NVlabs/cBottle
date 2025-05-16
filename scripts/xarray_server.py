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
from dataclasses import dataclass
from typing import Optional, List, Any
import functools
import socket
import consul
import atexit

import numpy as np
import xarray as xr
import dask.array as da
import dask.delayed
import dask
import torch
from xpublish import Rest

from dask.distributed import Client
import earth2grid
from cbottle.checkpointing import Checkpoint
from cbottle.datasets import dataset_3d
from cbottle.denoiser_factories import get_denoiser, DenoiserType
from cbottle.diffusion_samplers import edm_sampler_from_sigma, StackedRandomGenerator
import queue
import threading

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


def get_dataset():
    return dataset_3d.get_dataset(
        split="",
        dataset="amip",  # Default to ERA5
        sst_input=True,
        infinite=False,
        shuffle=False,
    )


def load_model(state_path: str):
    with Checkpoint(state_path) as checkpoint:
        net = checkpoint.read_model()

    net.eval()
    net.cuda()
    net.requires_grad_(False)
    net.float()
    return net


queue = queue.Queue()
_model = None
_dataset = None

_counter = threading.Semaphore(2)

@dask.delayed
def run_inference(
    times: List[np.datetime64],
    config: InferenceConfig,
) -> xr.Dataset:
    with _counter:
        return _run_inference(times, config)

def _run_inference(
    times: List[np.datetime64],
    config: InferenceConfig,
) -> xr.Dataset:
    """Run inference for given timestamps

    Args:
        times: List of timestamps to run inference for
        net: Neural network model
        dataset: Dataset for getting conditions
        config: Configuration parameters

    Returns:
        xr.Dataset: Inference results
    """
    logging.info(f"Running inference for {len(times)} times")

    # cache the model and dataset
    global _model, _dataset
    if _model is None:
        _model = load_model(config.state_path)

    if _dataset is None:
        _dataset = get_dataset()

    net = _model
    dataset = _dataset

    device = next(net.parameters()).device
    batch_size = config.batch_size

    # Process in batches
    results = []
    for i in range(0, len(times), batch_size):
        batch_times = times[i : i + batch_size]

        # Get conditions from dataset
        batch_data = [dataset.sel_time(t) for t in batch_times]

        # Prepare inputs
        condition = torch.stack([d["condition"] for d in batch_data]).to(device)
        labels = torch.stack([d["labels"] for d in batch_data]).to(device)
        second_of_day = torch.stack([d["second_of_day"] for d in batch_data]).to(device)
        day_of_year = torch.stack([d["day_of_year"] for d in batch_data]).to(device)

        # just used for the mask
        images = torch.stack([d["target"] for d in batch_data]).to(device)

        # Setup random generator
        if config.seed is None:
            rnd = torch
        else:
            seeds = [
                config.seed + int(t.astype("datetime64[s]").astype("int"))
                for t in batch_times
            ]
            rnd = StackedRandomGenerator(device, seeds=seeds)

        # Generate initial noise
        latents = rnd.randn(
            (
                len(batch_times),
                net.img_channels,
                net.time_length,
                net.domain.numel(),
            ),
            device=device,
        )

        xT = latents * config.sigma_max

        # Get denoiser
        D = get_denoiser(
            net=net,
            images=images,
            labels=labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
            denoiser_type=config.denoiser_type,
            sigma_max=config.sigma_max,
        )

        # Run inference
        logging.info(f"Running inference for {i} to {i + batch_size}")
        with torch.autocast("cuda", enabled=config.bf16, dtype=torch.bfloat16):
            with torch.no_grad():
                out = edm_sampler_from_sigma(
                    D,
                    xT,
                    randn_like=rnd.randn_like,
                    sigma_max=int(config.sigma_max),
                )

        # Denormalize and convert to numpy
        out = dataset.batch_info.denormalize(out)
        results.append(out.cpu().float().numpy())

    # Combine results
    all_results = np.concatenate(results, axis=0)
    out = all_results.squeeze(2)  # squeeze the window dim
    logging.info("Inference done")
    return out


class LazyHealpixInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        # Initialize HEALPix grid
        self.npix = 12 * (2**config.hpx_level) ** 2
        self.hpx = earth2grid.healpix.Grid(config.hpx_level)
        self.dataset = get_dataset()
        self.net = load_model(config.state_path)

    def create_dataset(
        self, start_time: np.datetime64, end_time: np.datetime64, freq: str = "1H"
    ) -> xr.Dataset:
        """Create a lazy xarray dataset for the given time range"""

        # Generate time coordinates
        times = np.arange(
            start_time, end_time, np.timedelta64(1, freq[-1]) * int(freq[:-1])
        )

        lazy_arrays = [
            da.from_delayed(
                run_inference(
                    times[i : i + self.config.batch_size],
                    config=self.config,
                ),
                shape=(
                    self.config.batch_size,
                    len(self.dataset.batch_info.channels),
                    self.npix,
                ),
                dtype=np.float32,
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
            },
        )

        return ds


def register_with_consul(host: str, port: int, service_name: str = "xarray"):
    """Register the service with Consul"""
    try:
        c = consul.Consul("login18")
        service_id = f"{service_name}-{host}-{port}"

        # Get the actual IP address
        actual_ip = socket.gethostbyname(host)

        # Register the service
        c.agent.service.register(
            name=service_name,
            service_id=service_id,
            address=actual_ip,
            port=port,
            tags=["xarray", "inference"],
            check=consul.Check.http(
                f"http://{actual_ip}:{port}/health", interval="30s", timeout="10s"
            ),
        )

        # Register cleanup on exit
        atexit.register(lambda: c.agent.service.deregister(service_id))

        logger.info(f"Registered service {service_name} with Consul")
    except Exception as e:
        logger.error(f"Failed to register with Consul: {e}")


def main():
    """Main function to start the xarray server"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    parser.add_argument(
        "--start-time", type=str, required=True, help="Start time (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-time", type=str, required=True, help="End time (YYYY-MM-DD)"
    )
    parser.add_argument("--freq", default="1h", help="Time frequency (e.g. 1H, 3H)")
    parser.add_argument("--hpx-level", type=int, default=6, help="HEALPix level")
    parser.add_argument(
        "--sigma-max", type=float, default=80.0, help="Maximum sigma value"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for inference"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument(
        "--scheduler-file", type=str, default="single-threaded", help="Dask scheduler"
    )
    parser.add_argument("--consul-host", default="localhost", help="Consul host")
    parser.add_argument("--consul-port", type=int, default=8500, help="Consul port")
    parser.add_argument(
        "--service-name", default="xarray", help="Service name for Consul registration"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    if not args.scheduler_file == "single-threaded":
        Client(scheduler_file=args.scheduler_file)

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
    with dask.annotate(resources={"gpu": 1.0}):
        ds = inference.create_dataset(
            np.datetime64(args.start_time), np.datetime64(args.end_time), args.freq
        )

    # Create REST server
    rest = Rest({"inference": ds})

    # Add health check endpoint
    @rest.app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # Register with Consul
    register_with_consul(args.host, args.port, args.service_name)

    # Start server
    rest.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
