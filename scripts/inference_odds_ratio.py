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
"""Batch AMIP driver for :meth:`CBottle3d._calculate_odds_ratio_full`.

Paper-scale wrapper around the public odds-ratio API. Writes per-rank NetCDF
(forward samples) and a wide CSV whose column names match
``importance_sample.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
import time

import pandas as pd
import numpy as np
import torch
import tqdm
import warnings

import earth2grid

import cbottle.distributed as dist
import cbottle.inference
import cbottle.netcdf_writer
from cbottle.datasets.dataset_3d import get_dataset

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

UNIX_TO_1900_S = 2208988800


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AMIP odds-ratio / guided-sample driver (public cBottle API)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("output_path", type=str)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Root containing cBottle-3d/ and cBottle-3d-tc/ training-state files. "
        "Empty uses CHECKPOINT_ROOT from the environment.",
    )
    p.add_argument("--guidance-scale-prefactor", type=float, default=0.1)
    p.add_argument("--start-time", type=str, default="1979-09-01T00:00:00")
    p.add_argument("--end-time", type=str, default="2018-09-30T00:00:00")
    p.add_argument(
        "--month",
        type=int,
        default=9,
        help="Keep only this calendar month (paper uses September). 0 = all months.",
    )
    p.add_argument("--forward-guidance", action="store_true")
    p.add_argument("--run-backward-sampling", action="store_true")
    p.add_argument("--compute-forward-divergences", action="store_true")
    p.add_argument("--tc-lon", type=float, default=-80.0)
    p.add_argument("--tc-lat", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma-max", type=float, default=200.0)
    p.add_argument("--num-steps", type=int, default=36)
    p.add_argument(
        "--extra-steps",
        type=int,
        default=25,
        help="Extra sigma nodes in [guidance-on, guidance-off]. 0 = no densification.",
    )
    p.add_argument("--guidance-on", type=float, default=15.0)
    p.add_argument("--guidance-off", type=float, default=20.0)
    p.add_argument(
        "--extra-steps-window",
        type=str,
        default="",
        help="Sigma interval 'lo,hi' to densify with --extra-steps. "
        "Empty uses [guidance-on, guidance-off].",
    )
    p.add_argument("--divergence-samples", type=int, default=3)
    p.add_argument(
        "--x0-dir",
        type=str,
        default="",
        help="Directory of RING NetCDFs. If set, skip forward sampling and "
        "score saved states (two backward phases only).",
    )
    p.add_argument(
        "--skip-write-nc",
        action="store_true",
        help="Do not write forward samples (implied by --x0-dir).",
    )
    p.add_argument(
        "--save-divergence-traces",
        action="store_true",
        help="Write per-sigma divergence traces under <output>/traces, one file "
        "per phase that ran (forward, and backward when --run-backward-sampling).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Keep only the first N times after month filtering. 0 = all.",
    )
    p.add_argument(
        "--keep-gradient-checkpointing",
        action="store_true",
        help="Leave training-time checkpointing on (slower Hutchinson backwards).",
    )
    return p.parse_args()


def _times(args: argparse.Namespace) -> list:
    series = pd.date_range(
        start=args.start_time, end=args.end_time, freq="6h"
    ).to_series()
    if args.month:
        series = series.loc[lambda x: x.dt.month == args.month]
    times = series.tolist()
    if args.max_samples:
        times = times[: args.max_samples]
    return times


def _index_x0_dir(x0_dir: str) -> dict:
    """Map UTC timestamp string -> (nc_path, time_index)."""
    import glob

    import cftime
    import netCDF4 as nc

    index = {}
    for path in sorted(glob.glob(os.path.join(x0_dir, "*.nc"))):
        with nc.Dataset(path, "r") as ds:
            units = ds["time"].units
            calendar = ds["time"].calendar
            for i, raw in enumerate(ds["time"][:]):
                dt = cftime.num2date(raw, units=units, calendar=calendar)
                key = dt.strftime("%Y-%m-%d %H:%M:%S")
                index[key] = (path, int(i))
    return index


def _load_x0_ring(path: str, time_index: int, channels: list[str]) -> torch.Tensor:
    import netCDF4 as nc

    fields = []
    with nc.Dataset(path, "r") as ds:
        for ch in channels:
            fields.append(np.asarray(ds[ch][time_index], dtype=np.float32))
    stacked = np.stack(fields, axis=0)
    return torch.from_numpy(stacked)[None, :, None, :]


def _invert_written_target(model, x_ring: torch.Tensor) -> torch.Tensor:
    """Undo ``NetCDFWriter.write_target``: RING, denormalized -> model latents."""
    ring = earth2grid.healpix.Grid(6, pixel_order=earth2grid.healpix.PixelOrder.RING)
    x_nest = ring.reorder(model.output_grid.pixel_order, x_ring)
    return model._reorder(model._normalize(x_nest))


def _disable_gradient_checkpointing(model) -> int:
    disabled = 0
    modules = list(model.net.modules())
    if model.separate_classifier is not None:
        modules += list(model.separate_classifier.modules())
    for m in modules:
        if getattr(m, "checkpoint", False):
            m.checkpoint = False
            disabled += 1
    return disabled


def _unix_to_datetime_str(ts) -> str:
    if torch.is_tensor(ts):
        ts = ts.item()
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _unix_to_netcdf_1900(timestamps: torch.Tensor) -> torch.Tensor:
    return timestamps + UNIX_TO_1900_S


def _row_from_result(
    result,
    *,
    rank: int,
    batch_idx: int,
    timestamp: str,
    guided: bool,
    gpu_seconds: float,
) -> dict:
    fwd_phase = "forward" if guided else "forward_no_guidance"
    row = {
        "rank": rank,
        "batch_idx": batch_idx,
        "timestamp": timestamp,
        f"guidance_likelihood_{fwd_phase}": result.forward_guidance_div_integral,
        f"score_likelihood_{fwd_phase}": result.forward_score_div_integral,
        "gpu_seconds": gpu_seconds,
    }
    if result.backward_gaussian_logp is not None:
        row.update(
            {
                "gaussian_logp_backward_with_guidance": result.backward_gaussian_logp,
                "gaussian_logp_backward_without_guidance": result.backward_no_guidance_gaussian_logp,
                "guidance_likelihood_backward_with_guidance": result.backward_guidance_div_integral,
                "guidance_likelihood_backward_without_guidance": result.backward_no_guidance_guidance_div_integral,
                "score_likelihood_backward_with_guidance": result.backward_score_div_integral,
                "score_likelihood_backward_without_guidance": result.backward_no_guidance_score_div_integral,
            }
        )
    return row


def _append_csv(path: str, row: dict) -> None:
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def _write_divergence_traces(
    output_path: str, rank: int, batch_idx: int, result
) -> None:
    phases = {
        "forward": result.forward_divergence_data,
        "backward": result.backward_divergence_data,
    }
    trace_dir = os.path.join(output_path, "traces")
    for phase, data in phases.items():
        if not data:
            continue
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(
            trace_dir, f"trace_rank{rank}_batch{batch_idx}_{phase}.json"
        )
        with open(path, "w") as f:
            json.dump({"divergence_data": data, "phase": phase}, f, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    times = _times(args)
    guidance_scale = args.guidance_scale_prefactor * 64 * 20

    dist.init()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    slurm_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))
    slurm_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    rank = rank + world_size * (slurm_id - 1)
    world_size = world_size * slurm_count

    os.makedirs(args.output_path, exist_ok=True)
    if torch.cuda.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(
        "Rank %s/%s  prefactor=%s guidance_scale=%s  ntimes=%s  "
        "forward_guidance=%s run_backward=%s num_steps=%s extra_steps=%s "
        "k=%s x0=%s",
        rank,
        world_size,
        args.guidance_scale_prefactor,
        guidance_scale,
        len(times),
        args.forward_guidance,
        args.run_backward_sampling,
        args.num_steps,
        args.extra_steps,
        args.divergence_samples,
        bool(args.x0_dir),
    )

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    load_kw = dict(channels_last=False)
    if args.checkpoint:
        model = cbottle.inference.load(
            "cbottle-3d-moe-tc", root=args.checkpoint, **load_kw
        )
    else:
        model = cbottle.inference.load("cbottle-3d-moe-tc", **load_kw)
    model.sigma_max = args.sigma_max
    if model.separate_classifier is not None:
        model.separate_classifier.eval()
    if not args.keep_gradient_checkpointing:
        n = _disable_gradient_checkpointing(model)
        logger.info("Rank %s: disabled gradient checkpointing on %s blocks", rank, n)

    if args.extra_steps_window:
        densify_lo, densify_hi = (float(v) for v in args.extra_steps_window.split(","))
    else:
        densify_lo, densify_hi = args.guidance_on, args.guidance_off
    extra_steps_intervals = (
        ((densify_lo, densify_hi, args.extra_steps),) if args.extra_steps > 0 else ()
    )
    skip_write = args.skip_write_nc or bool(args.x0_dir)
    x0_index = _index_x0_dir(args.x0_dir) if args.x0_dir else None
    if rank == 0 and x0_index is not None:
        logger.info("Indexed %s saved times from %s", len(x0_index), args.x0_dir)

    dataset = get_dataset(
        dataset="amip",
        rank=rank,
        world_size=world_size,
        infinite=False,
        shuffle=False,
    )
    dataset.set_times(times)
    writer = None
    already = 0
    if not skip_write:
        writer = cbottle.netcdf_writer.NetCDFWriter(
            args.output_path,
            config=cbottle.netcdf_writer.NetCDFConfig(hpx_level=6),
            rank=rank,
            channels=model.batch_info.channels,
        )
        already = writer.time_index
        if hasattr(dataset, "_times") and already:
            logger.info("Rank %s: skipping %s already-written times", rank, already)
            dataset._times = dataset._times[already:]

    csv_path = os.path.join(args.output_path, f"likelihoods_rank{rank}.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    guidance_pixels = model.get_guidance_pixels([args.tc_lon], [args.tc_lat])

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, disable=rank != 0)):
        start_latents = None
        ts = _unix_to_datetime_str(batch["timestamp"][0])
        if x0_index is not None:
            if ts not in x0_index:
                raise KeyError(f"timestamp {ts} not found in --x0-dir {args.x0_dir}")
            path, tidx = x0_index[ts]
            ring = _load_x0_ring(path, tidx, list(model.batch_info.channels))
            ring = ring.to(batch["target"].device)
            start_latents = _invert_written_target(model, ring)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model._calculate_odds_ratio_full(
            batch,
            guidance_pixels,
            guidance_scale=guidance_scale,
            run_backward=args.run_backward_sampling,
            forward_guidance=args.forward_guidance,
            compute_forward_divergences=args.compute_forward_divergences,
            num_steps=args.num_steps,
            extra_steps_intervals=extra_steps_intervals,
            guidance_on=args.guidance_on,
            guidance_off=args.guidance_off,
            divergence_samples=args.divergence_samples,
            start_latents=start_latents,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - t0
        _append_csv(
            csv_path,
            _row_from_result(
                result,
                rank=rank,
                batch_idx=already + batch_idx,
                timestamp=ts,
                guided=args.forward_guidance,
                gpu_seconds=gpu_seconds,
            ),
        )
        if args.save_divergence_traces:
            _write_divergence_traces(
                args.output_path, rank, already + batch_idx, result
            )
        if writer is not None:
            processed = model._post_process(result.forward_latents)
            writer.write_target(
                processed,
                model.coords,
                timestamps=_unix_to_netcdf_1900(batch["timestamp"]),
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Rank %s: done", rank)


if __name__ == "__main__":
    main()
