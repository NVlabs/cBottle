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
"""Train unconditional icon model using the edm-chaos training loop, loss and architecture"""

import contextlib
import dataclasses
import datetime
import logging
import math
import os
import time
import json
import warnings
import functools
import random
import numpy as np
from typing import Any

import cbottle.config.environment as config
import cbottle.config.models
import cbottle.likelihood
import cbottle.loss
import earth2grid
import matplotlib.pyplot as plt
import torch
import torch.distributed
import torch.utils
import torch.utils.data
from cbottle import distributed as dist
from cbottle import training_stats
from cbottle.models.distributed import (
    compute_t_split,
    DistributedConfig,
    unwrap_net,
    make_mp_randn_like,
)
from cbottle.dataclass_parser import parse_args
from cbottle.datasets import dataset_2d
from cbottle.datasets.base import BatchInfo, TimeUnit
from cbottle.config.training.masking import MaskingConfig, base_masking_config
from cbottle.metrics import BinnedAverage
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets import samplers
from cbottle.datasets.samplers import ChunkedDistributedSampler
from cbottle.datasets.round_robin import RoundRobinLoader
from cbottle.datasets.dataset_3d import (
    VARIABLE_CONFIGS,
    VariableConfig,
    MAX_CLASSES,
    get_batch_info,
    get_dataset,
)
from cbottle.diagnostics import (
    sample_from_condition,
    sample_images,
    sample_regression,
    visualize,
)
from cbottle.training import loop

import cbottle.models.networks


@dataclasses.dataclass(frozen=True)
class SongUnetConfig:
    model_channels: int = 128
    include_legacy_calendar_bug: bool = False


@dataclasses.dataclass
class TrainingLoop(loop.TrainingLoopBase):
    """
    valid_samples_per_season: the number of samples to use when making season
        average plots
    """

    lr: float = 0.0001
    lr_min: float = 1e-6
    lr_rampup_img: int = 10_000
    lr_flat_imgs: int = 1_000_000_000
    lr_decay_imgs: int = 0

    regression: bool = False
    valid_min_samples: int = 128

    # loss options
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_min: float = 0.02
    sigma_max: float = 80.0
    noise_distribution: str = "log_normal"

    # data options
    with_era5: bool = True
    with_icon: bool = True
    use_labels: bool = False
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int = 200
    label_dropout: float = 0.0
    monthly_sst_input: bool = False
    ibtracs_input: bool = False  # Whether to include tropical cyclone labels
    ibtracs_loss_weight: float = 0.1  # Weight for tropical cyclone classification loss
    variables: str = "default"
    icon_chunk_size: int = 8
    era5_chunk_size: int = 48

    # network
    network: SongUnetConfig = SongUnetConfig()

    # deprecated parameters
    hpx_level: int = 6
    data_version: int = 6

    # video model configuration
    time_length: int = 1  # Number of frames per video
    time_step: int = 1  # Time step between frames in hours
    masking_config: MaskingConfig = dataclasses.field(
        default_factory=base_masking_config
    )

    channels_last: bool = False  # Use channels_last memory format for training.

    # parallelism settings
    model_parallel: int = 1  # Number of ranks per model parallel group
    check_mp_batch: bool = False  # Enable sanity checks for model parallel batches

    def __post_init__(self):
        super().__post_init__()
        self._device_mesh = None
        self._train_samplers = None

    def _sync_sampler_epoch(self):
        """Set sampler epoch based on cur_nimg for deterministic resumption."""
        if self._train_samplers is None:
            return
        for sampler in self._train_samplers:
            sampler.set_epoch(self.cur_nimg)

    @property
    def variable_config(self) -> VariableConfig:
        return VARIABLE_CONFIGS[self.variables]

    @property
    def memory_format(self):
        return torch.channels_last if self.channels_last else torch.contiguous_format

    def learning_rate(self, cur_nimg: int) -> float:
        """Linear ramp-up, constant period, followed by cosine decay."""
        total_imgs = self.lr_rampup_img + self.lr_flat_imgs + self.lr_decay_imgs

        base_lr = self.lr
        min_lr = self.lr_min
        min_factor = min_lr / base_lr

        if cur_nimg < self.lr_rampup_img:
            # linear ramp from 0 → 1
            return float(cur_nimg) / self.lr_rampup_img
        elif cur_nimg < self.lr_rampup_img + self.lr_flat_imgs:
            return 1.0
        elif cur_nimg < total_imgs:
            # cosine decay from 1 to min_factor
            progress = (
                float(cur_nimg - self.lr_rampup_img - self.lr_flat_imgs)
                / self.lr_decay_imgs
            )
            return min_factor + 0.5 * (1.0 - min_factor) * (
                1.0 + math.cos(math.pi * progress)
            )
        else:
            return min_factor

    def _finalize_network(self, net):
        if self.channels_last:
            net = net.to(memory_format=torch.channels_last)
        return net

    @staticmethod
    @contextlib.contextmanager
    def _disable_parameter_gradients(net):
        requires_grad_list = []
        for param in net.parameters():
            requires_grad_list.append(param.requires_grad)
            param.requires_grad_(False)

        try:
            yield
        finally:
            for param, requires_grad in zip(net.parameters(), requires_grad_list):
                param.requires_grad_(requires_grad)

    def setup_sigma_bins(self):
        # Loss by sigma metric
        self._sigma_metric_bin_edges = torch.tensor([0, 0.1, 1, 10, 100, 316, 1000])
        self._test_sigma_metric = BinnedAverage(self._sigma_metric_bin_edges).to(
            self.device
        )
        self._train_sigma_metric = BinnedAverage(self._sigma_metric_bin_edges).to(
            self.device
        )
        self._train_classifier_sigma_metric = BinnedAverage(
            self._sigma_metric_bin_edges
        ).to(self.device)
        self._test_classifier_sigma_metric = BinnedAverage(
            self._sigma_metric_bin_edges
        ).to(self.device)

    @property
    def distributed_config(self) -> DistributedConfig:
        """Get the distributed configuration for data and model parallelism."""
        if self._device_mesh is not None:
            data_rank = self._device_mesh.get_local_rank(0)
            model_rank = self._device_mesh.get_local_rank(1)
            data_world_size, model_world_size = self._device_mesh.mesh.shape
        else:
            data_rank = dist.get_rank()
            data_world_size = dist.get_world_size()
            model_rank = 0
            model_world_size = 1
        return DistributedConfig(
            data_rank, data_world_size, model_rank, model_world_size
        )

    @property
    def model_rank(self):
        """Get the model parallel rank within the model group."""
        return self.distributed_config.model_rank

    @property
    def model_size(self):
        """Get the model parallel world size."""
        return self.distributed_config.model_world_size

    @functools.cached_property
    def t_bounds(self) -> tuple[int, int, int]:
        """Get (t_start, t_end, t_full) for this rank's time slice.

        Used for RNG-aligned noise generation and time indexing.
        For MP=1: (0, T, T). For MP>1: (offset, offset+local, full).
        """
        cfg = self.distributed_config
        t_splits = compute_t_split(self.time_length, cfg.model_world_size)
        t_start = sum(t_splits[: cfg.model_rank])
        t_end = t_start + t_splits[cfg.model_rank]
        return (t_start, t_end, self.time_length)

    @functools.cached_property
    def mp_randn_like(self):
        """Randn_like function that handles MP sharding."""
        return make_mp_randn_like(self.t_bounds)

    @property
    def data_world_size(self) -> int:
        """Number of data parallel replicas."""
        return self.distributed_config.data_world_size

    @property
    def data_rank(self) -> int:
        """Rank for data parallelism."""
        return self.distributed_config.data_rank

    def setup(self):
        # Set up model parallel device mesh before calling super().setup()
        self._device_mesh = None
        if self.model_parallel > 1:
            self._device_mesh = torch.distributed.init_device_mesh(
                "cuda",
                [dist.get_world_size() // self.model_parallel, self.model_parallel],
                mesh_dim_names=["data", "model"],
            )
            dist.print0("Setting up model parallelism")
            dist.print0(f"{self.batch_gpu_total=}")
            rank = self._device_mesh.get_rank()
            data_rank = self._device_mesh.get_local_rank(0)
            model_rank = self._device_mesh.get_local_rank(1)
            print(f"{rank=} {data_rank=} {model_rank=}")

        super().setup()
        self.setup_sigma_bins()

        # Set parallel group on the network for model parallelism
        if self.model_parallel > 1:
            model_group = self._device_mesh.get_group(1)
            self.net.set_parallel_group(model_group, t_full=self.time_length)

    def finish_sigma_by_loss_metrics(self):
        values = {}
        loss_avg = values["train"] = self._train_sigma_metric.compute()
        values["test"] = self._test_sigma_metric.compute()
        classifier_loss_avg = {}
        classifier_loss_avg["train"] = self._train_classifier_sigma_metric.compute()
        classifier_loss_avg["test"] = self._test_classifier_sigma_metric.compute()

        self._train_sigma_metric.reset()
        self._test_sigma_metric.reset()
        self._train_classifier_sigma_metric.reset()
        self._test_classifier_sigma_metric.reset()

        for split in values:
            for i in range(loss_avg.size(0)):
                bin = self._sigma_metric_bin_edges[i].item()
                # Denoising loss
                name = f"loss_by_sigma{bin:.2e}/{split}"
                loss_bin = values[split][i].item()
                training_stats.report(name, loss_bin)
                # Classifier loss
                classifier_name = f"classifier_loss_by_sigma{bin:.2e}/{split}"
                classifier_loss_bin = classifier_loss_avg[split][i].item()
                training_stats.report(classifier_name, classifier_loss_bin)

    @functools.cached_property
    def batch_info(self) -> BatchInfo:
        return get_batch_info(
            config=self.variable_config,
            time_step=self.time_step,
            time_unit=TimeUnit.HOUR,
        )

    @property
    def out_channels(self):
        return len(self.batch_info.channels)

    def _get_frame_masker(self, train):
        if self.time_length == 1:
            return None
        else:
            cfg = self.distributed_config
            if train:
                return FrameMasker(
                    masking_config=self.masking_config,
                    model_rank=cfg.model_rank,
                    model_world_size=cfg.model_world_size,
                    t_full=self.time_length,
                )
            else:
                # use a fixed unconditional generation task for test set
                return FrameMasker(
                    keep_frames=[],
                    model_rank=cfg.model_rank,
                    model_world_size=cfg.model_world_size,
                    t_full=self.time_length,
                )

    def get_dataset(self, train: bool):
        """Returns the final wrapped dataset for both image and video training."""
        cfg = self.distributed_config

        if not self.with_era5 and not self.with_icon:
            raise ValueError("At least one of with_era5 or with_icon must be True")

        if self.with_era5 and self.with_icon:
            if cfg.data_world_size % 2 != 0:
                warnings.warn(
                    RuntimeWarning(
                        "data world size not divisible by 2. ERA5 and ICON will not be 50-50."
                    )
                )
            n_era5 = math.ceil(cfg.data_world_size / 2)
        elif self.with_era5:
            n_era5 = cfg.data_world_size
        else:
            n_era5 = 0

        # Determine which dataset to use based on data rank
        if cfg.data_rank < n_era5:
            dataset = "era5"
            effective_rank = cfg.data_rank
            effective_world_size = n_era5
            chunk_size = self.era5_chunk_size
        else:
            dataset = "icon"
            effective_rank = cfg.data_rank - n_era5
            effective_world_size = cfg.data_world_size - n_era5
            chunk_size = self.icon_chunk_size

        map_style_no_chunking = self.model_parallel > 1 and not train
        return get_dataset(
            split="train" if train else "test",
            dataset=dataset,
            rank=effective_rank,
            world_size=effective_world_size,
            model_rank=cfg.model_rank,
            model_world_size=cfg.model_world_size,
            sst_input=self.monthly_sst_input,
            infinite=True,
            shuffle=True,
            chunk_size=chunk_size if not map_style_no_chunking else 0,
            time_step=self.time_step,
            time_length=self.time_length,
            frame_masker=self._get_frame_masker(train),
            ibtracs_input=self.ibtracs_input,
            variable_config=VARIABLE_CONFIGS[self.variables],
            map_style=self.model_parallel > 1,
        )

    # unused settings only for backwards compatibility
    valid_samples_per_season: Any = None

    def _create_dataloader(
        self,
        dataset,
        batch_size: int,
        *,
        sampler=None,
        num_workers: int = None,
        prefetch_factor: int = None,
    ):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def _get_loader(self, dataset, batch_size: int, train: bool = True):
        """Create the appropriate loader for train or test.

        For model parallel training, uses RoundRobinLoader with ChunkedDistributedSamplers.
        For test, uses fewer workers and a simpler sampler setup.
        """
        workers = self.dataloader_num_workers
        prefetch_factor = self.dataloader_prefetch_factor
        cfg = self.distributed_config

        # For test with no model parallelism, use fewer workers
        if not train and self.model_parallel == 1 and workers > 0:
            workers = min(
                self.valid_min_samples // (self.batch_size),
                workers,
            )
            prefetch_factor = min(prefetch_factor, 4)

        if isinstance(dataset, torch.utils.data.IterableDataset):
            # Iterable datasets don't use samplers
            return self._create_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                prefetch_factor=prefetch_factor,
            )

        if self.model_parallel > 1 and train:
            # For model parallel: use round-robin loader with chunked samplers
            # Create one dataloader per worker, each with its own chunk assignment
            # Then round-robin between them to simulate iterable-style behavior
            num_loaders = max(workers, 1)
            dataloaders = []

            samplers_list = []
            for worker_id in range(num_loaders):
                worker_sampler = ChunkedDistributedSampler(
                    dataset,
                    chunk_size=self.era5_chunk_size,
                    num_replicas=cfg.data_world_size * num_loaders,
                    rank=cfg.data_rank * num_loaders + worker_id,
                    shuffle=train,
                    shuffle_within_chunk=train,
                    drop_last=True,
                    seed=self.seed,
                )
                samplers_list.append(worker_sampler)
                worker_loader = self._create_dataloader(
                    dataset,
                    batch_size=batch_size,
                    sampler=worker_sampler,
                    num_workers=1,  # single worker per loader
                    prefetch_factor=prefetch_factor,
                )
                dataloaders.append(worker_loader)

            loader = RoundRobinLoader(
                dataloaders,
                shuffle_order=True,
                shuffle_every=1,
                seed=self.seed + cfg.data_rank,
            )
            if train:
                self._train_samplers = samplers_list
            return loader

        sampler = samplers.distributed_split(
            samplers.subsample(
                dataset,
                min_samples=self.valid_min_samples,
                world_size=cfg.data_world_size,
            ),
            rank=cfg.data_rank,
            world_size=cfg.data_world_size,
        )

        return self._create_dataloader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            prefetch_factor=prefetch_factor,
        )

    def get_data_loaders(self, batch_gpu):
        dataset = self.get_dataset(train=True)
        train_loader = self._get_loader(dataset, batch_size=batch_gpu, train=True)

        test_dataset = self.get_dataset(train=False)
        test_loader = self._get_loader(test_dataset, batch_size=batch_gpu, train=False)

        self._test_dataset = test_dataset
        return dataset, train_loader, test_loader

    def _curry_net(self, net, batch):
        def D(x, t):
            return net(
                x.to(memory_format=self.memory_format),
                torch.as_tensor(t, device=x.device),
                batch["labels"],
                condition=batch["condition"],
                day_of_year=batch["day_of_year"],
                second_of_day=batch["second_of_day"],
            )

        return D

    def _curry_net_discard_classifier(self, net, batch):
        def D(x, t):
            out = net(
                x.to(memory_format=self.memory_format),
                torch.as_tensor(t, device=x.device),
                batch["labels"],
                condition=batch["condition"],
                day_of_year=batch["day_of_year"],
                second_of_day=batch["second_of_day"],
            )
            return out.out

        return D

    def _check_mp_timestamps(self, timestamp):
        """Verify timestamps are identical across model parallel ranks.

        All MP ranks process the same sample (different time slices), so they
        should all have the same starting timestamp.
        """
        if not self.check_mp_batch or self.model_parallel <= 1:
            return

        model_world_size = self.distributed_config.model_world_size
        model_group = self._device_mesh.get_group(1)

        # Gather timestamps from all model parallel ranks
        gather_list = [torch.empty_like(timestamp) for _ in range(model_world_size)]
        torch.distributed.all_gather(
            gather_list, timestamp.contiguous(), group=model_group
        )

        # All ranks should have identical timestamps (same sample)
        reference = gather_list[0]
        for i, ts in enumerate(gather_list[1:], 1):
            if not torch.equal(reference, ts):
                raise ValueError(
                    f"Timestamp mismatch across MP ranks!\n"
                    f"Rank 0: {reference.cpu()}\n"
                    f"Rank {i}: {ts.cpu()}\n"
                    "MP ranks are not processing the same sample."
                )

    def _check_mp_sigma(self, sigma):
        """Verify sigma is identical across model parallel ranks.

        All MP ranks processing the same sample should have the same sigma
        (ensured by synchronized RNG seeding).
        """
        if not self.check_mp_batch or self.model_parallel <= 1:
            return

        model_world_size = self.distributed_config.model_world_size
        model_group = self._device_mesh.get_group(1)

        # Gather sigma from all model parallel ranks
        gather_list = [torch.empty_like(sigma) for _ in range(model_world_size)]
        torch.distributed.all_gather(gather_list, sigma.contiguous(), group=model_group)

        # All sigmas should be identical
        reference = gather_list[0]
        for i, s in enumerate(gather_list[1:], 1):
            if not torch.allclose(reference, s):
                raise ValueError(
                    f"Sigma mismatch across MP ranks!\n"
                    f"Rank 0 sigma: {reference.cpu()}\n"
                    f"Rank {i} sigma: {s.cpu()}\n"
                    "This indicates RNG is not synchronized across MP ranks."
                )

    def _check_mp_rng_sync(self):
        """Verify RNG state is identical across model parallel ranks.

        Generates one value from each RNG and compares across ranks.
        If in sync, values match; if not, they differ. Uses minimal
        RNG consumption to avoid affecting subsequent operations.
        """
        if not self.check_mp_batch or self.model_parallel <= 1:
            return

        model_world_size = self.distributed_config.model_world_size
        model_group = self._device_mesh.get_group(1)
        device = torch.device("cuda")

        # Check Python random state matches
        py_state = random.getstate()[1][:4]
        state_check = torch.tensor(py_state, device=device, dtype=torch.int64)
        state_gather = [torch.empty_like(state_check) for _ in range(model_world_size)]
        torch.distributed.all_gather(state_gather, state_check, group=model_group)
        for i, s in enumerate(state_gather[1:], 1):
            if not torch.equal(state_gather[0], s):
                raise ValueError(
                    f"Python random state mismatch! Rank 0: {state_gather[0].tolist()}, Rank {i}: {s.tolist()}"
                )

        # Check numpy random state matches (first 4 values of MT19937 state array)
        np_state = np.random.get_state()[1][:4]
        np_state_check = torch.tensor(np_state, device=device, dtype=torch.int64)
        np_state_gather = [
            torch.empty_like(np_state_check) for _ in range(model_world_size)
        ]
        torch.distributed.all_gather(np_state_gather, np_state_check, group=model_group)
        for i, s in enumerate(np_state_gather[1:], 1):
            if not torch.equal(np_state_gather[0], s):
                raise ValueError(
                    f"Numpy random state mismatch! Rank 0: {np_state_gather[0].tolist()}, Rank {i}: {s.tolist()}"
                )

        # Generate one test value from each RNG
        test_vals = torch.tensor(
            [
                torch.randn(1, device=device).item(),
                np.random.rand(),
                random.random(),
            ],
            device=device,
            dtype=torch.float32,
        )

        gather_list = [torch.empty_like(test_vals) for _ in range(model_world_size)]
        torch.distributed.all_gather(
            gather_list, test_vals.contiguous(), group=model_group
        )

        reference = gather_list[0]
        for i, val in enumerate(gather_list[1:], 1):
            if not torch.allclose(reference, val, rtol=1e-5, atol=1e-5):
                rng_names = ["torch", "numpy", "random"]
                for j, name in enumerate(rng_names):
                    if abs(reference[j] - val[j]) > 1e-5:
                        raise ValueError(
                            f"{name} RNG mismatch across MP ranks!\n"
                            f"Rank 0: {reference[j].item():.6f}\n"
                            f"Rank {i}: {val[j].item():.6f}\n"
                            "RNG is not synchronized across MP ranks."
                        )

    def train_step(self, *, target, timestamp=None, **batch):
        # Sanity checks for model parallel setup
        if timestamp is not None:
            self._check_mp_timestamps(timestamp)
        self._check_mp_rng_sync()

        loss: cbottle.loss.Output = self.loss_fn(
            self._curry_net(self.ddp, batch),
            target,
            classifier_labels=batch.get("classifier_labels"),
            randn_like=self.mp_randn_like,
        )

        # Check sigma is identical across MP ranks (RNG sync check)
        self._check_mp_sigma(loss.sigma)

        if self.distributed_config.model_rank == 0:
            try:
                sigma_vals = loss.sigma.squeeze().cpu().tolist()
                if isinstance(sigma_vals, float):
                    sigma_vals = [sigma_vals]
                sigma_str = ", ".join(f"{s:.4f}" for s in sigma_vals[:8])
                if len(sigma_vals) > 8:
                    sigma_str += f"... ({len(sigma_vals)} total)"
                print(f"[Sigma] batch sigmas: [{sigma_str}]")
            except Exception as e:
                print(f"[Sigma] error logging sigmas: {e}. {loss.sigma}")

            if timestamp is not None:
                try:
                    import datetime

                    raw_vals = [t.item() for t in timestamp[:4]]
                    ts_strs = [
                        datetime.datetime.fromtimestamp(
                            raw, datetime.timezone.utc
                        ).strftime("%Y-%m-%d %H:%M")
                        for raw in raw_vals
                    ]
                    print(
                        f"[Batch] timestamps raw={raw_vals[:2]} decoded={ts_strs}{'...' if len(timestamp) > 4 else ''}"
                    )
                except Exception as e:
                    print(f"[Batch] error logging timestamps: {e}. raw={timestamp}")

        training_stats.report("Loss/denoising", loss.denoising)
        self._train_sigma_metric.update(loss.sigma, loss.denoising)
        if loss.classification is not None:
            training_stats.report("Loss/classification", loss.classification)
            self._train_classifier_sigma_metric.update(loss.sigma, loss.classification)
        return loss.total

    def test_step(self, *, target, timestamp=None, **batch):
        self._check_mp_rng_sync()
        loss: cbottle.loss.Output = self.loss_fn(
            self._curry_net(self.ddp, batch),
            target,
            classifier_labels=batch.get("classifier_labels"),
            randn_like=self.mp_randn_like,
        )
        training_stats.report("Loss/test_denoising", loss.denoising)
        self._test_sigma_metric.update(loss.sigma, loss.denoising)
        if loss.classification is not None:
            training_stats.report("Loss/test_classification", loss.classification)
            self._test_classifier_sigma_metric.update(loss.sigma, loss.classification)
        return loss.total

    @classmethod
    def loads(cls, s):
        fields = json.loads(s)
        # remove aument_kwargs if present
        # this is in some older checkpoint
        fields.pop("augment_kwargs", None)
        fields.pop("auto_tuning_loss", None)
        fields.pop("auto_tuning_z", None)
        fields.pop("auto_tuning_z", None)
        fields.pop("dynamic_channels", None)

        is_video = fields.pop("video", None)
        if is_video:
            fields.pop("video_config", None)
            fields.pop("icon_v6_video_config", None)
            fields.pop("era5_v6_video_config", None)
            fields.pop("decoder_start_with_temporal_attention", None)
            fields.pop("all_decoder_temporal_attention", None)
            fields.pop("upsample_temporal_attention", None)
            fields.pop("transfer_learning_config", None)
            fields["time_length"] = 12
            fields["network"].pop("num_heads", None)

        network = SongUnetConfig(**fields.pop("network", {}))
        return cls(network=network, **fields)

    @property
    def model_config(self) -> cbottle.config.models.ModelConfigV1:
        label_dim = MAX_CLASSES if self.use_labels else 0
        out_channels = self.out_channels

        condition_channels = 1 if self.monthly_sst_input else 0
        is_video = self.time_length > 1
        if is_video:
            condition_channels += out_channels + 1  # all channels conditioning + mask
            return cbottle.config.models.ModelConfigV1(
                label_dim=label_dim,
                out_channels=out_channels,
                condition_channels=condition_channels,
                model_channels=self.network.model_channels,
                time_length=self.time_length,
                label_dropout=self.label_dropout,
                calendar_include_legacy_bug=self.network.include_legacy_calendar_bug,
            )
        else:
            return cbottle.config.models.ModelConfigV1(
                label_dim=label_dim,
                out_channels=out_channels,
                condition_channels=condition_channels,
                model_channels=self.network.model_channels,
                calendar_include_legacy_bug=self.network.include_legacy_calendar_bug,
                enable_classifier=self.ibtracs_input,
            )

    def get_optimizer(self, parameters):
        return torch.optim.Adam(params=parameters, lr=self.lr)

    def get_loss_fn(self):
        if self.regression:
            return cbottle.loss.RegressLoss()
        else:
            return cbottle.loss.EDMLoss(
                P_mean=self.p_mean,
                P_std=self.p_std,
                sigma_data=1.0,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                distribution=self.noise_distribution,
                classifier_weight=self.ibtracs_loss_weight,
            )

    def sample(self, batch):
        assert (
            self.model_parallel == 1
        ), "sample() not supported with model_parallel > 1"
        if self.regression:
            return sample_regression(self.net, batch, batch_info=self.batch_info)
        else:
            return sample_images(self.net, batch, batch_info=self.batch_info)

    @staticmethod
    def print_network_info(net, device):
        dist.print0(net)

    def _report_log_likelihood(self, net, batch):
        # likelihood

        with self._disable_parameter_gradients(net):
            target = batch["target"]
            mask = ~torch.isnan(target)
            log_prob, _ = cbottle.likelihood.log_prob(
                # TODO replace with denoiser classifier?
                self._curry_net_discard_classifier(net, batch),
                target,
                mask,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                divergence_samples=1,
                randn_like=self.mp_randn_like,
            )

        # Log data by label
        log_prob_per_dim = log_prob / mask.sum(dim=(1, 2, 3))  # n
        labels_one_hot = batch["labels"]  # n L

        training_stats.report("log_prob", log_prob_per_dim)
        for i, dataset_name in enumerate(dataset_2d.LABELS):
            training_stats.report(
                f"log_prob_{dataset_name}", log_prob_per_dim[labels_one_hot[:, i] != 0]
            )

    @staticmethod
    def _reorder_classifier_output(x):
        x = torch.as_tensor(x)
        return earth2grid.healpix.reorder(
            x, earth2grid.healpix.HEALPIX_PAD_XY, earth2grid.healpix.PixelOrder.RING
        )

    def validate(self, net=None):
        # show plots for a single batch
        val_start = time.time()
        dist.print0("[Validate] Starting validation...")

        if net is None:
            net = self.net
        net.eval()
        base_net = unwrap_net(net)  # unwrap for net attribute access
        seasons = {
            "DJF": (12, 1, 2),
            "MAM": (3, 4, 5),
            "JJA": (6, 7, 8),
            "SON": (9, 10, 11),
        }
        seasons_by_month = {
            m: season for season, months in seasons.items() for m in months
        }

        # build keys in a deterministic manner so that all ranks have the same keys
        totals = {}

        def update(key, array):
            n, total = totals.get(key, (torch.tensor(0, device=base_net.device), 0))
            totals[key] = (n + 1, total + array)

        def finish():
            averages = {}
            if dist.get_world_size() > 1:
                # reduce all fields (need to generate the list of potential output names)
                # sync the keys across all the ranks
                objs = [None for _ in range(dist.get_world_size())]
                torch.distributed.all_gather_object(objs, list(totals))
                keys = set()
                for obj in objs:
                    keys = keys.union(set(obj))
                keys = sorted(keys)

                # reduce the total onto rank=0
                for key in keys:
                    buf = torch.zeros([base_net.domain.numel()], device=base_net.device)
                    n = torch.tensor(0, device=base_net.device)
                    if key in totals:
                        n, total = totals[key]
                        buf.copy_(total)

                    # the reduce hung when transporting the "total" tensor
                    # directly, so need to use a temporary buffer

                    torch.distributed.reduce(buf, dst=0)
                    torch.distributed.reduce(n, dst=0)
                    averages[key] = buf / n
            else:
                for key, (n, buf) in totals.items():
                    averages[key] = buf / n

            return averages

        if self.time_step == 6:
            global_times_to_visualize = [0, 1, 2, 3]
        elif self.time_step == 1:
            global_times_to_visualize = [0, 1, 6, 12, 18]
        else:
            global_times_to_visualize = list(range(min(self.time_length, 4)))

        d = None
        total_batches = (
            self.valid_min_samples + self.batch_size - 1
        ) // self.batch_size
        dist.print0(f"[Validate] Processing up to {total_batches} batches...")

        for batch_num, batch in enumerate(self.valid_loader):
            iter_start = time.time()
            if batch_num * self.batch_size >= self.valid_min_samples:
                break
            batch = self._stage_dict_batch(batch)
            images = batch["target"]

            timestamp = batch.pop("timestamp")
            times = [
                datetime.datetime.fromtimestamp(t.item(), datetime.timezone.utc)
                for t in timestamp
            ]

            hpx = base_net.domain._grid
            ring_images = hpx.reorder(earth2grid.healpix.PixelOrder.RING, images)

            # Use t_bounds for time offset calculations
            t_start, t_end, _ = self.t_bounds
            t_local = t_end - t_start
            time_length_to_log = min(t_local, 4)
            cfg = self.distributed_config

            sample_start = time.time()
            with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                d = sample_from_condition(
                    net,
                    batch,
                    batch_info=self.batch_info,
                    regression=self.regression,
                    sigma_max=self.sigma_max,
                    sigma_min=self.sigma_min,
                    randn_like=self.mp_randn_like,
                )
            sample_time = time.time() - sample_start

            for j in range(len(times)):
                first_frame_time = times[j]
                for t in range(time_length_to_log):
                    # Use global time index for time delta calculation
                    global_t = t_start + t
                    time_idx = first_frame_time + self.batch_info.get_time_delta(
                        global_t
                    )

                    season = seasons_by_month[time_idx.month]
                    for field in d:
                        update(
                            f"seasonal_cycle/{field}/{season}/generated",
                            d[field][j, t],
                        )

                    for c in range(ring_images.size(1)):
                        b = self.batch_info
                        field = b.channels[c]
                        update(
                            f"seasonal_cycle/{field}/{season}/truth",
                            ring_images[j, c, t] * b.scales[c] + b.center[c],
                        )
            test_start = time.time()
            with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Classifier validation
                    self.validate_classifier(images, net, batch)
                    loss = self.test_step(**batch)
                    training_stats.report("Loss/test_loss", loss)
            test_time = time.time() - test_start

            # Log likelihood is expensive for video - only compute on first batch
            if self.model_parallel == 1 and (
                (batch_num % 2 == 0) or self.time_length == 1
            ):
                ll_start = time.time()
                with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                    self._report_log_likelihood(net, batch)
                ll_time = time.time() - ll_start
            else:
                ll_time = 0.0

            iter_time = time.time() - iter_start
            ll_str = f"ll={ll_time:.2f}s, " if ll_time > 0 else ""
            dist.print0(
                f"[Validate] Batch {batch_num+1}/{total_batches}: "
                f"sample={sample_time:.2f}s, test={test_time:.2f}s, {ll_str}total={iter_time:.2f}s"
            )

        dist.print0(f"[Validate] Batch loop done in {time.time() - val_start:.2f}s")

        finish_start = time.time()
        averages = finish()
        self.finish_sigma_by_loss_metrics()

        if d is None:
            raise RuntimeError(
                "No inference was performed. Check that the test dataset is not empty."
            )

        # show the seasonal composite
        for field in self.batch_info.channels:
            for source in ["generated", "truth"]:
                try:
                    jja = averages[f"seasonal_cycle/{field}/JJA/{source}"]
                    djf = averages[f"seasonal_cycle/{field}/DJF/{source}"]
                except KeyError:
                    pass
                else:
                    averages[f"JJA-DJF/{field}/{source}"] = jja - djf

        dist.print0(
            f"[Validate] Finish/reduce done in {time.time() - finish_start:.2f}s"
        )

        # Gather frames for visualization to rank 0
        gather_start = time.time()
        if cfg.model_world_size > 1:
            model_group = self._device_mesh.get_group(1)
            t_splits = compute_t_split(self.time_length, cfg.model_world_size)

            b, c, _, x = ring_images.shape
            ring_list = [
                torch.empty(
                    (b, c, t_splits[r], x),
                    device=ring_images.device,
                    dtype=ring_images.dtype,
                )
                for r in range(cfg.model_world_size)
            ]
            torch.distributed.all_gather(
                ring_list, ring_images.contiguous(), group=model_group
            )
            full_ring_images = torch.cat(ring_list, dim=2)

            field_names = list(d.keys())
            stacked_d = torch.stack(
                [d[f] for f in field_names], dim=0
            )  # (F, B, T_local, X)
            f_dim, b, _, x = stacked_d.shape
            d_list = [
                torch.empty(
                    (f_dim, b, t_splits[r], x),
                    device=stacked_d.device,
                    dtype=stacked_d.dtype,
                )
                for r in range(cfg.model_world_size)
            ]
            torch.distributed.all_gather(
                d_list, stacked_d.contiguous(), group=model_group
            )
            full_stacked_d = torch.cat(d_list, dim=2)  # (F, B, T_full, X)
            full_d = {f: full_stacked_d[i] for i, f in enumerate(field_names)}
        else:
            full_ring_images = ring_images
            full_d = d

        dist.print0(f"[Validate] Gather done in {time.time() - gather_start:.2f}s")

        # Visualize on rank 0
        viz_start = time.time()
        if dist.get_rank() == 0:
            for global_t in global_times_to_visualize:
                for field in full_d:
                    visualize(full_d[field][0, global_t])
                    self.writer.add_figure(
                        f"sample/{field}/generated/{global_t}",
                        plt.gcf(),
                        global_step=self.cur_nimg,
                    )

                for c in range(full_ring_images.size(1)):
                    b = self.batch_info
                    visualize(
                        full_ring_images[0, c, global_t] * b.scales[c] + b.center[c]
                    )
                    field = b.channels[c]
                    self.writer.add_figure(
                        f"sample/{field}/truth/{global_t}",
                        plt.gcf(),
                        global_step=self.cur_nimg,
                    )

        if dist.get_rank() == 0:
            for key in averages:
                visualize(averages[key])
                self.writer.add_figure(key, plt.gcf(), global_step=self.cur_nimg)

        dist.print0(f"[Validate] Visualization done in {time.time() - viz_start:.2f}s")
        dist.print0(f"[Validate] Complete! Total time: {time.time() - val_start:.2f}s")

    def validate_classifier(self, images, net, batch):
        if self.ibtracs_input:
            sigmas = torch.tensor(
                [self.sigma_min, self.sigma_max / 4],
                device=images.device,
            )
            for sigma in sigmas:
                noise = torch.randn_like(images) * sigma
                noisy_images = images + noise

                out = net(
                    x=noisy_images,
                    sigma=sigma.expand(images.shape[0], 1),
                    condition=batch.get("condition"),
                    day_of_year=batch.get("day_of_year"),
                    second_of_day=batch.get("second_of_day"),
                )

                if out.logits is not None:
                    tc_probs = torch.sigmoid(out.logits)
                    visualize(self._reorder_classifier_output(tc_probs[0, 0, 0]))
                    self.writer.add_figure(
                        f"sample/tc_probability/classifier/sigma_{sigma:.3f}",
                        plt.gcf(),
                        global_step=self.cur_nimg,
                    )
                    visualize(self._reorder_classifier_output(noisy_images[0, 0, 0]))
                    self.writer.add_figure(
                        f"sample/noisy_image/sigma_{sigma:.3f}",
                        plt.gcf(),
                        global_step=self.cur_nimg,
                    )


@dataclasses.dataclass
class CLI:
    name: str = ""
    output_dir: str = config.CHECKPOINT_ROOT
    validate_only: bool = False
    validate_all: bool = False
    resume_dir: str = ""
    resume_checkpoint: str = ""
    test_fast: bool = False
    regression: bool = False

    # training loop settings
    loop: TrainingLoop = dataclasses.field(
        default_factory=lambda: TrainingLoop(
            total_ticks=100_000,
            steps_per_tick=500,
            state_dump_ticks=5,
            snapshot_ticks=5,
            batch_size=64,
            batch_gpu=2,
            lr_rampup_img=10_000,
            with_era5=False,
            ibtracs_input=False,
            ibtracs_loss_weight=0.1,
        )
    )


def main():
    cli = parse_args(CLI, convert_underscore_to_hyphen=False)
    logging.basicConfig(level=logging.INFO)
    loop = cli.loop
    if not cli.name:
        name = "iconHpx64Uncond-" + str(int(time.time()))
    else:
        name = cli.name

    loop.run_dir = os.path.join(cli.output_dir, name)
    dist.init()
    data_world_size = dist.get_world_size() // loop.model_parallel

    if cli.test_fast:
        loop.total_ticks = 2
        loop.steps_per_tick = 1
        loop.state_dump_ticks = 1
        loop.snapshot_ticks = 1
        loop.batch_size = 1 * data_world_size
        loop.batch_gpu = 1
        loop.lr_rampup_img = 10_000
        loop.valid_min_samples = 2 * data_world_size

    if cli.regression:
        loop.lr = 0.0001
        loop.batch_size = loop.batch_gpu * data_world_size
        loop.steps_per_tick = 1000
    loop.valid_min_samples = max(loop.valid_min_samples, 64)
    loop.steps_per_tick = max(loop.steps_per_tick, 4000)
    if loop.model_parallel == 1:
        loop.steps_per_tick = 3250
    loop.era5_chunk_size = min(loop.era5_chunk_size, 24)

    # Reduce dataloader workers for large GPU counts to avoid S3 rate limiting
    total_gpus = dist.get_world_size()
    if total_gpus >= 64:
        loop.dataloader_num_workers = min(loop.dataloader_num_workers, 4)
        loop.dataloader_prefetch_factor = min(loop.dataloader_prefetch_factor, 8)
        print(
            f"Large scale run ({total_gpus} GPUs): reduced workers={loop.dataloader_num_workers}, prefetch={loop.dataloader_prefetch_factor}"
        )

    print("Training with:", loop)
    loop.setup()

    # attempt resuming from output-dir, and then try the resume_dir CLI
    # this behavoir makes it easy to submit multiple segments of the run using
    # the same CLI arguments
    resume_dirs_in_priority = [loop.run_dir, cli.resume_dir]
    new_training = True
    for rundir in resume_dirs_in_priority:
        try:
            loop.resume_from_rundir(rundir, require_all=False)
            new_training = False
            break
        except FileNotFoundError:
            pass

    if new_training and cli.resume_checkpoint:
        loop.resume_from_rundir(cli.resume_checkpoint, require_all=False)
        new_training = False
        print(f"Resumed from checkpoint: {cli.resume_checkpoint}")

    if new_training:
        print("Starting new training")

    if cli.validate_only:
        loop.validate()
        loop.log_training_stats()
    elif cli.validate_all:
        loop.validate_all_checkpoints()
    else:
        loop.train()


if __name__ == "__main__":
    main()
