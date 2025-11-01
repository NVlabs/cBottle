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

import dataclasses
import datetime
import logging
import math
import os
import time
import json
import warnings
import functools
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
from cbottle.dataclass_parser import parse_args
from cbottle.datasets import dataset_2d
from cbottle.datasets.base import BatchInfo, TimeUnit
from cbottle.config.training.masking import MaskingConfig, base_masking_config
from cbottle.metrics import BinnedAverage
from cbottle.training.video.frame_masker import FrameMasker
from cbottle.datasets import samplers
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

    @property
    def variable_config(self) -> VariableConfig:
        return VARIABLE_CONFIGS[self.variables]

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

    def setup(self):
        super().setup()
        self.setup_sigma_bins()

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
            if train:
                return FrameMasker(masking_config=self.masking_config)
            else:
                # use a fixed unconditional generation task for test set
                return FrameMasker(keep_frames=[])

    def get_dataset(self, train: bool):
        """Returns the final wrapped dataset for both image and video training."""
        if self.with_era5 and dist.get_world_size() % 2 != 0:
            warnings.warn(
                RuntimeWarning(
                    "world size not divisible by 2. ERA5 and ICON will not be 50-50."
                )
            )

        if self.with_era5:
            n_era5 = math.ceil(dist.get_world_size() / 2)
        else:
            n_era5 = 0

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Determine which dataset to use based on rank
        if self.with_era5 and rank < n_era5:
            dataset = "era5"
            effective_rank = rank
            effective_world_size = n_era5
            chunk_size = self.era5_chunk_size
        else:
            dataset = "icon"
            effective_rank = rank - n_era5
            effective_world_size = world_size - n_era5
            chunk_size = self.icon_chunk_size

        return get_dataset(
            split="train" if train else "test",
            dataset=dataset,
            rank=effective_rank,
            world_size=effective_world_size,
            sst_input=self.monthly_sst_input,
            infinite=True,
            shuffle=True,
            chunk_size=chunk_size,
            time_step=self.time_step,
            time_length=self.time_length,
            frame_masker=self._get_frame_masker(train),
            ibtracs_input=self.ibtracs_input,
            variable_config=VARIABLE_CONFIGS[self.variables],
        )

    # unused settings only for backwards compatibility
    valid_samples_per_season: Any = None

    def get_data_loaders(self, batch_gpu):
        dataset = self.get_dataset(train=True)
        batch_size = batch_gpu

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=self.dataloader_num_workers,
            prefetch_factor=self.dataloader_prefetch_factor,
            multiprocessing_context=(
                "spawn" if self.dataloader_num_workers > 0 else None
            ),
            persistent_workers=self.dataloader_num_workers > 0,
        )

        test_dataset = self.get_dataset(train=False)
        num_test_batches_per_rank = self.valid_min_samples // (
            dist.get_world_size() * batch_gpu
        )
        num_workers = min(num_test_batches_per_rank, self.dataloader_num_workers)
        test_sampler = None
        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            test_sampler = samplers.distributed_split(
                samplers.subsample(test_dataset, min_samples=self.valid_min_samples)
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_gpu,
            pin_memory=True,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            num_workers=num_workers,
            sampler=test_sampler,
        )

        self._test_dataset = test_dataset
        return dataset, train_loader, test_loader

    def _curry_net(self, net, batch):
        def D(x, t):
            return net(
                x,
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
                x,
                torch.as_tensor(t, device=x.device),
                batch["labels"],
                condition=batch["condition"],
                day_of_year=batch["day_of_year"],
                second_of_day=batch["second_of_day"],
            )
            return out.out

        return D

    def train_step(self, *, target, timestamp=None, **batch):
        loss: cbottle.loss.Output = self.loss_fn(
            self._curry_net(self.ddp, batch),
            target,
            classifier_labels=batch.get("classifier_labels"),
        )
        training_stats.report("Loss/denoising", loss.denoising)
        self._train_sigma_metric.update(loss.sigma, loss.denoising)
        if loss.classification is not None:
            training_stats.report("Loss/classification", loss.classification)
            self._train_classifier_sigma_metric.update(loss.sigma, loss.classification)
        return loss.total

    def test_step(self, *, target, timestamp=None, **batch):
        loss: cbottle.loss.Output = self.loss_fn(
            self._curry_net(self.ddp, batch),
            target,
            classifier_labels=batch.get("classifier_labels"),
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
        if self.regression:
            return sample_regression(self.net, batch, batch_info=self.batch_info)
        else:
            return sample_images(self.net, batch, batch_info=self.batch_info)

    @staticmethod
    def print_network_info(net, device):
        dist.print0(net)

    def _report_log_likelihood(self, net, batch):
        # likelihood
        # log likelihood only requires grad wrt the input x
        # can turn off grad wrt parameters to save memory
        requires_grad_list = []
        for param in net.parameters():
            requires_grad_list.append(param.requires_grad)
            param.requires_grad_(False)

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
        )

        for param, requires_grad in zip(net.parameters(), requires_grad_list):
            param.requires_grad_(requires_grad)

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
        if net is None:
            net = self.net
        net.eval()
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
            n, total = totals.get(key, (torch.tensor(0, device=net.device), 0))
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
                    buf = torch.zeros([net.domain.numel()], device=net.device)
                    n = torch.tensor(0, device=net.device)
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

        d = None
        for batch_num, batch in enumerate(self.valid_loader):
            if (
                batch_num * dist.get_world_size() * self.batch_gpu
                >= self.valid_min_samples
            ):
                break
            batch = self._stage_dict_batch(batch)
            images = batch["target"]

            timestamp = batch.pop("timestamp")
            times = [
                datetime.datetime.fromtimestamp(t.item(), datetime.timezone.utc)
                for t in timestamp
            ]

            hpx = net.domain._grid
            ring_images = hpx.reorder(earth2grid.healpix.PixelOrder.RING, images)

            with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                d = sample_from_condition(
                    net,
                    batch,
                    batch_info=self.batch_info,
                    regression=self.regression,
                    sigma_max=self.sigma_max,
                    sigma_min=self.sigma_min,
                )

            time_length = self.time_length

            for j in range(len(times)):
                first_frame_time = times[j]
                for t in range(time_length):
                    time_idx = first_frame_time + self.batch_info.get_time_delta(t)

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
            with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                with torch.no_grad():
                    # Classifier validation
                    self.validate_classifier(images, net, batch)
                    loss = self.test_step(**batch)
                    training_stats.report("Loss/test_loss", loss)

                self._report_log_likelihood(net, batch)

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

        if dist.get_rank() == 0:
            for t in range(time_length):
                for field in d:
                    array = d[field]
                    visualize(array[0, t])
                    self.writer.add_figure(
                        f"sample/{field}/generated/{t}",
                        plt.gcf(),
                        global_step=self.cur_nimg,
                    )

                for j in range(images.size(1)):
                    b = self.batch_info
                    visualize(ring_images[0, j, t] * b.scales[j] + b.center[j])
                    field = b.channels[j]
                    self.writer.add_figure(
                        f"sample/{field}/truth/{t}",
                        plt.gcf(),
                        global_step=self.cur_nimg,
                    )

            for key in averages:
                visualize(averages[key])
                self.writer.add_figure(key, plt.gcf(), global_step=self.cur_nimg)

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
    if cli.test_fast:
        loop.total_ticks = 2
        loop.steps_per_tick = 1
        loop.state_dump_ticks = 1
        loop.snapshot_ticks = 1
        loop.batch_size = 1 * dist.get_world_size()
        loop.batch_gpu = 1
        loop.lr_rampup_img = 10_000
        loop.valid_min_samples = 2 * dist.get_world_size()

    if cli.regression:
        loop.lr = 0.0001
        loop.batch_size = loop.batch_gpu * dist.get_world_size()
        loop.steps_per_tick = 1000

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
