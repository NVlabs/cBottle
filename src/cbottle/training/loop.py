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
import abc
import copy
import dataclasses
import json
import gc
import os
import pickle
import time
import warnings
import glob
import math
from functools import partial
from typing import Iterable, Union

from cbottle.config.training import loop
import cbottle.config.models
import cbottle.models
import cbottle.checkpointing
import cbottle.profiling
import numpy as np
import psutil
import torch
import torch.utils.tensorboard
from cbottle import distributed as dist
from cbottle import training_stats
from cbottle.training import utils as misc
from cbottle.training.event_log import SAVE_NETWORK_SNAPSHOT, EventLog
from cbottle.datasets.base import SpatioTemporalDataset

DATASET_METADATA_FILENAME = "dataset-metadata.pth"
TRAINER_METADATA_FILENAME = "loop.json"
EVENT_LOG_FILE = "events.jsonl"


def _to_batch(x, device):
    if isinstance(x, dict):
        return {k: _to_batch(v, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.is_floating_point(x):
            x = x.float()
        return x.to(device)
    else:
        raise NotImplementedError(x)


def _format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(
            s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60
        )


@dataclasses.dataclass
class TrainingLoopBase(loop.TrainingLoopBase, abc.ABC):
    """Abstract base class for diffusion trainings loops

    Implementations should define
    - get_data_loaders
    - get_network

    """

    device: torch.device | None = None

    def __post_init__(self):
        if self.steps_per_tick <= 0:
            ValueError(self.steps_per_tick)

    @abc.abstractmethod
    def get_data_loaders(
        self, batch_gpu: int
    ) -> tuple[SpatioTemporalDataset, Iterable, Iterable]:
        pass

    def get_network(self) -> torch.nn.Module:
        return cbottle.models.get_model(self.model_config)

    @abc.abstractmethod
    def get_optimizer(self, parameters):
        pass

    @abc.abstractmethod
    def get_loss_fn(self):
        pass

    @property
    def model_config(self) -> cbottle.config.models.ModelConfigV1 | None:
        """Model configuration used for the network. This is used for checkpointing.

        If you are overriding get_network, then be sure to make this consistent.
        """
        return None

    def _setup_datasets(self):
        self.dataset_obj, self.train_loader, self.valid_loader = self.get_data_loaders(
            self.batch_gpu
        )

    def _setup_networks(self):
        self.ddp = self.net = self.get_network()
        self.net.train().requires_grad_(True).to(self.device)
        if self.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)
        if dist.get_world_size() > 1:
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.device],
                broadcast_buffers=False,
            )
        self.ema = copy.deepcopy(self.net).eval().requires_grad_(False)

    @cbottle.profiling.nvtx
    def log_tick(
        self,
        maintenance_time,
        tick_start_time,
        tick_end_time,
        start_time,
        cur_tick,
        cur_nimg,
    ):
        # Print status line, accumulating the same information in training_stats.
        images_per_tick = self.steps_per_tick * self.batch_size
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {_format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / images_per_tick * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(self.device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(self.device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(" ".join(fields))

    def setup_logs(self):
        self.event_log = EventLog(os.path.join(self.run_dir, EVENT_LOG_FILE))
        self.writer = torch.utils.tensorboard.SummaryWriter(self.run_dir)

    @property
    def batch_gpu_total(self) -> int:
        world_size: int = dist.get_world_size()
        return self.batch_size // world_size

    def setup_batching(self):
        # Select batch size per GPU.
        if self.batch_gpu is None or self.batch_gpu > self.batch_gpu_total:
            self.batch_gpu = self.batch_gpu_total
        num_accumulation_rounds = self.batch_gpu_total // self.batch_gpu
        assert (
            self.batch_size
            == self.batch_gpu * num_accumulation_rounds * dist.get_world_size()
        )
        self.num_accumulation_rounds = num_accumulation_rounds

    @staticmethod
    def print_network_info(net, device):
        pass

    def resume_from_snapshot(self, resume_pkl, init_net_from_ema=False):
        """

        Args:
            resume_pkl: path to networks pkl file. Should have "ema" field
            init_net_from_ema: whether ``self.net`` should be updated with the
                weights in ``ema``. This is useful if the training state is not
                dumped out.
        """

        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first

        with open(resume_pkl, mode="rb") as f:
            data = pickle.load(f)

        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow

        if init_net_from_ema:
            misc.copy_params_and_buffers(
                src_module=data["ema"], dst_module=self.net, require_all=False
            )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=self.ema, require_all=False
        )

    def resume_from_state(self, resume_state_dump, optimizer=True, require_all=True):
        dist.print0(f'Loading training state from "{resume_state_dump}"...')

        with cbottle.checkpointing.Checkpoint(resume_state_dump, "r") as checkpoint:
            self._load_net_state(checkpoint, require_all)
            gc.collect()
            if optimizer and self.optimizer is not None:
                self._load_optimizer_state(checkpoint)

    def _load_net_state(self, checkpoint, require_all):
        with checkpoint.open("net_state.pth", "r") as f:
            net_state = torch.load(f, weights_only=True, map_location="cpu")
            self.net.load_state_dict(net_state, strict=require_all)

    def _load_optimizer_state(self, checkpoint):
        with checkpoint.open("optimizer_state.pth", "r") as f:
            # load to cpu to avoid copies in gpu memory
            optimizer_state = torch.load(f, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)

    def train_step(
        self, *, condition=None, target, labels, augment_labels=None, **kwargs
    ):
        return self.loss_fn(
            net=partial(
                self.ddp,
                condition=condition,
                class_labels=labels,
                augment_labels=augment_labels,
                **kwargs,
            ),
            images=target,
        )

    def _stage_tuple_batch(self, batch):
        indict = {}
        images, labels, condition = batch[:3]
        assert images.ndim == 4
        indict["target"] = images.to(self.device)
        indict["condition"] = condition.to(self.device)
        indict["labels"] = labels.to(self.device)

        if len(batch) == 4:
            augment_labels = batch[3]
            if augment_labels is not None:
                augment_labels.to(self.device).float()
            indict["augment_labels"] = batch[3]
        return indict

    def _stage_dict_batch(self, batch):
        return _to_batch(batch, self.device)

    @cbottle.profiling.nvtx
    def backward_batch(self, dataset_iterator):
        self.ddp.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for round_idx in range(self.num_accumulation_rounds):
            with misc.ddp_sync(
                self.ddp, (round_idx == self.num_accumulation_rounds - 1)
            ):
                torch.cuda.nvtx.range_push("load data")
                batch = next(dataset_iterator)

                if isinstance(batch, dict):
                    indict = self._stage_dict_batch(batch)
                else:
                    warnings.warn(
                        DeprecationWarning(
                            "tuple based dataloaders will be removed soon. please refactor to use dicts."
                        )
                    )
                    indict = self._stage_tuple_batch(batch)

                torch.cuda.nvtx.range_pop()
                with torch.autocast("cuda", enabled=self.bf16, dtype=torch.bfloat16):
                    loss = self.train_step(**indict)
                training_stats.report("Loss/loss", loss)
                time_length = loss.shape[2]  # (b, c, t, x)
                loss_mean = loss.sum().mul(
                    self.loss_scaling / (self.batch_gpu_total * time_length)
                )
                torch.cuda.nvtx.range_push("training_loop:backward")
                loss_mean.backward()
                torch.cuda.nvtx.range_pop()

                for param in self.ddp.parameters():
                    if param.grad is not None:
                        training_stats.report("grad_norm", param.grad.norm(2))

                total_loss += loss_mean.detach().cpu() / self.num_accumulation_rounds

    @cbottle.profiling.nvtx
    def step_optimizer(self, cur_nimg):
        # Update weights.
        torch.cuda.nvtx.range_push("training_loop:step")

        warmup_imgs = self.lr_rampup_img
        flat_imgs = self.lr_flat_imgs
        decay_imgs = self.lr_decay_imgs
        total_imgs = warmup_imgs + flat_imgs + decay_imgs

        def lr_lambda(cur_nimg):
            base_lr = self.lr
            min_lr = self.lr_min

            min_factor = min_lr / base_lr
            if cur_nimg < warmup_imgs:
                # linear ramp from 0 → 1
                return float(cur_nimg) / warmup_imgs
            elif cur_nimg < warmup_imgs + flat_imgs:
                return 1.0
            elif cur_nimg < total_imgs:
                # cosine decay from 1 to min_factor
                progress = float(cur_nimg - warmup_imgs - flat_imgs) / decay_imgs
                return min_factor + 0.5 * (1.0 - min_factor) * (
                    1.0 + math.cos(math.pi * progress)
                )
            else:
                return min_factor

        scale = lr_lambda(self.cur_nimg)
        for g in self.optimizer.param_groups:
            base_lr = g.setdefault(
                "base_lr", g.get("lr", self.optimizer.defaults["lr"])
            )
            lr = base_lr * scale
            g["lr"] = lr
            self.writer.add_scalar("lr", lr, global_step=self.cur_nimg)
        for param in self.net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        if self.gradient_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=self.gradient_clip_max_norm
            )

        self.optimizer.step()
        torch.cuda.nvtx.range_pop()

    @cbottle.profiling.nvtx
    def update_ema(self, cur_nimg):
        # Update EMA.
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * self.ema_rampup_ratio)
        ema_beta = 0.5 ** (self.batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def on_tick(self):
        pass

    def validate_all_checkpoints(self):
        event_log = EventLog(os.path.join(self.run_dir, EVENT_LOG_FILE))
        for training_state_file, nimg in event_log.states():
            self.cur_nimg = nimg
            self.resume_from_state(os.path.join(self.run_dir, training_state_file))
            self.validate(self.net)
            self.flush_training_stats()

    @cbottle.profiling.nvtx
    def validate(self, net):
        loss_key = "Loss/test_loss"

        with torch.no_grad():
            for batch in self.valid_loader:
                if len(batch) == 4:
                    images, labels, condition, augment_labels = batch
                else:
                    images, labels, condition = batch
                    augment_labels = None

                assert images.ndim == 4

                images = images.to(self.device).to(torch.float32)
                condition = condition.to(self.device).to(torch.float32)
                labels = labels.to(self.device)

                loss = self.train_step(
                    condition=condition,
                    target=images,
                    labels=labels,
                    augment_labels=augment_labels,
                )
                training_stats.report(loss_key, loss)

    @cbottle.profiling.nvtx
    def save_network_snapshot(self, cur_nimg):
        data = dict(
            ema=self.ema,
            loss_fn=self.loss_fn,
            dataset_kwargs=dict(getattr(self, "dataset_kwargs", {})),
        )

        for key, value in data.items():
            if isinstance(value, torch.nn.Module):
                value = copy.deepcopy(value).eval().requires_grad_(False)
                misc.check_ddp_consistency(value)
                data[key] = value.cpu()
            del value  # conserve memory
        if dist.get_rank() == 0:
            snapshot_filename = f"network-snapshot-{cur_nimg:09d}.pkl"
            self.event_log.log_network_snapshot(snapshot_filename, cur_nimg)
            with open(os.path.join(self.run_dir, snapshot_filename), "wb") as f:
                pickle.dump(data, f)

    @cbottle.profiling.nvtx
    def save_training_state(self, cur_nimg):
        state_filename = f"training-state-{cur_nimg:09d}.checkpoint"
        self.event_log.log_training_state(state_filename, cur_nimg)
        with cbottle.checkpointing.Checkpoint(
            os.path.join(self.run_dir, state_filename), "w"
        ) as checkpoint:
            checkpoint.write_model(self.net)
            checkpoint.write_batch_info(self.dataset_obj.batch_info)
            with checkpoint.open("optimizer_state.pth", "w") as f:
                torch.save(self.optimizer.state_dict(), f)

            with checkpoint.open("loop.json", "w") as f:
                f.write(self.dumps().encode())

            if self.model_config is not None:
                checkpoint.write_model_config(self.model_config)

    def flush_training_stats(self):
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            info = training_stats.default_collector.as_dict()
            try:
                nimg = info["Progress/kimg"]["mean"] * 1000
            except KeyError:
                nimg = self.cur_nimg

            for k, v in info.items():
                for moment in v:
                    self.writer.add_scalar(f"{k}/{moment}", v[moment], global_step=nimg)

            stats_path = os.path.join(self.run_dir, "stats.jsonl")
            with open(stats_path, "at") as f:
                stats = training_stats.default_collector.as_dict()
                for stat in stats:
                    mean = stats[stat]["mean"]
                    print(f"{stat} = {mean:4g}")
                f.write(
                    json.dumps(
                        dict(
                            training_stats.default_collector.as_dict(),
                            timestamp=time.time(),
                        )
                    )
                    + "\n"
                )

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.loads(f.read())

    @classmethod
    def from_rundir(cls, run_dir):
        path = os.path.join(run_dir, TRAINER_METADATA_FILENAME)
        loop = cls.from_json(path)
        loop.run_dir = run_dir
        return loop

    def dumps(self):
        fields = dataclasses.asdict(self)
        fields.pop("device", None)
        return json.dumps(fields)

    @classmethod
    def loads(cls, s):
        return cls(**json.loads(s))

    def save_metadata(self):
        with open(os.path.join(self.run_dir, TRAINER_METADATA_FILENAME), "w") as f:
            fields = dataclasses.asdict(self)
            fields.pop("device", None)
            f.write(self.dumps())

    def setup(self):
        self.setup_logs()
        self.save_metadata()
        self.device = self.device or torch.device("cuda", torch.cuda.current_device())
        self.loss_fn = self.get_loss_fn()
        self.cur_nimg = 0
        self._setup_datasets()
        self._setup_networks()
        self.print_network_info(self.net, self.device)
        self.setup_batching()
        self.optimizer = self.get_optimizer(self.net.parameters())
        self._state_checkpoint_handler = CheckpointHandler(self.run_dir)
        self._snapshot_checkpoint_handler = CheckpointHandler(
            self.run_dir, "network-snapshot-{}.checkpoint"
        )

    def resume_from_rundir(self, run_dir=None, require_all=True):
        run_dir = run_dir or self.run_dir
        event_log = EventLog(os.path.join(run_dir, EVENT_LOG_FILE))
        training_state_file, nimg = event_log.last_state()
        training_state = os.path.join(run_dir, training_state_file)

        resume_pkl = ""
        for event in event_log.query(SAVE_NETWORK_SNAPSHOT):
            if event["nimg"] == nimg:
                resume_pkl = os.path.join(run_dir, event["filename"])

        self.cur_nimg = nimg

        if resume_pkl:
            self.resume_from_snapshot(resume_pkl, init_net_from_ema=False)

        self.resume_from_state(training_state, require_all=require_all)

    def train(self):
        # Initialize.
        dist.print0("Loss function", self.loss_fn)
        start_time = time.time()
        np.random.seed(
            (self.seed * dist.get_world_size() + dist.get_rank() + self.cur_nimg)
            % (1 << 31)
        )
        torch.manual_seed(np.random.randint(1 << 31))
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        # Train.
        tick_start_time = time.time()
        maintenance_time = tick_start_time - start_time
        dist.update_progress(0, self.total_ticks)
        dataset_iterator = iter(self.train_loader)
        for cur_tick in range(self.total_ticks):
            for _ in range(self.steps_per_tick):
                self.backward_batch(dataset_iterator)
                self.cur_nimg += self.batch_size
                self.step_optimizer(self.cur_nimg)
                self.update_ema(self.cur_nimg)

            tick_end_time = time.time()
            self.log_tick(
                maintenance_time,
                tick_start_time,
                tick_end_time,
                start_time,
                cur_tick,
                self.cur_nimg,
            )
            self.net.eval()
            self.validate(self.net)
            self.net.train()

            # Save network snapshot.
            if (self.snapshot_ticks is not None) and (
                cur_tick % self.snapshot_ticks == 0
            ):
                self.save_network_snapshot(self.cur_nimg)
            if (
                (self.state_dump_ticks is not None)
                and (cur_tick % self.state_dump_ticks == 0)
                and dist.get_rank() == 0
            ):
                self.save_training_state(self.cur_nimg)

            # Update logs.
            self.flush_training_stats()
            dist.update_progress(cur_tick, self.total_ticks)

            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

        # Done.
        dist.print0()
        dist.print0("Exiting...")
