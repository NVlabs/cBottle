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
import os
import time
from typing import Callable

import earth2grid
import torch
import torch.distributed as dist

import cbottle.checkpointing
import cbottle.config.environment as config
import cbottle.models
from cbottle.datasets import samplers
from cbottle.datasets.dataset_2d import HealpixDatasetV5
from cbottle import distributed as cbottle_dist
from cbottle._distill_helper import (
    DistillLoss,
    CMModel,
    SCMModel,
    DMD2Model,
)
from cbottle.config.training.distill import DistillConfig
from cbottle.patchify import BatchedPatchIterator
from torch.optim.lr_scheduler import LambdaLR

from fastgen.callbacks.ct_schedule import CTScheduleCallback
from fastgen.callbacks.ema import EMACallback
from fastgen.utils.distributed.ddp import DDPWrapper
from fastgen.utils import lr_scheduler

import dataclasses
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import importlib
from scipy.signal import windows

# Import apex GroupNorm if installed only
_is_apex_available = False
if torch.cuda.is_available():
    try:
        apex_gn_module = importlib.import_module("apex.contrib.group_norm")
        ApexGroupNorm = getattr(apex_gn_module, "GroupNorm")
        _is_apex_available = True
    except ImportError:
        pass


MODEL_MAP = {
    "cm": CMModel,
    "scm": SCMModel,
    "dmd2": DMD2Model,
}


def get_distill_model(teacher_model, model_cfg, distill_cfg, device):
    # set teacher model and device
    model_cfg.net = teacher_model
    model_cfg.device = device
    model = MODEL_MAP[distill_cfg.mode](model_cfg)
    return model


def get_window_function(patch_size, window_alpha, type="KBD", **kwargs):
    functions = {
        "uniform": torch.ones,
        "hann": lambda ps: windows.hann(ps, sym=True),
        "hamming": lambda ps: windows.hamming(ps, sym=True),
        "general_hamming": lambda ps: windows.general_hamming(
            ps, window_alpha, sym=True
        ),
        "kaiser": lambda ps: windows.kaiser(ps, beta=window_alpha * np.pi, sym=True),
        "tukey": lambda ps: windows.tukey(ps, alpha=window_alpha, sym=True),
        "gaussian": lambda ps: windows.gaussian(
            ps, std=window_alpha * ps / 2, sym=True
        ),
        "KBD": lambda ps: windows.kaiser_bessel_derived(ps, window_alpha * np.pi),
    }
    if type not in functions.keys():
        raise ValueError(
            f"Unknown window function type {type}. Supported types are {list(functions.keys())}"
        )

    window = torch.tensor(functions[type](patch_size), **kwargs)
    window = window.unsqueeze(0) * window.unsqueeze(1)
    return window


def load_checkpoint(path: str, *, network, optimizer, scheduler, map_location) -> int:
    with cbottle.checkpointing.Checkpoint(path) as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.read_model(net=network.module)
        else:
            checkpoint.read_model(net=network)

        with checkpoint.open("loop_state.pth", "r") as f:
            training_state = torch.load(
                f, weights_only=False, map_location=map_location
            )
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            step = training_state["step"]

    return step


# loading function for teacher model
def load_teacher_checkpoint(path: str, *, network, map_location) -> int:
    with cbottle.checkpointing.Checkpoint(path) as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.read_model(net=network.module)
        else:
            checkpoint.read_model(net=network)

        with checkpoint.open("loop_state.pth", "r") as f:
            training_state = torch.load(f, weights_only=True, map_location=map_location)
            step = training_state["step"]

    return step


def _save_checkpoint(path, *, model_config, network, optimizer, scheduler, step, loss):
    with cbottle.checkpointing.Checkpoint(path, "w") as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.write_model(network.module)
        else:
            checkpoint.write_model(network)
        checkpoint.write_model_config(model_config)

        with checkpoint.open("loop_state.pth", "w") as f:
            torch.save(
                {
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                },
                f,
            )


def save_checkpoint(
    output_path: str,
    torch_compile: bool = False,
    *,
    model_config,
    network,
    optimizer,
    scheduler,
    step,
    loss,
):
    # save distilled checkpoint
    file_name = "cBottle-SR-Distill-{}.zip".format(step)
    output_path = os.path.join(output_path, file_name)
    if torch_compile:
        _save_checkpoint(
            path=output_path,
            model_config=model_config,
            network=network._orig_mod,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            loss=loss,
        )
    else:
        _save_checkpoint(
            path=output_path,
            model_config=model_config,
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            loss=loss,
        )


def find_latest_checkpoint(output_path: str) -> str:
    max_index_file = " "
    max_index = -1
    for filename in os.listdir(output_path):
        if filename.startswith("cBottle-SR-") and filename.endswith(".zip"):
            index_str = filename.split("-")[-1].split(".")[0]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_index_file = filename
            except ValueError:
                continue
    path = os.path.join(output_path, max_index_file)
    return path


class Mockdataset(torch.utils.data.Dataset):
    grid = earth2grid.healpix.Grid(
        level=10, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    fields_out = HealpixDatasetV5.fields_out

    def __getitem__(self, i):
        npix = 12 * 4**self.grid.level
        return {"target": torch.randn(len(HealpixDatasetV5.fields_out), 1, npix)}

    def __len__(self):
        return 1


def denormalize(dataset, inp):
    denormalized_inp = inp * dataset._scale.view(1, -1, 1, 1) + dataset._mean.view(
        1, -1, 1, 1
    )
    return denormalized_inp.numpy()


def to_channels_last(lpe, ltarget, llr):
    if _is_apex_available:
        lpe = lpe.to(memory_format=torch.channels_last)
        ltarget = ltarget.to(memory_format=torch.channels_last)
        llr = llr.to(memory_format=torch.channels_last)
    return lpe, ltarget, llr


def get_optimizer(distill_cfg, model):
    params = list(filter(lambda kv: "pos_embed" in kv[0], model.named_parameters()))
    base_params = list(
        filter(lambda kv: "pos_embed" not in kv[0], model.named_parameters())
    )

    params = [i[1] for i in params]
    base_params = [i[1] for i in base_params]
    optim_cls = getattr(torch.optim, distill_cfg.optimizer_name)
    lr_param = distill_cfg.get("lr_param", 5e-4)

    return optim_cls(
        [
            {
                "params": base_params,
            },  # will get lr=0.0001 (from global `lr`)
            {"params": params, "lr": lr_param},  # override lr for this group
        ],
        **distill_cfg.optimizer,
    )


def get_scheduler(distill_cfg, optimizer):
    if distill_cfg.scheduler_name is None:
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )  # nul scheduler, lr stays constant
    else:
        name = distill_cfg.scheduler_name
        cfg = getattr(distill_cfg.scheduler, distill_cfg.scheduler_name)
        schedule = getattr(lr_scheduler, name)(**cfg)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=schedule,
        )
    return scheduler


def validate_patch(
    output_batch_valid: dict,
    ltarget: torch.Tensor,
    test_dataset: HealpixDatasetV5,
    bf16: bool = True,
):
    wandb_log = {}
    # generate images and log to wandb
    image_out = output_batch_valid["gen_rand"]
    if isinstance(image_out, Callable):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            image_out = image_out()
    assert isinstance(image_out, torch.Tensor)

    # log first element in batch
    images = {
        "pred": image_out,
        "truth": ltarget,
    }

    # denormalization
    images = {
        name: denormalize(test_dataset, img.float().cpu())
        for name, img in images.items()
    }
    for batch_idx in range(images["pred"].shape[0]):
        for channel_idx in range(images["pred"].shape[1]):
            channel_name = test_dataset.fields_out[channel_idx]
            channel_min = np.min(images["truth"][batch_idx, channel_idx])
            channel_max = np.max(images["truth"][batch_idx, channel_idx])
            span = (channel_max - channel_min) * 1.5
            channel_images = []
            for name, img in images.items():
                img = img[batch_idx, channel_idx]
                img = (img - channel_min + 0.25 * span) / (1.5 * span)
                img = (img * 255).clip(0, 255).astype(np.uint8)
                channel_images.append(wandb.Image(img, caption=name))
            wandb_log[f"images/{channel_name}_{batch_idx}"] = channel_images

    # free memory
    del images, image_out
    return wandb_log


def train_for_step(
    *,
    model,
    lpe,
    ltarget,
    llr,
    optimizer,
    loss_fn: DistillLoss,
    distill_cfg,
    callbacks,
    unwrapped_model,
    bf16,
    step,
    WORLD_RANK,
    WORLD_SIZE,
    train_batch_size,
):
    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        loss, loss_map, output_batch = loss_fn(
            model,
            img_clean=ltarget,
            img_lr=llr,
            pos_embed=lpe,
            iteration=step,
        )
    loss = loss.sum()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.module.net.net.parameters(), distill_cfg.grad_clip_threshold
    )
    optimizer.step()
    dist.all_reduce(loss)
    loss_value = loss.item()

    if WORLD_RANK == 0:
        training_loss = loss / WORLD_SIZE / train_batch_size
        loss_map = {f"training/{k}": v for k, v in loss_map.items()}
        loss_map.update({"training/loss": training_loss})
        if hasattr(unwrapped_model, "ratio"):
            loss_map["schedule/ratio"] = unwrapped_model.ratio
        wandb.log(loss_map, step=step)

    for callback in callbacks:
        callback.on_training_step_end(
            unwrapped_model,
            data_batch=ltarget,
            output_batch=output_batch,
            loss_dict=loss_map,
            iteration=step,
        )
    del loss_map, output_batch

    return loss_value, grad_norm


def validate_step(
    *,
    model,
    test_loader,
    patch_iterator_val,
    test_batch_size,
    loss_fn,
    step,
    is_superpatch,
    bf16,
    WORLD_RANK,
    WORLD_SIZE,
):
    val_running_loss = 0
    count = 0
    ltarget_out = None
    output_batch_valid = None

    for batch in test_loader:
        for lpe, ltarget, llr in patch_iterator_val(batch, test_batch_size):
            lpe, ltarget, llr = to_channels_last(lpe, ltarget, llr)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss_valid, loss_map_valid, output_batch_valid = loss_fn(
                    model,
                    img_clean=ltarget,
                    img_lr=llr,
                    pos_embed=lpe,
                    iteration=step,
                    is_superpatch_eval=is_superpatch,
                )
            loss_valid = loss_valid.sum()
            dist.all_reduce(loss_valid)
            count += 1
            val_running_loss += loss_valid
            ltarget_out = ltarget

            if WORLD_RANK == 0:
                loss_map_valid = {f"valid/{k}": v for k, v in loss_map_valid.items()}
                loss_map_valid.update(
                    {
                        "valid/loss": loss_valid / WORLD_SIZE / test_batch_size,
                    }
                )
                wandb.log(loss_map_valid, step=step)
        break

    return val_running_loss, count, ltarget_out, output_batch_valid


def log_diagnostics(
    *,
    model,
    step,
    train_loss,
    val_loss,
    grad_norm,
    old_pos,
    old_pos2,
    old_conv,
    old_conv2,
    tic,
):
    pos = model.module.net.net.model.pos_embed.detach().clone()

    conv = None
    for name, para in model.module.net.net.named_parameters():
        if "enc.128x128_conv.weight" in name:
            conv = para.detach().clone()

    gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    toc = time.time()

    if old_pos is not None and old_pos2 is not None:
        a = torch.sqrt(torch.sum((pos - old_pos) ** 2))
        b = torch.sqrt(torch.sum((old_pos - old_pos2) ** 2))
        corr_pos = (
            (torch.sum((pos - old_pos) * (old_pos - old_pos2)) / (a * b))
            .cpu()
            .detach()
            .numpy()
        )

        a = torch.sqrt(torch.sum((conv - old_conv) ** 2))
        b = torch.sqrt(torch.sum((old_conv - old_conv2) ** 2))
        corr_conv = (
            (torch.sum((conv - old_conv) * (old_conv - old_conv2)) / (a * b))
            .cpu()
            .detach()
            .numpy()
        )

        print(
            "  step {:8d} | loss: {:.2e}, val loss: {:.2e}, diff pos: {:.2e},"
            " corr pos: {:.2f}, diff conv: {:.2e}, corr conv: {:.2f},"
            " grad norm: {:.2e}, gpu usage: {:.3f}, time: {:6.1f} sec".format(
                step,
                train_loss,
                val_loss,
                torch.sum(torch.abs(old_pos - pos) / torch.numel(pos))
                .cpu()
                .detach()
                .numpy(),
                corr_pos,
                torch.sum(torch.abs(old_conv - conv) / torch.numel(conv))
                .cpu()
                .detach()
                .numpy(),
                corr_conv,
                grad_norm,
                gpu_memory_used,
                (toc - tic),
            ),
            flush=True,
        )
    else:
        print(
            "  step {:8d} | loss: {:.2e}, val loss: {:.2e},"
            " grad norm: {:.2e}, gpu usage: {:.3f}, time: {:6.1f} sec".format(
                step,
                train_loss,
                val_loss,
                grad_norm,
                gpu_memory_used,
                (toc - tic),
            ),
            flush=True,
        )

    new_old_pos2 = old_pos.detach().clone() if old_pos is not None else None
    new_old_conv2 = old_conv.detach().clone() if old_pos is not None else None
    return pos.detach().clone(), new_old_pos2, conv.detach().clone(), new_old_conv2


def train(
    output_path: str,
    customized_dataset=None,
    lr_level=6,
    train_batch_size=64,
    test_batch_size=64,
    valid_min_samples: int = 1,
    num_steps: int = int(4e7),
    log_freq: int = 1000,
    test_fast: bool = False,
    dataloader_num_workers: int = 3,
    bf16: bool = False,
    torch_compile=False,
    window_function=None,
    window_alpha=None,
):
    """
    Args:
        test_fast: used for rapid testing. E.g. uses mocked data to avoid
            network I/O.
    """
    cbottle_dist.init()

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    WORLD_SIZE = cbottle_dist.get_world_size()
    WORLD_RANK = cbottle_dist.get_rank()

    distill_cfg = OmegaConf.create(dataclasses.asdict(DistillConfig()))
    distill_net_config = getattr(distill_cfg, distill_cfg.mode)
    num_steps = distill_cfg.training_duration // WORLD_SIZE // train_batch_size

    if WORLD_RANK == 0:
        print(f"total batch size across gpu is {train_batch_size * WORLD_SIZE}")
        print(f"total number of training steps is {num_steps}")

    # TODO: authenticate with wandb

    os.makedirs(output_path, exist_ok=True)
    training_sampler = None
    test_sampler = None
    # dataloader
    if test_fast:
        training_dataset = Mockdataset()
        test_dataset = Mockdataset()
    elif customized_dataset:
        training_dataset = customized_dataset(
            split="train",
        )
        test_dataset = customized_dataset(
            split="test",
        )
    else:
        training_dataset = HealpixDatasetV5(
            path=config.RAW_DATA_URL,
            train=True,
            healpixpad_order=False,
            land_path=config.LAND_DATA_URL_10,
        )
        test_dataset = HealpixDatasetV5(
            path=config.RAW_DATA_URL,
            train=False,
            healpixpad_order=False,
            land_path=config.LAND_DATA_URL_10,
        )
        training_sampler = samplers.InfiniteSequentialSampler(training_dataset)
        valid_min_samples = max(valid_min_samples, WORLD_SIZE)
        test_sampler = samplers.distributed_split(
            samplers.subsample(
                test_dataset, min_samples=max(WORLD_SIZE, valid_min_samples)
            )
        )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=None,
        num_workers=dataloader_num_workers,
        sampler=training_sampler,
        pin_memory=True,
        multiprocessing_context="spawn" if dataloader_num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=None,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=0,
    )

    out_channels = len(training_dataset.fields_out)

    # initialize teacher model
    # the model takes in both local and global lr channels
    local_lr_channels = out_channels
    global_lr_channels = out_channels
    model_config = cbottle.models.ModelConfigV1(
        architecture="unet_hpx1024_patch",
        condition_channels=local_lr_channels + global_lr_channels,
        out_channels=out_channels,
    )
    img_resolution = model_config.img_resolution  # patch shape
    model_config.level = training_dataset.grid.level
    teacher_model = cbottle.models.get_model(model_config)
    teacher_model.train().requires_grad_(True).cuda()
    teacher_model.cuda(LOCAL_RANK)
    try:
        # load superresolution ckp
        map_location = {
            "cuda:%d" % 0: "cuda:%d" % int(LOCAL_RANK)
        }  # map_location='cuda:{}'.format(self.params.local_rank)
        _ = load_teacher_checkpoint(
            path=distill_cfg.teacher_ckp_path,
            network=teacher_model,
            map_location=map_location,
        )
        if WORLD_RANK == 0:
            print(
                f"Loaded teacher model checkpoint from {distill_cfg.teacher_ckp_path}"
            )
    except FileNotFoundError:
        if WORLD_RANK == 0:
            print(f"Could not load teacher model from {distill_cfg.teacher_ckp_path}")

    # set up superpatch or regular patch training
    if distill_cfg.get("patching", None) is not None:
        patch_size = img_resolution
        # compute super-patch shape for distillation
        subpatch_num = distill_cfg.patching.get("subpatch_num", 2)
        overlap_pix = distill_cfg.patching.get("overlap_pixel", 32)
        super_patch_size = subpatch_num * (patch_size - overlap_pix) + overlap_pix
        patching_cfg = {
            "patch_shape": (patch_size, patch_size),
            "overlap_pix": overlap_pix,
        }

        super_patch_shape = (super_patch_size, super_patch_size)
        is_superpatch = True
        print(f"Enabling superpatch training, superpatch shape is {super_patch_shape}")
    else:
        super_patch_size = img_resolution
        super_patch_shape = (img_resolution, img_resolution)
        patching_cfg = {}
        is_superpatch = False
        print(f"Enabling regular patch training, patch shape is {super_patch_shape}")

    device = torch.device(f"cuda:{LOCAL_RANK}")

    # whether to use window function in superpatch training
    window = None
    if window_function and is_superpatch:
        print(
            f"Using window function {window_function} with alpha {window_alpha} in superpatch training"
        )
        window = get_window_function(
            img_resolution,
            window_alpha=window_alpha,
            type=window_function,
            dtype=torch.float32,
            device=device,
        )
        window = window.reshape((1, 1, window.shape[0], window.shape[1]))
    elif window_function and not is_superpatch:
        raise ValueError(
            "Window function is not supposed to be used with regular patch training"
        )

    # Instantiate the FastGenModel
    model_cfg_update = DictConfig(
        {
            "precision": "float32",  # cannot set to bf16 as healpix pad does not support bf16
            "precision_amp": "bfloat16" if bf16 else "float32",
            "input_shape": (teacher_model.img_channels, *super_patch_shape),
            "window": window,
            **patching_cfg,
        },
        flags={"allow_objects": True},
    )

    model_cfg = OmegaConf.merge(model_cfg_update, distill_net_config.model)
    model = get_distill_model(
        teacher_model=teacher_model,
        model_cfg=model_cfg,
        distill_cfg=distill_cfg,
        device=device,
    )

    # set up callbacks for distillation training
    callbacks = []
    callbacks_cfg = distill_net_config.callbacks
    if "ema" in callbacks_cfg:
        callbacks.append(EMACallback(**distill_net_config.callbacks.ema))
    if "ct_schedule" in callbacks_cfg:
        callbacks.append(
            CTScheduleCallback(
                **distill_net_config.callbacks.ct_schedule,
                batch_size=train_batch_size * WORLD_SIZE,
            )
        )

    # initialize loss function
    loss_fn = DistillLoss(net=teacher_model)

    model.on_train_begin()

    save_models = [model.net.net, model.net.logvar_linear]
    for name in ["fake_score", "discriminator"]:
        if hasattr(model, name):
            save_models.append(getattr(model, name))
            print(f"Saving {name} model")
    if model.use_ema:
        save_models += [model.ema.net, model.ema.logvar_linear]
        print("Saving EMA model")

    # save unwrapped model for callbacks
    unwrapped_model = model

    # wrap model with DDP
    model = DDPWrapper(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)

    # initialize patch iterator
    patch_iterator = BatchedPatchIterator(
        model,
        training_dataset.grid,
        lr_level,
        super_patch_size,  # img_resolution,
    )

    if is_superpatch:  # evaluate on regular patches
        patch_iterator_val = BatchedPatchIterator(
            model,
            training_dataset.grid,
            lr_level,
            img_resolution,
        )
    else:
        patch_iterator_val = patch_iterator

    # initialize optimizer
    optimizer = get_optimizer(distill_cfg, model)

    # initialize scheduler
    scheduler = get_scheduler(distill_cfg, optimizer)

    tic = time.time()
    step = 0
    train_loss_list = []
    val_loss_list = []

    # load checkpoint
    path = find_latest_checkpoint(output_path)

    try:
        map_location = {
            "cuda:%d" % 0: "cuda:%d" % int(LOCAL_RANK)
        }  # map_location='cuda:{}'.format(self.params.local_rank)
        step = load_checkpoint(
            path,
            network=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )
        step = step + 1
        print(f"Loaded network and optimizer states from {path}")
        if WORLD_RANK == 0:
            for p in optimizer.param_groups:
                print(p["lr"], p["initial_lr"])
    except FileNotFoundError:
        if WORLD_RANK == 0:
            print("Could not load network and optimizer states")

    # torch.compile the model if specified
    if torch_compile:
        torch._dynamo.reset()
        model = torch.compile(model)
        print("torch.compiling the model")

    # training loop
    old_pos = None
    old_pos2 = None
    old_conv = None
    old_conv2 = None
    running_loss = 0

    # FastGen Initialization
    for callback in callbacks:
        callback.on_train_begin(unwrapped_model, iteration=step)

    if WORLD_RANK == 0:
        print("training begin...", flush=True)

    while True:
        for batch in training_loader:
            for lpe, ltarget, llr in patch_iterator(batch, train_batch_size):
                lpe, ltarget, llr = to_channels_last(lpe, ltarget, llr)
                step += 1

                loss_value, grad_norm = train_for_step(
                    model=model,
                    lpe=lpe,
                    ltarget=ltarget,
                    llr=llr,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    distill_cfg=distill_cfg,
                    callbacks=callbacks,
                    unwrapped_model=unwrapped_model,
                    bf16=bf16,
                    step=step,
                    WORLD_RANK=WORLD_RANK,
                    WORLD_SIZE=WORLD_SIZE,
                    train_batch_size=train_batch_size,
                )
                running_loss += loss_value

                if step % log_freq == 0:
                    with torch.no_grad():
                        val_running_loss, count, val_ltarget, output_batch_valid = (
                            validate_step(
                                model=model,
                                test_loader=test_loader,
                                patch_iterator_val=patch_iterator_val,
                                test_batch_size=test_batch_size,
                                loss_fn=loss_fn,
                                step=step,
                                is_superpatch=is_superpatch,
                                bf16=bf16,
                                WORLD_RANK=WORLD_RANK,
                                WORLD_SIZE=WORLD_SIZE,
                            )
                        )

                    if WORLD_RANK == 0:
                        train_loss_list.append(
                            running_loss / log_freq / WORLD_SIZE / train_batch_size
                        )
                        val_loss_list.append(
                            val_running_loss
                            / len(test_loader)
                            / count
                            / WORLD_SIZE
                            / test_batch_size
                        )

                        wandb.log({"training loss": train_loss_list[-1]}, step=step)
                        wandb.log({"val loss": val_loss_list[-1]}, step=step)

                        for i, param_group in enumerate(optimizer.param_groups):
                            wandb.log({f"lr/group_{i}": param_group["lr"]}, step=step)

                        wandb_log = validate_patch(
                            output_batch_valid,
                            val_ltarget,
                            test_dataset,
                            bf16,
                        )
                        wandb.log(wandb_log, step=step)

                        old_pos, old_pos2, old_conv, old_conv2 = log_diagnostics(
                            model=model,
                            step=step,
                            train_loss=train_loss_list[-1],
                            val_loss=val_loss_list[-1],
                            grad_norm=grad_norm,
                            old_pos=old_pos,
                            old_pos2=old_pos2,
                            old_conv=old_conv,
                            old_conv2=old_conv2,
                            tic=tic,
                        )

                        save_checkpoint(
                            output_path=output_path,
                            torch_compile=torch_compile,
                            model_config=model_config,
                            network=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=step,
                            loss=train_loss_list,
                        )
                        running_loss = 0.0

                        del output_batch_valid

                if step >= num_steps:
                    print("training finished!")
                    return

                scheduler.step()
