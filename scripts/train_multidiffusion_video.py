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
import torch
import time
import random
import numpy as np
import earth2grid
import argparse
import einops
import cbottle.models
from cbottle.datasets import samplers
from cbottle import healpix_utils
import cbottle.checkpointing
from cbottle.datasets.dataset_2d import HealpixDatasetV5
import cbottle.config.environment as config
from cbottle import distributed as cbottle_dist
import torch.distributed as dist


class EDMLossSR:
    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, pos_embed):
        labels = None
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(img_clean) * sigma
        sigma_lr = None
        D_yn = net(
            img_clean + n,
            sigma,
            class_labels=labels,
            condition=img_lr,
            position_embedding=pos_embed,
            augment_labels=sigma_lr,
        )
        loss = weight * ((D_yn - img_clean) ** 2)
        return loss


def load_checkpoint(path: str, *, network, optimizer, scheduler, map_location, transfer_learning) -> int:
    with cbottle.checkpointing.Checkpoint(path) as checkpoint:
        if transfer_learning:
            if isinstance(network, torch.nn.parallel.DistributedDataParallel):
                _, unmatched_weights = checkpoint.read_model(net=network.module, transfer_learning=transfer_learning)
            else:
                _, unmatched_weights = checkpoint.read_model(net=network, transfer_learning=transfer_learning)
            if unmatched_weights:
                return 0
        else:
            if isinstance(network, torch.nn.parallel.DistributedDataParallel):
                checkpoint.read_model(net=network.module, transfer_learning=transfer_learning)
            else:
                checkpoint.read_model(net=network, transfer_learning=transfer_learning)
       
        with checkpoint.open("loop_state.pth", "r") as f:
            training_state = torch.load(f, weights_only=True, map_location=map_location)
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            step = training_state["step"]
        return step


def save_checkpoint(path, *, model_config, network, optimizer, scheduler, step, loss):
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
        return {"target": torch.zeros(len(HealpixDatasetV5.fields_out), 1, npix)}

    def __len__(self):
        return 1

def build_optimizer(net, temporal_lr=4e-2, pos_embed_lr=1e-1, warmup_lr=1e-4):
    groups = []
    groups.append({
        "params": [p for n, p in net.named_parameters()
                   if "temporal_attention" not in n and "pos_embed" not in n],
        "lr": warmup_lr,
        "tag": "base"
    })
    groups.append({
        "params": [p for n, p in net.named_parameters() if "temporal_attention" in n],
        "lr": temporal_lr,
        "tag": "temporal"
    })
    groups.append({
        "params": [p for n, p in net.named_parameters() if "pos_embed" in n],
        "lr": pos_embed_lr,
        "tag": "pos_embed"
    })
    return torch.optim.SGD(groups, momentum=0.9)

class BatchedPatchStreamer:
    def __init__(
        self,
        net: torch.nn.Module,
        training_dataset_grid: earth2grid.healpix.Grid,
        lr_level: int,
        time_length: int,
        img_resolution: int,
        padding: int = None,
    ):
        self.net = net
        self.lr_level = lr_level
        self.time_length = time_length
        self.img_resolution = img_resolution
        self.sr_level = training_dataset_grid.level
        self.padding = padding or img_resolution // 2

        # Setup regridders
        low_res_grid = earth2grid.healpix.Grid(
            lr_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        lat = torch.linspace(-90, 90, 128)[:, None]
        lon = torch.linspace(0, 360, 128)[None, :]
        self.regrid_to_latlon = low_res_grid.get_bilinear_regridder_to(lat, lon).cuda()
        self.regrid = earth2grid.get_regridder(low_res_grid, training_dataset_grid)
        self.regrid.cuda().float()
        self.coordinate_map = self.make_coordinate_map(self.sr_level, self.padding)

    @staticmethod
    def make_coordinate_map(level: int, padding: int, device="cuda") -> torch.Tensor:
        """
        Returns a tensor of shape (1, 12 * X * Y)
        Pixel ID layout:
            id = face * X * Y + row * Y + col
        """
        nside = 2**level
        nside_padded = nside + 2 * padding
        ids = torch.arange(12 * nside_padded**2, dtype=torch.float32, device=device)
        ids = ids.view(1, 12, nside_padded, nside_padded)
        return ids

    def __call__(self, batch, batch_size):
        data = batch["target"].cuda(non_blocking=True)
        data = einops.rearrange(data, "t c x -> (t c) x")
        target = data

        # Get low res version
        lr = data.clone()
        lr[1:-1] = 0  # set lr to zero for middle frames
        for _ in range(self.sr_level - self.lr_level):
            lr = healpix_utils.average_pool(lr)
        global_lr = self.regrid_to_latlon(lr.double())[None,].cuda()
        lr = self.regrid(lr)

        # Create patches
        patches = healpix_utils.to_patches(
            [target, lr],
            patch_size=self.img_resolution,
            batch_size=batch_size,
            padding=self.padding,
            skip_pad_tensors=[self.coordinate_map],
        )
        del target, data, lr

        for ltarget, llr, patch_coord_map, _ in patches:
            # decode id to get the patch coordinates
            ids = patch_coord_map[:, 0, 0, 0].long()  # top-left ID of every patch
            npix_padded = self.coordinate_map.shape[-1]
            face, rem = (
                torch.div(ids, npix_padded**2, rounding_mode="floor"),
                torch.remainder(ids, npix_padded**2),
            )
            row, col = (
                torch.div(rem, npix_padded, rounding_mode="floor"),
                torch.remainder(rem, npix_padded),
            )

            faces_pe = healpix_utils.to_faces(self.net.module.model.pos_embed)
            padded_pe = earth2grid.healpix.pad(faces_pe, padding=self.padding)

            # get the positional embedding slice corresponding to each patch
            lpe = torch.stack(
                [
                    padded_pe[
                        :,
                        int(f),
                        int(r) : int(r) + self.img_resolution,
                        int(c) : int(c) + self.img_resolution,
                    ]
                    for f, r, c in zip(face, row, col)
                ],
                dim=0,
            )

            global_lr_repeat = einops.repeat(
                global_lr,
                "1 (t c) x y -> (b t) c x y",
                b=llr.shape[0],
                t=self.time_length,
            )
            lpe = einops.repeat(lpe, "b c x y -> (b t) c x y", t=self.time_length)
            llr = einops.rearrange(
                llr.cuda(), "b (t c) x y -> (b t) c x y", t=self.time_length
            )
            ltarget = einops.rearrange(
                ltarget.cuda(), "b (t c) x y -> (b t) c x y", t=self.time_length
            )

            llr = torch.cat((llr, global_lr_repeat), dim=1)

            yield lpe, ltarget, llr


def train(
    output_path: str,
    customized_dataset=None,
    lr_level=6,
    train_batch_size=8,
    test_batch_size=16,
    valid_min_samples: int = 1,
    num_steps: int = int(4e7),
    log_freq: int = 1000,
    test_fast: bool = False,
    time_length: int = 7,
    transfer_learning: bool = False,
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

    os.makedirs(output_path, exist_ok=True)
    training_sampler = None
    test_sampler = None
    time_interval = 2
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
            time_length=time_length,
            time_interval=time_interval,
            land_path=config.LAND_DATA_URL_10,
        )
        test_dataset = HealpixDatasetV5(
            path=config.RAW_DATA_URL,
            train=False,
            healpixpad_order=False,
            time_length=time_length,
            time_interval=time_interval,
            land_path=config.LAND_DATA_URL_10,
        )
        training_sampler = samplers.InfiniteSequentialSampler(training_dataset)
        test_sampler = samplers.distributed_split(
            samplers.subsample(test_dataset, min_samples=valid_min_samples)
        )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=None,
        num_workers=0,
        sampler=training_sampler,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=None,
        sampler=test_sampler,
        num_workers=0,
    )

    loss_fn = EDMLossSR(P_mean=0.0)
    out_channels = len(training_dataset.fields_out)

    # the model takes in both local and global lr channels
    local_lr_channels = out_channels
    global_lr_channels = out_channels
    model_config = cbottle.models.ModelConfigV1(
        architecture="unet_hpx1024_patch",
        condition_channels=local_lr_channels + global_lr_channels,
        out_channels=out_channels,
    )
    img_resolution = model_config.img_resolution
    model_config.level = training_dataset.grid.level
    model_config.time_length = time_length
    net = cbottle.models.get_model(model_config)
    net.train().requires_grad_(True).cuda()
    net.cuda(LOCAL_RANK)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[LOCAL_RANK], find_unused_parameters=False
    )

    # optimizer
    optimizer = build_optimizer(net)
    warmup_steps = 2000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.6)
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
            network=net,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
            transfer_learning=transfer_learning,
        )
        step = step + 1
        print(f"Loaded network and optimizer states from {path}")
        if WORLD_RANK == 0:
            for p in optimizer.param_groups:
                print(p["lr"], p["initial_lr"])
    except FileNotFoundError:
        if WORLD_RANK == 0:
            print("Could not load network and optimizer states")

    # seed based on step and rank
    np.random.seed((WORLD_RANK + step) % (1 << 31))
    np_seed = np.random.randint(1 << 31)
    torch.manual_seed(np_seed)
    random.seed(np_seed)

    patch_streamer = BatchedPatchStreamer(
        net=net,
        training_dataset_grid=training_dataset.grid,
        lr_level=lr_level,
        time_length=time_length,
        img_resolution=img_resolution,
    )

    # training loop
    old_pos = None
    old_pos2 = None
    old_conv = None
    old_conv2 = None
    old_temporal = None
    old_temporal2 = None
    running_loss = 0

    if WORLD_RANK == 0:
        print("training begin...", flush=True)

    while True:
        for batch in training_loader:
            for lpe, ltarget, llr in patch_streamer(batch, train_batch_size):
                step += 1
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Compute the loss and its gradients
                    loss = loss_fn(net, img_clean=ltarget, img_lr=llr, pos_embed=lpe)
                    loss = loss.mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
                optimizer.step()
                # avoid synchronizing gpu
                loss = loss.detach()
                dist.all_reduce(loss)
                running_loss += loss.item()

                # logging
                if step % log_freq == 0:
                    with torch.no_grad():
                        val_running_loss = 0
                        for batch in test_loader:
                            count = 0
                            for lpe, ltarget, llr in patch_streamer(batch, test_batch_size):
                                loss = loss_fn(net, img_clean=ltarget, img_lr=llr, pos_embed=lpe)
                                loss = loss.mean()
                                dist.all_reduce(loss)
                                count += 1
                                val_running_loss += loss
                                if count == 10:
                                    break
                            break

                        # print out
                        if WORLD_RANK == 0:
                            train_loss_list.append(
                                running_loss / log_freq / WORLD_SIZE
                            )
                            val_loss_list.append(
                                val_running_loss
                                / len(test_loader)
                                / count
                                / WORLD_SIZE
                            )
                            pos = net.module.model.pos_embed.detach().clone()
                            for name, para in net.named_parameters():
                                if "enc.128x128_conv.weight" in name:
                                    conv = para.detach().clone()
                                    break
                            for name, para in net.named_parameters():
                                if "temporal_attention" in name:
                                    temporal = para.detach().clone()
                                    break
                            peak_gpu_memory_reserved = torch.cuda.max_memory_reserved() / (
                                1024 * 1024 * 1024
                            )
                            peak_gpu_memory_allocated = torch.cuda.max_memory_allocated() / (
                                1024 * 1024 * 1024
                            )
                            toc = time.time()
                            if old_pos is not None and old_pos2 is not None:
                                a = torch.sqrt(torch.sum((pos - old_pos) ** 2))
                                b = torch.sqrt(torch.sum((old_pos - old_pos2) ** 2))
                                corr_pos = (
                                    (
                                        torch.sum(
                                            (pos - old_pos) * (old_pos - old_pos2)
                                        )
                                        / (a * b)
                                    )
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                                a = torch.sqrt(torch.sum((conv - old_conv) ** 2))
                                b = torch.sqrt(torch.sum((old_conv - old_conv2) ** 2))
                                corr_conv = (
                                    (
                                        torch.sum(
                                            (conv - old_conv) * (old_conv - old_conv2)
                                        )
                                        / (a * b)
                                    )
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                                a = torch.sqrt(torch.sum((temporal - old_temporal) ** 2))
                                b = torch.sqrt(torch.sum((old_temporal - old_temporal2) ** 2))
                                corr_temporal = (
                                    (
                                        torch.sum(
                                            (temporal - old_temporal) * (old_temporal - old_temporal2)
                                        )
                                        / (a * b)
                                    )
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                                print(
                                    "  step {:8d} | loss: {:.2e}, val loss: {:.2e}, lrs: ({:.2e}, {:.2e}, {:.2e}), diff pos: {:.2e}, corr pos: {:.2f}, diff conv: {:.2e}, corr conv: {:.2f}, diff temporal: {:.2e}, corr temporal: {:.2f}, grad norm: {:.2e}, peak gpu reserved: {:.3f}, peak gpu allocated: {:.3f}, time: {:6.1f} sec".format(
                                        step,
                                        train_loss_list[-1],
                                        val_loss_list[-1],
                                        optimizer.param_groups[0]['lr'],
                                        optimizer.param_groups[1]['lr'],
                                        optimizer.param_groups[2]['lr'],
                                        torch.mean(torch.abs(old_pos - pos)).cpu().detach().numpy(),
                                        corr_pos,
                                        torch.mean(torch.abs(old_conv - conv)).cpu().detach().numpy(),
                                        corr_conv,
                                        torch.mean(torch.abs(old_temporal - temporal)).cpu().detach().numpy(),
                                        corr_temporal,
                                        grad_norm,
                                        peak_gpu_memory_reserved,
                                        peak_gpu_memory_allocated,
                                        (toc - tic),
                                    ),
                                    flush=True,
                                )
                            else:
                                print(
                                    "  step {:8d} | loss: {:.2e}, val loss: {:.2e}, lrs: ({:.2e}, {:.2e}, {:.2e}), grad norm: {:.2e}, peak gpu reserved: {:.3f}, peak gpu allocated: {:.3f}, time: {:6.1f} sec".format(
                                        step,
                                        train_loss_list[-1],
                                        val_loss_list[-1],
                                        optimizer.param_groups[0]['lr'],
                                        optimizer.param_groups[1]['lr'],
                                        optimizer.param_groups[2]['lr'],
                                        grad_norm,
                                        peak_gpu_memory_reserved,
                                        peak_gpu_memory_allocated,
                                        (toc - tic),
                                    ),
                                    flush=True,
                                )
                            if old_pos is not None:
                                old_pos2 = old_pos.detach().clone()
                                old_conv2 = old_conv.detach().clone()
                                old_temporal2 = old_temporal.detach().clone()
                            old_pos = pos.detach().clone()
                            old_conv = conv.detach().clone()
                            old_temporal = temporal.detach().clone()
                            file_name = "cBottle-SR-{}.zip".format(step)
                            save_checkpoint(
                                os.path.join(output_path, file_name),
                                model_config=model_config,
                                network=net,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                step=step,
                                loss=train_loss_list,
                            )
                            running_loss = 0.0

                if step == warmup_steps:
                    for i, group in enumerate(optimizer.param_groups):
                        if group.get("tag") == "base":
                            group["lr"] = 4e-2
                            scheduler.base_lrs[i] = 4e-2  # update scheduler reference

                if step >= num_steps:
                    print("training finished!")
                    exit(1)

                # break after a single batch if in testing mode
                scheduler.step()


def parse_args():
    parser = argparse.ArgumentParser(description="global CorrDiff")
    parser.add_argument(
        "--output-path", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "--log-freq", type=int, default=500, help="Log every N steps (default: 500)"
    )
    parser.add_argument(
        "--lr-level", type=int, default=6, help="HPX level of the low-resolution map"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=8, help="training batch size per GPU"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=8, help="validation batch size per GPU"
    )
    parser.add_argument(
        "--transfer-learning", action="store_true", help="Enable transfer learning from an image model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        output_path=args.output_path,
        num_steps=1e6,
        log_freq=args.log_freq,
        lr_level=args.lr_level,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        transfer_learning=args.transfer_learning,
    )
