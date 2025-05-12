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
import earth2grid
import torch.distributed as dist
import argparse
import cbottle.models
from cbottle.datasets import samplers
from cbottle import healpix_utils
import cbottle.checkpointing
from cbottle.datasets.dataset_2d import HealpixDatasetV5
import cbottle.config.environment as config


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


def load_checkpoint(path: str, *, network, optimizer, scheduler, map_location) -> int:
    with cbottle.checkpointing.Checkpoint(path) as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.read_model(net=network.module)
        else:
            checkpoint.read_model(net=network)

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


def train(
    output_path: str,
    customized_dataset=None,
    train_batch_size=15,
    test_batch_size=30,
    valid_min_samples: int = 1,
    num_steps: int = int(4e7),
    log_freq: int = 1000,
    test_fast: bool = False,
):
    """
    Args:
        test_fast: used for rapid testing. E.g. uses mocked data to avoid
            network I/O.
    """
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    WORLD_RANK = int(os.getenv("RANK", 0))

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12345")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=WORLD_SIZE, rank=WORLD_RANK
        )

    torch.cuda.set_device(LOCAL_RANK)

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
    # TODO use upscaling rate instead of hardcoded 6
    low_res_grid = earth2grid.healpix.Grid(
        6, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    lat = torch.linspace(-90, 90, 128)[:, None]
    lon = torch.linspace(0, 360, 128)[None, :]
    regrid_to_latlon = low_res_grid.get_bilinear_regridder_to(lat, lon).cuda()
    regrid = earth2grid.get_regridder(low_res_grid, training_dataset.grid)
    regrid.cuda().float()
    loss_fn = EDMLossSR()
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
    net = cbottle.models.get_model(model_config)
    net.train().requires_grad_(True).cuda()
    net.cuda(LOCAL_RANK)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[LOCAL_RANK], find_unused_parameters=False
    )

    # optimizer
    params = list(filter(lambda kv: "pos_embed" in kv[0], net.named_parameters()))
    base_params = list(
        filter(lambda kv: "pos_embed" not in kv[0], net.named_parameters())
    )
    params = [i[1] for i in params]
    base_params = [i[1] for i in base_params]
    optimizer = torch.optim.SGD(
        [{"params": base_params}, {"params": params, "lr": 5e-4}], lr=1e-7, momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.6)
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
        )
        step = step + 1
        print(f"Loaded network and optimizer states from {path}")
        if WORLD_RANK == 0:
            for p in optimizer.param_groups:
                print(p["lr"], p["initial_lr"])
    except FileNotFoundError:
        if WORLD_RANK == 0:
            print("Could not load network and optimizer states")

    # training loop
    old_pos = None
    old_pos2 = None
    old_conv = None
    old_conv2 = None
    running_loss = 0

    if WORLD_RANK == 0:
        print("training begin...", flush=True)

    while True:
        for batch in training_loader:
            data = batch["target"][:, 0].cuda()
            with torch.no_grad():
                data = data.cuda(non_blocking=True)
                target = data
                # get low res
                lr = data
                for _ in range(training_dataset.grid.level - low_res_grid.level):
                    lr = healpix_utils.average_pool(lr)
                global_lr = regrid_to_latlon(lr.double())[None,].cuda()
                lr = regrid(lr)
            for lpe, ltarget, llr, end_flag in healpix_utils.to_patches(
                [net.module.model.pos_embed, target, lr],
                patch_size=img_resolution,
                batch_size=train_batch_size,
            ):
                step += 1
                optimizer.zero_grad()
                # Compute the loss and its gradients
                llr = torch.cat((llr, global_lr.repeat(llr.shape[0], 1, 1, 1)), dim=1)
                loss = loss_fn(net, img_clean=ltarget, img_lr=llr, pos_embed=lpe)
                loss = loss.sum()
                # destroy the current graph if end_flag is given
                if end_flag:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e6)
                optimizer.step()
                # avoid synchronizing gpu
                dist.all_reduce(loss)
                running_loss += loss.item()

                # logging
                if step % log_freq == 0:
                    with torch.no_grad():
                        val_running_loss = 0
                        for batch in test_loader:
                            data = batch["target"][:, 0].cuda()
                            target = data
                            # get low res
                            lr = data
                            for _ in range(
                                training_dataset.grid.level - low_res_grid.level
                            ):
                                lr = healpix_utils.average_pool(lr)
                            lr = regrid(lr)
                            # validation loss
                            count = 0
                            for lpe, ltarget, llr, _ in healpix_utils.to_patches(
                                [net.module.model.pos_embed, target, lr],
                                patch_size=img_resolution,
                                batch_size=test_batch_size,
                            ):
                                llr = torch.cat(
                                    (llr, global_lr.repeat(llr.shape[0], 1, 1, 1)),
                                    dim=1,
                                )
                                loss = loss_fn(
                                    net, img_clean=ltarget, img_lr=llr, pos_embed=lpe
                                )
                                loss = loss.sum()
                                dist.all_reduce(loss)
                                count += 1
                                val_running_loss += loss
                            break

                        # print out
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
                            pos = net.module.model.pos_embed.detach().clone()
                            for name, para in net.named_parameters():
                                if "enc.128x128_conv.weight" in name:
                                    conv = para.detach().clone()
                            gpu_memory_used = torch.cuda.memory_allocated() / (
                                1024 * 1024 * 1024
                            )  # Convert to GB
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
                                print(
                                    "  step {:8d} | loss: {:.2e}, val loss: {:.2e}, diff pos: {:.2e}, corr pos: {:.2f}, diff conv: {:.2e}, corr conv: {:.2f}, grad norm: {:.2e}, gpu usage: {:.3f}, time: {:6.1f} sec".format(
                                        step,
                                        train_loss_list[-1],
                                        val_loss_list[-1],
                                        torch.sum(
                                            torch.abs(old_pos - pos) / torch.numel(pos)
                                        )
                                        .cpu()
                                        .detach()
                                        .numpy(),
                                        corr_pos,
                                        torch.sum(
                                            torch.abs(old_conv - conv)
                                            / torch.numel(conv)
                                        )
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
                                    "  step {:8d} | loss: {:.2e}, val loss: {:.2e}, grad norm: {:.2e}, gpu usage: {:.3f}, time: {:6.1f} sec".format(
                                        step,
                                        train_loss_list[-1],
                                        val_loss_list[-1],
                                        grad_norm,
                                        gpu_memory_used,
                                        (toc - tic),
                                    ),
                                    flush=True,
                                )
                            if old_pos is not None:
                                old_pos2 = old_pos.detach().clone()
                                old_conv2 = old_conv.detach().clone()
                            old_pos = pos.detach().clone()
                            old_conv = conv.detach().clone()
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
        "--log-freq", type=int, default=100, help="Log every N steps (default: 100)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(output_path=args.output_path, num_steps=1e6, log_freq=args.log_freq)
