import cbottle.datasets.zarr_loader as zl
import xarray as xr
from earth2grid import healpix
import torch
import cbottle.datasets.merged_dataset as md
import numpy as np
import tqdm
from train_multidiffusion import train as train_super_resolution

ds_path="/work/bm1235/m301120/dy3ha-rechunked/d3hp003.zarr/P1D_inst_z10_atm"

variable_list_2d = ["rlut", "pr"]
loaders = [
    zl.ZarrLoader(
        path=f"{ds_path}",
        variables_3d=[],
        variables_2d=[var],
        levels=[]
    )
    for var in variable_list_2d
]


def encode_task(t, d):
    t = t[0]
    d = d[0]
    condition = []  # empty; will be inferred during training
    target = [d[(var, -1)][None] for var in variable_list_2d]
    return {
        "condition": condition,
        "target": np.stack(target),
        "timestamp": t.timestamp()
    }

dataset = md.TimeMergedDataset(
    loaders[0].times,
    time_loaders=loaders,
    transform=encode_task,
    chunk_size=48,
    shuffle=True
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    num_workers=3
)

with tqdm.tqdm(unit='B', unit_scale=True) as pb:
    for i, b in enumerate(data_loader):
        if i == 20:
            break
        pb.update(b["target"].nbytes)


def dataset_wrapper(*, split: str = ""):
    valid_times = loaders[0].times
    train_times = valid_times[:int(len(valid_times) * 0.75)]
    test_times = valid_times[-1:]
    times = {"train": train_times, "test": test_times, "": valid_times}[split]
    chunk_size = {"train": 48, "test": 1, "": 1}[split]

    if times.size == 0:
        raise RuntimeError("No times are selected.")

    dataset = md.TimeMergedDataset(
        times,
        time_loaders=loaders,
        transform=encode_task,
        chunk_size=chunk_size,
        shuffle=True
    )

    # Additional metadata required for training
    dataset.grid = healpix.Grid(level=10, pixel_order=healpix.PixelOrder.NEST)
    dataset.fields_out = variable_list_2d

    return dataset


train_super_resolution(
    output_path="training_output",
    customized_dataset=dataset_wrapper,
    log_freq=100,
    train_batch_size=16,
    test_batch_size=32
)
