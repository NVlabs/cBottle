import cbottle.datasets.zarr_loader as zl
import cbottle.datasets.merged_dataset as md
from earth2grid import healpix
import numpy as np
from functools import partial
from types import SimpleNamespace
from inference_multidiffusion import inference as inference_super_resolution
from train_multidiffusion import train as train_super_resolution

target_hpx_level = 6
input_hpx_level = 3

# dataset build
variable_list_2d = ["rlut", "pr"]
loaders = [
    zl.ZarrLoader(
        path=f"/global/cfs/cdirs/m4581/gsharing/hackathon/scream-cess-healpix/scream2D_hrly_{var}_hp{target_hpx_level}_v7.zarr",
        variables_3d=[],
        variables_2d=[var],
        levels=[],
    )
    for var in variable_list_2d
]


def encode_task(t, d, mean, scale):
    t = t[0]
    d = d[0]
    condition = []  # empty; will be inferred during training
    target = [d[(var, -1)][None] for var in variable_list_2d]
    target = (np.stack(target) - mean) / scale
    return {
        "condition": condition,
        "target": target.astype(np.float32),
        "timestamp": t.timestamp(),
    }


def dataset_wrapper(*, split: str = ""):
    valid_times = loaders[0].times
    train_times = valid_times[: int(len(valid_times) * 0.75)]
    test_times = valid_times[-1:]
    times = {"train": train_times, "test": test_times, "": valid_times}[split]
    chunk_size = {"train": 48, "test": 1, "": 1}[split]
    infinite = {"train": True, "test": False, "": False}[split]
    if times.size == 0:
        raise RuntimeError("No times are selected.")
    mean = np.zeros((len(variable_list_2d), 1, 1))
    scale = np.ones((len(variable_list_2d), 1, 1))
    dataset = md.TimeMergedDataset(
        times,
        time_loaders=loaders,
        transform=partial(encode_task, mean=mean, scale=scale),
        chunk_size=chunk_size,
        infinite=infinite,
        shuffle=True,
    )

    # Additional metadata required for training and inference
    dataset.grid = healpix.Grid(
        level=target_hpx_level, pixel_order=healpix.PixelOrder.NEST
    )
    dataset.fields_out = variable_list_2d
    dataset._scale = scale
    dataset._mean = scale
    dataset.batch_info = SimpleNamespace()
    dataset.batch_info.channels = variable_list_2d
    dataset.time_units = "seconds since 1970-1-1 0:0:0"
    dataset.calendar = "standard"
    return dataset


# super-res model training


train_super_resolution(
    output_path="training_output",
    customized_dataset=dataset_wrapper,
    num_steps=1000,
    log_freq=100,
    lr_level=input_hpx_level,
    train_batch_size=50,
    test_batch_size=50,
)

# super-res model inferencing

arg_list = [
    "training_output/cBottle-SR-800.zip",
    "inference_superres/",
    "--min-samples",
    "1",
    "--level",
    f"{target_hpx_level}",
    "--level-lr",
    f"{input_hpx_level}",
    "--patch-size",
    "128",
    "--overlap-size",
    "32",
    "--num-steps",
    "18",
    "--plot-sample",
]
inference_super_resolution(arg_list, dataset_wrapper)

# plots will be stored in the working directory
