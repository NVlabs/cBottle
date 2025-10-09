# 🌍 Generate KM-Scale Weather Maps with Pre-Trained Cascaded cBottle

## Step 1: Configure Data Paths

cBottle uses environment variables to configure system-specific details like the
location of datasets. This makes makes it easy to train models across different
systems and environments (see [this](https://12factor.net/config).

You may configure these env vars however you like, e.g. in your .bashrc or
submission script, but cBottle will automatically load any variables listed in
the popular `.env` format. You can start by creating a .env in the current directory
containing this

```
###############
# ERA5 inputs #
###############
V6_ERA5_ZARR=path/to/era5


########
# ICON #
########
RAW_DATA_URL=path/to/high-res/icon
V6_ICON_ZARR=<path>

LAND_DATA_URL_10=<path>
LAND_DATA_URL_6=<path>

SST_MONMEAN_DATA_URL_6=<path>

# location for training outputs
PROJECT_ROOT=<path>
```

## Step 2: Run Inference with the Coarse Generator

```bash
python scripts/inference_coarse.py cBottle-3d.zip inference_output --sample.min_samples 1
```


## Step 3 (Optional): Plot the Generated Coarse Maps

This command creates a ZIP archive in the current directory containing visualizations of all output variables.

```bash
python scripts/plot.py inference_output/0.nc coarse.zip
```


## Step 4: Super-Resolve a Subregion of the Coarse Map

If `--input-path` is not provided, the script defaults to using ICON HPX64 data.

```bash
python scripts/inference_multidiffusion.py cBottle-SR.zip superres_output \
    --input-path inference_output/0.nc \
    --overlap-size 32 \
    --super-resolution-box 0 -120 50 -40
```


## Step 5 (Optional): Plot the Super-Resolved Output

```bash
python scripts/plot.py superres_output/0.nc high_res.zip
```


# Load and Explore Zarr Datasets

## Load a Zarr Dataset and Extract a Variable

```python
import xarray as xr
ds = xr.open_zarr('/global/cfs/cdirs/m4581/gsharing/hackathon/scream-cess-healpix/scream2D_hrly_pr_hp10_v7.zarr')
pr = ds.pr[:10].load()
```

## Convert to RING Order and Compute Zonal Average

```python
from earth2grid import healpix
import torch

pr_r = healpix.reorder(torch.from_numpy(pr.values), healpix.PixelOrder.NEST, healpix.PixelOrder.RING)
avg = healpix.zonal_average(pr_r)
```

## Load Data with `ZarrLoader`

```python
import cbottle.datasets.zarr_loader as zl

loader = zl.ZarrLoader(
    path="/global/cfs/cdirs/m4581/gsharing/hackathon/scream-cess-healpix/scream2D_hrly_rlut_hp10_v7.zarr",
    variables_3d=[],
    variables_2d=["rlut"],
    levels=[]
)
```

## Create a Time-Chunked Dataset

```python
import cbottle.datasets.merged_dataset as md

dataset = md.TimeMergedDataset(
    loader.times,
    time_loaders=[loader],
    transform=lambda t, x: x[0],
    chunk_size=48,
    shuffle=True
)
```


# Train on a Custom Dataset

## Step 1: Build a Dataloader

### Load Multiple Zarr Datasets

```python
variable_list_2d = ["rlut", "pr"]
loaders = [
    zl.ZarrLoader(
        path=f"/global/cfs/cdirs/m4581/gsharing/hackathon/scream-cess-healpix/scream2D_hrly_{var}_hp10_v7.zarr",
        variables_3d=[],
        variables_2d=[var],
        levels=[]
    )
    for var in variable_list_2d
]
```

### Define a Transform Function for Each Sample

```python
import numpy as np

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
```

### Create a DataLoader

```python
dataset = md.TimeMergedDataset(
    loaders[0].times,
    time_loaders=loaders,
    transform=encode_task,
    chunk_size=48,
    shuffle=True
)

import torch
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    num_workers=3
)
```

### Monitor I/O Throughput

```python
import tqdm

with tqdm.tqdm(unit='B', unit_scale=True) as pb:
    for i, b in enumerate(data_loader):
        if i == 20:
            break
        pb.update(b["target"].nbytes)
```


## Step 2: Wrap the Dataset with a Train/Test Split

```python
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
```


## Step 3: Train the Super-Resolution Model

Requires at least 60 GB of GPU memory.  
To run on **Perlmutter**, set `-C 'gpu&hbm80g'` to request A100 80GB nodes.

```python
from train_multidiffusion import train as train_super_resolution

train_super_resolution(
    output_path="training_output",
    customized_dataset=dataset_wrapper,
    num_steps=10,
    log_freq=5
)
```
