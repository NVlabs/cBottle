# Zarr Data Reading and Dataloader Construction

## Step 1. Load and Understand Zarr Data

### Load Zarr Dataset and Extract a Variable

```python
import xarray as xr
from train_multidiffusion import train as train_super_resolution

ds = xr.open_zarr('scream2D_hrly_pr_hp10_v7.zarr')
pr = ds.pr[:10].load()
```

### Reorder HEALPix from NEST to RING and Compute Zonal Average

```python
from earth2grid import healpix
import torch

pr_r = healpix.reorder(torch.from_numpy(pr.values), healpix.PixelOrder.NEST, healpix.PixelOrder.RING)
avg = healpix.zonal_average(pr_r)
```

### Load Zarr Data Using ZarrLoader

```python
import cbottle.datasets.zarr_loader as zl

loader = zl.ZarrLoader(
    path="scream2D_hrly_rlut_hp10_v7.zarr",
    variables_3d=[],
    variables_2d=["rlut"],
    levels=[]
)
```

### Create a Time-Chunked Dataset with Optional Shuffling

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

---

## Step 2. Build a Dataloader

### Load Zarr Datasets for Multiple Variables

```python
variable_list_2d = ["rlut", "pr"]
loaders = [
    zl.ZarrLoader(
        path=f"scream2D_hrly_{var}_hp10_v7.zarr",
        variables_3d=[],
        variables_2d=[var],
        levels=[]
    )
    for var in variable_list_2d
]
```

### Define Transform Function for Encoding Each Sample

```python
import numpy as np

def encode_task(t, d):
    t = t[0]
    d = d[0]
    # Dummy condition; the actual condition will be inferred from the target during training
    condition = []
    target = []
    for var in variable_list_2d:
        target.append(d[(var, -1)][None])
    return {
        "condition": condition,
        "target": np.stack(target),
        "timestamp": t.timestamp()
    }
```

### Create a Merged Dataset and DataLoader

```python
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
```

### Monitor I/O Bandwidth Over 20 Batches

```python
import tqdm

with tqdm.tqdm(unit='B', unit_scale=True) as pb:
    for i, b in enumerate(data_loader):
        if i == 20:
            break
        pb.update(b["target"].nbytes)
```

---

# Super-Resolution Training

## Step 3. Wrap Dataset with Train/Test Split

```python
def dataset_wrapper(*, split: str = ""):
    valid_times = loaders[0].times
    train_times = valid_times[:int(len(valid_times) * 0.75)]
    test_times = valid_times[-1:]
    times = {"train": train_times, "test": test_times, "": valid_times}[split]

    if times.size == 0:
        raise RuntimeError("No times are selected.")

    dataset = md.TimeMergedDataset(
        times,
        time_loaders=loaders,
        transform=encode_task,
        chunk_size=48,
        shuffle=True
    )

    # Additional metadata required for training
    dataset.grid = healpix.Grid(level=10, pixel_order=healpix.PixelOrder.NEST)
    dataset.fields_out = variable_list_2d

    return dataset
```

## Step 4. Start Training with the Custom Dataset

At least 60 GB of GPU memory is required. To train the super-resolution model on Perlmutter, set -C 'gpu&hbm80g' to request A100 80GB nodes.

```python
train_super_resolution(
    output_path="training_output",
    customized_dataset=dataset_wrapper,
    num_steps=10,
    log_freq=5
)
```

# Performing super-resolution on a sub-region
```bash
python scripts/inference_multidiffusion.py cBottle-SR.zip inference_output --overlap-size 32 --super-resolution-box 0 -120 50 -40
```
