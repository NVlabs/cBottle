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
# %%
import argparse

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import xarray as xr
from earth2grid import healpix

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument(
    "--background", type=str, default="black", choices=["black", "white"]
)
args = parser.parse_args()

input = args.input
output = args.output


# os.makedirs(output, exist_ok=True)

# Define min/max values for each field based on the data
field_ranges = {
    "clivi": (-1e-12, 1.5),
    "cllvi": (0, 11),
    "hfls": (-1500, 400),
    "hfss": (-1700, 1600),
    "hydro_canopy_cond_limited_box": (0, 0.04),
    "hydro_discharge_ocean_box": (-800, 50000),
    "hydro_drainage_box": (0, 0.06),
    "hydro_runoff_box": (-0.0004, 0.03),
    "hydro_snow_soil_dens_box": (0, 300),
    "hydro_transpiration_box": (-0.0003, 0.00003),
    "hydro_w_snow_box": (0, 160),
    "pr": (0, 0.07),
    "pres_msl": (95000, 104000),
    "pres_sfc": (39000, 107000),
    "prls": (0, 0.008),
    "prw": (0, 70),
    "qgvi": (-1e-12, 31),
    "qrvi": (-1e-13, 40),
    "qsvi": (-1e-12, 15),
    "rlds": (70, 500),
    "rlus": (110, 800),
    "rlut": (70, 360),
    "rsds": (0, 1130),
    "rsdt": (0, 1380),
    "rsus": (0, 620),
    "rsut": (0, 1010),
    "sfcwind": (0, 45),
    "sic": (0, 1),
    "sit": (0, 12),
    "sse_grnd_hflx_old_box": (-280, 120),
    "tauu": (-10, 10),
    "tauv": (-11, 10),
    "uas": (-40, 40),
    "vas": (-40, 40),
    "speed": (0, 30),
}

if input.endswith(".npz"):
    fields_out = [
        "cllvi",
        "clivi",
        "tas",
        "uas",
        "vas",
        "rlut",
        "rsut",
        "pres_msl",
        "pr",
        "rsds",
        "sst",
        "sic",
    ]

    mean = [
        0.05499429255723953,
        0.01109028048813343,
        286.0905456542969,
        -0.15406793355941772,
        -0.38197827339172363,
        243.5808563232422,
        88.92689514160156,
        101160.1953125,
        1.7416477930964902e-05,
        213.8172607421875,
        290.97320556640625,
        0.025404179468750954,
    ]
    scale = [
        0.14847485721111298,
        0.02724744752049446,
        15.605487823486328,
        5.174601078033447,
        4.648515224456787,
        41.99623489379883,
        128.31866455078125,
        1109.4359130859375,
        0.00010940144420601428,
        304.65899658203125,
        8.51423454284668,
        0.1454102247953415,
    ]
    data = np.load(input)["prediction"]
    data = data[0]
    data_vars = {}
    for i, field in enumerate(fields_out):
        array = data[i]
        array = healpix.reorder(
            torch.as_tensor(array), healpix.PixelOrder.NEST, healpix.PixelOrder.RING
        ).numpy()
        field = fields_out[i]
        array = array * scale[i] + mean[i]
        data_vars[field] = array

else:
    ds = xr.open_dataset(input)
    ds.info()

    data_vars = {}
    for field in ds.data_vars:
        mapping_var = ds[field].attrs.get("grid_mapping", None)
        if not mapping_var:
            continue

        if not ds[mapping_var].attrs["grid_mapping_name"] == "healpix":
            continue

        data = ds[field][0].values
        if ds[mapping_var].attrs["healpix_order"] == "nest":
            data = healpix.reorder(
                torch.as_tensor(data), healpix.PixelOrder.NEST, healpix.PixelOrder.RING
            ).numpy()
        data_vars[field] = data

data_vars["speed"] = np.sqrt(data_vars["uas"] ** 2 + data_vars["vas"] ** 2)

for field in data_vars:
    data = data_vars[field]
    vmin, vmax = field_ranges.get(field, (data.min(), data.max()))

    if field == "pr":
        data = 10 * np.log10((data * 3600).clip(0.1))
        vmin, vmax = -5, 20
        cmap = plt.cm.gist_ncar
    elif field == "tas":
        data = data - 273.15
        vmin, vmax = -40, 40
        cmap = plt.cm.RdBu_r
    else:
        cmap = plt.cm.bone

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    x = healpix.to_double_pixelization(data, fill_value=float("nan"))
    x = np.ma.masked_invalid(x)

    # Convert the normalized data to RGB using the colormap
    out_rgb = (cmap(norm(x)) * 255).astype(np.uint8)
    # Create background based on user choice
    bg_value = 255 if args.background == "white" else 0
    bg = np.full_like(out_rgb, bg_value)
    # Where the data is masked, use background color
    out_rgb = np.where(x.mask[..., None], bg, out_rgb)

    # use matplotlib to save the image

    # plt.imshow(x)
    # plt.colorbar()
    # plt.savefig(f"{output}/{field}.png")
    # plt.close()
    # Create the image from the RGB array and convert to RGB mode
    img = PIL.Image.fromarray(out_rgb).convert("RGB")

    import zipfile

    with zipfile.ZipFile(output, "a") as zip:
        with zip.open(f"{field}.png", "w") as f:
            img.save(f, "PNG")
    # img.save(f"{output}/{field}.jpg")
