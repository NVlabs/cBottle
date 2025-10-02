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
import cbottle.inference
import torch
import cbottle.netcdf_writer
from cbottle.datasets.dataset_3d import get_dataset
import earth2grid
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Cannot do a zero-copy NCHW to NHWC")

times = pd.date_range(start="2018-09-01T16:00:00", end="2018-09-01T16:00:00", freq="1h")
lons = [-80, -53.25, -119.26]
lats = [25, 21.77, 22.97]
output_path = sys.argv[1]
for_superresolution = sys.argv[2] == "for_superresolution"

ds = get_dataset(dataset="amip")
ds.set_times(times)
loader = torch.utils.data.DataLoader(ds, batch_size=10)
batch = next(iter(loader))
model = cbottle.inference.load(
    "cbottle-3d-moe-tc",
)
indices_where_tc = model.get_guidance_pixels(lons, lats)

if for_superresolution:
    out, coords = cbottle.inference.sample_for_superresolution(
        model, batch, indices_where_tc
    )
else:
    out, coords = model.sample(
        batch,
        guidance_pixels=indices_where_tc,
    )

writer = cbottle.netcdf_writer.NetCDFWriter(
    output_path,
    config=cbottle.netcdf_writer.NetCDFConfig(hpx_level=coords.grid.level),
    rank=0,
    channels=coords.batch_info.channels,
)
writer.write_target(out, coords, timestamps=batch["timestamp"])

# convert to pixel in NEST convention
nest = earth2grid.healpix.xy2nest(
    8, indices_where_tc, model.classifier_grid.pixel_order
)
np.save(os.path.join(output_path, "indices_where_tc.npy"), nest)
