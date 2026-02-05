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
import pandas as pd

# ERA5 data processing configuration
TIME_CHUNK = 24  # results in around 5MB chunks
TIME_RANGE = pd.date_range("1940", "2100", freq="1h")
START_TIME = pd.Timestamp("1900-01-01")
TIME_UNITS = "minutes since 1900-1-1 0:0:0"
LEVELS = 37
OUTPUT_LEVEL = 6

# 0.25 Degree ERA5 Data
# List of paths to directories containing ERA5 netCDF files
ERA5_ROOTS = ["ERA5_sfc", "ERA5_pl", "ERA5_1h_acc_precip"]
# rclone profile to use if using a remote output location
# see https://rclone.org/ for more information on how to configure
# currently only supports S3-compatible output
ERA5_PROFILE = ""

# regridded output. netCDF format
OUTPUT_BUCKET = "path/to/ERA5_HPX6"
OUTPUT_PROFILE = ""

# curated output
ZARR_PATH = "path/to/output.zarr"
ZARR_PROFILE = ""
CELERY_BROKER = "redis://YOUR_HOSTNAME:6379/0"
