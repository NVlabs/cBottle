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
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

# slurm submission info
SUBMIT_ACCOUNT = os.getenv("SUBMIT_ACCOUNT")
SUBMIT_SCRIPT = os.getenv("SUBMIT_SCRIPT", "../../ord_scripts/submit_ord.sh")

# data
V6_ERA5_ZARR = os.getenv(
    "V6_ERA5_ZARR", "/global/cfs/cdirs/trn006/data/nvidia/era5_hpx_6.zarr/"
)
V6_ERA5_ZARR_PROFILE = os.getenv("V6_ERA5_ZARR_PROFILE", "")
RAW_DATA_URL_7 = "s3://ICON_cycle3_ngc3028/ngc3028_PT30M_7.zarr/"
RAW_DATA_URL_6 = os.getenv(
    "RAW_DATA_URL_6", "/global/cfs/cdirs/trn006/data/nvidia/ngc3028_PT30M_6.zarr/"
)
RAW_DATA_URL_4 = os.getenv(
    "RAW_DATA_URL_4", "s3://ICON_cycle3_ngc3028/ngc3028_PT30M_4.zarr/"
)
RAW_DATA_URL = "/global/cfs/cdirs/trn006/data/nvidia/ngc3028_PT30M_4weeks_10.zarr/"

V6_ICON_ZARR = os.getenv(
    "V6_ICON_ZARR", "/global/cfs/cdirs/trn006/data/nvidia/ICON_v6_dataset.zarr/"
)
V6_ICON_ZARR_PROFILE = ""
RAW_DATA_PROFILE = ""
SST_MONMEAN_DATA_PROFILE = ""
LAND_DATA_PROFILE = ""

LAND_DATA_URL_10 = os.getenv(
    "LAND_DATA_URL_10",
    "/global/cfs/cdirs/trn006/data/nvidia/landfraction/ngc3028_P1D_10.zarr/",
)
LAND_DATA_URL_6 = os.getenv(
    "LAND_DATA_URL_6",
    "/global/cfs/cdirs/trn006/data/nvidia/landfraction/ngc3028_P1D_6.zarr/",
)
LAND_DATA_URL_4 = os.getenv(
    "LAND_DATA_URL_6", "s3://ICON_cycle3_ngc3028/landfraction/ngc3028_P1D_4.zarr/"
)

SST_MONMEAN_DATA_URL_6 = os.getenv(
    "SST_MONMEAN_DATA_URL_6",
    "/global/cfs/cdirs/trn006/data/nvidia/ngc3028_P1D_ts_monmean_6.zarr",
)
SST_MONMEAN_DATA_URL_4 = os.getenv("SST_MONMEAN_DATA_URL_4", None)

# ERA5 Data Paths
ERA5_HPX64_PATH = os.getenv("ERA5_HPX64_PATH", "")
ERA5_NPY_PATH_4 = os.getenv("ERA5_NPY_PATH_4", "")

# SST data from CMIP
# Data is downloaded from the https://aims2.llnl.gov/search
AMIP_MID_MONTH_SST = os.getenv(
    "AMIP_MID_MONTH_SST",
    "s3://input4MIPs/tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc",
)
AMIP_MID_MONTH_SST_PROFILE = os.getenv("AMIP_MID_MONTH_SST_PROFILE", "pbss")

# project file
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT", "/global/cfs/cdirs/trn006/data/nvidia/cBottle/"
)
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(PROJECT_ROOT, "datasets"))
CHECKPOINT_ROOT = os.getenv(
    "CHECKPOINT_ROOT", os.path.join(PROJECT_ROOT, "training-runs")
)

# ERA5 data processing
BEANSTALKD_HOST = os.getenv("BEANSTALKD_HOST", "fb7510c-lcedt.dyn.nvidia.com")
BEANSTALKD_PORT = int(os.getenv("BEANSTALKD_PORT", "11300"))
BEANSTALKD_TUBE = "era5hpx64"
