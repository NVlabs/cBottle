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

CACHE_DIR = os.path.expanduser("~/.cache/cbottle")

###############
# ERA5 inputs #
###############
V6_ERA5_ZARR = os.getenv("V6_ERA5_ZARR", "")
V6_ERA5_ZARR_PROFILE = os.getenv("V6_ERA5_ZARR_PROFILE", "")

# SST data from CMIP
AMIP_MID_MONTH_SST = os.getenv(
    "AMIP_MID_MONTH_SST",
    os.path.join(
        CACHE_DIR,
        "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc",
    ),
)
AMIP_MID_MONTH_SST_PROFILE = os.getenv("AMIP_MID_MONTH_SST_PROFILE", "")
AIMIP_1ST_MONTH_SST = os.getenv("AIMIP_1ST_MONTH_SST", "")
AIMIP_1ST_MONTH_SST_PROFILE = os.getenv("AIMIP_1ST_MONTH_SST_PROFILE", "")

########
# ICON #
########
RAW_DATA_URL = os.getenv("RAW_DATA_URL", "")
RAW_DATA_PROFILE = os.getenv("RAW_DATA_PROFILE", "")
RAW_DATA_URL_6 = os.getenv("RAW_DATA_URL_6", "")

V6_ICON_ZARR = os.getenv("V6_ICON_ZARR", "")
V6_ICON_ZARR_PROFILE = os.getenv("V6_ICON_ZARR_PROFILE", "")

LAND_DATA_URL_10 = os.getenv(
    "LAND_DATA_URL_10",
    "",
)
LAND_DATA_URL_6 = os.getenv(
    "LAND_DATA_URL_6",
    "",
)
LAND_DATA_PROFILE = os.getenv("LAND_DATA_PROFILE", "")


SST_MONMEAN_DATA_URL_6 = os.getenv(
    "SST_MONMEAN_DATA_URL_6",
    "",
)
SST_MONMEAN_DATA_PROFILE = os.getenv("SST_MONMEAN_DATA_PROFILE", "")


# project file
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(PROJECT_ROOT, "datasets"))
CHECKPOINT_ROOT = os.getenv(
    "CHECKPOINT_ROOT", os.path.join(PROJECT_ROOT, "training-runs")
)

# IBTrACS data path
IBTRACS_DATA_PATH = os.getenv(
    "IBTRACS_DATA_PATH", os.path.join(CACHE_DIR, "ibtracs.nc")
)
