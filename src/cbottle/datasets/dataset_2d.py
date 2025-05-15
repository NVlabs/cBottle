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
"""
Schema for ICON data::


xarray.Dataset {
dimensions:
        time = 96480 ;
        cell = 12582912 ;
        crs = 1 ;

variables:
        float32 clivi(time, cell) ;
                clivi:cell_methods = time: mean ;
                clivi:component = atmo ;
                clivi:grid_mapping = crs ;
                clivi:long_name = vertically integrated cloud ice ;
                clivi:units = kg m-2 ;
                clivi:vgrid = atmosphere ;
        float32 cllvi(time, cell) ;
                cllvi:cell_methods = time: mean ;
                cllvi:component = atmo ;
                cllvi:grid_mapping = crs ;
                cllvi:long_name = vertically integrated cloud water ;
                cllvi:units = kg m-2 ;
                cllvi:vgrid = atmosphere ;
        float32 crs(crs) ;
                crs:grid_mapping_name = healpix ;
                crs:healpix_nside = 1024 ;
                crs:healpix_order = nest ;
        float32 hfls(time, cell) ;
                hfls:cell_methods = time: mean ;
                hfls:component = atmo ;
                hfls:grid_mapping = crs ;
                hfls:long_name = latent heat flux ;
                hfls:standard_name = surface_downward_latent_heat_flux ;
                hfls:units = W m-2 ;
                hfls:vgrid = surface ;
        float32 hfss(time, cell) ;
                hfss:cell_methods = time: mean ;
                hfss:component = atmo ;
                hfss:grid_mapping = crs ;
                hfss:long_name = sensible heat flux ;
                hfss:standard_name = surface_downward_sensible_heat_flux ;
                hfss:units = W m-2 ;
                hfss:vgrid = surface ;
        float32 hydro_canopy_cond_limited_box(time, cell) ;
                hydro_canopy_cond_limited_box:cell_methods = time: point ;
                hydro_canopy_cond_limited_box:component = jsbach ;
                hydro_canopy_cond_limited_box:grid_mapping = crs ;
                hydro_canopy_cond_limited_box:vgrid = surface ;
        float32 hydro_discharge_ocean_box(time, cell) ;
                hydro_discharge_ocean_box:cell_methods = time: mean ;
                hydro_discharge_ocean_box:component = jsbach ;
                hydro_discharge_ocean_box:grid_mapping = crs ;
                hydro_discharge_ocean_box:units = m3 s-1 ;
                hydro_discharge_ocean_box:vgrid = surface ;
        float32 hydro_drainage_box(time, cell) ;
                hydro_drainage_box:cell_methods = time: mean ;
                hydro_drainage_box:component = jsbach ;
                hydro_drainage_box:grid_mapping = crs ;
                hydro_drainage_box:units = kg m-2 s-1 ;
                hydro_drainage_box:vgrid = surface ;
        float32 hydro_runoff_box(time, cell) ;
                hydro_runoff_box:cell_methods = time: mean ;
                hydro_runoff_box:component = jsbach ;
                hydro_runoff_box:grid_mapping = crs ;
                hydro_runoff_box:long_name = surface runoff ;
                hydro_runoff_box:units = kg m-2 s-1 ;
                hydro_runoff_box:vgrid = surface ;
        float32 hydro_snow_soil_dens_box(time, cell) ;
                hydro_snow_soil_dens_box:cell_methods = time: point ;
                hydro_snow_soil_dens_box:component = jsbach ;
                hydro_snow_soil_dens_box:grid_mapping = crs ;
                hydro_snow_soil_dens_box:long_name = Density of snow on soil ;
                hydro_snow_soil_dens_box:units = kg m-3 ;
                hydro_snow_soil_dens_box:vgrid = surface ;
        float32 hydro_transpiration_box(time, cell) ;
                hydro_transpiration_box:cell_methods = time: mean ;
                hydro_transpiration_box:component = jsbach ;
                hydro_transpiration_box:grid_mapping = crs ;
                hydro_transpiration_box:long_name = Transpiration from surface ;
                hydro_transpiration_box:units = kg m-2 s-1 ;
                hydro_transpiration_box:vgrid = surface ;
        float32 hydro_w_snow_box(time, cell) ;
                hydro_w_snow_box:cell_methods = time: point ;
                hydro_w_snow_box:component = jsbach ;
                hydro_w_snow_box:grid_mapping = crs ;
                hydro_w_snow_box:long_name = Water content of snow reservoir on surface ;
                hydro_w_snow_box:units = m ;
                hydro_w_snow_box:vgrid = surface ;
        float32 pr(time, cell) ;
                pr:cell_methods = time: mean ;
                pr:component = atmo ;
                pr:grid_mapping = crs ;
                pr:long_name = precipitation flux ;
                pr:units = kg m-2 s-1 ;
                pr:vgrid = surface ;
        float32 pres_msl(time, cell) ;
                pres_msl:cell_methods = time: point ;
                pres_msl:component = atmo ;
                pres_msl:grid_mapping = crs ;
                pres_msl:long_name = mean sea level pressure ;
                pres_msl:units = Pa ;
                pres_msl:vgrid = meansea ;
        float32 pres_sfc(time, cell) ;
                pres_sfc:cell_methods = time: point ;
                pres_sfc:component = atmo ;
                pres_sfc:grid_mapping = crs ;
                pres_sfc:long_name = surface pressure ;
                pres_sfc:standard_name = surface_air_pressure ;
                pres_sfc:units = Pa ;
                pres_sfc:vgrid = surface ;
        float32 prls(time, cell) ;
                prls:cell_methods = time: mean ;
                prls:component = atmo ;
                prls:grid_mapping = crs ;
                prls:vgrid = surface ;
        float32 prw(time, cell) ;
                prw:cell_methods = time: mean ;
                prw:component = atmo ;
                prw:grid_mapping = crs ;
                prw:long_name = vertically integrated water vapour ;
                prw:units = kg m-2 ;
                prw:vgrid = atmosphere ;
        float32 qgvi(time, cell) ;
                qgvi:cell_methods = time: mean ;
                qgvi:component = atmo ;
                qgvi:grid_mapping = crs ;
                qgvi:long_name = vertically integrated graupel ;
                qgvi:units = kg m-2 ;
                qgvi:vgrid = atmosphere ;
        float32 qrvi(time, cell) ;
                qrvi:cell_methods = time: mean ;
                qrvi:component = atmo ;
                qrvi:grid_mapping = crs ;
                qrvi:long_name = vertically integrated rain ;
                qrvi:units = kg m-2 ;
                qrvi:vgrid = atmosphere ;
        float32 qsvi(time, cell) ;
                qsvi:cell_methods = time: mean ;
                qsvi:component = atmo ;
                qsvi:grid_mapping = crs ;
                qsvi:long_name = vertically integrated snow ;
                qsvi:units = kg m-2 ;
                qsvi:vgrid = atmosphere ;
        float32 rlds(time, cell) ;
                rlds:cell_methods = time: mean ;
                rlds:component = atmo ;
                rlds:grid_mapping = crs ;
                rlds:long_name = surface downwelling longwave radiation ;
                rlds:standard_name = surface_downwelling_longwave_flux_in_air ;
                rlds:units = W m-2 ;
                rlds:vgrid = surface ;
        float32 rlus(time, cell) ;
                rlus:cell_methods = time: mean ;
                rlus:component = atmo ;
                rlus:grid_mapping = crs ;
                rlus:long_name = surface upwelling longwave radiation ;
                rlus:standard_name = surface_upwelling_longwave_flux_in_air ;
                rlus:units = W m-2 ;
                rlus:vgrid = surface ;
        float32 rlut(time, cell) ;
                rlut:cell_methods = time: mean ;
                rlut:component = atmo ;
                rlut:grid_mapping = crs ;
                rlut:long_name = toa outgoing longwave radiation ;
                rlut:standard_name = toa_outgoing_longwave_flux ;
                rlut:units = W m-2 ;
                rlut:vgrid = toa ;
        float32 rsds(time, cell) ;
                rsds:cell_methods = time: mean ;
                rsds:component = atmo ;
                rsds:grid_mapping = crs ;
                rsds:long_name = surface downwelling shortwave radiation ;
                rsds:standard_name = surface_downwelling_shortwave_flux_in_air ;
                rsds:units = W m-2 ;
                rsds:vgrid = surface ;
        float32 rsdt(time, cell) ;
                rsdt:cell_methods = time: mean ;
                rsdt:component = atmo ;
                rsdt:grid_mapping = crs ;
                rsdt:long_name = toa incident shortwave radiation ;
                rsdt:standard_name = toa_incoming_shortwave_flux ;
                rsdt:units = W m-2 ;
                rsdt:vgrid = toa ;
        float32 rsus(time, cell) ;
                rsus:cell_methods = time: mean ;
                rsus:component = atmo ;
                rsus:grid_mapping = crs ;
                rsus:long_name = surface upwelling shortwave radiation ;
                rsus:standard_name = surface_upwelling_shortwave_flux_in_air ;
                rsus:units = W m-2 ;
                rsus:vgrid = surface ;
        float32 rsut(time, cell) ;
                rsut:cell_methods = time: mean ;
                rsut:component = atmo ;
                rsut:grid_mapping = crs ;
                rsut:long_name = toa outgoing shortwave radiation ;
                rsut:standard_name = toa_outgoing_shortwave_flux ;
                rsut:units = W m-2 ;
                rsut:vgrid = toa ;
        float32 sfcwind(time, cell) ;
                sfcwind:cell_methods = time: point ;
                sfcwind:component = atmo ;
                sfcwind:grid_mapping = crs ;
                sfcwind:long_name = 10m windspeed ;
                sfcwind:units = m s-1 ;
                sfcwind:vgrid = height_10m ;
        float32 sic(time, cell) ;
                sic:cell_methods = time: point ;
                sic:component = atmo ;
                sic:grid_mapping = crs ;
                sic:long_name = fraction of ocean covered by sea ice ;
                sic:vgrid = surface ;
        float32 sit(time, cell) ;
                sit:cell_methods = time: point ;
                sit:component = atmo ;
                sit:grid_mapping = crs ;
                sit:long_name = sea ice thickness ;
                sit:units = m ;
                sit:vgrid = surface ;
        float32 sse_grnd_hflx_old_box(time, cell) ;
                sse_grnd_hflx_old_box:cell_methods = time: mean ;
                sse_grnd_hflx_old_box:component = jsbach ;
                sse_grnd_hflx_old_box:grid_mapping = crs ;
                sse_grnd_hflx_old_box:long_name = Ground heat flux (old) ;
                sse_grnd_hflx_old_box:units = J m-2 s-1 ;
                sse_grnd_hflx_old_box:vgrid = surface ;
        float32 tas(time, cell) ;
                tas:cell_methods = time: point ;
                tas:component = atmo ;
                tas:grid_mapping = crs ;
                tas:long_name = temperature in 2m ;
                tas:standard_name = air_temperature ;
                tas:units = K ;
                tas:vgrid = height_2m ;
        float32 tauu(time, cell) ;
                tauu:cell_methods = time: mean ;
                tauu:component = atmo ;
                tauu:grid_mapping = crs ;
                tauu:long_name = u-momentum flux at the surface ;
                tauu:units = N m-2 ;
                tauu:vgrid = surface ;
        float32 tauv(time, cell) ;
                tauv:cell_methods = time: mean ;
                tauv:component = atmo ;
                tauv:grid_mapping = crs ;
                tauv:long_name = v-momentum flux at the surface ;
                tauv:units = N m-2 ;
                tauv:vgrid = surface ;
        datetime64[ns] time(time) ;
                time:axis = T ;
        float32 ts(time, cell) ;
                ts:cell_methods = time: point ;
                ts:component = atmo ;
                ts:grid_mapping = crs ;
                ts:long_name = surface temperature ;
                ts:standard_name = surface_temperature ;
                ts:units = K ;
                ts:vgrid = surface ;
        float32 uas(time, cell) ;
                uas:cell_methods = time: point ;
                uas:component = atmo ;
                uas:grid_mapping = crs ;
                uas:long_name = zonal wind in 10m ;
                uas:units = m s-1 ;
                uas:vgrid = height_10m ;
        float32 vas(time, cell) ;
                vas:cell_methods = time: point ;
                vas:component = atmo ;
                vas:grid_mapping = crs ;
                vas:long_name = meridional wind in 10m ;
                vas:units = m s-1 ;
                vas:vgrid = height_10m ;

// global attributes:
}



## Land fraction data
s3://ICON_cycle3_ngc3028/landfraction/ngc3028_P1D_6.zarr
xarray.Dataset {
dimensions:
        time = 2010 ;
        cell = 49152 ;
        soil_depth_water_level = 5 ;
        soil_depth_energy_level = 5 ;

variables:
        float32 hydro_canopy_cond_limited_box(time, cell) ;
                hydro_canopy_cond_limited_box:cell_methods = time: mean cell: mean ;
                hydro_canopy_cond_limited_box:component = jsbach ;
                hydro_canopy_cond_limited_box:grid_mapping = crs ;
                hydro_canopy_cond_limited_box:vgrid = surface ;
        float32 hydro_discharge_ocean_box(time, cell) ;
                hydro_discharge_ocean_box:cell_methods = time: mean cell: mean ;
                hydro_discharge_ocean_box:component = jsbach ;
                hydro_discharge_ocean_box:grid_mapping = crs ;
                hydro_discharge_ocean_box:units = m3 s-1 ;
                hydro_discharge_ocean_box:vgrid = surface ;
        float32 hydro_drainage_box(time, cell) ;
                hydro_drainage_box:cell_methods = time: mean cell: mean ;
                hydro_drainage_box:component = jsbach ;
                hydro_drainage_box:grid_mapping = crs ;
                hydro_drainage_box:units = kg m-2 s-1 ;
                hydro_drainage_box:vgrid = surface ;
        float32 hydro_runoff_box(time, cell) ;
                hydro_runoff_box:cell_methods = time: mean cell: mean ;
                hydro_runoff_box:component = jsbach ;
                hydro_runoff_box:grid_mapping = crs ;
                hydro_runoff_box:long_name = surface runoff ;
                hydro_runoff_box:units = kg m-2 s-1 ;
                hydro_runoff_box:vgrid = surface ;
        float32 hydro_snow_soil_dens_box(time, cell) ;
                hydro_snow_soil_dens_box:cell_methods = time: mean cell: mean ;
                hydro_snow_soil_dens_box:component = jsbach ;
                hydro_snow_soil_dens_box:grid_mapping = crs ;
                hydro_snow_soil_dens_box:long_name = Density of snow on soil ;
                hydro_snow_soil_dens_box:units = kg m-3 ;
                hydro_snow_soil_dens_box:vgrid = surface ;
        float32 hydro_transpiration_box(time, cell) ;
                hydro_transpiration_box:cell_methods = time: mean cell: mean ;
                hydro_transpiration_box:component = jsbach ;
                hydro_transpiration_box:grid_mapping = crs ;
                hydro_transpiration_box:long_name = Transpiration from surface ;
                hydro_transpiration_box:units = kg m-2 s-1 ;
                hydro_transpiration_box:vgrid = surface ;
        float32 hydro_w_ice_sl_box(time, soil_depth_water_level, cell) ;
                hydro_w_ice_sl_box:cell_methods = time: mean cell: mean ;
                hydro_w_ice_sl_box:component = jsbach ;
                hydro_w_ice_sl_box:grid_mapping = crs ;
                hydro_w_ice_sl_box:long_name = Ice content in soil layers ;
                hydro_w_ice_sl_box:units = m ;
                hydro_w_ice_sl_box:vgrid = soil_depth_water ;
        float32 hydro_w_snow_box(time, cell) ;
                hydro_w_snow_box:cell_methods = time: mean cell: mean ;
                hydro_w_snow_box:component = jsbach ;
                hydro_w_snow_box:grid_mapping = crs ;
                hydro_w_snow_box:long_name = Water content of snow reservoir on surface ;
                hydro_w_snow_box:units = m ;
                hydro_w_snow_box:vgrid = surface ;
        float32 hydro_w_soil_sl_box(time, soil_depth_water_level, cell) ;
                hydro_w_soil_sl_box:cell_methods = time: mean cell: mean ;
                hydro_w_soil_sl_box:component = jsbach ;
                hydro_w_soil_sl_box:grid_mapping = crs ;
                hydro_w_soil_sl_box:long_name = Water content in soil layers ;
                hydro_w_soil_sl_box:units = m ;
                hydro_w_soil_sl_box:vgrid = soil_depth_water ;
        float32 land_fraction(cell) ;
                land_fraction:component = jsbach ;
                land_fraction:grid_mapping = crs ;
                land_fraction:units = 1 ;
                land_fraction:vgrid = surface ;
        float32 sse_grnd_hflx_old_box(time, cell) ;
                sse_grnd_hflx_old_box:cell_methods = time: mean cell: mean ;
                sse_grnd_hflx_old_box:component = jsbach ;
                sse_grnd_hflx_old_box:grid_mapping = crs ;
                sse_grnd_hflx_old_box:long_name = Ground heat flux (old) ;
                sse_grnd_hflx_old_box:units = J m-2 s-1 ;
                sse_grnd_hflx_old_box:vgrid = surface ;
        float32 sse_t_soil_sl_box(time, soil_depth_energy_level, cell) ;
                sse_t_soil_sl_box:cell_methods = time: mean cell: mean ;
                sse_t_soil_sl_box:component = jsbach ;
                sse_t_soil_sl_box:grid_mapping = crs ;
                sse_t_soil_sl_box:standard_name = soil_temperature ;
                sse_t_soil_sl_box:units = K ;
                sse_t_soil_sl_box:vgrid = soil_depth_energy ;

// global attributes:
}
"""

import datetime
import os
import pickle
import random
import warnings

import cftime
import earth2grid
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
import xarray
import zarr
import zict
from cbottle.config import environment as config
from cbottle.datasets.base import BatchInfo, TimeUnit
from cbottle.datetime import as_cftime, second_of_day

from cbottle.models.embedding import FrequencyEmbedding

# Number of classes to label datasets with. Use a high number for some breathing
# room.
MAX_CLASSES = 1024
LABELS = [
    "icon",
    "era5",
]
SST_LAND_FILL_VALUE = 290
MONTHLY_SST = "monthly_sst"

SST_MEAN = 287.6897  # K
SST_SCALE = 15.5862  # K
# index for converting the channel name to integer. For backward compatibility
# add new channel names to the end of the list.
CHANNEL_TOKENS = pd.Index(
    [
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
        MONTHLY_SST,
    ]
)


def cftime_to_timestamp(time: cftime.DatetimeGregorian) -> float:
    return datetime.datetime(
        *cftime.to_tuple(time), tzinfo=datetime.timezone.utc
    ).timestamp()


def compute_stats(data, stats_path):
    # compute mean and std from data v3. More efficient method required for larger dataset
    data_list = []
    for i, sample in enumerate(data):
        data_list.append(np.stack((sample["cllvi"], sample["clivi"]), axis=0))
    data_list = np.stack(data_list, axis=0)
    mean = np.mean(data_list, axis=(0, 2, 3), keepdims=True)
    std = np.std(data_list, axis=(0, 2, 3), keepdims=True)
    if torch.cuda.current_device() == 0:
        stats = {}
        stats["mean"] = mean
        stats["std"] = std
        np.save(stats_path, stats)
    return mean, std


def coarsen_then_interp(target, upscaling_rate):
    lr = torch.nn.functional.avg_pool2d(target, (upscaling_rate, upscaling_rate))
    lr = torch.nn.functional.interpolate(
        lr, scale_factor=(upscaling_rate, upscaling_rate), mode="bicubic"
    )
    return lr


def get_total_size(directory):
    total_size = 0
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        total_files += len(filenames)
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size / (1024**3), total_files


class _CachedZarr:
    """Cache zarr chunks for faster sequential access"""

    def __init__(self, array, cache_path, chunk_size=16, lru_size=6):
        # used to in the __reduce__ method below
        self._args = (array, cache_path, chunk_size, lru_size)
        self.chunk_size = chunk_size
        self.array = array
        self.stop_caching = False

        self.read_from_local = False
        # TODO can probably delete this, doesn't look used
        if cache_path == "":
            self.read_from_local = True
            self.on_disk_cache = None
        else:
            a = zict.File(cache_path)
            self.on_disk_cache = zict.Func(pickle.dumps, pickle.loads, a)
        # zict objects are not not pickleable so need to override pickle's ser/
        # deser logic using __reduce__
        self.in_memory_cache = zict.LRU(lru_size, {})

    def __reduce__(self):
        return (self.__class__, self._args)

    def _get_chunk(self, i):
        start = self.chunk_size * i
        stop = start + self.chunk_size
        return self.array[start:stop]

    def __getitem__(self, idx: int):
        chunk = idx // self.chunk_size
        index_in_chunk = idx % self.chunk_size
        key_in_cache = str(chunk)

        if key_in_cache in self.in_memory_cache:
            return self.in_memory_cache[key_in_cache][index_in_chunk]
        elif self.on_disk_cache and (key_in_cache in self.on_disk_cache):
            self.in_memory_cache[key_in_cache] = self.on_disk_cache[key_in_cache]
        elif self.stop_caching:
            self.in_memory_cache[key_in_cache] = self.on_disk_cache[
                random.choice(list(self.on_disk_cache.keys()))
            ]
        else:
            chunk = self._get_chunk(chunk)
            self.in_memory_cache[key_in_cache] = chunk
            if self.on_disk_cache:
                try:
                    self.on_disk_cache[key_in_cache] = chunk
                except OSError:
                    self.stop_caching = True
                    total_size, total_files = get_total_size("/tmp/healpix_icon/")
                    warnings.warn(
                        "No space left on /tmp/healpix_icon/. %d files takes %f GB of space. Starting to read the data on disk randomly. "
                        % (total_files, total_size)
                    )

        return self.in_memory_cache[key_in_cache][index_in_chunk]


class HealpixDatasetV5(torch.utils.data.Dataset):
    """Healpix Dataset with on disk and in-memory caching of the ICON data in S3

    Optimized for sequential access in time.

    """

    calendar = "standard"
    time_units = "seconds since 1970-1-1 0:0:0"
    LABEL = 0
    NGRID = 8

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

    variables_needed = [
        "cllvi",
        "clivi",
        "tas",
        "uas",
        "vas",
        "rsdt",
        "rlut",
        "rsut",
        "pres_msl",
        "pr",
        "rsds",
        "ts",
        "sic",
    ]

    condition_variables = [
        "ts_monmean",
    ]

    # to compute run:
    # $ python3 healpix_v5_compute_normalizations.py
    # and then copy paste here
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
    cond_mean = [
        SST_MEAN,
    ]
    cond_scale = [
        SST_SCALE,
    ]

    in_channels = 0
    condition_channels = len(condition_variables)
    out_channels = len(fields_out)

    batch_info = BatchInfo(
        fields_out,
        scales=scale,
        center=mean,
        time_step=30,
        time_unit=TimeUnit.MINUTE,
    )
    cond_batch_info = BatchInfo(
        condition_variables,
        scales=cond_scale,
        center=cond_mean,
        time_step=30,
        time_unit=TimeUnit.MINUTE,
    )

    def __init__(
        self,
        path: str = config.RAW_DATA_URL_6,
        train: bool = True,
        normalize: bool = True,
        yield_index: bool = False,
        healpixpad_order: bool = True,
        sst: bool = False,
        land_path: str = config.LAND_DATA_URL_6,
        sst_monmean_path: str = config.SST_MONMEAN_DATA_URL_6,
        cache: bool = False,
    ):
        """
        Args:
            shuffle_every: reshuffle inits at this frequencies. in between
                sequential samples in time are returned to optimize cache usage.
            sst: if true then the returned `condition` field will contain the monthly mean sst
            sst_monmean_path: used if sst is True. Path to the monthly mean sst dataset
        """
        self.patch_size = 128
        self.normalize = normalize
        self.train = train
        self.yield_index = yield_index
        self.healpixpad_order = healpixpad_order

        if path.startswith("s3://"):
            storage_options = dict(
                client_kwargs=dict(endpoint_url="https://pbss.s8k.io")
            )
        else:
            storage_options = None

        self.group = zarr.open_group(path, storage_options=storage_options)
        self.npix = self.group[self.variables_needed[0]].shape[-1]
        self.res_level = earth2grid.healpix.npix2level(self.npix)
        self.sst = sst
        self.cache = cache

        if self.sst:
            self.sst_ds = xarray.open_zarr(sst_monmean_path)
            self.sst_time_lower_bound = self.sst_ds.time[0].values
            self.sst_time_upper_bound = self.sst_ds.time[-1].values
            self.sst_lower = self.sst_ds.ts_monmean.sel(
                time=self.sst_ds.time[0].values
            ).values
            self.sst_upper = self.sst_ds.ts_monmean.sel(
                time=self.sst_ds.time[-1].values
            ).values

        if land_path.startswith("s3://"):
            storage_options = dict(
                client_kwargs=dict(endpoint_url="https://pbss.s8k.io")
            )
        else:
            storage_options = None

        land_data = zarr.open_group(land_path, storage_options=storage_options)
        self.land_fraction = land_data["land_fraction"][:]

        self._mean = torch.tensor(self.mean).unsqueeze(-1)
        self._scale = torch.tensor(self.scale).unsqueeze(-1)
        if self.sst:
            self._cond_mean = torch.tensor(self.cond_mean).unsqueeze(-1)
            self._cond_scale = torch.tensor(self.cond_scale).unsqueeze(-1)

        self._variables = {
            field: _CachedZarr(self.group[field], cache_path="", chunk_size=48)
            if self.cache
            else self.group[field]
            for field in self.variables_needed
        }
        num_time = self.group["time"].shape[0]

        # train test split
        num_time = self.group["cllvi"].shape[0]
        n_valid = num_time // 4
        n_train = num_time - n_valid

        time_v = self.group["time"]
        # use the xarray cftime index for convenience
        # this suppors e.g. times.season
        times = xarray.CFTimeIndex(
            cftime.num2date(
                time_v[:],
                calendar=time_v.attrs["calendar"],
                units=time_v.attrs["units"],
            )
        )

        if train:
            self.time_index = np.arange(n_train)
        else:
            self.time_index = np.arange(n_train, num_time)

        self.times = times[self.time_index]

    def compute_normalization(self):
        """
        Will compute mean scale, values are hardcoded in __init__ for simplicity

        field mean scale
        clivi 0.011090281 0.030287422
        tas 286.0906 15.653724
        uas -0.15406793 5.281562
        vas -0.3819784 4.766307
        rsdt 351.3995 453.20602
        rlut 243.58089 43.33734
        pres_msl 101160.22 1108.9742
        pr 1.7416476e-05 0.00026218474
        rsds 213.81732 307.96356
        """

        mom1 = 0
        mom2 = 0
        cmom1 = 0
        cmom2 = 0
        n = 10

        def global_average(x: np.ndarray):
            return x.mean(-1, keepdims=True)

        start = 100
        end = self.time_index.shape[0] - start
        stride = (end - start) // n
        for i in range(start, end, stride):
            raw = self.get_nest_map_for_time_index(i)
            raw = {k: torch.tensor(v) for k, v in raw.items()}
            outp = self.pack_outputs(raw)

            mom1 += global_average(outp) / n
            mom2 += global_average(outp**2) / n

            if self.sst:
                cond = self.get_ts_monmean_for_time_index(i)
                cond = torch.tensor(cond, dtype=torch.float32)

                cmom1 += global_average(cond) / n
                cmom2 += global_average(cond**2) / n

        if self.sst:
            return mom1, np.sqrt(mom2 - mom1**2), cmom1, np.sqrt(cmom2 - cmom1**2)
        else:
            return mom1, np.sqrt(mom2 - mom1**2)

    def metadata(self):
        return {}

    def __len__(self):
        return len(self.time_index)

    def randomize_initial_time(self):
        self.initial_time = random.randint(0, self.time_index.size - 1)

    @property
    def grid(self):
        return earth2grid.healpix.Grid(
            level=self.res_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )

    def get_time(self, i: int):
        time_v = self.group["time"]
        return cftime.num2date(
            time_v[i],
            calendar=time_v.attrs["calendar"],
            units=time_v.attrs["units"],
        )

    @classmethod
    def pack_inputs(cls, d):
        """prepare output tensor from inputs"""
        a = FrequencyEmbedding(cls.NGRID)(d["local_solar_second"] / 86400)
        doy = d["day_of_year"].unsqueeze(-1)
        b = FrequencyEmbedding(cls.NGRID)((doy / 365.25) % 1)
        a, b = torch.broadcast_tensors(a, b)
        out = torch.concat([a, b], dim=1)  # (npix, 32)
        return out.T  # (32, npix)

    @classmethod
    def pack_outputs(cls, d):
        arrays = []
        for field in cls.fields_out:
            arrays.append(d[field])
        return torch.stack(arrays, dim=0)

    @classmethod
    def time_embeddings(cls, day_of_year, second_of_day, lon):
        """prepare output tensor from inputs"""
        out = {}
        local_time = (second_of_day - lon * 86400 // 360) % 86400

        out["day_of_year"] = day_of_year
        out["local_solar_second"] = local_time
        return cls.pack_inputs(out)

    def get_nest_map_for_time_index(self, i: int) -> np.ndarray:
        """Get input data the entire globe for a certain time

        Returns
            out: target data in NEST format (nchan, npix)
        """
        out = {field: self._variables[field][i] for field in self.variables_needed}
        time = self.get_time(i)
        out["day_of_year"] = time.dayofyr
        out["second_of_day"] = second_of_day(time)
        # TODO return some mask for the loss function
        out["sst"] = np.where(self.land_fraction > 0, SST_LAND_FILL_VALUE, out["ts"])
        return out

    def get_ts_monmean_for_time_index(self, i: int) -> np.ndarray:
        """Get monthly mean SST data for the entire globe for a certain time

        Returns
            out: target data in NEST format (nchan, npix)
        """
        time = self.get_time(i)
        dtime = datetime.datetime.strptime(time.isoformat(), "%Y-%m-%dT%H:%M:%S")
        dtime64 = np.datetime64(dtime)
        if dtime64 <= self.sst_time_lower_bound:
            out = self.sst_lower
        elif dtime64 >= self.sst_time_upper_bound:
            out = self.sst_upper
        else:
            out = self.sst_ds.ts_monmean.interp(time=dtime).values
        return np.expand_dims(out, axis=0)

    def __getitem__(self, i):
        raw = self.get_nest_map_for_time_index(self.time_index[i])
        raw = {k: torch.tensor(v) for k, v in raw.items()}
        outp = self.pack_outputs(raw)

        if self.normalize:
            outp = (outp - self._mean) / self._scale

        if self.sst:
            cond = self.get_ts_monmean_for_time_index(self.time_index[i])
            cond = torch.tensor(cond, dtype=torch.float32)

            if self.normalize:
                cond = (cond - self._cond_mean) / self._cond_scale

        if self.healpixpad_order:
            outp = self.grid.reorder(
                earth2grid.healpix.HEALPIX_PAD_XY, outp
            )  # (cin, npix)
            if self.sst:
                cond = self.grid.reorder(
                    earth2grid.healpix.HEALPIX_PAD_XY, cond
                )  # (cin, npix)

        outp = outp.unsqueeze(1)
        if self.sst:
            cond = cond.unsqueeze(1)
        else:
            cond = outp[
                0:0
            ]  # empty condition rather than None makes downstream code easier

        labels = torch.nn.functional.one_hot(
            torch.tensor(self.LABEL), num_classes=MAX_CLASSES
        )
        out = {
            "target": outp,
            "labels": labels,
            "condition": cond,
            "second_of_day": raw["second_of_day"][None],
            "day_of_year": raw["day_of_year"][None],
            "timestamp": cftime_to_timestamp(self.get_time(i)),
        }

        if self.yield_index:
            out["index"] = i

        return out


def encode_sst(sstk, is_land=None, offset=0.0):
    # get land mask if necessary
    if is_land is None:
        if np.ma.isMaskedArray(sstk):
            is_land = sstk.mask
        else:
            is_land = np.isnan(sstk)

    monthly_sst = sstk + offset
    monthly_sst = np.where(is_land, SST_LAND_FILL_VALUE, monthly_sst)
    condition = (monthly_sst - SST_MEAN) / SST_SCALE
    condition = condition[None, None, :]
    return condition


class NetCDFWrapperV1(torch.utils.data.Dataset):
    """Wraps a inference netcdf result with diffusion training outputs"""

    batch_info = HealpixDatasetV5.batch_info
    fields_out = HealpixDatasetV5.fields_out
    calendar = "standard"
    time_units = "seconds since 1970-1-1 0:0:0"

    def __init__(
        self,
        ds,
        normalize=True,
        hpx_level=10,
        healpixpad_order: bool = True,
        yield_index: bool = True,
    ):
        """
        Args:
            ds: source dataset
        """
        self.yield_index = yield_index
        self._ds = ds
        self.lr_level = int(np.log2(ds["crs"].healpix_nside))
        if ds["crs"].healpix_order == "ring":
            self.in_grid = earth2grid.healpix.Grid(
                level=self.lr_level, pixel_order=earth2grid.healpix.PixelOrder.RING
            )
        else:
            self.in_grid = earth2grid.healpix.Grid(
                level=self.lr_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
            )

        self.target_grid = earth2grid.healpix.Grid(
            level=hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        self._mean = torch.tensor(self.batch_info.center).unsqueeze(-1)
        self._scale = torch.tensor(self.batch_info.scales).unsqueeze(-1)
        self.normalize = normalize
        self.healpixpad_order = healpixpad_order

    @property
    def times(self):
        return pd.to_datetime(self._ds["time"].values).tolist()

    def __len__(self):
        return len(self._ds["time"].values)

    def get_time(self, i: int):
        return as_cftime(self.times[i])

    def __getitem__(self, i):
        ds = self._ds.isel(time=i)
        numpy_arrays = []

        for var in self.batch_info.channels:
            if var in ds:
                numpy_arrays.append(ds[var].values)
            else:
                print(f"Variable {var} not found in the dataset.")

        condition = torch.from_numpy(np.stack(numpy_arrays, axis=0))
        # prepare outputs
        condition = torch.as_tensor(condition)

        if self.normalize:
            condition = (condition - self._mean) / self._scale

        # TODO
        if self.healpixpad_order:
            condition = self.in_grid.reorder(
                earth2grid.healpix.HEALPIX_PAD_XY, condition
            )
        else:
            condition = self.in_grid.reorder(
                earth2grid.healpix.PixelOrder.NEST, condition
            )

        # add singleton time dimension
        condition = condition.unsqueeze(-2)
        condition = condition.float()

        out = {
            "target": torch.zeros_like(condition),
            "condition": condition,
            "timestamp": cftime_to_timestamp(self.get_time(i)),
        }

        if self.yield_index:
            out["index"] = i

        return out

    def metadata(self):
        return {}
