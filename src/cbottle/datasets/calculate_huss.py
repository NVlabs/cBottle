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
import numpy as np

# source: https://prod.ecmwf-forum-prod.compute.cci2.ecmwf.int/t/how-to-calculate-hus-at-2m-huss/1254/3


def calculate_surface_specific_humidity(sp, td):
    Rdry = 287.0597
    Rvap = 461.5250
    a1 = 611.21
    a3 = 17.502
    a4 = 32.19
    T0 = 273.16

    E = a1 * np.exp(a3 * (td - T0) / (td - a4))
    qsat = (Rdry / Rvap) * E / (sp - ((1 - Rdry / Rvap) * E))

    return qsat
