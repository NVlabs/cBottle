#!/bin/bash
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
#SBATCH -A coreai_climate_earth2
#SBATCH -J coreai_climate_earth2-era5-prep
#SBATCH -t 04:00:00
#SBATCH -p cpu_short
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err

set -e


# mount the data and code directories
readonly _cont_mounts="/lustre:/lustre,$HOME:/root"

# Use the local image
readonly _cont_image=/lustre/fsw/portfolios/nvr/projects/nvr_earth2_e2/images/nbrenowitz/edm-chaos/latest.sqsh

cpus=$(( SLURM_JOB_CPUS_PER_NODE / 4 ))

srun -A coreai_climate_earth2\
	--container-image=${_cont_image}\
	--container-name=main2 \
	--container-mounts=${_cont_mounts}\
	--pty \
	bash -ec "
    export TMPDIR=/lustre/fsw/portfolios/coreai/projects/coreai_climate_earth2/tmp
    export PATH=$PWD:$PATH

    cd $PWD
	python3 -m pip install -r requirements.txt
  celery -A tasks worker -c $SLURM_CPUS_ON_NODE --loglevel=info
	"
