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
#!/bin/bash

#SBATCH --account=trn006
#SBATCH --qos=shared
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu&hbm80g
#SBATCH --module=gpu,nccl-plugin
#SBATCH --image=registry.nersc.gov/m4935/cbottle
#SBATCH -o cBottle_inference_1node_%j.out 
#SBATCh -e cBottle_inference_1node_%j.err

set -x

# run inference job
ROOT=$(git rev-parse --show-toplevel)
cd ${ROOT}
srun --nodes 1 --ntasks-per-node 1 --gpus-per-node 1 shifter \
    bash -c "
    unset NCCL_CROSS_NIC
    pip install -e .
    python scripts/inference_coarse.py \
        /global/cfs/cdirs/trn006/data/nvidia/cBottle/cBottle-3d.zip \
        inference_output \
        --sample.min_samples 1
    "
