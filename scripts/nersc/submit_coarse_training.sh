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
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu&hbm80g
#SBATCH --module=gpu,nccl-plugin
#SBATCH --image=registry.nersc.gov/m4935/cbottle
#SBATCH -o cBottle_training_2node_%j.out 
#SBATCh -e cBottle_training_2node_%j.err

set -x

# run training job
ROOT=$(git rev-parse --show-toplevel)
cd ${ROOT}
srun --nodes 2 --ntasks-per-node 4 --gpus-per-node 4 shifter \
    bash -c "
    unset NCCL_CROSS_NIC
    pip install -e .
    python3 scripts/train_coarse.py \
        --loop.noise_distribution log_uniform \
        --loop.sigma_min 0.02 \
        --loop.sigma_max 200 \
        --loop.label_dropout 0.25 \
        --loop.batch_gpu 4 \
        --loop.batch_size 64 \
        --loop.dataloader_num_workers 8 \
        --loop.with_era5 \
        --loop.use_labels  \
        --loop.data_version 6 \
        --loop.monthly_sst_input \
        --name v6data \
        --loop.dataloader_prefetch_factor 100
    "
