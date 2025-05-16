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
#SBATCH --account=trn006
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu&hbm80g
#SBATCH --module=gpu,nccl-plugin
#SBATCH --image=registry.nersc.gov/m4935/cbottle
#SBATCH -o cBottle_training_2node_%j.out 
#SBATCh -e cBottle_training_2node_%j.err

source ~/cbottle-env/bin/activate

set -a
source scripts/nersc/env
set +a

export CHECKPOINT=/global/cfs/cdirs/trn006/data/nvidia/cBottle/cBottle-3d.zip
export CONSUL_PORT=8500
export CONSUL_HOST=$(cat .load-balancer)

srun bash -c '
WORKER_PORT=$((9000 + $SLURM_PROCID))
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
python3 scripts/xarray_server.py \
    --sigma-max 200 --bf16  \
    --state-path "$CHECKPOINT" \
    --start-time 2000-01-01 \
    --end-time 2001-12-31  \
    --host $HOSTNAME \
    --port $WORKER_PORT \
    --batch-size 4 \
    --freq 3h  \
    --consul-port $CONSUL_PORT \
    --consul-host $CONSUL_HOST
'
