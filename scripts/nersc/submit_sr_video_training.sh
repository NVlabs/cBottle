#!/bin/bash
#SBATCH --account=trn006
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu&hbm80g
#SBATCH --module=gpu,nccl-plugin
#SBATCH --image=registry.nersc.gov/m4935/cbottle
#SBATCH -o cBottle_training_%j.out 
#SBATCh -e cBottle_training_%j.err
#SBATCH --dependency=singleton

set -x

# run training job
ROOT=$(git rev-parse --show-toplevel)
cd ${ROOT}
srun --nodes 8 --ntasks-per-node 4 --gpus-per-node 4 shifter \
    bash -c "
    unset NCCL_CROSS_NIC
    pip install -e .
    python scripts/train_multidiffusion_video.py \
    --output-path /lustre/fsw/portfolios/coreai/projects/coreai_climate_earth2/icon_hpx_hack/training-runs/hpx64cond1024_video \
    --transfer-learning"
