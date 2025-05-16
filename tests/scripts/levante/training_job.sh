#!/bin/bash
#SBATCH --job-name=cbottle-train-d3hp-z10      # Specify job name
#SBATCH --partition=gpu            # Specify partition name
#SBATCH --nodes=8                  # Specify number of nodes
#SBATCH --ntasks-per-node=4        # Specify number of (MPI) tasks on each node
#SBATCH --gpus-per-task=1          # Specify number of GPUs per task
#SBATCH --exclusive                # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                    # Request all memory available on all nodes
#SBATCH --time=08:00:00            # Set a limit on the total run time
#SBATCH --mail-type=FAIL           # Notify user by email in case of job failure
#SBATCH --account=                 # Charge resources on this project account
#SBATCH --output=training_job.o%j        # File name for standard output

export OMPI_MCA_pml=ucx                                 # Use UCX to support InfiniBand devices and CUDA [1]
export OMPI_MCA_btl="self"                              # Only use self transport to reduce overhead [2]
export UCX_RNDV_SCHEME=put_zcopy                        # Preferred communication scheme with Rendezvous protocol
export UCX_RNDV_THRESH=16384                            # Threshold when to switch transport from TCP to NVLINK [3]
export UCX_IB_GPU_DIRECT_RDMA=yes                       # Allow remote direct memory access from/to GPU
export UCX_TLS=cma,rc,mm,cuda_ipc,cuda_copy,gdr_copy    # Include cuda and gdr based transport layers for communication [4]
export UCX_MEMTYPE_CACHE=n      

base_dir=PATH_TO_C-BOTTLE
training_wrapper="${base_dir}/cBottle/scripts/training_wrapper.sh"

cat > ${training_wrapper} << EOF
#!/bin/bash
# Torch distributed env variables
# Set master address and port (only needs to be done once per job)
#export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
#export MASTER_PORT=29502  # Choose any free port

# Set PyTorch distributed vars from SLURM
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

source ${base_dir}/_venv/bin/activate
python ${base_dir}/cBottle/scripts/train_dkrz.py
EOF

chmod 777 ${training_wrapper}

srun bash ${training_wrapper}
