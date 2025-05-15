# AMIP Inference (HPX64)


## Prerequisites:

- [Install cBottle](./installation.md)
- Download the cBottle-3d.zip checkpoint from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle/files).


## Tutorial

You can run an SST-conditioned (AMIP) inference like this

    python scripts/inference_coarse.py \
        --sample.min_samples 1 \
        --dataset amip \
        cBottle-3d.zip \
        inference_output 

!!! note 
    If not already download, this will first download the SST data from ESGF.

    If you plan to run a big parallel job, you can prefetch this first by runing
    
        python3 scripts/download_amip_sst.py
    
    By default this will download to `~/.cache/cbottle`, but if desired you can configure the environment variables to point to the downloaded data:
    
        export AMIP_MID_MONTH_SST="/path/to/data"
    
    To persist this setting across multiple sessions, you can put this in
    your .bashrc, submission scripts, or a [dotenv file](https://hexdocs.pm/
    dotenvy/0.2.0/dotenv-file-format.html).


