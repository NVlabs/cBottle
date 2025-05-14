# Setting up cBottle on NERSC (Perlmutter)

## Using containers
We can use containers on Perlmutter using [Shifter](https://docs.nersc.gov/development/containers/shifter/).

Login to the NERSC container registry using:
```bash
shifterimg login registry.nersc.gov
```

Pull the cbottle container using 
```bash
shifterimg pull registry.nersc.gov/m4935/cbottle
```

Launch a coarse res generation batch job using the submission script:
```bash
sbatch -A <ACCOUNT> scripts/nersc/submit_coarse_inference.sh
```

## Training quickstart (NERSC)

Prerequisites: 
- [install cBottle](./installation.md)
- Request access to project (trn006) using Iris (?)

To get an interactive session run
```
scripts/nersc/env/interactive.sh
```

This will request a single 80 Gb A100 and also configure the environment to
point to point to the datasets at NERSC. 

> [!NOTE]
> If you are interested, you can see the configured options in the `scripts/nersc/env` file. These are loaded in [this python file](../../src/cbottle/config/environment.py).

Once you have obtained the interactive session, you can run the coarse-resolution training like this

```
# Source your venv again
source ~/cbottle-env/bin/activate

# modify these as desired
MY_OUTPUT_PATH=path/to/the/output
NAME=my_experiment

# launch the training
python3 scripts/train_coarse.py --loop.noise_distribution log_uniform \
--loop.sigma_min 0.02 --loop.sigma_max 200 --loop.label_dropout 0.25 \
--loop.batch_gpu 4 --loop.batch_size 64 --loop.dataloader_num_workers 8 \
--loop.with_era5 --loop.use_labels  --loop.data_version 6 \
--loop.monthly_sst_input --name v6data  --loop.dataloader_prefetch_factor 100 \
--output_dir $MY_OUTPUT_PATH --name $NAME
```

The outputed checkpoints will be stored at "$MY_OUTPUT_PATH/$NAME".
