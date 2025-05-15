# Climate in a Bottle
![](https://github.com/user-attachments/assets/a2cab939-48ce-421a-8008-00b17fd6fa9f)

cBottle is an diffusion model that generates atmospheric states at kilometer resolution using a cascaded diffusion architecture.

This model is for research and development only.

[📖 arXiv](https://arxiv.org/abs/2505.06474v1) [📦 Checkpoints](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle) [📚 Documentation](https://nvlabs.github.io/cBottle/) 

## Setup

See [installation](docs/installation.md) instructions.

## Coarse Model (cBottle-3d)

### Training

```
python3 scripts/train_coarse.py --loop.noise_distribution log_uniform --loop.sigma_min 0.02 --loop.sigma_max 200 --loop.label_dropout 0.25 --loop.batch_gpu 4 --loop.batch_size 64 --loop.dataloader_num_workers 8 --loop.with_era5 --loop.use_labels  --loop.data_version 6 --loop.monthly_sst_input --name v6data  --loop.dataloader_prefetch_factor 100
```

## Coarse Video Model (cBottle-video)

###
Video training requires larger chunk sizes than image training.

```
python3 scripts/train_coarse.py
    --name v6-video \
    --loop.time_length 12 \
    --loop.time_step 6 \
    --loop.icon_chunk_size 56 \
    --loop.era5_chunk_size 96 \
    --loop.use_labels \
    --loop.label_dropout 0.05 \
    --loop.with_era5 \
    --loop.monthly_sst_input \
    --loop.noise_distribution log_uniform \
    --loop.sigma_min 0.02 \
    --loop.sigma_max 1000 \
    --loop.network.model_channels 256 \
    --loop.snapshot_ticks 1 \
    --loop.state_dump_ticks 1 \
    --loop.steps_per_tick 2000 \
    --loop.batch_gpu 1 \
    --loop.batch_size 32 \
    --loop.valid_min_samples 32 \
    --loop.dataloader_prefetch_factor 10
    --loop.dataloader_num_workers 8
```

### Inference
To create netcdf files of the generations (and optionally the corresponding ground truth), run the following:
```
torchrun --nproc-per-node 8 scripts/inference_coarse_video.py \
    /path/to/your/model.checkpoint \
    --output_path /output/path \
    --sample.save_mode all \
    --sample.frame_selection_strategy unconditional \
    --sample.denoiser_type standard \
    --sample.sigma_max 0.02 \
    --sample.sigma_max 1000
```

## Super-resolution model (cBottle-SR)

### Training
```
python3 scripts/train_multidiffusion.py --output-path OUTPUT 
```

### Inference

```
python3 scripts/inference_multidiffusion.py --input-path path/to/checkpoint output/
```


## Disclaimer

This project will download and install additional third-party open source
software projects. Review the license terms of these open source projects before
use.

