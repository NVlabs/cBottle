# Installation

cBottle has been tested on these software configurations
- Linux OS (Ubuntu 22.04)
- CUDA 12
- pytorch 2.6 and 2.7
- [earth2grid](https://github.com/nvlabs/earth2grid) v2025.6.1, v2025.7.1

Inference has been tested on
- Data center GPUS: H100, A100
- Desktop GPU: RTX 6000 Ada Generation

Training has been tested on
- H100, A100

It typically takes about 10 minutes to install earth2grid from scratch on a desktop computer, assuming that CUDA has  been installed. 5 of that is to build earth2grid from source. This could be avoided if there are binaries available for your platform [(see instructions)](https://github.com/NVlabs/earth2grid?tab=readme-ov-file#binary-release).

## PIP

If necessary install pytorch. Then, we need to install earth2grid
```
pip install torch # if necessary
pip install setuptools hatchling
pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz
```

**Warning: make sure to include `--no-build-isolation` above to avoid building earth2grid against the wrong version of pytorch. This will cause very confusing error messages during runtime.**

Now, install cbottle
```
pip install -e .
```

## NERSC (instructions)

Install the pre-requisites
```
module load pytorch/2.6.0
python3 -m venv --system-site-packages cbottle-env
source cbottle-env/bin/activate
CC=gcc CXX=g++ pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz
```

Then clone cBottle and install some other dependencies
```
git clone https://github.com/NVlabs/cBottle
cd cBottle
pip install -e .
```
