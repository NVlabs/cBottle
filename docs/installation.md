# Installation

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

## NERSC

```
module load pytorch/2.6.0
python3 -m venv --system-site-packages cbottle-env
source cbottle-env/bin/activate
CC=gcc CXX=g++ pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/c51739da94596c5520d8963644daa3d20d224154.tar.gz
```
