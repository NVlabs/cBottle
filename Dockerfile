# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
     --no-install-recommends \
    ffmpeg \
    git-lfs \
    nco \
    netcdf-bin \
    && rm -rf /var/lib/apt/lists/*

# install cartopy dependencies
RUN apt-get update && apt-get install -y \
     --no-install-recommends \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/* \
    pip install cartopy

RUN pip install --no-build-isolation https://github.com/NVIDIA/torch-harmonics/archive/refs/tags/v0.7.4.tar.gz

COPY docker/install_cdo.sh /tmp/install_cdo.sh
RUN apt-get update && bash /tmp/install_cdo.sh && rm -rf /var/lib/apt/lists/*

ARG EARTH2GRID_VERSION=main
#baa00d28e647fc731dfaa44ee45a86131d50bcf5
RUN pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/${EARTH2GRID_VERSION}.tar.gz

RUN cd /opt && \
    git clone https://github.com/NVlabs/cBottle.git && \
    cd cBottle && pip install -e .

WORKDIR /opt
RUN curl -L https://github.com/ClimateGlobalChange/tempestextremes/archive/refs/tags/v2.3.1.tar.gz | tar xz && \
    cd tempestextremes-* && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ -DCMAKE_PREFIX_PATH=MPI_ROOT .. && \
    make && make install && \
    cd /tmp && rm -rf tempestextremes*

WORKDIR /workspace
