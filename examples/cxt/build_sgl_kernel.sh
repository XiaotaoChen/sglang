#!/bin/bash
export CCACHE_DIR=/cfs/xtchen/ccache/sgl-kernel-cu124
export CCACHE_BACKEND=""
export CCACHE_KEEP_LOCAL_STORAGE="TRUE"
unset CCACHE_READONLY
export CC="ccache gcc"
export CXX="ccache g++"
export CUDA_NVCC="ccache nvcc"


### sometime build long time no response, ctrl-c and re-compile it.

# CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) python3 -m uv build --wheel -Cbuild-dir=build --color=always .

# same to python3 -m uv xxx
make build