#!/bin/bash

set -e

# Load the right modules
module load PrgEnv-gnu
module load cmake
module load cudatoolkit

export HOME_BASE=$(pwd)

# Load the e4s and kokkos modules
ml e4s/22.05

spack_kokkos_cuda_version="kokkos@3.6.00%gcc@11.2.0 +cuda+wrapper cuda_arch=80"
spack load ${spack_kokkos_cuda_version}

export KOKKOS_ROOT=$(spack location -i ${spack_kokkos_cuda_version})

if [ ! -d build_cuda ]; then
    mkdir build_cuda 
fi
cd build_cuda
rm -rf *

# Build the test 
cmake \
    -DCMAKE_CXX_EXTENSIONS=Off \
    -DCMAKE_CXX_COMPILER=${KOKKOS_ROOT}/bin/nvcc_wrapper \
    -DKokkos_ROOT=$KOKKOS_ROOT \
    ${HOME_BASE}
make -j64

# Run the test
./test
