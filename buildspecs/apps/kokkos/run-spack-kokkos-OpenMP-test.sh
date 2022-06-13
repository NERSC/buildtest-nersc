#!/bin/bash

set -e

# Load the right modules
module load PrgEnv-gnu
module load cmake
module load cudatoolkit

export HOME_BASE=$(pwd)

# Load the e4s and kokkos modules
module load e4s/21.11-tcl
#module load kokkos-kernels/3.4.01-gcc-11.2.0-mpi-cuda
spack load kokkos +openmp cuda_arch=80

if [ ! -d build_OpenMP ]; then
    mkdir build_OpenMP
fi
cd build_OpenMP
rm -rf *

# Build the test 
cmake \
    -DCMAKE_CXX_EXTENSIONS=Off \
    -DCMAKE_CXX_COMPILER=g++ \
    -DKokkos_ROOT=$KOKKOS_ROOT \
    ${HOME_BASE}
make -j64

# Run the test
./test
