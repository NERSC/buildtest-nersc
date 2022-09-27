#!/bin/bash

set -e

# Load the right modules
module load PrgEnv-gnu

export HOME_BASE=$(pwd)

# Load the e4s and kokkos modules
module load e4s/22.05

spack load kokkos +openmp %gcc

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
