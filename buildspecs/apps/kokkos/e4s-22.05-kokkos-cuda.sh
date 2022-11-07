#!/bin/bash

set -e

# Load the right modules
module load PrgEnv-gnu

export HOME_BASE=$(pwd)

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Load the e4s and kokkos modules
module load e4s/22.05
spack load --first cmake %gcc
spack load kokkos +cuda %gcc

build_dir=$(pwd)/e4s_22.05_build_cuda
if [ ! -d $build_dir ]; then
    mkdir $build_dir
fi
cd $build_dir
rm -rf *

# Build the test 
cmake \
    -DCMAKE_CXX_EXTENSIONS=Off \
    -DCMAKE_CXX_COMPILER=CC \
    -DKokkos_ROOT=$KOKKOS_ROOT \
    ${HOME_BASE}
make -j8

# Run the test
./test
