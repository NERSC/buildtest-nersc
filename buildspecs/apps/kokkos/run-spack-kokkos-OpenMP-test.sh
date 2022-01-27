#!/bin/bash

set -xe

# Load the right modules
module load PrgEnv-gnu
module load cmake
module load cudatoolkit

export HOME_BASE=$(pwd)

# Load the e4s and kokkos modules
ml e4s/21.11-tcl
ml kokkos/3.4.01-gcc-9.3.0

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
