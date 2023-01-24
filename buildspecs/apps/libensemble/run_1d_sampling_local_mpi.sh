#!/bin/bash

module load python
module load spack/e4s-22.05

spack env activate -V e4s
spack load py-libensemble
spack load py-mpi4py@3.1.2

python test_1d_sampling.py --comms local --nworkers 4
srun --ntasks=5 --nodes=1 python test_1d_sampling.py
