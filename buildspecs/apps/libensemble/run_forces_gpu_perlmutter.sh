#!/bin/bash

module load python
module load gpu
module load spack/e4s-22.05
module load PrgEnv-nvidia

spack env activate -V gcc
spack load py-libensemble

cc -O3 -DGPU -fopenmp -mp=gpu -target-accel=$CRAY_ACCEL_TARGET -o forces.x forces.c
python run_libe_forces.py --comms local --nworkers 8
