#!/bin/bash

module load python PrgEnv-gnu cudatoolkit
module load spack/e4s-22.05
module load PrgEnv-nvidia

spack env activate -V e4s
spack load py-libensemble

cc -O3 -DGPU -fopenmp -mp=gpu -target-accel=nvidia80 -o forces.x forces.c
# run on buildtest-nersc distributed test
python run_libe_forces.py --comms local --nworkers 8
