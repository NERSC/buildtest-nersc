#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 5
#SBATCH -A m3503_g

srun -n4 ./vec_add
