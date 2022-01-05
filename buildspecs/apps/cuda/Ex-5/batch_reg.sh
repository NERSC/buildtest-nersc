#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -t 5
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH -A m3503_g

lscpu | grep NUMA

echo -e "\n"
srun -n8 --cpu-bind=cores ./vec_add


