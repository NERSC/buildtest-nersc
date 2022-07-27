#!/bin/bash
#SBATCH --nodes 2
#SBATCH -q debug
#SBATCH -t 5
#SBATCH -C gpu
#SBATCH -G 4 
#SBATCH --account=m3503_g

module use /global/cfs/cdirs/m3896/shared/modulefiles
module load mvapich2/2.3.7
nvidia-smi
mpicc foo.c -o foo
srun -n 2 ./foo

