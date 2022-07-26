#!/bin/bash
#SBATCH --nodes 2
#SBATCH -q regular 
#SBATCH -t 60
#SBATCH -C cpu
#SBATCH --account=nstaff

module use /global/cfs/cdirs/m3896/shared/modulefiles
module load mvapich2/2.3.7
mpicc foo.c -o foo
srun -n 2 ./foo

