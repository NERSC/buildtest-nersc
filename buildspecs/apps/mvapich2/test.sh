#!/bin/bash
#SBATCH --nodes 2
#SBATCH -q regular 
#SBATCH -t 60
#SBATCH -C gpu
#SBATCH --gpus 4
#SBATCH --account=nstaff_g

module use /global/common/software/spackecp/perlmutter/mvapich2/modulefiles
module load mvapich2
mpicc foo.c -o foo
mpirun -np 2 ./foo

