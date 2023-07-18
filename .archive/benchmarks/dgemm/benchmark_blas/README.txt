module swap PrgEnv-intel PrgEnv-gnu
make # select line 67,68,69 or 70 of Makefile to use different blas lib available on Cori.
allocate 1 interactive haswell node

export OMP_NUM_THREADS=4 # 1 4 16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 1 -c 64 --cpu_bind=cores ./run
