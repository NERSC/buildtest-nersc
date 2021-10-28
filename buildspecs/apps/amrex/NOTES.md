


Known requirements:

module load e4s/21.05

module load impi  -- need the intel MPI 

spack load cmake@3.20.2  -- default cmake version not high enough

*Store AMReX install dir*
export AMR_DIR=$(spack location -i amrex)

Might need:
module load PrgEnv-intel


Flags for building with CMake:

-DAMReX_DIR=${AMR_DIR}/lib/cmake/AMReX
-DCMAKE_CXX_COMPILER=$(which icpc)
-DCMAKE_C_COMPILER=$(which icc)
-DAMReX_SPACEDIM=2


after build use

srun -n 4 ./SingleVortex ../inputs max_step=1
 
SUCCESS!

Next step, adjust the yml file.


For e4s build details, see: https://github.com/spack/spack-configs/blob/main/NERSC/cori/e4s-21.05/spack.yaml 



Intel compiler versions don't match

