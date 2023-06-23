## Small shell script that builds and runs SLATE examples

## salloc -A xxxxx -C gpu -q regular -t 0:20:00 -N 1 --ntasks-per-node=4
## ["-N 1", "-n 8", "-t 0:10:00"]
## bash slate-e4s-build-and-run.sh

module load spack/e4s-22.05
spack env activate gcc
spack load slate lapackpp blaspp
export SLATE_DIR=$(spack location -i slate~cuda)
export BLASPP_DIR=$(spack location -i blaspp~cuda)
export LAPACKPP_DIR=$(spack location -i lapackpp)

echo $SLATE_DIR
echo $BLASPP_DIR
echo $LAPACKPP_DIR

echo "Compiling SLATE examples using CMake in build subdir"
cd examples/
mkdir build
cd build
env CXX=CC cmake ..
make

echo "Running examples using CTest"
## Use CMake/CTest to run tests
make test
