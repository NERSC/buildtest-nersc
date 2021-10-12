// gpus_for_tasks.cpp
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <mpi.h>

int main(int argc, char **argv) {
  int deviceCount = 0;
  int rank, nprocs;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  cudaGetDeviceCount(&deviceCount);

  printf("Rank %d out of %d processes: I see %d GPU(s).\n", rank, nprocs, deviceCount);

  int dev, len = 15;
  char gpu_id[15];
  cudaDeviceProp deviceProp;

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaDeviceGetPCIBusId(gpu_id, len, dev);
    printf("%d for rank %d: %s\n", dev, rank, gpu_id);
  }

  MPI_Finalize ();

  return 0;
}
