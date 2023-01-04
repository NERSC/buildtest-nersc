#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

#include "kernel.cu"

#define DEFAULT_N 1024
#define NBLOCKS(n, block_size) ((n + block_size - 1) / block_size)

#define CUDA_CALL(call) \
  { \
    cudaError err = call; \
    if (err != cudaSuccess) { \
      char* name = NULL; sprintf(name, "%s", call); \
      std::cout << "ERROR: " << name << " returned " << err << ": " << std::endl; \
      std::cout << cudaGetErrorString(err) << std::endl; \
    } \
  }

void usage (char* arg0)
{
  std::cout << "Usage: " << arg0 << " N " << std::endl;
}

void init (const long N, const double scale, double* v)
{
#pragma omp parallel for shared(scale)
  for (long i = 0; i < N; i++) {
    v[i] = scale * (double)i;
  }
}

long verify (const long N, const double* h_a, const double* h_b, const double* h_c)
{
  double tol = 1e-7;
  long nerror = 0;
#pragma omp parallel for private(chk) reduction(+:nerror)
  for (long i = 0; i < N; i++) {
    double chk = h_c[i] - (h_a[i] + h_b[i]);
    if (chk >= tol) nerror += 1;
  }
  return nerror;
}

int main (int argc, char* argv[])
{
  // Parse N from argv[1]
  long N = DEFAULT_N;
  if (argc >= 2) {
    char* arg1 = argv[1];
    if (strcmp(arg1, "-h") == 0 || strcmp(arg1, "--help") == 0) {
      usage(argv[0]);
      return 0;
    }
    N = std::atol(arg1);
    if (N <= 0) {
      std::cout << "ERROR: N must be positive" << std::endl;
      usage(argv[0]);
      return 0;
    }
  } // argc

  std::cout << "Using N = " << N << std::endl;

  // Allocate host vectors
  double* h_A = (double*)malloc((size_t)N * sizeof(double));
  double* h_B = (double*)malloc((size_t)N * sizeof(double));
  double* h_C = (double*)malloc((size_t)N * sizeof(double));

  // Initialize host vectors
  init(N, 0.001, h_A);
  init(N, -0.001, h_B);

  // Allocate device vectors
  double* d_A; CUDA_CALL(cudaMalloc(&d_A, (size_t)N * sizeof(double)));
  double* d_B; CUDA_CALL(cudaMalloc(&d_B, (size_t)N * sizeof(double)));
  double* d_C; CUDA_CALL(cudaMalloc(&d_C, (size_t)N * sizeof(double)));

  // Transfer input vectors to device
  CUDA_CALL(cudaMemcpy(d_A, h_A, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_B, h_B, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));

  // Perform vector addition using CUDA kernel
  dadd<<<NBLOCKS(N, 1024), 1024>>>(N, d_A, d_B, d_C);
  CUDA_CALL(cudaDeviceSynchronize());

  // Transfer result vector to host
  CUDA_CALL(cudaMemcpy(h_C, d_C, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));

  // Verify results
  long nerr = verify(N, h_A, h_B, h_C);
  if (nerr > 0) {
    std::cout << "ERROR: " << nerr << " elements differ" << std::endl;
  } else {
    std::cout << "OK" << std::endl;
  }

  // Clean up device vectors
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));

  // Clean up host vectors
  free(h_A);
  free(h_B);
  free(h_C);
  
  return 0;
}
