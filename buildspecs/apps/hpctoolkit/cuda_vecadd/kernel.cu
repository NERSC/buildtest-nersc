__global__ void dadd (const long N, const double* a, const double* b, double* c)
{
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) c[i] = a[i] + b[i];
}
