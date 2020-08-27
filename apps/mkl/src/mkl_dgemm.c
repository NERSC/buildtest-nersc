#include <stdio.h>
#include <mkl.h>
#include <time.h>
int main(void) 
{
  double *a, *b, *c; 
  int n, i; 
  double alpha, beta; 
  MKL_INT64 AllocatedBytes; 
  int N_AllocatedBuffers;
  clock_t begin,end;
  double time;
  double mflops;
  alpha = 1.1; 
  beta = -1.2; 
  n = 5000;
  mflops = (double)n*n*n*1E-6;
  printf("flops: %f\n",mflops);
  a = (double*)mkl_malloc(n*n*sizeof(double),128);
  b = (double*)mkl_malloc(n*n*sizeof(double),128);
  c = (double*)mkl_malloc(n*n*sizeof(double),128);

  for (i=0;i<(n*n);i++) 
  { 
    a[i] = (double)(i+1); 
    b[i] = (double)(-i-1); 
    c[i] = 0.0; 
  }
  begin = clock();
  dgemm("N","N",&n,&n,&n,&alpha,a,&n,b,&n,&beta,c,&n);
  end = clock();
  time = (double)(end-begin)/CLOCKS_PER_SEC;
  printf("MFLOPS/sec: %f \t Time(sec): %f \n",mflops/time, time );
 
  return 0;
}
