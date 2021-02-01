#include <stdio.h>
#include <mkl.h>
#include <time.h>
int main(void) 
{
  
  float *a, *b, *c; 
  int n, i; 
  float alpha, beta; 
  //MKL_INT64 AllocatedBytes; 
  int N_AllocatedBuffers;
  clock_t begin,end;
  double time;
  double mflops;
  alpha = 1.1; 
  beta = -1.2; 
  n = 5000;
  mflops = (double)n*n*n*1E-6;
   
  printf("flops: %f\n",mflops);
  a = (float*)mkl_malloc(n*n*sizeof(float),64);
  b = (float*)mkl_malloc(n*n*sizeof(float),64);
  c = (float*)mkl_malloc(n*n*sizeof(float),64);

  for (i=0;i<(n*n);i++) 
  { 
    a[i] = (float)(i+1); 
    b[i] = (float)(-i-1); 
    c[i] = 0.0; 
  }
  begin = clock();
  sgemm("N","N",&n,&n,&n,&alpha,a,&n,b,&n,&beta,c,&n);
  end = clock();
  time = (double)(end-begin)/CLOCKS_PER_SEC;
  printf("MFLOPS/sec: %f \t Time(sec): %f \n",mflops/time, time );
  
  return 0;
}
