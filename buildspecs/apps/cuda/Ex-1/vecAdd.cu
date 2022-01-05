#include <iostream>
#include <stdlib.h>
#include <math.h>
 
#define VECSIZE 100000


#define ERRCHECK(ans)                                                                  \
{                                                                                    \
    gpuAssert((ans), __FILE__, __LINE__);                                            \
}
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){
    if(code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d cpu:%d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void vec_add_gpu(double *h_a, double *h_b, double *h_c, int n){
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
    // Allocate memory for each vector on GPU
    size_t bytes = n*sizeof(double);
 
    ERRCHECK(cudaMalloc(&d_a, bytes));
    ERRCHECK(cudaMalloc(&d_b, bytes));
    ERRCHECK(cudaMalloc(&d_c, bytes));

    // Copy host vectors to device
    ERRCHECK(cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice));
    ERRCHECK(cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice));
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);
 
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
    // Copy array back to host
    ERRCHECK(cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost ));

        // Release device memory
    ERRCHECK(cudaFree(d_a));
    ERRCHECK(cudaFree(d_b));
    ERRCHECK(cudaFree(d_c));

}

int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = VECSIZE;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 

    // Allocate memory for each vector on host
    h_a = new double [VECSIZE];//(double*)malloc(bytes);
    h_b = new double [VECSIZE];
    h_c = new double [VECSIZE];
 

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }
 
    vec_add_gpu(h_a, h_b, h_c, n);

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/(double)n);
 

    // Release host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
