
#include <iostream>
#include "kernels.h"

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


int find_gpus(void) {
    int ngpu;
    cudaGetDeviceCount(&ngpu);
    return ngpu;
}
void gpu_pci_id(char* device_id, int device_num) {
    int len=15;
    cudaDeviceGetPCIBusId(device_id, len, device_num);
}

void set_my_device(int my_device){
    ERRCHECK(cudaSetDevice(my_device));
}
int get_current_device(){
    int my_device = -1;
    ERRCHECK(cudaGetDevice(&my_device));

    return my_device;
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
