#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "kernels.h"
 
#define VECSIZE 100000
 
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
