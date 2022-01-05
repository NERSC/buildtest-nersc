#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <hwloc.h>
#include "kernels.h"

#define VECSIZE 100000

int main( int argc, char* argv[] )
{
    int myid, namelen, world_size;
    char myname[MPI_MAX_PROCESSOR_NAME];
    double final_result = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(myname, &namelen);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    hwloc_topology_t topo;
    hwloc_topology_init ( &topo);
    hwloc_topology_load ( topo);
    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    int check_err;
    check_err = hwloc_get_proc_cpubind(topo, getpid(), set, HWLOC_CPUBIND_PROCESS);
    if (check_err) {
    printf ("***Error: Something went wrong***\n"), exit(1);
    }
 
    int ngpu = find_gpus();
    int my_gpu = myid%ngpu;
    char my_gpu_id[15];
    gpu_pci_id(my_gpu_id, my_gpu);
    fprintf(stdout, "Rank %d/%d (PID:%d on Core: %d) from %s sees %d GPUs, GPU assigned to me is: = %s\n",myid, world_size, getpid(),hwloc_bitmap_first(set), myname, ngpu, my_gpu_id);
    fprintf(stdout, "Other %d GPUs are: \n", (ngpu-1));


    for (int j = 0; j < ngpu; j++) {
    if (j != my_gpu) {
        char gpu_id[15];
        gpu_pci_id(gpu_id, j);
        fprintf(stdout, "**rank = %d: %s ** \n", j, gpu_id);
    }
    }

    //setting device for current GPU
    set_my_device(my_gpu);
    int curr_device = get_current_device();
    if(my_gpu != curr_device){
        fprintf(stderr, "********Device was not set properly for some ranks*******\n");
    }
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
    sum /=n;
    
    
    double global_sum = 0;

    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Release host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    if(myid == 0)
        final_result = global_sum/(double)world_size;

    MPI_Finalize();
    
    if(myid == 0){
        if(final_result > 1.0){
            fprintf(stderr, "*****Result is incorrect, something went wrong, program will be terminated*****\n");
            exit(-1);
        }
        fprintf(stdout,"****final result: %f ******\n", final_result);
    }

    hwloc_bitmap_free(set);
    hwloc_topology_destroy(topo);
    return 0;
}
