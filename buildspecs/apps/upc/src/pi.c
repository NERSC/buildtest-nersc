// Compute pi by approximating the area of a circle of radius 1. 
// Algorithm: generate random points in [0,1]x[0,1] and measure the fraction 
// of them falling in a circle centered at the origin (approximates pi/4)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <upc.h>

int hit() { // return non-zero for a hit in the circle
  double x = rand()/(double)RAND_MAX;
  double y = rand()/(double)RAND_MAX;
  return (x*x + y*y) <= 1.0;
}

// shared array for the results computed by each thread
shared int64_t all_hits[THREADS];

int main(int argc, char **argv) {
    int64_t trials = 100000000;
    if (argc > 1) trials = (int64_t)atoll(argv[1]);
    int64_t my_trials = (trials + THREADS - 1 - MYTHREAD)/THREADS;

    srand(MYTHREAD); // seed each thread's PRNG differently

    int64_t my_hits = 0;
    for (int64_t i=0; i < my_trials; i++)
        my_hits += hit(); // compute in parallel

    all_hits[MYTHREAD] = my_hits; // publish results
    upc_barrier;

    if (MYTHREAD == 0) { // fetch results from each thread
        // (could alternatively call upc_all_reduce())
        int64_t total_hits = 0;
        for (int i=0; i < THREADS; i++)
            total_hits += all_hits[i];
        double pi = 4.0*total_hits/(double)trials;
        printf("PI estimated to %10.7f from %lld trials on %d threads.\n",
               pi, (long long)trials, THREADS);
    }

    return 0;
}
