#include <stdio.h>
#include <stdlib.h>

float sum(const float *a, const size_t n);

int main(int argc, char* argv[]) {
  const size_t n = 1<<10;
  size_t i;
  float *a;

  a = malloc(n*sizeof(float));

  for (i = 0; i < n; i++) {
    a[i] = (float)i;
  }

  printf("Sum: %f\n", sum(a, n));

  free(a);
}

float sum(const float *a, const size_t n)
{
    float total = 0.;
    size_t i;

    #pragma omp parallel for reduction(+:total)
    for (i = 0; i < n; i++) {
        total += a[i];
    }
    return total;
}

