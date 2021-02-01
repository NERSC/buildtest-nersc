#include <stdio.h>
#include <upc.h>

int main() {
  upc_barrier;
  printf("Hello from %i/%i\n",MYTHREAD,THREADS);
  upc_barrier;
  return 0;
}
