#include <mpi.h>
int main(int argc, char **argv)
{
  int rank, size;
  MPI_Init(&argc, &argv); 
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("%d\n", size);
  }
  MPI_Finalize();
}
