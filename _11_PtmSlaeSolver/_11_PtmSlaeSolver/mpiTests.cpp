#include "stdio.h"
#include "mpi.h"

// Источники
// https://www.youtube.com/watch?v=BA_Mqi3a9HI

inline void mpiTests()
{
	printf("\n----------MPI Tests---------\n");
	int commsize;
	int rank;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &commsize);
	printf("rank = %d\n", rank);
	printf("commsize = %d\n", commsize);
	MPI_Finalize();
}