#include "stdio.h"
#include "omp.h"

// Источники
// https://pro-prof.com/archives/4335#more-4335

inline void openMPTests()
{
	printf("\n----------OpenMP Tests---------\n");

#pragma omp parallel num_threads(4)
	{
		printf("thread...\n");
	}
}