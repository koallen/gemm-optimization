#include "my_dgemm.h"
/*
 * Specialized kernel to compute MM
 */
static void MicroKernel(int k, double *packed_a, double *packed_b, double *C, int ldc)
{
	for (int kr = 0; kr < k; ++kr)
	{
		// perform rank-1 update
		// TODO: fully unroll this
		for (int x = 0; x < MR; x++)
		{
			for (int y = 0; y < NR; y++)
				C[y * ldc + x] += *(packed_a) * *(packed_b + y);
			packed_a++;
		}
		packed_b += NR;
	}
	// TODO: update C in memory in the end
}
