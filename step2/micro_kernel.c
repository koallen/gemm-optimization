#include "my_dgemm.h"
/*
 * Specialized kernel to compute MM
 */
static void MicroKernel(int k, double *packed_a, double *packed_b, double *C, int ldc)
{
	register double a;
	register double b0, b1, b2, b3, b4, b5;
	for (int kr = 0; kr < k; ++kr)
	{
		// perform rank-1 update
		b0 = *(packed_b);
		b1 = *(packed_b + 1);
		b2 = *(packed_b + 2);
		b3 = *(packed_b + 3);
		b4 = *(packed_b + 4);
		b5 = *(packed_b + 5);

		// 1st iteration
		a = *packed_a;
		C[0] += a * b0;
		C[1 * ldc] += a * b1;
		C[2 * ldc] += a * b2;
		C[3 * ldc] += a * b3;
		C[4 * ldc] += a * b4;
		C[5 * ldc] += a * b5;
		packed_a++;

		// 2nd iteration
		a = *packed_a;
		C[1] += a * b0;
		C[1 * ldc + 1] += a * b1;
		C[2 * ldc + 1] += a * b2;
		C[3 * ldc + 1] += a * b3;
		C[4 * ldc + 1] += a * b4;
		C[5 * ldc + 1] += a * b5;
		packed_a++;

		a = *packed_a;
		C[2] += a * b0;
		C[1 * ldc + 2] += a * b1;
		C[2 * ldc + 2] += a * b2;
		C[3 * ldc + 2] += a * b3;
		C[4 * ldc + 2] += a * b4;
		C[5 * ldc + 2] += a * b5;
		packed_a++;

		a = *packed_a;
		C[3] += a * b0;
		C[1 * ldc + 3] += a * b1;
		C[2 * ldc + 3] += a * b2;
		C[3 * ldc + 3] += a * b3;
		C[4 * ldc + 3] += a * b4;
		C[5 * ldc + 3] += a * b5;
		packed_a++;

		a = *packed_a;
		C[4] += a * b0;
		C[1 * ldc + 4] += a * b1;
		C[2 * ldc + 4] += a * b2;
		C[3 * ldc + 4] += a * b3;
		C[4 * ldc + 4] += a * b4;
		C[5 * ldc + 4] += a * b5;
		packed_a++;

		a = *packed_a;
		C[5] += a * b0;
		C[1 * ldc + 5] += a * b1;
		C[2 * ldc + 5] += a * b2;
		C[3 * ldc + 5] += a * b3;
		C[4 * ldc + 5] += a * b4;
		C[5 * ldc + 5] += a * b5;
		packed_a++;

		a = *packed_a;
		C[6] += a * b0;
		C[1 * ldc + 6] += a * b1;
		C[2 * ldc + 6] += a * b2;
		C[3 * ldc + 6] += a * b3;
		C[4 * ldc + 6] += a * b4;
		C[5 * ldc + 6] += a * b5;
		packed_a++;

		a = *packed_a;
		C[7] += a * b0;
		C[1 * ldc + 7] += a * b1;
		C[2 * ldc + 7] += a * b2;
		C[3 * ldc + 7] += a * b3;
		C[4 * ldc + 7] += a * b4;
		C[5 * ldc + 7] += a * b5;
		packed_a++;
		packed_b += NR;
	}
	// TODO: update C in memory in the end
}
