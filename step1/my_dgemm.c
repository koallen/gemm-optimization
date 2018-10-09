/*
 * DGEMM implementation in step0
 *
 * Objective:
 *   - apply basic code optimizations
 */
#include "my_dgemm.h"

#define MC 8
#define NC 8
#define KC 8

void my_dgemm(int m,
	int n,
	int k,
	double *A,
	int lda,
	double *B,
	int ldb,
	double *C,
	int ldc)
{
	int i, j, p, x, y, z;
	for (i = 0; i < m; i += MC)
		for (p = 0; p < k; p += KC)
			for (j = 0; j < n; j += NC)
				for (x = i; x < i + MC; x++)
					for (y = p; y < p + KC; y++)
						for (z = j; z < j + NC; z++)
							C[x * ldc + z] += A[x * lda + y] * B[y * ldb + z];
}
