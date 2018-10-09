/*
 * DGEMM implementation in step0
 *
 * Objective:
 *   - apply basic code optimizations
 */
#include "my_dgemm.h"

#define MC 32
#define NC 32
#define KC 32

void after_step_gemm(int m,
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
				for (x = 0; x < MC; x++)
					for (y = 0; y < KC; y++)
						for (z = 0; z < NC; z++)
							C[x * ldc + z] += A[x * lda + y] * B[y * ldb + z];
}
