/*
 * DGEMM implementation in step0
 *
 * Objective:
 *   - apply basic code optimizations
 */
#include "my_dgemm.h"

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
	int i, j, p;
	for (i = 0; i < m; i++)
		for (p = 0; p < k; p++)
		{
			double* bp = B + p * ldb;
			double* cp = C + i * ldc;
			register double a = A[i * lda + p];
			for (j = 0; j < n; j += 8)
			{
				*(cp+0) += a * *(bp+0);
				*(cp+1) += a * *(bp+1);
				*(cp+2) += a * *(bp+2);
				*(cp+3) += a * *(bp+3);
				*(cp+4) += a * *(bp+4);
				*(cp+5) += a * *(bp+5);
				*(cp+6) += a * *(bp+6);
				*(cp+7) += a * *(bp+7);
				bp += 8; cp += 8;
			}
		}
}
