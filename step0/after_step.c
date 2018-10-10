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
	for (j = 0; j < n; j++)
		for (p = 0; p < k; p++)
		{
			double* ap = A + p * lda;
			double* cp = C + j * ldc;
			// with j and p fixed, b is fixed
			register double b = B[j * ldb + p];
			for (i = 0; i < m; i += 8)
			{
				*(cp+0) += *(ap+0) * b;
				*(cp+1) += *(ap+1) * b;
				*(cp+2) += *(ap+2) * b;
				*(cp+3) += *(ap+3) * b;
				*(cp+4) += *(ap+4) * b;
				*(cp+5) += *(ap+5) * b;
				*(cp+6) += *(ap+6) * b;
				*(cp+7) += *(ap+7) * b;
				ap += 8; cp += 8;
			}
		}
}
