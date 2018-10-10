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
			// with j and p fixed, b is fixed
			double* ap = A + p * lda;
			double* cp = C + j * ldc;
			register double b = B[j * ldb + p];
			for (i = 0; i < m; i += 8)
			{
				*(cp+0) += b * *(ap+0);
				*(cp+1) += b * *(ap+1);
				*(cp+2) += b * *(ap+2);
				*(cp+3) += b * *(ap+3);
				*(cp+4) += b * *(ap+4);
				*(cp+5) += b * *(ap+5);
				*(cp+6) += b * *(ap+6);
				*(cp+7) += b * *(ap+7);
				ap += 8; cp += 8;
			}
		}
}
