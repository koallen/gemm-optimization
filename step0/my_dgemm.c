#include "my_dgemm.h"

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
	int i, j, p;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			for (p = 0; p < k; p++)
				C[i * ldc + j] += A[i * lda + p] * B[p * ldb + j];
}
