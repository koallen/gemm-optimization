/*
 * DGEMM implementation in step0
 *
 * Objective:
 *   - apply basic code optimizations
 */
#include "my_dgemm.h"

#define MC 4
#define NC 8
#define KC 16

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
				// blocking of 32 * 32
				for (x = i; x < i + MC; x++)
					for (y = p; y < p + KC; y++)
					{
						double* bp = B + y * ldb + j;
						double* cp = C + x * ldb + j;
						register double a = A[x * lda + y];
						for (z = j; z < j + NC; z+=8)
						{
							*(cp+0) += a * *(bp+0);
							*(cp+1) += a * *(bp+1);
							*(cp+2) += a * *(bp+2);
							*(cp+3) += a * *(bp+3);
							*(cp+4) += a * *(bp+4);
							*(cp+5) += a * *(bp+5);
							*(cp+6) += a * *(bp+6);
							*(cp+7) += a * *(bp+7);
							cp += 8; bp += 8;
						}
					}
}
