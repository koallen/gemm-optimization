/*
 * DGEMM following the Goto approach
 */
#include <stdlib.h>
#include "my_dgemm.h"

#define NC 48
#define KC 48
#define MC 48
#define NR 6
#define MR 8

/*
 * Pack MC * KC small matrix
 */
static void PackA(double *A, double *packed_a, int ic, int pc, int lda)
{
	double* current_a = A + pc * lda + ic;
	for (int i = 0; i < MC; i += MR)
		for (int j = 0; j < KC; ++j)
			for (int k = 0; k < MR; ++k)
				*(packed_a++) = current_a[j * lda + (i + k)];
}

/*
 * Pack KC * NC small matrix
 */
static void PackB(double *B, double *packed_b, int pc, int jc, int ldb)
{
	double* current_b = B + jc * ldb + pc;
	for (int i = 0; i < NC; i += NR)
		for (int j = 0; j < KC; ++j)
			for (int k = 0; k < NR; ++k)
				*(packed_b++) = current_b[(i + k) * ldb + j];
}

/*
 * Specialized kernel to compute MM
 */
static void MicroKernel(double *C, double *packed_a, double *packed_b, int ldc)
{
	for (int kr = 0; kr < KC; ++kr)
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
	double *packed_a = NULL, *packed_b = NULL;
	posix_memalign((void**)&packed_a, sizeof(double), MC * KC * sizeof(double));
	posix_memalign((void**)&packed_b, sizeof(double), KC * NC * sizeof(double));
	int jc, pc, ic, jr, ir;
	for (jc = 0; jc < n; jc += NC)
		for (pc = 0; pc < k; pc += KC)
		{
			PackB(B, packed_b, pc, jc, ldb);
			for (ic = 0; ic < m; ic += MC)
			{
				PackA(A, packed_a, ic, pc, lda);
				for (jr = 0; jr < NC; jr += NR)
					for (ir = 0; ir < MC; ir += MR)
						MicroKernel(C + (jc + jr) * ldc + (ic + ir), packed_a + ir * KC, packed_b + jr * KC, ldc);
			}
		}
	free(packed_a); free(packed_b);
}
