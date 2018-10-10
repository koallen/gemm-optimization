/*
 * DGEMM following the Goto approach
 */
#include <stdlib.h>
#include "my_dgemm.h"

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define NC 48
#define KC 48
#define MC 48
#define NR 6
#define MR 8

/////////////////////////////////////////////////////////
// TODO:
//   - use ib, jb, and pb for block size
/////////////////////////////////////////////////////////

/*
 * This function packs MR x KC block of A each time,
 * to pack the whole MC x KC block, need to call this
 * function many times
 */
static void PackA(int m,
	int k,
	double *A,
	int lda,
	int offset_a,
	double *packed_a)
{
	int i, p;
	double *a_ptr[MR];

	for (i = 0; i < m; ++i)
		a_ptr[i] = A + offset_a + i;
	for (i = m; i < MR; ++i)
		a_ptr[i] = A + offset_a + 0; // if m smaller than MR, fill with same data

	for (p = 0; p < k; ++p)
		for (i = 0; i < MR; ++i)
		{
			*(packed_a++) = *a_ptr[i];
			a_ptr[i] += lda;
		}
}

/*
 * This function packs KC x NR block of B each time,
 * to pack the whole KC x NC block, need to call this
 * function many times
 */
static void PackB(int n,
	int k,
	double *B,
	int ldb,
	int offset_b,
	double *packed_b)
{
	int j, p;
	double *b_ptr[NR];

	for (j = 0; j < n; ++j)
		b_ptr[j] = B + ldb * (offset_b + j);
	for (j = n; j < NR; ++j)
		b_ptr[j] = B + ldb * (offset_b + 0);

	for (p = 0; p < k; ++p)
		for (j = 0; j < NR; ++j)
			*(packed_b++) = *(b_ptr[j]++);
}

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

static void MacroKernel(int m, int n, int k, double *packed_a, double *packed_b, double *C, int ldc)
{
	int ir, jr; // used as index
	int ib, jb; // used as block size

	for (jr = 0; jr < n; jr += NR) // 2nd loop
	{
		jb = MIN(n - jr, NR);
		for (ir = 0; ir < m; ir += MR) // 1st loop
		{
			ib = MIN(m - ir, MR);
			MicroKernel(k,
				&packed_a[ir * k],
				&packed_b[jr * k],
				&C[jr * ldc + ir],
				ldc);
		}
	}
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
	int jc, pc, ic; // used as index
	int i, j; // used as packing index
	int ib, jb, pb; // used as block size

	double *packed_a = NULL, *packed_b = NULL;
	posix_memalign((void**)&packed_a, sizeof(double), MC * KC * sizeof(double));
	posix_memalign((void**)&packed_b, sizeof(double), KC * NC * sizeof(double));

	for (jc = 0; jc < n; jc += NC) // 5th loop
	{
		jb = MIN(NC, n - jc);
		for (pc = 0; pc < k; pc += KC) // 4th loop
		{
			pb = MIN(KC, k - pc);
			// pack B
			for (j = 0; j < jb; j += NR)
			{
				PackB(MIN(jb - j, NR),
					pb,
					B + pc,
					k,
					jc + j,
					packed_b + j * pb);
			}
			for (ic = 0; ic < m; ic += MC) // 3rd loop
			{
				ib = MIN(MC, m - ic);
				// pack A
				for (i = 0; i < ib; i += MR)
				{
					PackA(MIN(ib - i, MR),
						pb,
						A + pc * lda,
						m,
						ic + i,
						packed_a + i * pb);
				}
				// MACRO KERNEL
				MacroKernel(ib, jb, pb, packed_a, packed_b, &C[jc * ldc + ic], ldc);
			}
		}
	}

	free(packed_a); free(packed_b);
}
