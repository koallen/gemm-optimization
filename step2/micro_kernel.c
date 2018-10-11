#include "my_dgemm.h"
/*
 * Specialized kernel to compute MM
 */
static void MicroKernel(int k, double *packed_a, double *packed_b, double *C, int ldc)
{
	register double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0, c04 = 0.0, c05 = 0.0;
	register double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0, c14 = 0.0, c15 = 0.0;
	register double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0, c24 = 0.0, c25 = 0.0;
	register double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0, c34 = 0.0, c35 = 0.0;
	register double c40 = 0.0, c41 = 0.0, c42 = 0.0, c43 = 0.0, c44 = 0.0, c45 = 0.0;
	register double c50 = 0.0, c51 = 0.0, c52 = 0.0, c53 = 0.0, c54 = 0.0, c55 = 0.0;
	register double c60 = 0.0, c61 = 0.0, c62 = 0.0, c63 = 0.0, c64 = 0.0, c65 = 0.0;
	register double c70 = 0.0, c71 = 0.0, c72 = 0.0, c73 = 0.0, c74 = 0.0, c75 = 0.0;
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
		c00 += a * b0;
		c01 += a * b1;
		c02 += a * b2;
		c03 += a * b3;
		c04 += a * b4;
		c05 += a * b5;
		packed_a++;

		// 2nd iteration
		a = *packed_a;
		c10 += a * b0;
		c11 += a * b1;
		c12 += a * b2;
		c13 += a * b3;
		c14 += a * b4;
		c15 += a * b5;
		packed_a++;

		a = *packed_a;
		c20 += a * b0;
		c21 += a * b1;
		c22 += a * b2;
		c23 += a * b3;
		c24 += a * b4;
		c25 += a * b5;
		packed_a++;

		a = *packed_a;
		c30 += a * b0;
		c31 += a * b1;
		c32 += a * b2;
		c33 += a * b3;
		c34 += a * b4;
		c35 += a * b5;
		packed_a++;

		a = *packed_a;
		c40 += a * b0;
		c41 += a * b1;
		c42 += a * b2;
		c43 += a * b3;
		c44 += a * b4;
		c45 += a * b5;
		packed_a++;

		a = *packed_a;
		c50 += a * b0;
		c51 += a * b1;
		c52 += a * b2;
		c53 += a * b3;
		c54 += a * b4;
		c55 += a * b5;
		packed_a++;

		a = *packed_a;
		c60 += a * b0;
		c61 += a * b1;
		c62 += a * b2;
		c63 += a * b3;
		c64 += a * b4;
		c65 += a * b5;
		packed_a++;

		a = *packed_a;
		c70 += a * b0;
		c71 += a * b1;
		c72 += a * b2;
		c73 += a * b3;
		c74 += a * b4;
		c75 += a * b5;
		packed_a++;
		packed_b += NR;
	}
	C[0 * ldc + 0] = c00;
	C[1 * ldc + 0] = c01;
	C[2 * ldc + 0] = c02;
	C[3 * ldc + 0] = c03;
	C[4 * ldc + 0] = c04;
	C[5 * ldc + 0] = c05;
	C[0 * ldc + 1] = c10;
	C[1 * ldc + 1] = c11;
	C[2 * ldc + 1] = c12;
	C[3 * ldc + 1] = c13;
	C[4 * ldc + 1] = c14;
	C[5 * ldc + 1] = c15;
	C[0 * ldc + 2] = c20;
	C[1 * ldc + 2] = c21;
	C[2 * ldc + 2] = c22;
	C[3 * ldc + 2] = c23;
	C[4 * ldc + 2] = c24;
	C[5 * ldc + 2] = c25;
	C[0 * ldc + 3] = c30;
	C[1 * ldc + 3] = c31;
	C[2 * ldc + 3] = c32;
	C[3 * ldc + 3] = c33;
	C[4 * ldc + 3] = c34;
	C[5 * ldc + 3] = c35;
	C[0 * ldc + 4] = c40;
	C[1 * ldc + 4] = c41;
	C[2 * ldc + 4] = c42;
	C[3 * ldc + 4] = c43;
	C[4 * ldc + 4] = c44;
	C[5 * ldc + 4] = c45;
	C[0 * ldc + 5] = c50;
	C[1 * ldc + 5] = c51;
	C[2 * ldc + 5] = c52;
	C[3 * ldc + 5] = c53;
	C[4 * ldc + 5] = c54;
	C[5 * ldc + 5] = c55;
	C[0 * ldc + 6] = c60;
	C[1 * ldc + 6] = c61;
	C[2 * ldc + 6] = c62;
	C[3 * ldc + 6] = c63;
	C[4 * ldc + 6] = c64;
	C[5 * ldc + 6] = c65;
	C[0 * ldc + 7] = c70;
	C[1 * ldc + 7] = c71;
	C[2 * ldc + 7] = c72;
	C[3 * ldc + 7] = c73;
	C[4 * ldc + 7] = c74;
	C[5 * ldc + 7] = c75;
}
