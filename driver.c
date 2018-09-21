/*
 * This is the driver program that tests the performance of
 * different GEMM implementations.
 */
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>

typedef double matrix_t;

void GenMatrix(matrix_t* matrix, size_t matrix_size)
{
	for (int i = 0; i < matrix_size; ++i)
		for (int j = 0; j < matrix_size; ++j)
		{
			matrix[i * matrix_size + j] = (matrix_t)(i + j);
		}
}

int main(int argc, char** argv)
{
	// setup the matrices
	size_t matrix_size = 100;
	matrix_t* A = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	matrix_t* B = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	matrix_t* C = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	GenMatrix(A, matrix_size);
	GenMatrix(B, matrix_size);
	printf("Generated all matrices\n");

	// call the GEMM routine
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			matrix_size, matrix_size, matrix_size,
			1.0, A, matrix_size, B, matrix_size, 1.0,
			C, matrix_size);
	printf("Finished computation\n");
}
