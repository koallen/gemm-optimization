/*
 * This is the driver program that tests the performance of
 * different GEMM implementations.
 */
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>

#define IDX(i, j) (i * matrix_size + j)

typedef double matrix_t;

void PrintMatrix(matrix_t* matrix, size_t matrix_size, char* matrix_name)
{
	printf("==========Printing matrix %c==========\n", *matrix_name);
	for (int i = 0; i < matrix_size; ++i)
	{
		for (int j = 0; j < matrix_size; ++j)
			printf("%10.3lf ", matrix[IDX(i,j)]);
		printf("\n");
	}
}

void GenMatrix(matrix_t* matrix, size_t matrix_size)
{
	for (int i = 0; i < matrix_size; ++i)
		for (int j = 0; j < matrix_size; ++j)
		{
			matrix[IDX(i,j)] = (matrix_t)(i + j);
		}
}

int main(int argc, char** argv)
{
	// setup the matrices
	size_t matrix_size = 5;
	matrix_t* A = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	matrix_t* B = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	matrix_t* C = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	GenMatrix(A, matrix_size);
	GenMatrix(B, matrix_size);
	printf("Generated all matrices\n");
#ifndef NDEBUG
	PrintMatrix(A, matrix_size, "A");
	PrintMatrix(B, matrix_size, "B");
#endif

	// call the GEMM routine
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			matrix_size, matrix_size, matrix_size,
			1.0, A, matrix_size, B, matrix_size, 1.0,
			C, matrix_size);
	printf("Finished computation\n");
#ifndef NDEBUG
	PrintMatrix(C, matrix_size, "C");
#endif
}
