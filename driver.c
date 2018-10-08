/*
 * This is the driver program that tests the performance of
 * different GEMM implementations.
 */
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "my_dgemm.h"

#define IDX(i, j) (i * matrix_size + j)
#define SIZE 8
#define EPSILON 1e-6

typedef double matrix_t;

void PrintMatrix(matrix_t* matrix, size_t matrix_size, char* matrix_name)
{
	printf("==========Printing matrix %s==========\n", matrix_name);
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

void CheckMatrix(matrix_t* A, matrix_t* B, size_t matrix_size)
{
	for (int i = 0; i < matrix_size; ++i)
		for (int j = 0; j < matrix_size; ++j)
			if (abs(A[IDX(i, j)] - B[IDX(i, j)]) > EPSILON)
			{
				printf("C[%d, %d] differs!\n", i, j);
			}
}

int main(int argc, char** argv)
{
	// setup the matrices
	size_t matrix_size = SIZE;

	// TODO: allocate cache line aligned matrices
	matrix_t* A = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	matrix_t* B = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	matrix_t* C_ref = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	matrix_t* C_opt = (matrix_t*)malloc(matrix_size * matrix_size * sizeof(matrix_t));
	memset(C_opt, 0, matrix_size * matrix_size * sizeof(matrix_t));
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
			C_ref, matrix_size);
	my_dgemm(matrix_size, matrix_size, matrix_size,
			A, matrix_size, B, matrix_size, C_opt, matrix_size);
	printf("Finished computation\n");
	CheckMatrix(C_ref, C_opt, matrix_size);
#ifndef NDEBUG
	PrintMatrix(C_ref, matrix_size, "C ref");
	PrintMatrix(C_opt, matrix_size, "C opt");
#endif
}
