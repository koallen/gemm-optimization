/*
 * This is the driver program that tests the performance of
 * different GEMM implementations.
 */
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

#include "my_dgemm.h"

#define IDX(i, j) (i * matrix_size + j)

#define MIN_SIZE 128
#define MAX_SIZE 768
#define STEP 32
#define REPETITIONS 3

#define EPSILON 1e-10

typedef double matrix_t;

inline double GetSecond(struct timeval start, struct timeval end)
{
	return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

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
	// TODO: allocate cache line aligned matrices
	matrix_t* A = (matrix_t*)malloc(MAX_SIZE * MAX_SIZE * sizeof(matrix_t));
	matrix_t* B = (matrix_t*)malloc(MAX_SIZE * MAX_SIZE * sizeof(matrix_t));
	matrix_t* C_ref = (matrix_t*)malloc(MAX_SIZE * MAX_SIZE * sizeof(matrix_t));
	matrix_t* C_opt = (matrix_t*)malloc(MAX_SIZE * MAX_SIZE * sizeof(matrix_t));
	GenMatrix(A, MAX_SIZE);
	GenMatrix(B, MAX_SIZE);
#ifndef NDEBUG
	PrintMatrix(A, matrix_size, "A");
	PrintMatrix(B, matrix_size, "B");
#endif

	size_t matrix_size;
	printf("MNK\t OPT  \t REF\n");
	for (matrix_size = MIN_SIZE; matrix_size <= MAX_SIZE; matrix_size += STEP)
	{
		double ref_time = 0.0, opt_time = 0.0;
		memset(C_ref, 0, matrix_size * matrix_size * sizeof(matrix_t));
		memset(C_opt, 0, matrix_size * matrix_size * sizeof(matrix_t));

		// call the GEMM routine
		for (int rep = 0; rep < REPETITIONS; ++rep)
		{
			struct timeval ref_start, ref_end;
			gettimeofday(&ref_start, NULL);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					matrix_size, matrix_size, matrix_size,
					1.0, A, matrix_size, B, matrix_size, 1.0,
					C_ref, matrix_size);
			gettimeofday(&ref_end, NULL);
			ref_time += GetSecond(ref_start, ref_end);

			struct timeval opt_start, opt_end;
			gettimeofday(&opt_start, NULL);
			my_dgemm(matrix_size, matrix_size, matrix_size,
					A, matrix_size, B, matrix_size, C_opt, matrix_size);
			gettimeofday(&opt_end, NULL);
			opt_time += GetSecond(opt_start, opt_end);
		}
		ref_time = ref_time / REPETITIONS;
		opt_time = opt_time / REPETITIONS;
		double flops = (matrix_size * matrix_size / (1000.0 * 1000.0 * 1000.0)) * (2 * matrix_size);
		printf("%5zd\t %5.2lf\t %5.2lf\n", matrix_size, flops / opt_time, flops / ref_time);

		// validate the result
		CheckMatrix(C_ref, C_opt, matrix_size);
#ifndef NDEBUG
		PrintMatrix(C_ref, matrix_size, "C ref");
		PrintMatrix(C_opt, matrix_size, "C opt");
#endif
	}
}
