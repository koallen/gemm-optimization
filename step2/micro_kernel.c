#include <stdint.h>
#include "my_dgemm.h"
/*
 * Specialized kernel to compute MM
 */
void MicroKernel(int k, double *packed_a, double *packed_b, double *C, int ldc)
{
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c = 1;
	uint64_t cs_c = ldc * sizeof(double);

	__asm__ volatile
	(
	"vzeroall                             \n\t"
	"                                     \n\t"
	"movq %2, %%rax                       \n\t" // load address of packed A
	"movq %3, %%rbx                       \n\t" // load address of packed B

	"addq $32 * 4, %%rax                  \n\t"

	"vmovapd -4 * 32 (%%rax), %%ymm0      \n\t" // load 8 elements from A
	"vmovapd -3 * 32 (%%rax), %%ymm1      \n\t"
	
	"movq %4, %%rcx                       \n\t" // load address of C
	"movq %6, %%rdi                       \n\t" // load ldc * 8 (stripe)

	"movq %0, %%rsi                       \n\t" // i = k_iter
	"testq %%rsi, %%rsi                   \n\t"
	"je .DCONSIDERKLEFT                   \n\t"

	".DKITER:                             \n\t" // MAIN LOOP

        "                                     \n\t" // iteration 0
	"vbroadcastsd 0 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 1 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm4   \n\t" // a[0-3] * b[0]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm5   \n\t" // a[4-7] * b[0]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm6   \n\t" // a[0-3] * b[1]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm7   \n\t" // a[4-7] * b[1]

	"vbroadcastsd 2 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 3 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm8   \n\t" // a[0-3] * b[2]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm9   \n\t" // a[4-7] * b[2]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm10  \n\t" // a[0-3] * b[3]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm11  \n\t" // a[4-7] * b[3]

	"vbroadcastsd 4 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 5 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm12  \n\t" // a[0-3] * b[4]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm13  \n\t" // a[4-7] * b[4]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm14  \n\t" // a[0-3] * b[5]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm15  \n\t" // a[4-7] * b[5]

	"vmovapd -2 * 32(%%rax), %%ymm0       \n\t" // load 8 elements from A
	"vmovapd -1 * 32(%%rax), %%ymm1       \n\t"

        "                                     \n\t" // iteration 1
	"vbroadcastsd 6 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 7 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm4   \n\t" // a[0-3] * b[0]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm5   \n\t" // a[4-7] * b[0]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm6   \n\t" // a[0-3] * b[1]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm7   \n\t" // a[4-7] * b[1]

	"vbroadcastsd 8 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 9 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm8   \n\t" // a[0-3] * b[2]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm9   \n\t" // a[4-7] * b[2]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm10  \n\t" // a[0-3] * b[3]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm11  \n\t" // a[4-7] * b[3]

	"vbroadcastsd 10 * 8(%%rbx), %%ymm2   \n\t"
	"vbroadcastsd 11 * 8(%%rbx), %%ymm3   \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm12  \n\t" // a[0-3] * b[4]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm13  \n\t" // a[4-7] * b[4]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm14  \n\t" // a[0-3] * b[5]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm15  \n\t" // a[4-7] * b[5]

	"vmovapd 0 * 32(%%rax), %%ymm0        \n\t" // load 8 elements from A
	"vmovapd 1 * 32(%%rax), %%ymm1        \n\t"

        "                                     \n\t" // iteration 2
	"vbroadcastsd 12 * 8(%%rbx), %%ymm2   \n\t"
	"vbroadcastsd 13 * 8(%%rbx), %%ymm3   \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm4   \n\t" // a[0-3] * b[0]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm5   \n\t" // a[4-7] * b[0]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm6   \n\t" // a[0-3] * b[1]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm7   \n\t" // a[4-7] * b[1]

	"vbroadcastsd 14 * 8(%%rbx), %%ymm2   \n\t"
	"vbroadcastsd 15 * 8(%%rbx), %%ymm3   \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm8   \n\t" // a[0-3] * b[2]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm9   \n\t" // a[4-7] * b[2]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm10  \n\t" // a[0-3] * b[3]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm11  \n\t" // a[4-7] * b[3]

	"vbroadcastsd 16 * 8(%%rbx), %%ymm2   \n\t"
	"vbroadcastsd 17 * 8(%%rbx), %%ymm3   \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm12  \n\t" // a[0-3] * b[4]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm13  \n\t" // a[4-7] * b[4]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm14  \n\t" // a[0-3] * b[5]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm15  \n\t" // a[4-7] * b[5]

	"vmovapd 2 * 32(%%rax), %%ymm0        \n\t" // load 8 elements from A
	"vmovapd 3 * 32(%%rax), %%ymm1        \n\t"

        "                                     \n\t" // iteration 3
	"vbroadcastsd 18 * 8(%%rbx), %%ymm2   \n\t"
	"vbroadcastsd 19 * 8(%%rbx), %%ymm3   \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm4   \n\t" // a[0-3] * b[0]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm5   \n\t" // a[4-7] * b[0]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm6   \n\t" // a[0-3] * b[1]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm7   \n\t" // a[4-7] * b[1]

	"vbroadcastsd 20 * 8(%%rbx), %%ymm2   \n\t"
	"vbroadcastsd 21 * 8(%%rbx), %%ymm3   \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm8   \n\t" // a[0-3] * b[2]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm9   \n\t" // a[4-7] * b[2]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm10  \n\t" // a[0-3] * b[3]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm11  \n\t" // a[4-7] * b[3]

	"vbroadcastsd 22 * 8(%%rbx), %%ymm2   \n\t"
	"vbroadcastsd 23 * 8(%%rbx), %%ymm3   \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm12  \n\t" // a[0-3] * b[4]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm13  \n\t" // a[4-7] * b[4]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm14  \n\t" // a[0-3] * b[5]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm15  \n\t" // a[4-7] * b[5]

	"addq $4 * 8 * 8, %%rax               \n\t" // increment address of A
	"addq $4 * 6 * 8, %%rbx               \n\t" // increment address of B

	"vmovapd -4 * 32(%%rax), %%ymm0       \n\t" // load 8 elements from A
	"vmovapd -3 * 32(%%rax), %%ymm1       \n\t"

	"decq %%rsi                           \n\t" // i -= 1
	"jne .DKITER                          \n\t" // if i != 0, loop again

        ".DCONSIDERKLEFT:                     \n\t"
	"movq %1, %%rsi                       \n\t" // i = k_left
	"testq %%rsi, %%rsi                   \n\t"
	"je .DPOSTACCUM                       \n\t"

	".DKLEFT:                             \n\t" // EDGE CASE (k % 4 != 0)

	"vbroadcastsd 0 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 1 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm4   \n\t" // a[0-3] * b[0]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm5   \n\t" // a[4-7] * b[0]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm6   \n\t" // a[0-3] * b[1]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm7   \n\t" // a[4-7] * b[1]

	"vbroadcastsd 2 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 3 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm8   \n\t" // a[0-3] * b[2]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm9   \n\t" // a[4-7] * b[2]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm10  \n\t" // a[0-3] * b[3]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm11  \n\t" // a[4-7] * b[3]

	"vbroadcastsd 4 * 8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd 5 * 8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd %%ymm0, %%ymm2, %%ymm12  \n\t" // a[0-3] * b[4]
	"vfmadd231pd %%ymm1, %%ymm2, %%ymm13  \n\t" // a[4-7] * b[4]
	"vfmadd231pd %%ymm0, %%ymm3, %%ymm14  \n\t" // a[0-3] * b[5]
	"vfmadd231pd %%ymm1, %%ymm3, %%ymm15  \n\t" // a[4-7] * b[5]

	"addq $1 * 8 * 8, %%rax               \n\t" // increment address of A
	"addq $1 * 6 * 8, %%rbx               \n\t" // increment address of B

	"vmovapd -4 * 32(%%rax), %%ymm0       \n\t" // load 8 elements from A
	"vmovapd -3 * 32(%%rax), %%ymm1       \n\t"

	"decq %%rsi                           \n\t" // i -= 1
	"jne .DKLEFT                          \n\t" // if i != 0, loop again

	".DPOSTACCUM:                         \n\t" // STORE RESULT TO C

	"vmovapd %%ymm4,  0  * 32(%%rcx)      \n\t"
	"vmovapd %%ymm5,  1  * 32(%%rcx)      \n\t"
	"addq %%rdi, %%rcx                    \n\t"
	"vmovapd %%ymm6,  0  * 32(%%rcx)      \n\t"
	"vmovapd %%ymm7,  1  * 32(%%rcx)      \n\t"
	"addq %%rdi, %%rcx                    \n\t"
	"vmovapd %%ymm8,  0  * 32(%%rcx)      \n\t"
	"vmovapd %%ymm9,  1  * 32(%%rcx)      \n\t"
	"addq %%rdi, %%rcx                    \n\t"
	"vmovapd %%ymm10, 0  * 32(%%rcx)      \n\t"
	"vmovapd %%ymm11, 1  * 32(%%rcx)      \n\t"
	"addq %%rdi, %%rcx                    \n\t"
	"vmovapd %%ymm12, 0  * 32(%%rcx)      \n\t"
	"vmovapd %%ymm13, 1  * 32(%%rcx)      \n\t"
	"addq %%rdi, %%rcx                    \n\t"
	"vmovapd %%ymm14, 0 * 32(%%rcx)       \n\t"
	"vmovapd %%ymm15, 1 * 32(%%rcx)       \n\t"

	: // output operands
	: // input operands
	  "m" (k_iter),   // 0
	  "m" (k_left),   // 1
	  "m" (packed_a), // 2
	  "m" (packed_b), // 3
	  "m" (C),        // 4
	  "m" (rs_c),     // 5
	  "m" (cs_c)      // 6
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
          "xmm0", "xmm1", "xmm2", "xmm3",
          "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11",
          "xmm12", "xmm13", "xmm14", "xmm15",
          "memory"
	);
}
