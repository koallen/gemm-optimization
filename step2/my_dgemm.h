// these values are copied from BLIS configuration for haswell
#define NC 4080
#define KC 256
#define MC 72
#define NR 6
#define MR 8

void my_dgemm(int, int, int, double*, int, double*, int, double*, int);
void after_step_gemm(int, int, int, double*, int, double*, int, double*, int);
