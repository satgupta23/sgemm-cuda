#include "sgemm.h"
#include "sgemm_v6_kernel.h"

static const int block_m = 128;
static const int block_n = 128;
static const int block_k = 8;
static const int thread_m = 8;
static const int thread_n = 8;

void launch_sgemm_v6(int n, const float *d_a, const float *d_b, float *d_c)
{
    const int threads = (block_m * block_n) / (thread_m * thread_n);
    const dim3 grid((n + block_n - 1) / block_n, (n + block_m - 1) / block_m);

    sgemm_v6_vectorized<block_m, block_n, block_k, thread_m, thread_n>
        <<<grid, threads>>>(n, d_a, d_b, d_c);
}
