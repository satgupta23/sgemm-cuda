#include "cuda_check.h"
#include "sgemm.h"

static const int block_dimension = 32;

__global__ void sgemm_v2_coalesced(int n, const float *a, const float *b, float *c)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y; 
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

void launch_sgemm_v2(int n, const float *d_a, const float *d_b, float *d_c)
{
    const dim3 block(block_dimension, block_dimension);
    const dim3 grid((n + block_dimension - 1) / block_dimension,
                    (n + block_dimension - 1) / block_dimension);

    sgemm_v2_coalesced<<<grid, block>>>(n, d_a, d_b, d_c);
}
