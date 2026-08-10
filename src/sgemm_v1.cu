#include "cuda_check.h"
#include "sgemm.h"

static const int block_dimension = 32;

__global__ void sgemm_v1_naive(int n, const float *a, const float *b, float *c)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

void launch_sgemm_v1(int n, const float *d_a, const float *d_b, float *d_c)
{
    const dim3 block(block_dimension, block_dimension);
    const dim3 grid((n + block_dimension - 1) / block_dimension,
                    (n + block_dimension - 1) / block_dimension);

    sgemm_v1_naive<<<grid, block>>>(n, d_a, d_b, d_c);
}
