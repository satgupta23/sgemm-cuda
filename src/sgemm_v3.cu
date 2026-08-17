#include "cuda_check.h"
#include "sgemm.h"

static const int tile_dimension = 32;

// Each block computes one tile_dimension square tile of c. The k dimension is walked in
// phases: every thread loads one element of a and one of b into shared memory, the block
// synchronizes, and then every thread consumes the whole tile from on-chip memory. That
// turns tile_dimension global reads per element into one.
__global__ void sgemm_v3_shared(int n, const float *a, const float *b, float *c)
{
    __shared__ float a_tile[tile_dimension][tile_dimension];
    __shared__ float b_tile[tile_dimension][tile_dimension];

    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int row = blockIdx.y * tile_dimension + thread_row;
    const int col = blockIdx.x * tile_dimension + thread_col;

    const int phases = (n + tile_dimension - 1) / tile_dimension;

    float sum = 0.0f;
    for (int phase = 0; phase < phases; ++phase) {
        const int a_col = phase * tile_dimension + thread_col;
        const int b_row = phase * tile_dimension + thread_row;

        // Out of range elements load as zero so they contribute nothing to the sum. The
        // conditional selects the value rather than skipping the store, which keeps every
        // thread on the path to the barrier below.
        a_tile[thread_row][thread_col] =
            (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        b_tile[thread_row][thread_col] =
            (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < tile_dimension; ++k) {
            sum += a_tile[thread_row][k] * b_tile[k][thread_col];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

void launch_sgemm_v3(int n, const float *d_a, const float *d_b, float *d_c)
{
    const dim3 block(tile_dimension, tile_dimension);
    const dim3 grid((n + tile_dimension - 1) / tile_dimension,
                    (n + tile_dimension - 1) / tile_dimension);

    sgemm_v3_shared<<<grid, block>>>(n, d_a, d_b, d_c);
}
