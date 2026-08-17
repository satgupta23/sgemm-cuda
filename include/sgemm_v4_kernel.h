#pragma once

#include "cuda_check.h"

// Each block computes a BlockM by BlockN tile of c. Each thread owns ThreadM outputs
// stacked in one column, so a value loaded from the b tile is used ThreadM times from a
// register instead of being fetched from shared memory once per output. BlockK sets how
// deep one phase is: it controls shared memory footprint and phase count, but not total
// global traffic, which depends only on BlockM and BlockN.
template <int BlockM, int BlockN, int BlockK, int ThreadM>
__global__ void sgemm_v4_1d_tiling(int n, const float *a, const float *b, float *c)
{
    constexpr int threads = (BlockM * BlockN) / ThreadM;

    __shared__ float a_tile[BlockM * BlockK];
    __shared__ float b_tile[BlockK * BlockN];

    const int thread_index = threadIdx.x;
    const int thread_row = thread_index / BlockN;
    const int thread_col = thread_index % BlockN;

    const int row_base = blockIdx.y * BlockM;
    const int col_base = blockIdx.x * BlockN;

    // Held in registers, which requires every index below to be a compile time constant.
    // If any loop over this array stops being unrolled it lands in local memory instead,
    // which is off-chip, and the kernel becomes slower than the version without blocking.
    float accumulator[ThreadM];
    #pragma unroll
    for (int i = 0; i < ThreadM; ++i) {
        accumulator[i] = 0.0f;
    }

    const int phases = (n + BlockK - 1) / BlockK;
    for (int phase = 0; phase < phases; ++phase) {
        const int k_base = phase * BlockK;

        for (int i = thread_index; i < BlockM * BlockK; i += threads) {
            const int load_row = i / BlockK;
            const int load_col = i % BlockK;
            const int global_row = row_base + load_row;
            const int global_col = k_base + load_col;
            a_tile[i] = (global_row < n && global_col < n)
                        ? a[global_row * n + global_col] : 0.0f;
        }

        for (int i = thread_index; i < BlockK * BlockN; i += threads) {
            const int load_row = i / BlockN;
            const int load_col = i % BlockN;
            const int global_row = k_base + load_row;
            const int global_col = col_base + load_col;
            b_tile[i] = (global_row < n && global_col < n)
                        ? b[global_row * n + global_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int dot = 0; dot < BlockK; ++dot) {
            const float b_value = b_tile[dot * BlockN + thread_col];
            #pragma unroll
            for (int i = 0; i < ThreadM; ++i) {
                accumulator[i] += a_tile[(thread_row * ThreadM + i) * BlockK + dot] * b_value;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < ThreadM; ++i) {
        const int row = row_base + thread_row * ThreadM + i;
        const int col = col_base + thread_col;
        if (row < n && col < n) {
            c[row * n + col] = accumulator[i];
        }
    }
}
