#pragma once

#include "cuda_check.h"

// Each thread owns a ThreadM by ThreadN patch of c. One step of the dot product is an
// outer product: ThreadM values from the a tile and ThreadN from the b tile are pulled
// into registers, then ThreadM x ThreadN products are accumulated. Loads grow as the sum
// of the patch dimensions while work grows as the product, which is why the patch is
// square and why this is cheaper per flop than the 1D version.
template <int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__global__ void sgemm_v5_2d_tiling(int n, const float *a, const float *b, float *c)
{
    constexpr int threads = (BlockM * BlockN) / (ThreadM * ThreadN);
    constexpr int thread_cols = BlockN / ThreadN;

    __shared__ float a_tile[BlockM * BlockK];
    __shared__ float b_tile[BlockK * BlockN];

    const int thread_index = threadIdx.x;
    const int thread_row = thread_index / thread_cols;
    const int thread_col = thread_index % thread_cols;

    const int row_base = blockIdx.y * BlockM;
    const int col_base = blockIdx.x * BlockN;

    float accumulator[ThreadM][ThreadN];
    #pragma unroll
    for (int i = 0; i < ThreadM; ++i) {
        #pragma unroll
        for (int j = 0; j < ThreadN; ++j) {
            accumulator[i][j] = 0.0f;
        }
    }

    float a_fragment[ThreadM];
    float b_fragment[ThreadN];

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
            #pragma unroll
            for (int i = 0; i < ThreadM; ++i) {
                a_fragment[i] = a_tile[(thread_row * ThreadM + i) * BlockK + dot];
            }
            #pragma unroll
            for (int j = 0; j < ThreadN; ++j) {
                b_fragment[j] = b_tile[dot * BlockN + thread_col * ThreadN + j];
            }

            #pragma unroll
            for (int i = 0; i < ThreadM; ++i) {
                #pragma unroll
                for (int j = 0; j < ThreadN; ++j) {
                    accumulator[i][j] += a_fragment[i] * b_fragment[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < ThreadM; ++i) {
        #pragma unroll
        for (int j = 0; j < ThreadN; ++j) {
            const int row = row_base + thread_row * ThreadM + i;
            const int col = col_base + thread_col * ThreadN + j;
            if (row < n && col < n) {
                c[row * n + col] = accumulator[i][j];
            }
        }
    }
}
