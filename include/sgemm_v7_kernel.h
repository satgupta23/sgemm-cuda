#pragma once

#include "cuda_check.h"


template <int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
__global__ void sgemm_v7_no_conflicts(int n, const float *a, const float *b, float *c)
{
    constexpr int threads = (BlockM * BlockN) / (ThreadM * ThreadN);
    constexpr int thread_cols = BlockN / ThreadN;

    __shared__ float a_tile[BlockK * BlockM];
    __shared__ float b_tile[BlockK * BlockN];

    const int thread_index = threadIdx.x;
    const int thread_row = thread_index / thread_cols;
    const int thread_col = thread_index % thread_cols;

    const int row_base = blockIdx.y * BlockM;
    const int col_base = blockIdx.x * BlockN;

    const int a_load_row = thread_index / (BlockK / 4);
    const int a_load_col = (thread_index % (BlockK / 4)) * 4;
    const int b_load_row = thread_index / (BlockN / 4);
    const int b_load_col = (thread_index % (BlockN / 4)) * 4;

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

        const int a_row = row_base + a_load_row;
        const int a_col = k_base + a_load_col;
        float4 a_vector = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (a_row < n && a_col + 3 < n) {
            a_vector = reinterpret_cast<const float4 *>(&a[(size_t)a_row * n + a_col])[0];
        }
        a_tile[(a_load_col + 0) * BlockM + a_load_row] = a_vector.x;
        a_tile[(a_load_col + 1) * BlockM + a_load_row] = a_vector.y;
        a_tile[(a_load_col + 2) * BlockM + a_load_row] = a_vector.z;
        a_tile[(a_load_col + 3) * BlockM + a_load_row] = a_vector.w;

        const int b_row = k_base + b_load_row;
        const int b_col = col_base + b_load_col;
        float4 b_vector = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (b_row < n && b_col + 3 < n) {
            b_vector = reinterpret_cast<const float4 *>(&b[(size_t)b_row * n + b_col])[0];
        }

        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int local_col = b_load_col + e;
            const int owner = local_col / ThreadN;
            const int step = local_col % ThreadN;
            const float value = (e == 0) ? b_vector.x
                              : (e == 1) ? b_vector.y
                              : (e == 2) ? b_vector.z
                                        : b_vector.w;
            b_tile[b_load_row * BlockN + step * thread_cols + owner] = value;
        }

        __syncthreads();

        #pragma unroll
        for (int dot = 0; dot < BlockK; ++dot) {
            #pragma unroll
            for (int i = 0; i < ThreadM; ++i) {
                a_fragment[i] = a_tile[dot * BlockM + thread_row * ThreadM + i];
            }
            #pragma unroll
            for (int j = 0; j < ThreadN; ++j) {
                b_fragment[j] = b_tile[dot * BlockN + j * thread_cols + thread_col];
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
                c[(size_t)row * n + col] = accumulator[i][j];
            }
        }
    }
}
