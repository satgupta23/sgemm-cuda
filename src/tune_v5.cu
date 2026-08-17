#include "cuda_check.h"
#include "benchmark.h"
#include "reference.h"
#include "sgemm_v5_kernel.h"

template <int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN>
static void run_one(int n, const float *h_a, const float *h_b, float *h_c,
                    const float *d_a, const float *d_b, float *d_c,
                    double total_flops, size_t bytes)
{
    constexpr int threads = (BlockM * BlockN) / (ThreadM * ThreadN);
    const void *kernel =
        (const void *)sgemm_v5_2d_tiling<BlockM, BlockN, BlockK, ThreadM, ThreadN>;

    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, 0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const double occupancy =
        100.0 * blocks_per_sm * threads / prop.maxThreadsPerMultiProcessor;

    const double intensity = 1.0 / (2.0 * (1.0 / BlockM + 1.0 / BlockN));
    const double loads_per_fma = 1.0 / ThreadM + 1.0 / ThreadN;

    if (blocks_per_sm == 0) {
        printf("%4d %4d %3d %3d %3d %8d %7d      does not fit\n",
              BlockM, BlockN, BlockK, ThreadM, ThreadN, threads, attributes.numRegs);
        return;
    }

    const dim3 grid((n + BlockN - 1) / BlockN, (n + BlockM - 1) / BlockM);

    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    sgemm_v5_2d_tiling<BlockM, BlockN, BlockK, ThreadM, ThreadN>
        <<<grid, threads>>>(n, d_a, d_b, d_c);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    const VerificationResult check = verify_sgemm(n, h_a, h_b, h_c, 2048, 1e-4);

    const BenchmarkResult timing = benchmark(
        [&] {
            sgemm_v5_2d_tiling<BlockM, BlockN, BlockK, ThreadM, ThreadN>
                <<<grid, threads>>>(n, d_a, d_b, d_c);
        },
        3, 10);

    printf("%4d %4d %3d %3d %3d %8d %7d %6.2f %7.1f %8.3f %9.1f%% %10.1f %6d\n",
          BlockM, BlockN, BlockK, ThreadM, ThreadN, threads, attributes.numRegs,
          attributes.sharedSizeBytes / 1024.0, intensity, loads_per_fma,
          occupancy, total_flops / (timing.mean_ms / 1e3) / 1e9, check.failures);
}

int main(int argc, char **argv)
{
    int n = 2048;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    const size_t bytes = (size_t)n * n * sizeof(float);
    const double total_flops = 2.0 * (double)n * (double)n * (double)n;

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        fprintf(stderr, "host allocation failed\n");
        return EXIT_FAILURE;
    }
    fill_random(n, h_a, 1u);
    fill_random(n, h_b, 2u);

    float *d_a;
    float *d_b;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    printf("n %d\n\n", n);
    printf("  BM   BN  BK  TM  TN  threads    regs  smem  intens  ld/fma  occupancy    GFLOP/s  fail\n");

    run_one<64, 64, 8, 4, 4>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<64, 64, 8, 8, 8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<128, 64, 8, 8, 8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<128, 128, 8, 4, 4>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<128, 128, 8, 4, 8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<128, 128, 8, 8, 8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<128, 128, 16, 8, 8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<256, 128, 8, 8, 8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<256, 256, 8, 8, 8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
