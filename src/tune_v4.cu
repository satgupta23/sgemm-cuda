#include "cuda_check.h"
#include "benchmark.h"
#include "reference.h"
#include "sgemm_v4_kernel.h"

static const int block_m = 64;
static const int block_n = 64;
static const int block_k = 8;

template <int ThreadM>
static void run_one(int n, const float *h_a, const float *h_b, float *h_c,
                    const float *d_a, const float *d_b, float *d_c,
                    double total_flops, size_t bytes)
{
    const int threads = (block_m * block_n) / ThreadM;
    const dim3 grid((n + block_n - 1) / block_n, (n + block_m - 1) / block_m);

    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, (const void *)sgemm_v4_1d_tiling<block_m, block_n, block_k, ThreadM>));

    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        (const void *)sgemm_v4_1d_tiling<block_m, block_n, block_k, ThreadM>, threads, 0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const double occupancy =
        100.0 * blocks_per_sm * threads / prop.maxThreadsPerMultiProcessor;

    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    sgemm_v4_1d_tiling<block_m, block_n, block_k, ThreadM><<<grid, threads>>>(n, d_a, d_b, d_c);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    const VerificationResult check = verify_sgemm(n, h_a, h_b, h_c, 2048, 1e-4);

    const BenchmarkResult timing = benchmark(
        [&] {
            sgemm_v4_1d_tiling<block_m, block_n, block_k, ThreadM>
                <<<grid, threads>>>(n, d_a, d_b, d_c);
        },
        3, 10);

    printf("%8d %8d %10d %8zu %11d %10.1f%% %11.1f %9d\n",
          ThreadM, threads, attributes.numRegs, attributes.localSizeBytes,
          blocks_per_sm, occupancy,
          total_flops / (timing.mean_ms / 1e3) / 1e9, check.failures);
}

int main(int argc, char **argv)
{
    int n = 2048;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    const size_t elements = (size_t)n * n;
    const size_t bytes = elements * sizeof(float);
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

    printf("n %d, 64 by 64 block tile, BK 8\n\n", n);
    printf("thread_m  threads  registers    lmem  blocks/SM   occupancy      GFLOP/s  failures\n");

    run_one<4>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<8>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<16>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);
    run_one<32>(n, h_a, h_b, h_c, d_a, d_b, d_c, total_flops, bytes);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
