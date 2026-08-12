#include "cuda_check.h"
#include "benchmark.h"
#include "sgemm.h"
#include "reference.h"

static const SgemmVersion versions[] = {
    {"v1_naive", launch_sgemm_v1}, 
    {"v2_coalesced", launch_sgemm_v2},
};

static const int version_count = (int)(sizeof(versions) / sizeof(versions[0]));


static double peak_gflops(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));

    int lanes = 0;
    if (prop.major == 12 || (prop.major == 8 && prop.minor >= 6)) {
        lanes = 128;
    } else if (prop.major == 8) {
        lanes = 64;
    }
    if (lanes == 0) {
        return 0.0;
    }

    return 2.0 * lanes * prop.multiProcessorCount * clock_khz * 1e3 / 1e9;
}

int main(int argc, char **argv)
{
    int n = 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    const int warmup_iterations = 3;
    const int timed_iterations = 10;
    const int verification_samples = 4096;
    const double tolerance = 1e-4;

    const size_t elements = (size_t)n * n;
    const size_t bytes = elements * sizeof(float);

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

    const double total_flops = 2.0 * (double)n * (double)n * (double)n;
    const double peak = peak_gflops();

    printf("n %d, %.3f GFLOP per multiply, peak %.0f GFLOP/s\n\n",
           n, total_flops / 1e9, peak);
    printf("version         ms     GFLOP/s   %% peak   max rel err   failures\n");

    for (int v = 0; v < version_count; ++v) {
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));
        versions[v].launch(n, d_a, d_b, d_c);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

        const VerificationResult check =
            verify_sgemm(n, h_a, h_b, h_c, verification_samples, tolerance);

        const BenchmarkResult timing = benchmark(
            [&] { versions[v].launch(n, d_a, d_b, d_c); },
            warmup_iterations, timed_iterations);

        const double gflops = total_flops / (timing.mean_ms / 1e3) / 1e9;

        printf("%-12s %8.3f  %9.1f  %6.2f%%  %12.2e  %9d\n",
               versions[v].name, timing.mean_ms, gflops,
               100.0 * gflops / peak, check.max_relative_error, check.failures);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
