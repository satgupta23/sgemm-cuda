#include "cuda_check.h"
#include "benchmark.h"
#include "sgemm.h"
#include "reference.h"

struct TrafficModel {
    const char *name;
    SgemmLauncher launch;
    int size_multiple;
    int block_m;      // 0 means no tiling: traffic is 8 N^3 bytes
    int block_n;
};

static const TrafficModel models[] = {
    {"v1_naive", launch_sgemm_v1, 1, 0, 0},
    {"v2_coalesced", launch_sgemm_v2, 1, 0, 0},
    {"v3_shared", launch_sgemm_v3, 1, 32, 32},
    {"v4_1d_tiling", launch_sgemm_v4, 1, 64, 64},
    {"v5_2d_tiling", launch_sgemm_v5, 1, 128, 128},
    {"v6_vectorized", launch_sgemm_v6, 4, 128, 128},
};

static const int model_count = (int)(sizeof(models) / sizeof(models[0]));

static double implied_traffic_bytes(const TrafficModel *model, double n)
{
    if (model->block_m == 0) {
        return 8.0 * n * n * n;
    }
    return 4.0 * n * n * n * (1.0 / model->block_m + 1.0 / model->block_n);
}

int main(int argc, char **argv)
{
    int n = 2048;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    const size_t bytes = (size_t)n * n * sizeof(float);

    int memory_clock_khz = 0;
    int bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    const double peak_bandwidth = 2.0 * memory_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    if (h_a == NULL || h_b == NULL) {
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

    printf("n %d, peak bandwidth %.0f GB/s\n\n", n, peak_bandwidth);
    printf("version         GFLOP/s   traffic GB   required GB/s   %% of peak   verdict\n");

    for (int m = 0; m < model_count; ++m) {
        if (n % models[m].size_multiple != 0) {
            continue;
        }

        const BenchmarkResult timing = benchmark(
            [&] { models[m].launch(n, d_a, d_b, d_c); }, 3, 10);

        const double seconds = timing.mean_ms / 1e3;
        const double traffic = implied_traffic_bytes(&models[m], (double)n);
        const double required = traffic / seconds / 1e9;
        const double fraction = 100.0 * required / peak_bandwidth;

        const char *verdict;
        if (fraction > 100.0) {
            verdict = "cache is absorbing most of the traffic";
        } else if (fraction > 70.0) {
            verdict = "DRAM bandwidth limited";
        } else if (fraction > 40.0) {
            verdict = "partly bandwidth limited";
        } else {
            verdict = "not bandwidth limited, look elsewhere";
        }

        printf("%-14s %9.1f  %11.2f  %14.1f  %9.1f%%   %s\n",
              models[m].name, total_flops / seconds / 1e9, traffic / 1e9,
              required, fraction, verdict);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    return 0;
}
