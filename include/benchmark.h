#pragma once

#include "cuda_check.h"

struct BenchmarkResult {
    double mean_ms;
    double min_ms;
    double max_ms;
    int iterations;
};

// `launch` is any callable that submits one unit of GPU work. The warmup runs absorb
// context creation, first-touch page faults, and the clock ramp. Each timed run gets
// its own event pair so that the spread across runs stays visible in the result.
template <typename Callable>
BenchmarkResult benchmark(Callable launch, int warmup_iterations, int timed_iterations)
{
    for (int i = 0; i < warmup_iterations; ++i) {
        launch();
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    for (int i = 0; i < timed_iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        total_ms += elapsed_ms;
        if (i == 0 || elapsed_ms < min_ms) {
            min_ms = elapsed_ms;
        }
        if (elapsed_ms > max_ms) {
            max_ms = elapsed_ms;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    BenchmarkResult result;
    result.mean_ms = total_ms / timed_iterations;
    result.min_ms = min_ms;
    result.max_ms = max_ms;
    result.iterations = timed_iterations;
    return result;
}

static inline double gigabytes_per_second(double bytes, double milliseconds)
{
    return bytes / (milliseconds / 1e3) / 1e9;
}
