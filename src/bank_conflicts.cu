#include "cuda_check.h"
#include "benchmark.h"

static const int scratch_floats = 1024;
static const int threads_per_block = 256;
static const int blocks = 1024;

// Every lane reads with the given word stride, so the bank each lane hits is
// (lane * stride) mod 32 and the conflict degree is gcd(stride, 32). The sink store is
// unreachable and exists only to stop the compiler discarding the loop.
__global__ void shared_stride_read(int iterations, int stride, float *sink)
{
    __shared__ float scratch[scratch_floats];

    for (int i = threadIdx.x; i < scratch_floats; i += blockDim.x) {
        scratch[i] = (float)i;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;

    float total = 0.0f;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        total += scratch[(lane * stride + iteration) & (scratch_floats - 1)];
    }

    if (total < 0.0f) {
        sink[0] = total;
    }
}

static int conflict_degree(int stride)
{
    int a = stride;
    int b = 32;
    while (b != 0) {
        const int remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

int main(void)
{
    const int iterations = 4096;
    const int warmup_iterations = 3;
    const int timed_iterations = 20;

    float *d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    printf("stride  predicted way  mean ms  relative\n");

    double baseline_ms = 0.0;
    for (int stride = 1; stride <= 32; stride *= 2) {
        const BenchmarkResult result = benchmark(
            [&] { shared_stride_read<<<blocks, threads_per_block>>>(iterations, stride, d_sink); },
            warmup_iterations, timed_iterations);

        if (stride == 1) {
            baseline_ms = result.mean_ms;
        }

        printf("%6d %14d %8.4f %8.2fx\n", stride, conflict_degree(stride),
              result.mean_ms, result.mean_ms / baseline_ms);
    }

    CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
