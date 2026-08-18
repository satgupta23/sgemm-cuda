# CUDA SGEMM Optimization

A from-scratch CUDA implementation and performance optimization of
single-precision general matrix multiplication (SGEMM).

The project starts with a naive one-thread-per-output-element CUDA kernel
and progressively optimizes it through memory coalescing, shared-memory
tiling, register blocking, 2D thread tiling, and vectorized memory access.

On a 4096 × 4096 matrix multiplication, the final kernel reaches
**6.37 TFLOP/s**, a **67.9× speedup** over the naive CUDA implementation
and approximately **90% of NVIDIA cuBLAS throughput**.

## Performance

Benchmark results for `N = 4096`:

| Version | Optimization | GFLOP/s | Speedup vs. Naive | % cuBLAS |
|---|---|---:|---:|---:|
| v1 | Naive CUDA | 93.8 | 1.0× | 1.3% |
| v2 | Coalesced memory access | 715.6 | 7.6× | 10.1% |
| v3 | Shared-memory tiling | 919.2 | 9.8× | 13.0% |
| v4 | 1D register/thread tiling | 2336.6 | 24.9× | 33.0% |
| v5 | 2D register/thread tiling | 3099.6 | 33.0× | 43.8% |
| v6 | Vectorized memory access | **6370.1** | **67.9×** | **90.0%** |
| cuBLAS | NVIDIA cuBLAS SGEMM | 7077.2 | — | 100% |

The benchmarked GPU reported an FP32 peak throughput of approximately
9.03 TFLOP/s, putting the final kernel at **70.5% of theoretical peak
throughput**.

Full results across `N = 512, 1024, 2048, 4096` are available in
[`results/final.txt`](results/final.txt).

## Optimization Ladder

### v1 — Naive CUDA

The baseline implementation assigns one CUDA thread to each output
matrix element. Each thread independently computes a full dot product.

This implementation performs redundant global-memory accesses and
provides the performance baseline for the project.

### v2 — Coalesced Global Memory Access

Reorganized the mapping between CUDA threads and matrix coordinates so
neighboring threads access neighboring memory locations.

This improves memory coalescing and increases throughput from
93.8 GFLOP/s to 715.6 GFLOP/s at `N = 4096`.

### v3 — Shared-Memory Tiling

Introduced shared-memory tiles for the input matrices.

Threads cooperatively load matrix tiles from global memory into shared
memory and reuse those values while computing a block of the output,
reducing redundant global-memory traffic.

### v4 — 1D Register Tiling

Changed the work assigned to each thread so that one thread computes
multiple output values.

This increases data reuse in registers and improves arithmetic intensity,
raising performance to 2.34 TFLOP/s at `N = 4096`.

### v5 — 2D Register Tiling

Extended register blocking into two dimensions.

Each thread computes an `8 × 8` output tile while blocks operate on
`128 × 128` output regions, substantially increasing reuse of values
loaded from shared memory.

This kernel reaches 3.10 TFLOP/s at `N = 4096`.

### v6 — Vectorized Memory Access

The final kernel adds vectorized `float4` global-memory loads, transferring
16 bytes per load instruction instead of 4.

The A tile is also stored transposed in shared memory to improve its
access pattern and reduce shared-memory bank conflicts.

At `N = 4096`, this kernel achieves:

- **6.37 TFLOP/s**
- **67.9× speedup over naive CUDA**
- **90.0% of cuBLAS throughput**
- **70.5% of measured theoretical FP32 peak**

## Scaling

The optimized kernel becomes increasingly competitive with cuBLAS as the
matrix size grows.

| Matrix Size | Naive GFLOP/s | v6 GFLOP/s | cuBLAS GFLOP/s | v6 / cuBLAS |
|---:|---:|---:|---:|---:|
| 512 | 64.9 | 1762.2 | 1295.9 | 136.0%* |
| 1024 | 90.7 | 4617.8 | 6164.7 | 74.9% |
| 2048 | 93.5 | 6104.9 | 7401.7 | 82.5% |
| 4096 | 93.8 | **6370.1** | **7077.2** | **90.0%** |

\* Small matrices are strongly affected by fixed launch/library overhead,
so the larger matrix sizes provide a more representative throughput
comparison.

## Correctness

Every kernel is checked against a reference SGEMM implementation before
performance measurements are reported.

The benchmark harness uses:

- deterministic random input matrices
- 3 warm-up iterations
- 10 timed iterations
- sampled output verification
- relative-error checking
- a cuBLAS SGEMM reference implementation

For `N = 4096`, the optimized kernel completed verification with:

- maximum relative error: `2.92e-06`
- verification failures: `0`

## Profiling and Performance Analysis

The repository includes additional experiments used to understand the
performance characteristics of the kernels:

- NVIDIA Nsight Compute profiling reports
- roofline-style bandwidth analysis
- shared-memory bank-conflict experiments
- parameter tuning for tiled kernels
- cuBLAS benchmarking
- theoretical peak-throughput comparison

These experiments were used to distinguish memory-access bottlenecks
from compute limitations and guide each optimization step.

## Project Structure

```text
sgemm-cuda/
├── include/
│   ├── benchmark.h
│   ├── cuda_check.h
│   ├── reference.h
│   ├── sgemm.h
│   ├── sgemm_v4_kernel.h
│   ├── sgemm_v5_kernel.h
│   └── sgemm_v6_kernel.h
│
├── src/
│   ├── main.cu
│   ├── reference.cu
│   ├── cublas_reference.cu
│   ├── sgemm_v1.cu
│   ├── sgemm_v2.cu
│   ├── sgemm_v3.cu
│   ├── sgemm_v4.cu
│   ├── sgemm_v5.cu
│   ├── sgemm_v6.cu
│   ├── bank_conflicts.cu
│   ├── roofline_measure.cu
│   ├── tune_v4.cu
│   └── tune_v5.cu
│
├── results/
│   ├── final.txt
│   ├── roofline_measured.md
│   └── *.ncu-rep
│
├── scripts/
│   └── roofline.py
│
└── Makefile
```

The repository also contains experimental kernels and diagnostic code
beyond the six versions used for the reported optimization results.

## Building

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit / `nvcc`
- NVIDIA cuBLAS
- GNU Make

The Makefile currently defaults to:

```make
ARCH ?= sm_89
```

For another NVIDIA GPU architecture, override it when building:

```bash
make ARCH=sm_86
```

or modify the architecture setting in the Makefile.

Build the project with:

```bash
make
```

To build only the SGEMM benchmark:

```bash
make sgemm
```

## Running

Run the benchmark by providing the matrix dimension:

```bash
./sgemm 4096
```

This benchmarks the CUDA optimization ladder and the cuBLAS reference on
square `N × N` matrices.

For example:

```bash
./sgemm 512
./sgemm 1024
./sgemm 2048
./sgemm 4096
```

## What I Learned

This project explores how GPU performance depends on more than the
asymptotic complexity of an algorithm. Successive kernels demonstrate
the effects of:

- global-memory coalescing
- GPU memory hierarchy
- shared-memory reuse
- arithmetic intensity
- register blocking
- thread-level work decomposition
- vectorized memory operations
- shared-memory bank behavior
- occupancy and resource usage
- profiling-driven performance engineering

The final result improves the original CUDA implementation by nearly
**68×** while approaching the performance of a highly optimized vendor
BLAS implementation.
