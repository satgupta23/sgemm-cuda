#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status_ = (call);                                          \
        if (status_ != cudaSuccess) {                                          \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Confirms a launch was accepted. Building with -DCUDA_DEBUG_SYNC additionally waits
// for the kernel to finish, which reports execution errors at the launch site instead
// of at whatever call happens to synchronize next.
#ifdef CUDA_DEBUG_SYNC
#define CUDA_CHECK_LAUNCH()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)
#else
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#endif
