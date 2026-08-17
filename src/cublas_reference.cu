#include "cuda_check.h"
#include "sgemm.h"
#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status_ = (call);                                       \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "%s:%d cuBLAS: %s\n", __FILE__, __LINE__,          \
                    cublasGetStatusName(status_));                             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static cublasHandle_t handle = NULL;

void cublas_init(void)
{
    CUBLAS_CHECK(cublasCreate(&handle));
}

void cublas_shutdown(void)
{
    if (handle != NULL) {
        CUBLAS_CHECK(cublasDestroy(handle));
        handle = NULL;
    }
}

// cuBLAS reads matrices column major. A row major matrix reinterpreted column major is
// its own transpose, so cuBLAS sees A transposed and B transposed. Asking it for the
// product in the order (b, a) therefore computes B^T A^T, which equals (A B)^T, and
// writing that column major into d_c leaves exactly A B in row major. No data is moved.
void launch_cublas_sgemm(int n, const float *d_a, const float *d_b, float *d_c)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n,
                            &alpha,
                            d_b, n,
                            d_a, n,
                            &beta,
                            d_c, n));
}
