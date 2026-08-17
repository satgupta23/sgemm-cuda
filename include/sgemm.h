#pragma once

typedef void (*SgemmLauncher)(int n, const float *d_a, const float *d_b, float *d_c);

struct SgemmVersion {
    const char *name;
    SgemmLauncher launch;
    // n must be a multiple of this. Vectorized loads need 16 byte alignment, which for a
    // row major matrix means the row length must be a multiple of four floats.
    int size_multiple;
};

void launch_sgemm_v1(int n, const float *d_a, const float *d_b, float *d_c);
void launch_sgemm_v2(int n, const float *d_a, const float *d_b, float *d_c);
void launch_sgemm_v3(int n, const float *d_a, const float *d_b, float *d_c);
void launch_sgemm_v4(int n, const float *d_a, const float *d_b, float *d_c);
void launch_sgemm_v5(int n, const float *d_a, const float *d_b, float *d_c);
void launch_sgemm_v6(int n, const float *d_a, const float *d_b, float *d_c);
void launch_sgemm_v7(int n, const float *d_a, const float *d_b, float *d_c);

void cublas_init(void);
void cublas_shutdown(void);
void launch_cublas_sgemm(int n, const float *d_a, const float *d_b, float *d_c);
