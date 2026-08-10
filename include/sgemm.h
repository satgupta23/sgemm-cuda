#pragma once

// Every kernel version exposes this signature, so the driver can hold them in a
// table and the benchmark loop never has to know which one it is running.
typedef void (*SgemmLauncher)(int n, const float *d_a, const float *d_b, float *d_c);

struct SgemmVersion {
    const char *name;
    SgemmLauncher launch;
};

void launch_sgemm_v1(int n, const float *d_a, const float *d_b, float *d_c);
