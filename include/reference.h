#pragma once

struct VerificationResult {
    double max_relative_error;
    int samples;
    int failures;
};

void fill_random(int n, float *matrix, unsigned int seed);

VerificationResult verify_sgemm(int n, const float *h_a, const float *h_b,
                                const float *h_c, int samples, double tolerance);
