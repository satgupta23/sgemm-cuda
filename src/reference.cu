#include "reference.h"
#include <math.h>

static double reference_element(int n, const float *h_a, const float *h_b, int row, int col)
{
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += (double)h_a[(size_t)row * n + k] * (double)h_b[(size_t)k * n + col];
    }
    return sum;
}

static unsigned int next_random(unsigned int *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

void fill_random(int n, float *matrix, unsigned int seed)
{
    unsigned int state = seed;
    const size_t elements = (size_t)n * n;
    for (size_t i = 0; i < elements; ++i) {
        matrix[i] = (float)(next_random(&state) >> 8) / (float)(1u << 24);
    }
}

VerificationResult verify_sgemm(int n, const float *h_a, const float *h_b,
                                const float *h_c, int samples, double tolerance)
{
    const int corner_rows[4] = {0, 0, n - 1, n - 1};
    const int corner_cols[4] = {0, n - 1, 0, n - 1};

    VerificationResult result;
    result.max_relative_error = 0.0;
    result.samples = 0;
    result.failures = 0;

    unsigned int state = 12345u;

    for (int s = 0; s < samples; ++s) {
        int row;
        int col;
        if (s < 4) {
            row = corner_rows[s];
            col = corner_cols[s];
        } else {
            row = (int)(next_random(&state) % (unsigned int)n);
            col = (int)(next_random(&state) % (unsigned int)n);
        }

        const double expected = reference_element(n, h_a, h_b, row, col);
        const double actual = (double)h_c[(size_t)row * n + col];
        const double relative_error = fabs(actual - expected) / fabs(expected);

        if (relative_error > result.max_relative_error) {
            result.max_relative_error = relative_error;
        }
        if (relative_error > tolerance) {
            ++result.failures;
        }
        ++result.samples;
    }

    return result;
}
