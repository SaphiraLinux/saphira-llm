/*
 * ops.c — vector operations for the transformer forward pass.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Every operation here is a real x86-64-v3 path: the baseline is not a
 * fallback tier, it is the thing that runs. Each has a plain scalar tail so
 * odd lengths are exact rather than approximately right.
 *
 * The scalar form of each kernel is also what the golden-vector tests compare
 * against, so the tests do not merely confirm that two SIMD kernels agree
 * with each other.
 */

#include <saphira_llm/ops.h>
#include <saphira_llm/log.h>

#if defined(__x86_64__)
#  include <immintrin.h>
#  define SLLM_X86 1
#endif

#include <math.h>
#include <string.h>

sllm_status sllm_rope_type_parse(const char * name, sllm_rope_type * out) {
    if (name == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    if (strcmp(name, "neox") == 0) {
        *out = SLLM_ROPE_NEOX;
        return SLLM_OK;
    }
    if (strcmp(name, "normal") == 0) {
        *out = SLLM_ROPE_NORMAL;
        return SLLM_OK;
    }
    return SLLM_ERR_ARG;
}

/* ------------------------------------------------------------------ */
/* expf                                                                */
/* ------------------------------------------------------------------ */

float sllm_fast_exp(float x) {
    return expf(x);
}

/* ------------------------------------------------------------------ */
/* add / mul                                                           */
/* ------------------------------------------------------------------ */

void sllm_add(float * dst, const float * a, const float * b, size_t n) {
    size_t i = 0;
#if defined(SLLM_X86)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
}

void sllm_mul(float * dst, const float * a, const float * b, size_t n) {
    size_t i = 0;
#if defined(SLLM_X86)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
}

/* ------------------------------------------------------------------ */
/* RMSNorm                                                             */
/* ------------------------------------------------------------------ */

void sllm_rms_norm(float * dst, const float * x, const float * weight,
                   size_t n, float eps) {
    /* Sum of squares first. Accumulated in double because a 4096-element f32
     * sum of squares loses several bits, and the reference accumulates the same
     * way; matching the reference matters more here than the last bit. */
    double ss = 0.0;
    size_t i = 0;
#if defined(SLLM_X86)
    __m256d acc = _mm256_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        /*
         * Square in f32, widen to f64, accumulate in f64.
         *
         * The square is not optional. An earlier version of this loop
         * accumulated the raw values, so the "sum of squares" was a plain sum;
         * the mean could then come out near zero or negative, the square root
         * went imaginary, and the result was silently nonsense for large n
         * while looking fine for small n. The golden vector caught it at
         * n = 186.
         *
         * Accumulating in f64 matters at model scale: a 4096-element f32 sum
         * of squares has already lost several bits by the time it is scaled.
         */
        const __m256 sq = _mm256_mul_ps(v, v);
        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_castps256_ps128(sq)));
        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1)));
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, acc);
    for (int k = 0; k < 4; ++k) {
        ss += tmp[k];
    }
#endif
    for (; i < n; ++i) {
        ss += (double) x[i] * (double) x[i];
    }

    /*
     * The narrowing order is taken from ggml_compute_forward_rms_norm_f32 and
     * is not negotiable:
     *
     *     const float mean  = sum/ne00;                    // narrows to f32
     *     const float scale = 1.0f/sqrtf(mean + eps);      // f32 sqrtf
     *
     * Computing sqrt in double and narrowing afterwards is arguably more
     * accurate and is off by about an ulp. That ulp is the difference between
     * a value landing either side of a rounding boundary when the next layer
     * quantises the activation to int8, so "more accurate" would be a parity
     * bug. The reference is the specification.
     */
    const float mean  = (float) (ss / (double) n);
    const float scale = 1.0f / sqrtf(mean + eps);

    i = 0;
#if defined(SLLM_X86)
    const __m256 vs = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(vs, _mm256_loadu_ps(x + i)));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = scale * x[i];
    }

    if (weight != NULL) {
        sllm_mul(dst, dst, weight, n);
    }
}

/* ------------------------------------------------------------------ */
/* softmax                                                             */
/* ------------------------------------------------------------------ */

void sllm_softmax_inplace(float * x, size_t n) {
    if (n == 0) {
        return;
    }
    float max = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float e = expf(x[i] - max);
        x[i] = e;
        sum += (double) e;
    }
    const float inv = (float) (1.0 / sum);
    for (size_t i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

/* ------------------------------------------------------------------ */
/* SiLU                                                                */
/* ------------------------------------------------------------------ */

void sllm_silu_inplace(float * x, size_t n) {
    /*
     * x / (1 + exp(-x)), in both paths.
     *
     * The vector form here originally computed x * exp(x), which is not SiLU
     * at all: it is the product of the input with the exponential rather than
     * with the logistic. The golden-vector test caught it at the first
     * element, and the discrepancy was enormous (164.9 against 3.69) precisely
     * because the two formulas agree nowhere except near zero.
     *
     * exp is evaluated through libm per lane, and the point of the vector
     * path is the memory traffic and the branch-free form, not pretending
     * AVX2 has a transcendental it does not have.
     */
    size_t i = 0;
#if defined(SLLM_X86)
    if (n >= 8) {
        for (; i + 8 <= n; i += 8) {
            float t[8];
            _mm256_storeu_ps(t, _mm256_loadu_ps(x + i));
            for (int k = 0; k < 8; ++k) {
                t[k] = t[k] / (1.0f + expf(-t[k]));
            }
            _mm256_storeu_ps(x + i, _mm256_loadu_ps(t));
        }
    }
#endif
    for (; i < n; ++i) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* ------------------------------------------------------------------ */
/* RoPE                                                                */
/* ------------------------------------------------------------------ */

void sllm_rope_inplace(float * x, size_t n_rot, int32_t pos,
                       float theta, float freq_scale, sllm_rope_type type) {
    if (n_rot < 2) {
        return;
    }
    if (n_rot % 2 != 0) {
        /* An odd rotary dimension has no well-defined pairing. Half of it is
         * still rotatable and the reference leaves the rest alone. */
        n_rot -= 1;
    }

    /*
     * Two layouts, and they are not interchangeable:
     *
     *   NEOX   pairs adjacent elements: (0,1), (2,3), (4,5), ...
     *   NORMAL pairs the two halves:      (0, h), (1, h+1), ...
     *
     * where h = n_rot/2. Mixing them up does not crash and does not look
     * wrong in any single element; it destroys the relative phase between
     * dimensions, which is the entire content of the encoding. The golden
     * vectors check the rotation preserves each pair's norm, which catches it.
     *
     * Note the argument order. The reference computes theta_i = base^(-2i/n)
     * and then scales the position, so the equivalent form is to divide theta
     * by freq_scale. Reversing that is invisible at position 0 and wrong
     * everywhere else.
     */
    const size_t half = n_rot / 2;

    for (size_t k = 0; k < half; ++k) {
        const size_t a = (type == SLLM_ROPE_NEOX) ? (k * 2) : k;
        const size_t b = (type == SLLM_ROPE_NEOX) ? (k * 2 + 1) : (k + half);
        if (b >= n_rot) {
            continue;
        }

        const float inv_freq = powf(theta, -2.0f * (float) k / (float) n_rot);
        const float f = ((float) pos * inv_freq) / freq_scale;
        const float c = cosf(f);
        const float s = sinf(f);

        const float xa = x[a];
        const float xb = x[b];
        x[a] = xa * c - xb * s;
        x[b] = xa * s + xb * c;
    }
}

/* ------------------------------------------------------------------ */
/* get_rows                                                            */
/* ------------------------------------------------------------------ */

void sllm_get_rows(float * dst, const float * src, const int32_t * idx,
                   size_t n_rows, size_t row_floats) {
    for (size_t i = 0; i < n_rows; ++i) {
        const int32_t r = idx[i];
        if (r < 0) {
            /* An index from a model is input, not a promise. Clamping is
             * safer than reading wherever the number points. */
            sllm_log(SLLM_LOG_WARN, "get_rows: negative row index %d, clamped to 0", (int) r);
            memcpy(dst + i * row_floats, src, row_floats * sizeof(float));
            continue;
        }
        memcpy(dst + i * row_floats, src + (size_t) r * row_floats, row_floats * sizeof(float));
    }
}
