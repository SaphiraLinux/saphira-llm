/*
 * i2s_gemm.c — the BitNet I2_S ternary matrix kernels.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Exactness before speed, and the speed comes from where the reference already
 * found it. The reference has two forms of this kernel: a GEMM with a 4x4
 * register block and a GEMV that walks one row, and both have an AVX-VNNI
 * path alongside the AVX2 one. The VNNI path is not an optimisation we are
 * inventing; it is the reference's own, and the reason is worth recording.
 *
 * The AVX2 path uses _mm256_maddubs_epi16, which is the *saturating*
 * signed-by-unsigned form, and accumulates in int16 lanes. The VNNI path uses
 * _mm256_dpbusd_epi32, which accumulates straight into int32 with no int16
 * intermediate and therefore cannot saturate at all. On the target CPU, which
 * has AVX-VNNI, the VNNI path is both the reference's choice and the
 * saturation-free one.
 *
 * The unsigned first operand is not an accident either. I2_S weight codes are
 * 0, 1 and 2, which are non-negative, so dpbusd's unsigned-first semantics are
 * exactly right for the weights and only the activations are signed. A
 * signed-by-signed port would need a bias and a correction, and would read
 * every negative activation as a large positive one.
 */

#include <saphira_llm/i2s_gemm.h>
#include <saphira_llm/log.h>

#if defined(__x86_64__)
#  include <immintrin.h>
#  define SLLM_X86 1
#endif

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* activation quantisation                                             */
/* ------------------------------------------------------------------ */

void sllm_i2s_quant_act(const float * x, size_t n, int8_t * q_out,
                        sllm_i2s_act * out) {
    /*
     * An exact port of the reference's quantize_row_i8_s.
     *
     * Two details that are easy to get wrong and that both change results:
     *
     *   - amax is NOT floored at 1e-5. An all-zero row gives s = 0, so every
     *     quantised value is 0 and the epilogue's division by act_scale is a
     *     division by zero. The reference behaves the same way, so we do too;
     *     it is not our job to paper over it, and the parity gate depends on
     *     reproducing it.
     *
     *   - roundf is round-half-away-from-zero. A magic-constant "nearest int"
     *     that other quantisers use gives the same result, but only because
     *     they are constructed to; using rint() here would give banker's
     *     rounding and differ on exact halves.
     */
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) {
            amax = a;
        }
    }
    const float s = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    out->scale = s;

    int32_t sum = 0;
    int8_t * q = q_out;
    for (size_t i = 0; i < n; ++i) {
        int v = (int) roundf(x[i] * s);
        if (v > 127)  { v = 127; }
        if (v < -128) { v = -128; }
        q[i] = (int8_t) v;
        sum += v;
    }
    out->sum = sum;
}

/* ------------------------------------------------------------------ */
/* the scalar oracle                                                   */
/* ------------------------------------------------------------------ */

int32_t sllm_i2s_dot_scalar(const uint8_t * w, const int8_t * a, size_t n) {
    /*
     * Walk the layout the way the format defines it, with no vectorisation at
     * all, so that disagreement with the SIMD kernels is unambiguous.
     *
     * Within each 128-element block, element j = 32 * field + lane, packed into
     * byte `lane` of the block at bit shift 6 - 2 * field. So field 0 is the top
     * of the byte and pairs with activations 0..31.
     */
    const size_t blocks = n / SLLM_I2S_QK;
    int32_t acc = 0;
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t * blk = w + b * (SLLM_I2S_QK / 4);
        const int8_t   * ab  = a + b * SLLM_I2S_QK;
        for (size_t lane = 0; lane < 32; ++lane) {
            const uint8_t byte = blk[lane];
            for (unsigned field = 0; field < 4; ++field) {
                const uint8_t code = (uint8_t) ((byte >> (6u - 2u * field)) & 0x3u);
                acc += (int32_t) code * (int32_t) ab[field * 32 + lane];
            }
        }
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/* SIMD dot products                                                   */
/* ------------------------------------------------------------------ */

#if defined(SLLM_X86)

static inline int32_t hsum8(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    __m128i h2 = _mm_unpackhi_epi64(s, s);
    s = _mm_add_epi32(s, h2);
    __m128i h3 = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(s, h3));
}

/*
 * One 128-element block is 32 packed bytes and 128 activations.
 *
 * Within the block, byte `lane` holds four elements: field a of byte `lane` is
 * element 32*a + lane, and it pairs with activation 32*a + lane. So a single
 * 32-byte load yields four code vectors, and each pairs with a different
 * quarter of the block's activations.
 *
 * The obvious wrong version of this loop broadcasts one byte across a vector
 * and runs 32 times over the block, which computes each element four times
 * and pairs nothing correctly. Loading the whole chunk is both the correct
 * reading and the faster one.
 */
static inline void unpack_block(const uint8_t * blk, __m256i * c0, __m256i * c1,
                                __m256i * c2, __m256i * c3, __m256i mask) {
    const __m256i packed = _mm256_loadu_si256((const __m256i *) blk);
    *c0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask);
    *c1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    *c2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask);
    *c3 = _mm256_and_si256(packed, mask);
}

/* The v3 baseline: AVX2, no attribute, no guarded section. */
static int32_t dot_v3(const uint8_t * w, const int8_t * a, size_t n) {
    const size_t blocks = n / SLLM_I2S_QK;
    const __m256i mask  = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    __m256i acc = _mm256_setzero_si256();
    for (size_t b = 0; b < blocks; ++b) {
        __m256i c0, c1, c2, c3;
        unpack_block(w + b * 32u, &c0, &c1, &c2, &c3, mask);

        const int8_t * ab = a + b * SLLM_I2S_QK;
        const __m256i b0 = _mm256_loadu_si256((const __m256i *) (ab +  0));
        const __m256i b1 = _mm256_loadu_si256((const __m256i *) (ab + 32));
        const __m256i b2 = _mm256_loadu_si256((const __m256i *) (ab + 64));
        const __m256i b3 = _mm256_loadu_si256((const __m256i *) (ab + 96));

        /*
         * maddubs is the SATURATING signed-by-unsigned form and the sum lands
         * in int16 lanes before widening. With codes in 0..2 and activations in
         * -128..127, a product is at most 254, so a block's 128 products stay
         * well inside int16 and the lanes only ever accumulate across blocks.
         * That is a property of this data, not a guarantee of the instruction.
         */
        const __m256i d0 = _mm256_maddubs_epi16(c0, b0);
        const __m256i d1 = _mm256_maddubs_epi16(c1, b1);
        const __m256i d2 = _mm256_maddubs_epi16(c2, b2);
        const __m256i d3 = _mm256_maddubs_epi16(c3, b3);
        const __m256i s16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                              _mm256_add_epi16(d2, d3));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s16, one16));
    }
    return hsum8(acc);
}

/* Above v3: AVX-VNNI, guarded. */
SLLM_ISA_TARGET_VNNI SLLM_ISA_EXT
static int32_t dot_vnni(const uint8_t * w, const int8_t * a, size_t n) {
    const size_t blocks = n / SLLM_I2S_QK;
    const __m256i mask  = _mm256_set1_epi8(0x03);

    __m256i acc = _mm256_setzero_si256();
    for (size_t b = 0; b < blocks; ++b) {
        __m256i c0, c1, c2, c3;
        unpack_block(w + b * 32u, &c0, &c1, &c2, &c3, mask);

        const int8_t * ab = a + b * SLLM_I2S_QK;
        const __m256i b0 = _mm256_loadu_si256((const __m256i *) (ab +  0));
        const __m256i b1 = _mm256_loadu_si256((const __m256i *) (ab + 32));
        const __m256i b2 = _mm256_loadu_si256((const __m256i *) (ab + 64));
        const __m256i b3 = _mm256_loadu_si256((const __m256i *) (ab + 96));

        /*
         * Four 4-way int8 dot products per call, accumulated straight into
         * int32. No int16 intermediate, so nothing can saturate at all. This
         * is the reference's own VNNI path, not an optimisation invented here.
         */
        acc = _mm256_dpbusd_epi32(acc, c0, b0);
        acc = _mm256_dpbusd_epi32(acc, c1, b1);
        acc = _mm256_dpbusd_epi32(acc, c2, b2);
        acc = _mm256_dpbusd_epi32(acc, c3, b3);
    }
    return hsum8(acc);
}

static int32_t (*g_dot)(const uint8_t *, const int8_t *, size_t) = NULL;
static sllm_isa_level g_dot_isa = SLLM_ISA_COUNT;

#endif /* SLLM_X86 */

/* ------------------------------------------------------------------ */

int32_t sllm_i2s_dot(const uint8_t * w, const int8_t * q, size_t n) {
    if (w == NULL || q == NULL || n < SLLM_I2S_QK) {
        return 0;
    }
#if defined(SLLM_X86)
    if (g_dot == NULL) {
        return dot_v3(w, q, n);
    }
    return g_dot(w, q, n);
#else
    return sllm_i2s_dot_scalar(w, q, n);
#endif
}

void sllm_i2s_gemv(const uint8_t * w, size_t n_rows, size_t n,
                   const int8_t * q, int32_t * rows_out) {
    if (w == NULL || q == NULL || rows_out == NULL) {
        return;
    }
    const size_t row_bytes = n / 4u;
    for (size_t r = 0; r < n_rows; ++r) {
        rows_out[r] = sllm_i2s_dot(w + r * row_bytes, q, n);
    }
}

float sllm_i2s_epilogue(int32_t dot, int32_t act_sum, float act_scale, float w_scale) {
    return ((float) (dot - act_sum)) / act_scale * w_scale;
}

sllm_isa_level sllm_i2s_installed_isa(void) {
#if defined(SLLM_X86)
    return g_dot_isa == SLLM_ISA_COUNT ? SLLM_ISA_V3 : g_dot_isa;
#else
    return SLLM_ISA_V3;
#endif
}

void sllm_i2s_select_isa(const sllm_isa_dispatch * d) {
#if defined(SLLM_X86)
    g_dot     = dot_v3;
    g_dot_isa = SLLM_ISA_V3;
    if (d != NULL && sllm_isa_level_supported(SLLM_ISA_VNNI, &d->caps) &&
        d->selected >= SLLM_ISA_VNNI) {
        g_dot     = dot_vnni;
        g_dot_isa = SLLM_ISA_VNNI;
    }
#else
    (void) d;
#endif
}
