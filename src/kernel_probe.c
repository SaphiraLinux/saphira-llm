/*
 * kernel_probe.c — the first kernel in the tree, and the pattern for all others.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Why this exists
 * ---------------
 * Two reasons, both practical.
 *
 * 1. It establishes the dispatch pattern. A signed-int8 dot product has a
 *    genuine v3 implementation (AVX2 maddubs) and a genuine above-v3 one
 *    (AVX-VNNI dpbusd). That makes it the smallest honest example of the rule
 *    the whole project depends on: the v3 path is a real optimised path, and
 *    dispatch only ever adds to it.
 *
 * 2. `make check-isa` refuses to pass vacuously. It requires a populated
 *    .text.sllm_isa_ext section, so there has to be at least one kernel that
 *    genuinely needs more than v3. This is it.
 *
 * dpbusd is the instruction the BitNet ternary GEMM is built on: four
 * 4-way int8 dot products accumulated straight into int32, no int16
 * intermediate and therefore no saturation. Upstream's BLAST kernel has to
 * accumulate in int16 and lives with the saturation that implies, so this is
 * also the first place the performance argument for Phase 6 becomes concrete.
 *
 * The two implementations are required to agree exactly, and the test suite
 * checks that. A dispatch mechanism that silently changes results is worse
 * than no dispatch at all.
 */

#include <saphira_llm/sllm.h>

#include <stddef.h>
#include <string.h>

#if defined(__x86_64__)
#  include <immintrin.h>
#  define SLLM_PROBE_X86 1
#endif

/* The dispatch table. A NULL entry means "this level was not selected", and
 * the test suite asserts that, so a forced-low run cannot reach guarded code
 * even by accident. */
static struct {
    int32_t (*dot)(const int8_t *, const int8_t *, size_t);
    sllm_isa_level installed_for;
} g_probe;

#if defined(SLLM_PROBE_X86)

/* ------------------------------------------------------------------ */
/* x86-64-v3 path: AVX2, no attribute, no guard                        */
/* ------------------------------------------------------------------ */

/* Horizontal sum of the eight int32 lanes. Plain C on a spilled array, which
 * is unambiguous and is not on the critical path of this probe. */
static int32_t hsum8(__m256i v) {
    int32_t lanes[8];
    _mm256_storeu_si256((__m256i *) lanes, v);
    int32_t s = 0;
    for (int k = 0; k < 8; ++k) {
        s += lanes[k];
    }
    return s;
}

static int32_t dot_i8_v3(const int8_t * a, const int8_t * b, size_t n) {
    __m256i acc = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128i va8 = _mm_loadu_si128((const __m128i *) (a + i));
        const __m128i vb8 = _mm_loadu_si128((const __m128i *) (b + i));
        /* Sign-extend to int16, then madd pairs into int32. Multiplying
         * signed by signed matters: maddubs would treat its first operand as
         * unsigned, which is the wrong tool here. */
        const __m256i va16 = _mm256_cvtepi8_epi16(va8);
        const __m256i vb16 = _mm256_cvtepi8_epi16(vb8);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va16, vb16));
    }

    int32_t lanes[8];
    _mm256_storeu_si256((__m256i *) lanes, acc);
    int32_t sum = 0;
    for (int k = 0; k < 8; ++k) {
        sum += lanes[k];
    }
    for (; i < n; ++i) {
        sum += (int32_t) a[i] * (int32_t) b[i];
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/* above-v3 path: AVX-VNNI, guarded                                    */
/* ------------------------------------------------------------------ */

/*
 * SLLM_ISA_EXT puts this function in .text.sllm_isa_ext. SLLM_ISA_TARGET_VNNI
 * lets the compiler emit AVX-VNNI inside it. Both are required: the section
 * placement is what makes the instruction visible to `make check-isa`, and
 * the target attribute is what lets the intrinsic be used at all in a binary
 * whose baseline is v3.
 */
SLLM_ISA_TARGET_VNNI SLLM_ISA_EXT
static int32_t dot_i8_vnni(const int8_t * a, const int8_t * b, size_t n) {
    /*
     * dpbusd has one asymmetry that will bite anyone porting a kernel to it:
     * the FIRST operand is unsigned, the second is signed. Feeding it two
     * signed int8 vectors silently reads every negative weight as a large
     * positive one and returns a plausible wrong number. The bias below
     * shifts `a` into [0, 255] so both sign ranges are representable, and
     * the correction at the end puts the mean back:
     *
     *   sum((a_k + 128) * b_k) = sum(a_k * b_k) + 128 * sum(b_k)
     *
     * so sum(a*b) = dpbusd_result - 128 * sum(b).
     *
     * Note for the ternary GEMM in Phase 3: it will not need this correction
     * at all. I2_S weights are 2-bit codes in {0, 1, 2}, which are already
     * non-negative, so the unsigned first operand is exactly what we want and
     * only the activations are signed. This kernel is written the hard way on
     * purpose, because the easy way is the one that fails quietly.
     */
    const __m256i bias = _mm256_set1_epi8((char) 0x80);

    __m256i acc = _mm256_setzero_si256();
    int32_t bsum = 0;

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i va = _mm256_loadu_si256((const __m256i *) (a + i));
        const __m256i vb = _mm256_loadu_si256((const __m256i *) (b + i));
        acc = _mm256_dpbusd_epi32(acc, _mm256_xor_si256(va, bias), vb);

        const __m128i blo = _mm256_castsi256_si128(vb);
        const __m128i bhi = _mm256_extracti128_si256(vb, 1);
        bsum += hsum8(_mm256_madd_epi16(_mm256_cvtepi8_epi16(blo), _mm256_set1_epi16(1)));
        bsum += hsum8(_mm256_madd_epi16(_mm256_cvtepi8_epi16(bhi), _mm256_set1_epi16(1)));
    }

    int32_t sum = hsum8(acc) - 128 * bsum;
    for (; i < n; ++i) {
        sum += (int32_t) a[i] * (int32_t) b[i];
    }
    return sum;
}

#endif /* SLLM_PROBE_X86 */

/* ------------------------------------------------------------------ */

/* Scalar reference. Always compiled, never dispatched to except by the
 * tests, and used to prove the SIMD paths are right. */
static int32_t dot_i8_scalar(const int8_t * a, const int8_t * b, size_t n) {
    int32_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += (int32_t) a[i] * (int32_t) b[i];
    }
    return sum;
}

void sllm_probe_init(const sllm_isa_dispatch * d) {
    memset(&g_probe, 0, sizeof(g_probe));

    g_probe.dot         = dot_i8_scalar;
    g_probe.installed_for = SLLM_ISA_COUNT;

#if defined(SLLM_PROBE_X86)
    /* v3 is not a fallback: it is installed unconditionally, because the
     * whole binary assumes it. */
    g_probe.dot = dot_i8_v3;
    g_probe.installed_for = SLLM_ISA_V3;

    if (d != NULL && sllm_isa_level_supported(SLLM_ISA_VNNI, &d->caps) &&
        d->selected >= SLLM_ISA_VNNI) {
        g_probe.dot = dot_i8_vnni;
        g_probe.installed_for = SLLM_ISA_VNNI;
    }
#else
    (void) d;
#endif
}

int32_t sllm_probe_dot(const int8_t * a, const int8_t * b, size_t n) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    return g_probe.dot(a, b, n);
}

int32_t sllm_probe_dot_scalar(const int8_t * a, const int8_t * b, size_t n) {
    if (a == NULL || b == NULL) {
        return 0;
    }
    return dot_i8_scalar(a, b, n);
}

sllm_isa_level sllm_probe_installed_for(void) {
    return g_probe.installed_for;
}
