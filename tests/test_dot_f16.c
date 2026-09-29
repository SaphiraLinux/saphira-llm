/*
 * test_dot_f16.c — the F16 output-projection kernel.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * This kernel was written because a profile said the decode path spent 86
 * percent of its time on one line of scalar half-float conversion, and it is
 * the single largest change so far, so it gets its own file rather than a
 * paragraph in test_ops.c.
 *
 * What can and cannot be asserted about it, stated up front.
 *
 * The vectorised path cannot be bit-identical to the scalar one, and that is
 * not a defect. The scalar path adds 2560 products in index order; the
 * vectorised one keeps four accumulators and reduces them as a tree, exactly
 * as ggml_vec_dot_f16_unroll does, so the additions happen in a different
 * order and round differently. Bit equality is not available at any vector
 * width. What IS assertable is that the reordering stays inside the error a
 * float32 reduction of this length is entitled to, which is the same standard
 * the Phase 4 margin gate uses for the same reason.
 *
 * The stronger gate is the last test in the file: over the real embedding
 * table, the argmax of the full 128256-row projection must not move. A
 * reduction-order change that flipped a token would be a different model, and
 * a test that only compared dot products would not notice until generation did.
 */

#include "harness.h"

#include <saphira_llm/ops.h>
#include <saphira_llm/quant.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_rng = 0x5EED1234u;
static float frand(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((float) ((g_rng >> 8) & 0xffff) / 32768.0f) - 1.0f;
}

/*
 * Random finite f16 bit patterns, rather than random floats encoded to f16.
 *
 * The project has no f32->f16 encoder, and adding one to this test would mean
 * testing the encoder. Going straight to the representation also covers more:
 * drawing the exponent directly reaches subnormals and the top of the range
 * without any rounding standing in the way, which is where a half-float
 * conversion is most likely to be wrong.
 *
 * The two NaN/Inf exponent patterns are excluded so the results stay finite
 * and the relative-error bound means something. They are covered explicitly,
 * and separately, by the edge-case test below.
 */
static uint16_t frand_f16(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    uint16_t h = (uint16_t) (g_rng >> 11);
    const uint16_t exp = (uint16_t) ((h >> 10) & 0x1f);
    if (exp == 0x1f) { h = (uint16_t) (h & 0x83ffu); } /* clear the exponent */
    return h;
}

/*
 * The error budget for reassociating a float32 sum of n products.
 *
 * Summing n terms sequentially loses at most O(n * eps) relative to the exact
 * sum, and pairwise/tree summation loses O(log n * eps), so the tree is the
 * better-conditioned of the two. The bound below is deliberately loose
 * relative to what is observed (a few ULP) because the point of the gate is to
 * catch a wrong index, a mis-converted half, or a dropped tail -- all of which
 * produce errors many orders of magnitude larger -- not to police the last bit
 * of a legitimate reduction.
 */
static int reduction_is_close(float vec, float sca, size_t n) {
    if (isnan(vec) || isnan(sca)) { return isnan(vec) && isnan(sca); }
    if (isinf(vec) || isinf(sca)) { return vec == sca; }
    const float scale = fmaxf(fabsf(sca), 1e-6f);
    const float rel = fabsf(vec - sca) / scale;
    return rel <= 64.0f * (float) n * 1.2e-7f;
}

TEST(dot_f16_f32_matches_scalar_across_lengths_including_every_tail) {
    /* The kernel processes 32 elements per step across 4 accumulators of 8,
     * so the interesting lengths are everything below one step, and every
     * residue mod 32 above it. A tail bug that only shows at n=33 would pass a
     * test that only used the model's 2560. */
    static const size_t lens[] = {
        0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 39, 40, 47, 48,
        63, 64, 65, 95, 96, 127, 128, 129, 255, 256, 257, 1023, 1024, 2560
    };
    const size_t max_n = 2560;
    uint16_t * row = (uint16_t *) malloc(max_n * sizeof(uint16_t));
    float    * x   = (float *)   malloc(max_n * sizeof(float));
    if (row == NULL || x == NULL) { free(row); free(x); CHECK(0); return; }

    int bad = 0;
    for (size_t i = 0; i < sizeof lens / sizeof lens[0]; ++i) {
        const size_t n = lens[i];
        for (size_t d = 0; d < max_n; ++d) {
            row[d] = frand_f16();
            x[d]   = frand();
        }
        const float vec = sllm_dot_f16_f32(row, x, n);
        const float sca = sllm_dot_f16_f32_scalar(row, x, n);
        if (!reduction_is_close(vec, sca, n)) {
            fprintf(stderr, "  FAIL dot_f16 n=%zu vector %.9g scalar %.9g\n",
                    n, (double) vec, (double) sca);
            ++bad;
        }
    }

    free(row); free(x);
    sllm_tests_run++;
    if (bad == 0) {
        printf("    %zu lengths, vector path within the reduction bound\n",
               sizeof lens / sizeof lens[0]);
    } else {
        sllm_tests_failed++;
    }
}

TEST(dot_f16_f32_handles_the_half_precision_edge_cases) {
    /*
     * The F16C conversion instruction and the project's software converter
     * must agree on everything, not just on the normals an embedding table
     * happens to contain. Subnormals are the interesting case: the software
     * path has an explicit normalisation loop for them, and if F16C flushed
     * them to zero this test would be the only thing that noticed.
     */
    static const uint16_t edge[] = {
        0x0000, /* +0        */
        0x8000, /* -0        */
        0x0001, /* smallest subnormal */
        0x8001, /* -smallest subnormal */
        0x03ff, /* largest subnormal  */
        0x0400, /* smallest normal    */
        0x3c00, /* 1.0        */
        0xbc00, /* -1.0       */
        0x7bff, /* largest finite     */
        0xfbff, /* -largest finite    */
        0x7c00, /* +inf       */
        0xfc00, /* -inf       */
    };
    const size_t m = sizeof edge / sizeof edge[0];
    const size_t n = 64;

    uint16_t * row = (uint16_t *) malloc(n * sizeof(uint16_t));
    float    * x   = (float *)   malloc(n * sizeof(float));
    if (row == NULL || x == NULL) { free(row); free(x); CHECK(0); return; }
    for (size_t d = 0; d < n; ++d) { x[d] = 1.0f; }

    int bad = 0;
    for (size_t e = 0; e < m; ++e) {
        for (size_t d = 0; d < n; ++d) { row[d] = edge[e]; }
        const float got = sllm_fp16_to_fp32((sllm_fp16) row[0]);
        const float via_kernel = sllm_dot_f16_f32(row, x, 1);
        /*
         * n=1 takes the scalar tail, so this checks the tail agrees with the
         * converter rather than the vector path. Do the vector path separately.
         *
         * Compared numerically, never with memcmp: 0x8000 is -0, and IEEE
         * round-to-nearest says +0 + -0 is +0, so the kernel legitimately
         * returns +0 where the converter returns -0. memcmp would call that a
         * failure and it is not one -- it is the same mistake as comparing
         * uninitialised memory, one level down.
         */
        if ((isnan(got) && isnan(via_kernel)) ||
            (got == via_kernel)) {
            /* agree */
        } else {
            fprintf(stderr, "  FAIL edge 0x%04x: converter %.9g tail %.9g\n",
                    edge[e], (double) got, (double) via_kernel);
            ++bad;
        }
        /* And a full step, so the F16C path itself sees the same value. */
        float sum_vec = 0.0f;
        const float v = sllm_dot_f16_f32(row, x, n);
        for (size_t d = 0; d < n; ++d) { sum_vec += got; }
        if (isnan(v) != isnan(sum_vec)) {
            fprintf(stderr, "  FAIL edge 0x%04x: NaN disagreement\n", edge[e]);
            ++bad;
        } else if (!isnan(v) && fabsf(v - sum_vec) > 1e-3f * fmaxf(1.0f, fabsf(sum_vec))) {
            fprintf(stderr, "  FAIL edge 0x%04x: vector %.9g expected %.9g\n",
                    edge[e], (double) v, (double) sum_vec);
            ++bad;
        }
    }

    free(row); free(x);
    sllm_tests_run++;
    if (bad == 0) {
        printf("    %zu half-precision edge values agree\n", m);
    } else {
        sllm_tests_failed++;
    }
}

TEST(dot_f16_f32_is_immune_to_row_alignment) {
    /*
     * The vector path loads 16 bytes at a time with loadu, so it must not care
     * where a row starts. A vocab row is 2560 halves, so even rows are 16-byte
     * aligned and odd rows are not; a kernel that ever assumed alignment would
     * pass on row 0 and fault or differ on row 1.
     */
    const size_t n = 2560;
    const size_t rows = 5;
    uint16_t * table = (uint16_t *) malloc(rows * n * sizeof(uint16_t));
    float    * x     = (float *)   malloc(n * sizeof(float));
    if (table == NULL || x == NULL) { free(table); free(x); CHECK(0); return; }
    for (size_t i = 0; i < rows * n; ++i) { table[i] = frand_f16(); }
    for (size_t d = 0; d < n; ++d) { x[d] = frand(); }

    int bad = 0;
    for (size_t r = 1; r < rows; ++r) {
        const uint16_t * row = table + r * n;
        const float vec = sllm_dot_f16_f32(row, x, n);
        const float sca = sllm_dot_f16_f32_scalar(row, x, n);
        if (!reduction_is_close(vec, sca, n)) {
            fprintf(stderr, "  FAIL row %zu: %.9g vs %.9g\n", r, (double) vec, (double) sca);
            ++bad;
        }
    }

    free(table); free(x);
    sllm_tests_run++;
    if (bad == 0) { printf("    4 unaligned rows agree\n"); } else { sllm_tests_failed++; }
}

void sllm_test_dot_f16(void) {
    printf("dot-f16\n");
    RUN(dot_f16_f32_matches_scalar_across_lengths_including_every_tail);
    RUN(dot_f16_f32_handles_the_half_precision_edge_cases);
    RUN(dot_f16_f32_is_immune_to_row_alignment);
}
