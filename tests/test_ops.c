/*
 * test_ops.c — golden vectors for the quantiser and vector kernels.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Every kernel is checked against a scalar reference written independently in
 * this file, not against another SIMD path. Two SIMD kernels agreeing with
 * each other proves very little; a SIMD kernel agreeing with the definition
 * proves something.
 */

#include "harness.h"

#include <saphira_llm/ops.h>
#include <saphira_llm/quant.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* A small deterministic generator, so a failure is always reproducible. */
static uint32_t g_rng = 0xC0FFEEu;
static float frand(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((float) ((g_rng >> 8) & 0xffff) / 32768.0f) - 1.0f;
}

TEST(fp16_round_trip_covers_the_awkward_values) {
    /* Exact, special and subnormal values. A conversion that is wrong for
     * subnormals is wrong for real model data, and quietly so. */
    struct { uint16_t h; float f; } cases[] = {
        { 0x0000, 0.0f }, { 0x8000, -0.0f },
        { 0x3c00, 1.0f }, { 0xbc00, -1.0f },
        { 0x4000, 2.0f }, { 0x3555, 0.33325195f },
        { 0x7bff, 65504.0f },                      /* largest finite */
        { 0x0001, 5.9604645e-8f },                 /* smallest subnormal */
        { 0x03ff, 6.0975552e-5f },                 /* largest subnormal */
        { 0x7c00, INFINITY }, { 0xfc00, -INFINITY },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float got = sllm_fp16_to_fp32(cases[i].h);
        sllm_tests_run++;
        const float want = cases[i].f;
        if (want == 0.0f) {
            if (got != 0.0f) {
                sllm_tests_failed++;
                fprintf(stderr, "  FAIL fp16 0x%04x: got %g want %g\n", cases[i].h, (double) got, (double) want);
            }
        } else if (fabsf(got - want) > fabsf(want) * 1e-6f + 1e-30f) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL fp16 0x%04x: got %g want %g\n",
                    cases[i].h, (double) got, (double) want);
        }
    }

    /* Round trip through the encoder side: every f16 in a range must decode
     * to something the encoder would produce back. */
    for (uint32_t h = 0x0400; h < 0x3c00; h += 37) {
        const float f = sllm_fp16_to_fp32((sllm_fp16) h);
        /* Re-encode by hand and check the identity. */
        uint32_t bits;
        memcpy(&bits, &f, 4);
        const uint32_t exp = (bits >> 23) & 0xff;
        const uint16_t re = (uint16_t) ((((exp - 127 + 15) & 0x1f) << 10) |
                                          ((bits >> 13) & 0x3ff));
        CHECK_EQ_INT(re, (int) h);
    }
}

/* ------------------------------------------------------------------ */
/* I2_S                                                                */
/* ------------------------------------------------------------------ */

TEST(i2s_layout_matches_the_converter_definition) {
    /*
     * This test used to derive the packing from sllm_i2s_code, which made it
     * structurally unable to detect that the reader had the four two-bit fields
     * backwards -- and it did not detect it, for two phases. It now packs with
     * sllm_pack_i2s_like_converter, written from utils/convert-hf-to-gguf-bitnet.py
     * rather than from the reader, so the two can disagree.
     */
    enum { N = 8192 };   /* two 4096-weight blocks */
    uint8_t * codes  = malloc(N);
    uint8_t * packed = malloc(N / 4 + 32);
    float * row = malloc(N * sizeof(float));
    CHECK(codes && packed && row);
    if (!codes || !packed || !row) { free(codes); free(packed); free(row); return; }

    for (size_t k = 0; k < N; ++k) { codes[k] = (uint8_t) ((k * 7 + k / 128) % 3); }
    sllm_pack_i2s_like_converter(packed, codes, N);

    int bad = 0;
    for (size_t k = 0; k < N; ++k) {
        if (sllm_i2s_code(packed, k) != codes[k]) {
            if (bad < 4) {
                fprintf(stderr, "  FAIL i2s weight %zu: got %u want %u\n",
                        k, sllm_i2s_code(packed, k), codes[k]);
            }
            ++bad;
        }
    }
    sllm_tests_run++;
    if (bad != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL i2s layout: %d of %d weights misread\n", bad, N);
    }

    sllm_i2s_row(packed, row, N);
    for (size_t k = 0; k < N; k += 97) {
        CHECK_SAME_BITS(row[k], (float) codes[k]);
    }
    free(codes); free(packed); free(row);
}

TEST(i2s_scale_sits_after_the_packed_weights) {
    enum { N = 1024 };
    uint8_t * packed = calloc(N / 4 + 32, 1);
    CHECK(packed != NULL);
    if (packed == NULL) { return; }

    const float want = 0.03125f;
    memcpy(packed + N / 4, &want, sizeof(want));
    const float got = sllm_i2s_scale(packed, N);
    CHECK(got == want);

    /* Getting this offset wrong is the single most dangerous thing in the
     * I2_S reader, so the boundary either side is checked too. */
    const float neighbour = 99.0f;
    memcpy(packed + N / 4 - 4, &neighbour, sizeof(neighbour));
    CHECK(sllm_i2s_scale(packed, N) == want);
    free(packed);
}

/* ------------------------------------------------------------------ */
/* row dequantisation                                                  */
/* ------------------------------------------------------------------ */

static void ref_dequant(sllm_ggml_type type, const void * src, float * dst, size_t n);

TEST(dequant_row_matches_a_scalar_reference) {
    for (sllm_ggml_type type = SLLM_TYPE_F32; type < SLLM_TYPE_I2_S; ++type) {
        /* Pick the types we implement; the rest must be refused, not guessed. */
        const bool implemented = (type == SLLM_TYPE_F32 || type == SLLM_TYPE_F16 ||
                                  type == SLLM_TYPE_BF16 || type == SLLM_TYPE_Q8_0 ||
                                  type == SLLM_TYPE_Q4_0);
        if (!implemented) {
            continue;
        }

        const size_t n = 512;
        uint64_t blck = 1, bytes_per = 4;
        switch (type) {
            case SLLM_TYPE_F32:  blck = 1;   bytes_per = 4; break;
            case SLLM_TYPE_F16:  blck = 1;   bytes_per = 2; break;
            case SLLM_TYPE_BF16: blck = 1;   bytes_per = 2; break;
            case SLLM_TYPE_Q8_0: blck = 32;  bytes_per = 34; break;
            case SLLM_TYPE_Q4_0: blck = 32;  bytes_per = 18; break;
            default: break;
        }
        const size_t nbytes = (n / blck) * bytes_per;

        uint8_t * src = malloc(nbytes ? nbytes : 1);
        float * got = malloc(n * sizeof(float));
        float * want = malloc(n * sizeof(float));
        CHECK(src && got && want);
        if (!src || !got || !want) { free(src); free(got); free(want); continue; }

        /* Fill with random bytes, then let the reference interpret them. Any
         * bit pattern must decode, including the ones that are not canonical
         * for the format. */
        for (size_t i = 0; i < nbytes; ++i) {
            src[i] = (uint8_t) (i * 7 + 13);
        }
        ref_dequant(type, src, want, n);

        CHECK_STATUS(sllm_dequant_row(type, src, got, n), SLLM_OK);
        for (size_t i = 0; i < n; ++i) {
            sllm_tests_run++;
            /*
             * Compare the bits, not the values. The fill pattern is raw bytes,
             * so it legitimately produces f16 encodings that decode to NaN and
             * a Q4_0 header that decodes to an infinite scale. NaN != NaN, so
             * a value comparison reports a mismatch that is not one, and a
             * bitwise comparison reports NaN payloads and signed zeros exactly
             * as they are. It is also the stronger assertion.
             *
             * The one exception is NaN payload bits. The scalar conversion
             * preserves the f16 payload while F16C quiets it, so the two paths
             * can both return a NaN with different bits. Requiring identical
             * payload propagation would be pinning a hardware detail rather
             * than testing a property anyone relies on, so the requirement is
             * the one that matters: both must produce a NaN, and every
             * non-NaN value must match bit for bit.
             */
            const int both_nan = isnan(got[i]) && isnan(want[i]);
            if (!both_nan && memcmp(&got[i], &want[i], sizeof(float)) != 0) {
                sllm_tests_failed++;
                /* Always name the index. A conditional report hid a real
                 * mismatch here, because the failing element happened to sit
                 * past the reporting threshold. */
                fprintf(stderr, "  FAIL %s element %zu of %zu: got %.9g want %.9g\n",
                        sllm_gguf_type_name(type), i, n, (double) got[i], (double) want[i]);
                break;
            }
        }
        free(src); free(got); free(want);
    }

    /* Types we have not implemented must be refused specifically. */
    float dst[8] = {0};
    uint8_t src[64] = {0};
    CHECK_STATUS(sllm_dequant_row(SLLM_TYPE_Q5_K, src, dst, 8), SLLM_ERR_TYPE_UNSUPPORTED);
    CHECK_STATUS(sllm_dequant_row(SLLM_TYPE_Q6_K, src, dst, 8), SLLM_ERR_TYPE_UNSUPPORTED);
    CHECK_STATUS(sllm_dequant_row(SLLM_TYPE_I2_S, src, dst, 8), SLLM_OK);
    CHECK_STATUS(sllm_dequant_row(SLLM_TYPE_F32, NULL, dst, 8), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_dequant_row(SLLM_TYPE_F32, src, NULL, 8), SLLM_ERR_ARG);
}

/* The reference: written from the format definition, not from the kernel. */
static void ref_dequant(sllm_ggml_type type, const void * src, float * dst, size_t n) {
    const uint8_t * s = (const uint8_t *) src;
    switch (type) {
        case SLLM_TYPE_F32:
            for (size_t i = 0; i < n; ++i) {
                float f; memcpy(&f, s + i * 4, 4); dst[i] = f;
            }
            break;
        case SLLM_TYPE_F16:
            for (size_t i = 0; i < n; ++i) {
                uint16_t h; memcpy(&h, s + i * 2, 2); dst[i] = sllm_fp16_to_fp32(h);
            }
            break;
        case SLLM_TYPE_BF16:
            for (size_t i = 0; i < n; ++i) {
                uint16_t b; memcpy(&b, s + i * 2, 2);
                uint32_t bits = (uint32_t) b << 16;
                float f; memcpy(&f, &bits, 4); dst[i] = f;
            }
            break;
        case SLLM_TYPE_Q8_0: {
            size_t i = 0;
            while (i < n) {
                uint16_t h; memcpy(&h, s, 2);
                const float d = sllm_fp16_to_fp32(h);
                for (int j = 0; j < 32 && i + j < n; ++j) {
                    dst[i + j] = d * (float) ((const int8_t *) (s + 2))[j];
                }
                i += 32; s += 34;
            }
            break;
        }
        case SLLM_TYPE_Q4_0: {
            size_t i = 0;
            while (i < n) {
                uint16_t h; memcpy(&h, s, 2);
                const float d = sllm_fp16_to_fp32(h);
                for (int j = 0; j < 32 && i + j < n; ++j) {
                    const uint8_t byte = s[2 + j / 2];
                    const int v = (j % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
                    dst[i + j] = d * (float) (v - 8);
                }
                i += 32; s += 18;
            }
            break;
        }
        case SLLM_TYPE_I2_S:
            sllm_i2s_row(src, dst, n);
            break;
        default:
            break;
    }
}

/* ------------------------------------------------------------------ */
/* vector kernels                                                      */
/* ------------------------------------------------------------------ */

TEST(ops_add_mul_match_the_scalar_definition) {
    for (size_t n = 0; n < 100; ++n) {
        float a[128], b[128], got[128];
        for (size_t i = 0; i < n; ++i) { a[i] = frand(); b[i] = frand(); }

        sllm_add(got, a, b, n);
        for (size_t i = 0; i < n; ++i) { CHECK_SAME_BITS(got[i], a[i] + b[i]); }

        sllm_mul(got, a, b, n);
        for (size_t i = 0; i < n; ++i) { CHECK_SAME_BITS(got[i], a[i] * b[i]); }
    }
}

TEST(ops_rms_norm_matches_the_scalar_definition) {
    for (size_t n = 1; n <= 260; n += 37) {
        float x[512], w[512], got[512];
        for (size_t i = 0; i < n; ++i) { x[i] = frand() * 4.0f; w[i] = frand(); }

        double ss = 0.0;
        for (size_t i = 0; i < n; ++i) { ss += (double) x[i] * x[i]; }
        const float scale = (float) (1.0 / sqrt(ss / (double) n + 1e-5));

        sllm_rms_norm(got, x, w, n, 1e-5f);
        for (size_t i = 0; i < n; ++i) {
            const float want = scale * x[i] * w[i];
            sllm_tests_run++;
            if (fabsf(got[i] - want) > 1e-5f * (fabsf(want) + 1.0f)) {
                sllm_tests_failed++;
                fprintf(stderr, "  FAIL rms_norm n=%zu i=%zu: got %g want %g\n",
                        n, i, (double) got[i], (double) want);
                break;
            }
        }
    }
}

TEST(ops_softmax_sums_to_one_and_is_finite) {
    for (size_t n = 1; n <= 200; n += 13) {
        float x[256];
        for (size_t i = 0; i < n; ++i) { x[i] = frand() * 20.0f; }
        sllm_softmax_inplace(x, n);

        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            CHECK(isfinite(x[i]));
            sum += x[i];
        }
        sllm_tests_run++;
        if (fabs(sum - 1.0) > 1e-5) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL softmax n=%zu sums to %g\n", n, sum);
        }
    }

    /* Large inputs must not overflow: the max is subtracted first. */
    float big[4] = { 1000.0f, 1001.0f, 999.0f, 1000.5f };
    sllm_softmax_inplace(big, 4);
    for (int i = 0; i < 4; ++i) { CHECK(isfinite(big[i])); }
}

TEST(ops_silu_matches_the_scalar_definition) {
    for (size_t n = 0; n < 100; ++n) {
        float x[128], orig[128];
        for (size_t i = 0; i < n; ++i) { x[i] = frand() * 6.0f; orig[i] = x[i]; }
        sllm_silu_inplace(x, n);
        for (size_t i = 0; i < n; ++i) {
            const float want = orig[i] / (1.0f + expf(-orig[i]));
            sllm_tests_run++;
            if (fabsf(x[i] - want) > 1e-6f * (fabsf(want) + 1.0f)) {
                sllm_tests_failed++;
                fprintf(stderr, "  FAIL silu n=%zu i=%zu: got %g want %g\n",
                        n, i, (double) x[i], (double) want);
                break;
            }
        }
    }
}

TEST(ops_rope_is_a_rotation_and_position_zero_is_the_identity) {
    const size_t n_rot = 64;
    float x[64];
    for (size_t i = 0; i < n_rot; ++i) { x[i] = frand(); }

    /* Position zero must leave the vector alone, in both layouts. A RoPE that
     * is not the identity at pos 0 is wrong in a way that only shows up later
     * as mysteriously degraded output. */
    float y[64];
    memcpy(y, x, sizeof(x));
    sllm_rope_inplace(y, n_rot, 0, 10000.0f, 1.0f, SLLM_ROPE_NEOX);
    for (size_t i = 0; i < n_rot; ++i) { CHECK(y[i] == x[i]); }
    memcpy(y, x, sizeof(x));
    sllm_rope_inplace(y, n_rot, 0, 10000.0f, 1.0f, SLLM_ROPE_NORMAL);
    for (size_t i = 0; i < n_rot; ++i) { CHECK(y[i] == x[i]); }

    /*
     * Each rotated pair must preserve its norm -- AND the pairs must be the
     * right ones.
     *
     * The norm property alone is not enough, and this is the test's original
     * defect. Both layouts preserve the norms of whatever pairs they rotate, so
     * checking adjacent pairs passes under NEOX even when NEOX is rotating the
     * two halves, and the two had in fact been swapped here for two phases
     * without this noticing. It is a norm-preserving relabelling of the whole
     * vector: every element looks right and the relative phase between
     * dimensions, which is the entire content of the encoding, is destroyed.
     *
     * So the pair indices are checked explicitly as well. NEOX rotates
     * (k, k + n/2) and NORMAL rotates (2k, 2k+1), from ggml's rotate_pairs
     * called with (n, n/2) and with (n, 1, scale=1) respectively.
     */
    for (int pos = 1; pos <= 5; ++pos) {
        const size_t half = n_rot / 2;
        struct { sllm_rope_type type; const char * name; } layouts[] = {
            { SLLM_ROPE_NEOX,   "neox"   },
            { SLLM_ROPE_NORMAL, "normal" },
        };
        for (size_t li = 0; li < 2; ++li) {
            memcpy(y, x, sizeof(x));
            sllm_rope_inplace(y, n_rot, pos, 10000.0f, 1.0f, layouts[li].type);
            for (size_t k = 0; k < half; ++k) {
                const size_t a = (layouts[li].type == SLLM_ROPE_NEOX) ? k : (k * 2);
                const size_t b = (layouts[li].type == SLLM_ROPE_NEOX) ? (k + half) : (k * 2 + 1);
                const double before = (double) x[a] * x[a] + (double) x[b] * x[b];
                const double after  = (double) y[a] * y[a] + (double) y[b] * y[b];
                sllm_tests_run++;
                if (fabs(before - after) > 1e-4 * (before + 1.0)) {
                    sllm_tests_failed++;
                    fprintf(stderr, "  FAIL rope %s pos=%d pair k=%zu (%zu,%zu): "
                                    "norm %g -> %g\n",
                            layouts[li].name, pos, k, a, b, before, after);
                    break;
                }
                /* And the rotation must actually have happened, or a
                 * no-op would satisfy the norm check too. */
                sllm_tests_run++;
                if (y[a] == x[a] && y[b] == x[b] && before > 0.0) {
                    sllm_tests_failed++;
                    fprintf(stderr, "  FAIL rope %s pos=%d pair k=%zu did not rotate\n",
                            layouts[li].name, pos, k);
                    break;
                }
            }
        }
    }

    /*
     * The two layouts must produce DIFFERENT vectors from the same input. This
     * is the assertion that would have caught the swap outright: identical
     * output from two supposedly different encodings means one of them is
     * implemented as the other.
     */
    {
        float a[64], b[64];
        memcpy(a, x, sizeof(x));
        memcpy(b, x, sizeof(x));
        sllm_rope_inplace(a, n_rot, 3, 10000.0f, 1.0f, SLLM_ROPE_NEOX);
        sllm_rope_inplace(b, n_rot, 3, 10000.0f, 1.0f, SLLM_ROPE_NORMAL);
        sllm_tests_run++;
        if (memcmp(a, b, sizeof(a)) == 0) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL neox and normal produced identical output; "
                            "one of them is implemented as the other\n");
        }
    }

    /*
     * freq_scale MULTIPLIES the angle: rope_yarn computes
     * theta_interp = freq_scale * theta_extrap, and the reference derives the
     * factor as 1/rope.scale_linear so that a scale of 2 in the metadata
     * becomes a factor below 1 and the rotation slows down for extrapolation.
     *
     * This test asserted the opposite -- that freq_scale divides, so a scale of
     * 2 equals doubling the position -- and it passed for two phases, because
     * it was never checked against the reference. It is the same lesson as the
     * NEOX/NORMAL swap: an assumption that reads as reasonable and is never
     * tested against the thing it is supposed to mirror.
     */
    float a[64], b[64];
    memcpy(a, x, sizeof(x));
    memcpy(b, x, sizeof(x));
    sllm_rope_inplace(a, n_rot, 4, 10000.0f, 1.0f, SLLM_ROPE_NEOX);
    sllm_rope_inplace(b, n_rot, 2, 10000.0f, 2.0f, SLLM_ROPE_NEOX);
    for (size_t i = 0; i < n_rot; ++i) {
        sllm_tests_run++;
        if (fabs((double) a[i] - (double) b[i]) > 1e-5) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL freq_scale should multiply the angle: "
                            "pos=4,scale=1 gave %g, pos=2,scale=2 gave %g\n",
                    (double) a[i], (double) b[i]);
            break;
        }
    }
}

TEST(ops_get_rows_gathers_and_clamps_bad_indices) {
    enum { ROWS = 4, COLS = 8 };
    float src[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; ++i) { src[i] = (float) i; }
    int32_t idx[3] = { 2, 0, 3 };
    float dst[3 * COLS];

    sllm_get_rows(dst, src, idx, 3, COLS);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < COLS; ++c) {
            CHECK_EQ_U64((uint64_t) dst[r * COLS + c],
                         (uint64_t) src[idx[r] * COLS + c]);
        }
    }

    /* A negative index is input from a model, not a promise. */
    int32_t bad[1] = { -5 };
    sllm_get_rows(dst, src, bad, 1, COLS);
    for (int c = 0; c < COLS; ++c) { CHECK_SAME_BITS(dst[c], src[c]); }
}

/*
 * Known-output RoPE vectors, for both layouts.
 *
 * The layout swap that Phase 4 found -- NEOX and NORMAL exchanging pairings --
 * survived two phases of property-based testing, because "the rotation
 * preserves each pair's norm" is true of BOTH pairings. A property cannot
 * distinguish two implementations when both satisfy it.
 *
 * These are literal expected outputs, computed from the reference's own
 * algorithm: the angle cache built by repeated multiplication in
 * ggml_rope_cache_init, then rotate_pairs with that layout's offsets. They are
 * for n_rot = 8, position 3, theta 10000, freq_scale 1, on the input x[i] = 1+i.
 *
 * They are not derived from our implementation, which is the point. If someone
 * exchanges the two pairings again, the numerics still look reasonable and
 * every norm still holds, but these constants do not move and the test fails.
 */
TEST(ops_rope_matches_known_output_for_both_layouts) {
    float x[8];
    for (int i = 0; i < 8; ++i) { x[i] = 1.0f + (float) i; }

    static const float ROPE_NEOX_GOLD[8] = {
        -1.69559252f, 0.137551665f, 2.78868151f, 3.97598219f,
        -4.80884266f, 6.32305956f, 7.08683681f, 8.01196384f
    };
    static const float ROPE_NORMAL_GOLD[8] = {
        -1.27223253f, -1.83886504f, 1.68392873f, 4.70790672f,
        4.81777716f, 6.14727783f, 6.97596884f, 8.02096462f
    };

    float neox[8], normal[8];
    memcpy(neox, x, sizeof(x));
    memcpy(normal, x, sizeof(x));
    sllm_rope_inplace(neox, 8, 3, 10000.0f, 1.0f, SLLM_ROPE_NEOX);
    sllm_rope_inplace(normal, 8, 3, 10000.0f, 1.0f, SLLM_ROPE_NORMAL);

    for (int i = 0; i < 8; ++i) {
        sllm_tests_run++;
        if (fabs((double) neox[i] - (double) ROPE_NEOX_GOLD[i]) > 1e-5) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL neox known output, element %d: %g, expected %g\n",
                    i, (double) neox[i], (double) ROPE_NEOX_GOLD[i]);
            break;
        }
    }
    for (int i = 0; i < 8; ++i) {
        sllm_tests_run++;
        if (fabs((double) normal[i] - (double) ROPE_NORMAL_GOLD[i]) > 1e-5) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL normal known output, element %d: %g, expected %g\n",
                    i, (double) normal[i], (double) ROPE_NORMAL_GOLD[i]);
            break;
        }
    }
}

void sllm_test_ops(void) {
    printf("ops\n");
    RUN(fp16_round_trip_covers_the_awkward_values);
    RUN(i2s_layout_matches_the_converter_definition);
    RUN(i2s_scale_sits_after_the_packed_weights);
    RUN(dequant_row_matches_a_scalar_reference);
    RUN(ops_add_mul_match_the_scalar_definition);
    RUN(ops_rms_norm_matches_the_scalar_definition);
    RUN(ops_softmax_sums_to_one_and_is_finite);
    RUN(ops_silu_matches_the_scalar_definition);
    RUN(ops_rope_is_a_rotation_and_position_zero_is_the_identity);
    RUN(ops_rope_matches_known_output_for_both_layouts);
    RUN(ops_get_rows_gathers_and_clamps_bad_indices);
}
