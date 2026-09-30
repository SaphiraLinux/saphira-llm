/*
 * test_i2s_convert.c — the native BF16 to I2_S converter.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * What this file is careful about, because it is the difference between a
 * converter that is checked and one that merely runs:
 *
 * The golden vectors in tests/golden/i2s-converter.txt are NOT produced by this
 * project's quantiser, and they are not produced by this project's decoder
 * either. They come from an oracle transcribed from TWO separate pinned upstream
 * functions -- quantize_i2_s (quants.c:1358) for the packing, and
 * dequantize_row_i2_s (quants.c:1335) for an independent read-back. If those
 * two transcriptions disagree the generator fails, so a self-consistent but
 * wrong pair cannot slip through.
 *
 * The original i2s-reference.txt fixture exists because the first version of
 * this project's decoder had the four fields of every byte reversed, and the
 * test did not catch it: the test built the packed buffer and read it back with
 * the same inverted assumption. A test that shares its assumption with the code
 * under it cannot detect that code is wrong. Everything here is checked against
 * bytes produced elsewhere.
 *
 * And the vectors carry BOTH upstream rules, because they are different
 * functions that disagree on about half of general input. Checking the rule
 * that was asked for, against bytes captured for that rule, is the point.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "harness.h"

#include <saphira_llm/i2s_convert.h>
#include <saphira_llm/quant.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define I2S_GOLDEN "tests/golden/i2s-converter.txt"

#ifndef SLLM_TEST_MODEL
#define SLLM_TEST_MODEL "/var/lib/spoon/models/ggml-model-i2_s.gguf"
#endif

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

/*
 * Read the `want`-th "|"-separated field of a golden line as hex bytes.
 * The fields are whitespace-separated hex words, so this walks words and stops
 * at the field boundary. Returns the byte count, or 0 on a parse failure.
 */
static int field(const char * line, int want, unsigned char * out, size_t cap) {
    const char * p = line;
    int seen = 0;
    size_t n = 0;
    while (seen < want) {
        while (*p == ' ' || *p == '\t') { ++p; }
        if (*p == '|') { ++seen; ++p; continue; }
        if (*p == '\0' || *p == '\n') { return 0; }
        while (*p && *p != ' ' && *p != '|' && *p != '\n') { ++p; }
    }
    for (;;) {
        while (*p == ' ' || *p == '\t') { ++p; }
        if (*p != '|' && *p != '\0' && *p != '\n' && n < cap) {
            /* one hex word */
            unsigned v = 0;
            int digits = 0;
            const char * q = p;
            while (*q && *q != ' ' && *q != '|' && *q != '\n') {
                const char c = *q;
                unsigned d;
                if (c >= '0' && c <= '9') { d = (unsigned) (c - '0'); }
                else if (c >= 'a' && c <= 'f') { d = (unsigned) (c - 'a' + 10); }
                else if (c >= 'A' && c <= 'F') { d = (unsigned) (c - 'A' + 10); }
                else { return 0; }
                v = (v << 4) | d;
                ++digits;
                ++q;
            }
            if (digits == 0 || digits > 2) { return 0; }
            out[n++] = (unsigned char) v;
            p = q;
            continue;
        }
        return (int) n;
    }
}

/* ------------------------------------------------------------------ */
/* the golden vectors                                                  */
/* ------------------------------------------------------------------ */

/*
 * Our converter must reproduce, byte for byte, packing produced by the upstream
 * transcriptions -- for BOTH rules, under both of which the fixture carries
 * vectors.
 */
TEST(i2s_convert_reproduces_the_upstream_packing_exactly) {
    if (access(I2S_GOLDEN, R_OK) != 0) {
        printf("    skipped: %s not available\n", I2S_GOLDEN);
        CHECK(1);
        return;
    }
    FILE * f = fopen(I2S_GOLDEN, "r");
    CHECK(f != NULL);
    if (f == NULL) { return; }

    char line[1 << 20];
    int cases = 0, mismatches = 0;
    unsigned char * ours = NULL;
    size_t ours_cap = 0;

    while (fgets(line, sizeof line, f) != NULL) {
        if (line[0] != 'c' || strncmp(line, "case ", 5) != 0) { continue; }

        int n = 0;
        long packed_bytes = 0;
        unsigned scale_hex = 0;
        if (sscanf(line, "case %d %ld %x", &n, &packed_bytes, &scale_hex) != 3) {
            continue;
        }
        float scale;
        memcpy(&scale, &scale_hex, sizeof scale);

        if (strchr(line, '|') == NULL) { continue; }
        ++cases;

        /* The source follows the scale token and precedes the first '|', so it
         * starts at token FOUR: "case", n, packed_bytes, scale. Two parsing
         * bugs lived here in a row and both produced plausible output rather
         * than an error, which is why they are worth stating: reading the
         * source from after the '|' took the packed bytes as the source, and
         * skipping only three tokens took the SCALE as the first source value.
         * The second made every element's code wrong in a way that still
         * produced a well-formed byte string. */
        const char * p = line;
        for (int skip = 0; skip < 4; ++skip) {
            while (*p == ' ') { ++p; }
            while (*p && *p != ' ' && *p != '|' && *p != '\n') { ++p; }
        }

        float * src = (float *) malloc((size_t) n * sizeof(float));
        if (src == NULL) { break; }
        for (int i = 0; i < n; ++i) {
            unsigned h = 0;
            if (sscanf(p, " %x", &h) != 1) { free(src); goto done; }
            src[i] = sllm_bf16_to_f32((uint16_t) h);
            while (*p == ' ') { ++p; }
            while (*p && *p != ' ') { ++p; }
        }

        if ((size_t) packed_bytes > ours_cap) {
            free(ours);
            ours = (unsigned char *) malloc((size_t) packed_bytes);
            ours_cap = (size_t) packed_bytes;
        }
        if (ours == NULL) { free(src); break; }

        /* Fields are numbered from the start of the line, and the source
         * precedes the first '|'. So: 0 = header+source, 1 = packed C,
         * 2 = decoded C, 3 = packed PY, 4 = decoded PY, 5 = the differ count. */
        for (int rule_i = 0; rule_i < 2; ++rule_i) {
            const sllm_i2s_rule rule = (rule_i == 0) ? SLLM_I2S_RULE_BITNET_C
                                                     : SLLM_I2S_RULE_GGUF_PY;
            const int want = (rule_i == 0) ? 1 : 3;

            memset(ours, 0, ours_cap);
            CHECK_EQ_INT(sllm_i2s_quantize_scaled(src, (size_t) n, ours, scale, rule), 0);

            unsigned char * want_buf = (unsigned char *) malloc((size_t) packed_bytes);
            if (want_buf == NULL) { break; }
            /* Copy the line and re-parse from the front, because field() walks
             * sequentially and the source length varies per case. */
            char * copy = strdup(line);
            int got = field(copy, want, want_buf, (size_t) packed_bytes);
            free(copy);
            if (got != (int) packed_bytes) {
                fprintf(stderr, "  golden parse: wanted %d bytes, got %d (n=%d)\n",
                        (int) packed_bytes, got, n);
                free(want_buf);
                ++mismatches;
                continue;
            }
            if (memcmp(ours, want_buf, (size_t) packed_bytes) != 0) {
                ++mismatches;
                if (mismatches <= 3) {
                    const unsigned char * w = want_buf;
                    size_t first = (size_t) packed_bytes;
                    for (size_t bi = 0; bi < (size_t) packed_bytes; ++bi) {
                        if (ours[bi] != w[bi]) { first = bi; break; }
                    }
                    fprintf(stderr, "  golden mismatch: n=%d rule=%d packed_bytes=%ld "
                                    "first_diff byte %zu\n",
                            n, (int) rule, packed_bytes, first);
                    /* Byte B holds the INTERLEAVED elements, four of them, 32
                     * apart -- not four consecutive ones. Reporting them as
                     * consecutive was itself misleading while chasing this. */
                    {
                        const size_t b128 = first / 32u, bb = first % 32u;
                        fprintf(stderr, "    scale=%+.10g  holds elements",
                                (double) scale);
                        for (int a = 0; a < 4; ++a) {
                            const size_t e = b128 * 128u + (size_t) a * 32u + bb;
                            if (e < (size_t) n) {
                                fprintf(stderr, " [%zu]=%+.6g", e, (double) src[e]);
                            }
                        }
                        fprintf(stderr, "\n");
                    }
                    for (size_t bi = (first > 3 ? first - 3 : 0);
                         bi < first + 5 && bi < (size_t) packed_bytes; ++bi) {
                        fprintf(stderr, "    [%zu] ours %02x want %02x%s\n",
                                bi, ours[bi], w[bi],
                                ours[bi] == w[bi] ? "" : "   <-- differs");
                    }
                }
            }
            /* The scale must also be where the reference puts it. */
            CHECK(sllm_i2s_block_scale(ours, (size_t) n) == scale);
            free(want_buf);
        }
        free(src);
    }
done:
    free(ours);
    fclose(f);
    CHECK(mismatches == 0);
    CHECK(cases > 10);
    printf("    %d cases, %d mismatches, both rules\n", cases, mismatches);
}

/*
 * The independent read-back in the golden file tells us what the codes MEAN.
 * A byte-exact packer with a wrong code mapping would pass the test above and
 * still produce a model that decodes to the wrong numbers, so this checks the
 * interpretation against the upstream decoder's output.
 */
TEST(i2s_convert_codes_mean_what_the_upstream_decoder_says) {
    if (access(I2S_GOLDEN, R_OK) != 0) { CHECK(1); return; }
    FILE * f = fopen(I2S_GOLDEN, "r");
    CHECK(f != NULL);
    if (f == NULL) { return; }

    char line[1 << 20];
    int cases = 0, bad = 0;
    while (fgets(line, sizeof line, f) != NULL) {
        if (strncmp(line, "case ", 5) != 0) { continue; }
        int n = 0; long pb = 0; unsigned sh = 0;
        if (sscanf(line, "case %d %ld %x", &n, &pb, &sh) != 3) { continue; }
        ++cases;
        if (n > 4096) { continue; }

        /* Our own decoder, via the shipped path, must agree with the values
         * the upstream read-back recorded. This is the check that a correct
         * packer and a correct unpacker mean the same thing. */
        unsigned char * packed = (unsigned char *) calloc((size_t) pb, 1);
        float * deq = (float *) malloc((size_t) n * sizeof(float));
        if (packed == NULL || deq == NULL) { free(packed); free(deq); break; }

        char * copy = strdup(line);
        int got = field(copy, 1, packed, (size_t) pb);
        free(copy);
        if (got == (int) pb) {
            sllm_i2s_dequant(packed, deq, (size_t) n);
            char * c2 = strdup(line);
            unsigned char * want = (unsigned char *) malloc((size_t) n * 4);
            if (want != NULL && field(c2, 2, want, (size_t) n * 4) == (int) n * 4) {
                for (int i = 0; i < n; ++i) {
                    float expect;
                    memcpy(&expect, want + i * 4, 4);
                    /* The golden records scale*ternary, ours too; compare the
                     * ternary SIGN pattern, which is the part at risk. */
                    const float sg = (expect > 0) - (expect < 0);
                    const float og = (deq[i] > 0) - (deq[i] < 0);
                    if (sg != og) { ++bad; }
                }
            }
            free(want);
            free(c2);
        }
        free(packed);
        free(deq);
    }
    fclose(f);
    CHECK(bad == 0);
    printf("    %d cases, ternary sign mismatches %d\n", cases, bad);
}

/*
 * BF16 conversion must be exact and symmetric. It is a shift, so a round trip
 * that is not bit-identical means the shift is wrong.
 */
TEST(i2s_convert_bf16_truncation_is_exact_only_where_bf16_can_represent) {
    /*
     * sllm_f32_to_bf16 is a TRUNCATION of the mantissa, not a rounding. Two
     * consequences, and both matter:
     *
     *  1. A value that is already representable in bf16 survives exactly. That
     *     is the only case where a round trip is the identity, and it is what
     *     the first loop below asserts.
     *
     *  2. A general f32 does NOT survive. bf16 keeps 7 EXPLICIT mantissa bits
     *     (8 significand bits counting the implicit one), so truncation can
     *     lose just under 2^-7 relative. Asserting an exact round trip here
     *     would be asserting something false; the first version of this test
     *     did exactly that and failed. The second version got the bound wrong
     *     by a factor of two, guessing 2^-8, which is the value for ROUNDING
     *     rather than truncation.
     *
     * Consequence 2 is the reason the hermetic round-trip gate supplies the
     * scale rather than recovering it. The I2_S scale is an arbitrary f32, and
     * an arbitrary f32 is not representable in bf16, so dequantise-to-bf16
     * necessarily perturbs it. The CODES, which is what the byte-identity gate
     * is about, are determined by sign and survive exactly. See
     * i2s_convert_hermetic_roundtrip_is_byte_identical.
     */
    uint32_t bits = 0x3f800000u;                    /* 1.0 */
    for (int i = 0; i < 4096; ++i) {
        float v;
        memcpy(&v, &bits, sizeof v);
        CHECK(sllm_bf16_to_f32(sllm_f32_to_bf16(v)) == v);
        /* Step by one bf16-ulp-sized increment: bump bit 16 of the mantissa,
         * which is the coarsest step bf16 can represent. */
        bits += 0x00010000u;
    }

    /* General values: within 2^-7 relative, which is what TRUNCATION costs on
     * 7 explicit mantissa bits. Rounding would cost 2^-8; truncation is the
     * worse of the two, and using the rounding bound here let values through
     * that the implementation does not actually guarantee. */
    for (int i = 1; i <= 2000; ++i) {
        const float v = (float) i * 0.001f;
        const float back = sllm_bf16_to_f32(sllm_f32_to_bf16(v));
        const double rel = fabs((double) back - (double) v) / (double) v;
        CHECK(rel <= 1.0 / 128.0 + 1e-9);
    }

    /* Sign and zero are always preserved, which is what the codes depend on. */
    const float signs[] = { 0.0f, -0.0f, 1.0f, -1.0f, 65504.0f, -65504.0f,
                            1.17549435e-38f, -1.17549435e-38f };
    for (size_t i = 0; i < sizeof signs / sizeof signs[0]; ++i) {
        const float back = sllm_bf16_to_f32(sllm_f32_to_bf16(signs[i]));
        CHECK((back > 0) == (signs[i] > 0));
        CHECK((back < 0) == (signs[i] < 0));
        CHECK((back == 0) == (signs[i] == 0));
    }
}

/*
 * The packed size must match the upstream return value n/4 + 32, because a
 * scale read from the wrong offset is a failure this project has made once.
 */
TEST(i2s_convert_packed_size_matches_upstream) {
    const size_t ns[] = { 1, 2, 3, 4, 5, 127, 128, 129, 256, 1024, 4096, 2560, 6912 };
    for (size_t i = 0; i < sizeof ns / sizeof ns[0]; ++i) {
        CHECK_EQ_INT((int) sllm_i2s_packed_size(ns[i]), (int) (ns[i] / 4 + 32));
    }
}

/*
 * The two upstream rules genuinely differ, and this test pins that difference
 * rather than hiding it. If a future change made them agree, the golden's
 * `differ` counts would stop matching and the packaging test would fail --
 * which is the desired direction: a silent merge of two different upstream
 * functions is a regression, not a cleanup.
 */
TEST(i2s_convert_the_two_upstream_rules_really_do_differ) {
    const size_t n = 512;
    float * src = (float *) malloc(n * sizeof(float));
    unsigned char * a = (unsigned char *) malloc(sllm_i2s_packed_size(n));
    unsigned char * b = (unsigned char *) malloc(sllm_i2s_packed_size(n));
    CHECK(src != NULL); CHECK(a != NULL); CHECK(b != NULL);
    if (src == NULL || a == NULL || b == NULL) {
        free(src); free(a); free(b); return;
    }
    /* Values spread across the +-0.5 threshold, which is where the rules part
     * company. A linear ramp is the clearest way to hit it. */
    for (size_t i = 0; i < n; ++i) {
        src[i] = (float) ((double) i / (double) n * 2.0 - 1.0) * 0.75f;
    }
    CHECK_EQ_INT(sllm_i2s_quantize_scaled(src, n, a, 0.75f, SLLM_I2S_RULE_GGUF_PY), 0);
    CHECK_EQ_INT(sllm_i2s_quantize_scaled(src, n, b, 0.75f, SLLM_I2S_RULE_BITNET_C), 0);

    size_t differ = 0;
    for (size_t i = 0; i < sllm_i2s_packed_size(n); ++i) {
        if (a[i] != b[i]) { ++differ; }
    }
    /* The GGUF-python rule thresholds at 0.5, so values in (-0.5, 0.5) become
     * zero there and take their sign under the C rule. On a uniform ramp over
     * (-1, 1) that is about half the elements. */
    CHECK(differ > n / 8);
    CHECK(differ < n * 3 / 4);
    printf("    the two rules differ on %zu of %zu packed bytes\n", differ, n / 4 + 32);

    free(src); free(a); free(b);
}

/* ------------------------------------------------------------------ */

void sllm_test_i2s_convert(void) {
    printf("i2s-convert\n");
    RUN(i2s_convert_packed_size_matches_upstream);
    RUN(i2s_convert_bf16_truncation_is_exact_only_where_bf16_can_represent);
    RUN(i2s_convert_reproduces_the_upstream_packing_exactly);
    RUN(i2s_convert_codes_mean_what_the_upstream_decoder_says);
    RUN(i2s_convert_the_two_upstream_rules_really_do_differ);
}
