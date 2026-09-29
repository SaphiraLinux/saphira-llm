/*
 * test_i2s.c — BitNet I2_S exactness gates.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * These gates exist because the I2_S reader was wrong twice and the first
 * golden vector could not see either error: it built its packed bytes and read
 * them back with the same wrong assumption. A test that shares an assumption
 * with the code under test is not a test.
 *
 * So the fixtures in tests/golden/i2s-reference.txt are the REFERENCE's own
 * dequantize_row_i2_s output on real tensors from the real model, and the
 * coverage is an FNV-1a 64 hash over every element -- millions of values per
 * tensor, checked without storing them. The reference library is not needed to
 * run these; only to regenerate the fixture.
 */

#include "harness.h"

#include <saphira_llm/quant.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The model holding the tensors the fixture was captured from. Not needed to
 * run the suite; the fixture is a text file of recorded values. */
#ifndef SLLM_TEST_MODEL
#define SLLM_TEST_MODEL "/var/lib/spoon/models/ggml-model-i2_s.gguf"
#endif
#ifndef SLLM_TEST_GOLDEN
#define SLLM_TEST_GOLDEN "tests/golden/i2s-reference.txt"
#endif

static uint64_t fnv1a64(const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* ------------------------------------------------------------------ */
/* the hard gate: our reader against the reference's output            */
/* ------------------------------------------------------------------ */

TEST(i2s_reader_matches_the_reference_on_real_tensors) {
    if (access(SLLM_TEST_GOLDEN, R_OK) != 0) {
        printf("    skipped: %s not present\n", SLLM_TEST_GOLDEN);
        CHECK(1);
        return;
    }

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) {
        printf("    skipped: %s not available (%s)\n", SLLM_TEST_MODEL, err);
        CHECK(1);
        return;
    }

    FILE * f = fopen(SLLM_TEST_GOLDEN, "r");
    CHECK(f != NULL);
    if (f == NULL) { sllm_gguf_close(&g); return; }

    char line[65536];
    char want_tensor[256] = "";
    int  tensors_checked = 0;
    int  elements_checked = 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        char name[256];
        size_t elems;
        double want_scale;
        unsigned long long want_hash;

        if (sscanf(line, "tensor %255s", name) == 1) {
            snprintf(want_tensor, sizeof(want_tensor), "%s", name);
            continue;
        }
        if (sscanf(line, "gemv %255s", name) == 1) {
            /* A gemv block follows; this parser only handles tensor blocks. */
            want_tensor[0] = '\0';
            int skipping = 1;
            while (fgets(line, sizeof(line), f) != NULL) {
                if (line[0] == '\n' || strncmp(line, "gemv ", 5) == 0) { skipping = 0; break; }
            }
            (void) skipping;
            continue;
        }
        if (want_tensor[0] == '\0') {
            continue;
        }
        if (sscanf(line, "  elements %zu", &elems) == 1) {
            continue;
        }
        if (sscanf(line, "  scale %lf", &want_scale) == 1) {
            continue;
        }
        if (sscanf(line, "  fnv1a64 %llx", &want_hash) != 1) {
            continue;
        }

        const sllm_gguf_tensor * t = sllm_gguf_find_tensor(&g, want_tensor);
        sllm_tests_run++;
        if (t == NULL) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL golden names tensor '%s' which the model lacks\n", want_tensor);
            continue;
        }
        if (t->type != SLLM_TYPE_I2_S) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL '%s' is %s, expected I2_S\n", want_tensor,
                    sllm_gguf_type_name(t->type));
            continue;
        }

        /* The scale, at the offset the format dictates. */
        const float got_scale = sllm_i2s_scale(t->data, elems);
        sllm_tests_run++;
        if (got_scale != (float) want_scale) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL %s scale: got %.9g want %.9g\n",
                    want_tensor, (double) got_scale, want_scale);
        }

        /* Every element, dequantised, hashed. */
        float * got = malloc(elems * sizeof(float));
        if (got == NULL) {
            fprintf(stderr, "  FAIL out of memory for %zu elements\n", elems);
            sllm_tests_failed++;
            continue;
        }
        sllm_i2s_dequant(t->data, got, elems);
        const unsigned long long got_hash = (unsigned long long)
            fnv1a64(got, elems * sizeof(float));
        free(got);

        if (got_hash != want_hash) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL %s: %zu elements hash %016llx, reference says %016llx\n",
                    want_tensor, elems, got_hash, want_hash);
        } else {
            ++tensors_checked;
            elements_checked += (int) (elems / 1000000);
            printf("    %-28s %9zu elements  %016llx  ok\n",
                   want_tensor, elems, got_hash);
        }
        want_tensor[0] = '\0';
    }

    fclose(f);
    sllm_gguf_close(&g);

    sllm_tests_run++;
    if (tensors_checked < 4) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL only %d tensors were checked, expected 4\n", tensors_checked);
    } else {
        printf("    %d tensors, ~%d million elements, all matching the reference\n",
               tensors_checked, elements_checked);
    }
}

/* ------------------------------------------------------------------ */
/* the layout, stated independently of both the reader and the model    */
/* ------------------------------------------------------------------ */

TEST(i2s_field_order_is_top_down_within_each_byte) {
    enum { N = 1024 };
    uint8_t * codes = malloc(N);
    uint8_t * packed = malloc(N / 4 + 32);
    float * row = malloc(N * sizeof(float));
    CHECK(codes && packed && row);
    if (!codes || !packed || !row) { free(codes); free(packed); free(row); return; }

    /* A pattern where the field index and the byte index are distinguishable:
     * every element in field a gets the code a, so a reversal is obvious. */
    for (size_t j = 0; j < N; ++j) {
        codes[j] = (uint8_t) ((j / 32) % 3);
    }
    sllm_pack_i2s_like_converter(packed, codes, N);

    int bad = 0;
    for (size_t j = 0; j < N; ++j) {
        const uint8_t got = sllm_i2s_code(packed, j);
        if (got != codes[j]) {
            if (bad < 4) {
                fprintf(stderr, "  FAIL element %zu: got code %u want %u\n",
                        j, got, codes[j]);
            }
            ++bad;
        }
    }
    sllm_tests_run++;
    if (bad != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL i2s field order: %d of %d elements wrong\n", bad, N);
    }

    free(codes); free(packed); free(row);
}

TEST(i2s_row_and_dequant_agree_and_differ_only_by_sign_and_scale) {
    enum { N = 512 };
    uint8_t * codes = malloc(N);
    uint8_t * packed = malloc(N / 4 + 32);
    float * raw = malloc(N * sizeof(float));
    float * deq = malloc(N * sizeof(float));
    CHECK(codes && packed && raw && deq);
    if (!codes || !packed || !raw || !deq) {
        free(codes); free(packed); free(raw); free(deq); return;
    }

    for (size_t j = 0; j < N; ++j) { codes[j] = (uint8_t) (j % 3); }
    sllm_pack_i2s_like_converter(packed, codes, N);

    const float scale = 0.5f;
    memcpy(packed + N / 4, &scale, sizeof(scale));

    sllm_i2s_row(packed, raw, N);
    sllm_i2s_dequant(packed, deq, N);

    for (size_t j = 0; j < N; ++j) {
        /* raw must be the unsigned code, which is what dpbusd needs. */
        CHECK_SAME_BITS(raw[j], (float) codes[j]);
        /* dequant must be scale * (code - 1). */
        CHECK_SAME_BITS(deq[j], scale * ((float) codes[j] - 1.0f));
    }
    CHECK(sllm_i2s_scale(packed, N) == scale);

    free(codes); free(packed); free(raw); free(deq);
}

/* ------------------------------------------------------------------ */
/* the kernel gate: our GEMV against the reference's, on real weights  */
/* ------------------------------------------------------------------ */

/*
 * Reproduces the reference's fixed activation generator exactly, because the
 * fixture pins the quantised row our quantiser has to produce. If this drifts,
 * the kernel comparison silently stops comparing like with like.
 */
static void make_reference_activation(size_t n, int8_t * q, sllm_i2s_act * act) {
    float * x = malloc(n * sizeof(float));
    CHECK(x != NULL);
    if (x == NULL) { return; }
    uint32_t rng = 0xBEEF01u;
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        x[i] = (((float) ((rng >> 8) & 0xffff) / 32768.0f) - 1.0f) * 3.0f;
    }
    sllm_i2s_quant_act(x, n, q, act);
    free(x);
}

TEST(i2s_kernels_match_the_reference_on_real_weights) {
    if (access(SLLM_TEST_GOLDEN, R_OK) != 0) {
        printf("    skipped: %s not present\n", SLLM_TEST_GOLDEN);
        CHECK(1);
        return;
    }

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) {
        printf("    skipped: %s not available (%s)\n", SLLM_TEST_MODEL, err);
        CHECK(1);
        return;
    }

    /* Force the v3 path and the VNNI path explicitly, so both are gated
     * regardless of what the host would select by default. */
    sllm_isa_caps caps;
    memset(&caps, 0, sizeof(caps));
    sllm_isa_dispatch d_v3, d_vnni;
    memset(&d_v3, 0, sizeof(d_v3));
    d_v3.caps = caps;
    d_v3.selected = SLLM_ISA_V3;
    caps.avx_vnni = true;
    memset(&d_vnni, 0, sizeof(d_vnni));
    d_vnni.caps = caps;
    d_vnni.selected = SLLM_ISA_VNNI;

    FILE * f = fopen(SLLM_TEST_GOLDEN, "r");
    CHECK(f != NULL);
    if (f == NULL) { sllm_gguf_close(&g); return; }

    char line[65536];
    char want_tensor[256] = "";
    int  records = 0;

    /*
     * The record fields must live OUTSIDE the loop. Declared inside, they are
     * re-initialised on every line, so the raw hash read on one line is wiped
     * before the line that triggers the comparison reads it -- and the gate
     * then confidently compares against zero. That is exactly what happened.
     */
    size_t n = 0, rows = 0;
    double act_scale = 0, w_scale = 0;
    int act_sum = 0, agree = 0;
    unsigned long long want_hash = 0, want_epi = 0;
    char name[256];

    while (fgets(line, sizeof(line), f) != NULL) {
        if (sscanf(line, "gemv %255s", name) == 1) {
            snprintf(want_tensor, sizeof(want_tensor), "%s", name);
            continue;
        }
        if (want_tensor[0] == '\0') { continue; }
        if (sscanf(line, "  n %zu", &n) == 1) { continue; }
        if (sscanf(line, "  rows %zu", &rows) == 1) { continue; }
        if (sscanf(line, "  act_scale %lf", &act_scale) == 1) { continue; }
        if (sscanf(line, "  act_sum %d", &act_sum) == 1) { continue; }
        if (sscanf(line, "  w_scale %lf", &w_scale) == 1) { continue; }
        if (sscanf(line, "  gemv_gemm_agree %d", &agree) == 1) { continue; }
        if (sscanf(line, "  fnv1a64 %llx", &want_hash) == 1) { continue; }

        /*
         * Run the comparison on the LAST field of the record, not the first.
         * The record carries two hashes, the raw dot products and the
         * epilogue, and comparing on the first one means the second has not
         * been read yet. That produced a confident "reference says 0" failure
         * for a hash that in fact matched exactly.
         */
        if (sscanf(line, "  epilogue_fnv1a64 %llx", &want_epi) != 1) { continue; }

        const sllm_gguf_tensor * t = sllm_gguf_find_tensor(&g, want_tensor);
        sllm_tests_run++;
        if (t == NULL) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL gemv fixture names unknown tensor '%s'\n", want_tensor);
            want_tensor[0] = '\0';
            want_hash = 0;
            continue;
        }

        size_t total = 1;
        for (uint32_t d = 0; d < t->n_dims; ++d) { total *= (size_t) t->ne[d]; }
        const size_t row_bytes = n / 4u;

        int8_t * q = malloc(n);
        int32_t * dots = malloc(rows * sizeof(int32_t));
        float * out = malloc(rows * sizeof(float));
        if (!q || !dots || !out) { sllm_tests_failed++; free(q); free(dots); free(out); return; }

        /* 1. The activation quantiser must reproduce the fixture's scale and
         *    sum exactly, or the kernel comparison below is meaningless. */
        sllm_i2s_act act;
        make_reference_activation(n, q, &act);
        sllm_tests_run++;
        if (act.scale != (float) act_scale || act.sum != act_sum) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL %s activation quant: got scale %.9g sum %d, "
                            "reference says %.9g sum %d\n", want_tensor,
                    (double) act.scale, act.sum, act_scale, act_sum);
        }

        /* 2. Both kernel paths, and the scalar oracle, against the reference's
         *    hash of its own GEMV output. */
        const char * label[3] = { "v3", "vnni", "scalar" };
        for (int path = 0; path < 3; ++path) {
            if (path == 0)      { sllm_i2s_select_isa(&d_v3); }
            else if (path == 1) { sllm_i2s_select_isa(&d_vnni); }
            else                { sllm_i2s_select_isa(NULL); }

            for (size_t r = 0; r < rows; ++r) {
                dots[r] = (path == 2)
                    ? sllm_i2s_dot_scalar((const uint8_t *) t->data + r * row_bytes, q, n)
                    : sllm_i2s_dot((const uint8_t *) t->data + r * row_bytes, q, n);
                out[r] = (float) dots[r];
            }
            const unsigned long long got = (unsigned long long) fnv1a64(out, rows * sizeof(float));
            sllm_tests_run++;
            if (got != want_hash) {
                sllm_tests_failed++;
                fprintf(stderr, "  FAIL %s [%s]: %zu rows hash %016llx, reference %016llx\n",
                        want_tensor, label[path], rows, got, want_hash);
            }
        }

        /* 3. The epilogue, applied to the raw dot products. */
        const float ws = sllm_i2s_scale(t->data, total);
        sllm_tests_run++;
        if (ws != (float) w_scale) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL %s w_scale: got %.9g want %.9g\n",
                    want_tensor, (double) ws, w_scale);
        }
        sllm_i2s_select_isa(&d_v3);
        sllm_i2s_gemv((const uint8_t *) t->data, rows, n, q, dots);
        for (size_t r = 0; r < rows; ++r) {
            out[r] = sllm_i2s_epilogue(dots[r], act.sum, act.scale, ws);
        }
        const unsigned long long got_epi = (unsigned long long) fnv1a64(out, rows * sizeof(float));
        sllm_tests_run++;
        if (got_epi != want_epi) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL %s epilogue: hash %016llx, reference %016llx\n",
                    want_tensor, got_epi, want_epi);
        } else {
            ++records;
            printf("    %-28s n=%-5zu rows=%-4zu %016llx  v3 vnni scalar epilogue all ok\n",
                   want_tensor, n, rows, want_hash);
        }

        free(q); free(dots); free(out);
        want_tensor[0] = '\0';
        want_hash = 0;
    }

    fclose(f);
    sllm_gguf_close(&g);

    sllm_tests_run++;
    if (records < 4) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL only %d gemv records checked, expected 4\n", records);
    }
}

TEST(i2s_epilogue_is_reproduced_verbatim) {
    /*
     * The -act_sum term is mathematically suspicious: it is only correct if
     * the ternary weights sum to zero over K. The reference applies it anyway,
     * so we apply it too, and this test pins the behaviour so a future
     * "improvement" cannot quietly change the numbers.
     */
    const int32_t dots[]   = { 0, 1, -1, 1000, -1000, 123456, -123456 };
    const int32_t sums[]   = { 0, 5, -5, 0, 0, 77, -77 };
    const float   scales[] = { 1.0f, 2.0f, 0.5f, 127.0f, 0.007874f, 42.0f, 3.5f };
    const float   w[]      = { 1.0f, -1.0f, 0.25f, 2.16316128f, 1.21885478f, 2.17397785f, 0.5f };

    for (size_t i = 0; i < sizeof(dots) / sizeof(dots[0]); ++i) {
        const float got = sllm_i2s_epilogue(dots[i], sums[i], scales[i], w[i]);
        /* The reference's forward divides once per column and then multiplies,
         * so that is the grouping asserted here. See sllm_i2s_epilogue. */
        const float post_scale = w[i] / scales[i];
        const float want = ((float) (dots[i] - sums[i])) * post_scale;
        CHECK_SAME_BITS(got, want);
    }

    /* The formula really does subtract the sum, rather than our reading of it
     * being a coincidence of these inputs. */
    CHECK(sllm_i2s_epilogue(1000, 100, 1.0f, 1.0f) == 900.0f);
    CHECK(sllm_i2s_epilogue(1000, 0, 1.0f, 1.0f) == 1000.0f);
}

TEST(i2s_activation_quantiser_matches_the_reference_rules) {
    /* An all-zero row: amax is not floored, so the scale is 0 and every value
     * quantises to 0. The reference does the same, division by zero in the
     * epilogue and all. Reproduced, not "fixed". */
    float z[256];
    int8_t q[256];
    sllm_i2s_act act;
    for (int i = 0; i < 256; ++i) { z[i] = 0.0f; }
    sllm_i2s_quant_act(z, 256, q, &act);
    CHECK(act.scale == 0.0f);
    CHECK_EQ_INT(act.sum, 0);
    for (int i = 0; i < 256; ++i) { CHECK_EQ_INT(q[i], 0); }

    /* A row whose maximum is 1: the scale is exactly 127 and values clamp. */
    float u[256];
    for (int i = 0; i < 256; ++i) { u[i] = (i % 3 == 0) ? 2.0f : -1.0f; }
    sllm_i2s_quant_act(u, 256, q, &act);
    CHECK(act.scale == 63.5f);
    for (int i = 0; i < 256; ++i) {
        const int want = (i % 3 == 0) ? 127 : -64;   /* roundf(2*63.5)=127, roundf(-63.5)=-64 */
        CHECK_EQ_INT(q[i], want);
    }
    int32_t sum = 0;
    for (int i = 0; i < 256; ++i) { sum += q[i]; }
    CHECK_EQ_INT(act.sum, sum);
}

void sllm_test_i2s(void) {
    printf("i2s\n");
    RUN(i2s_reader_matches_the_reference_on_real_tensors);
    RUN(i2s_kernels_match_the_reference_on_real_weights);
    RUN(i2s_epilogue_is_reproduced_verbatim);
    RUN(i2s_activation_quantiser_matches_the_reference_rules);
    RUN(i2s_field_order_is_top_down_within_each_byte);
    RUN(i2s_row_and_dequant_agree_and_differ_only_by_sign_and_scale);
}
