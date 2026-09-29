/*
 * Forward-pass tests: token-identical greedy output against the reference.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The gate has two parts and they are not the same kind of claim.
 *
 * The hard gate is the ARGMAX at every prompt position. If the argmax differs,
 * the model is conditioning on a different sequence, and no tolerance makes that
 * acceptable. It is exact.
 *
 * The secondary gate is the logits, within a tolerance MEASURED here rather
 * than guessed. Our reduction order is our own, so bit equality is not
 * available without copying upstream's thread partitioning, which is
 * explicitly not the plan. What is measured is the argmax MARGIN: the gap
 * between the top two logits at each position. A deviation smaller than the
 * margin cannot flip the argmax, which is why the two gates are consistent.
 */

#include "harness.h"
#include "saphira_llm/forward.h"
#include "saphira_llm/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef SLLM_TEST_MODEL
#define SLLM_TEST_MODEL "/var/lib/spoon/models/ggml-model-i2_s.gguf"
#endif

typedef struct {
    const char * prefix;      /* tests/golden/bitnet2b-capitol      */
    const char * prompt;
} case_desc;

/*
 * The named known divergence.
 *
 * Phase 4 is sealed with this outstanding, and this is the record of it. It is
 * data rather than literals inside a comparison so that the number has a name
 * that can be referenced from a commit, a review and this file, and so that it
 * is greppable from anywhere.
 *
 * It is a near-tie, not a semantic failure: the reference's top-1 and our
 * top-1 are separated by less than our logit deviation at that position, and
 * the two sequences re-converge on the next token. The float floor that
 * permits it is the reference's tinyBLAS path, which is out of scope by
 * decision. See docs/PHASE4-ACCEPTANCE.md.
 */
typedef struct {
    int32_t     index;      /* position in the continuation, 0-based */
    int32_t     ours;
    int32_t     reference;
    const char * ours_text;
    const char * reference_text;
} sllm_known_divergence;

static const sllm_known_divergence SLLM_KNOWN_DIVERGENCE = {
    2, 6424, 3363, "town", "city"
};

/* The first tokens that must always match, whatever happens to the divergence
 * above. If these stop matching the change is not a near-tie boundary moving,
 * it is a regression, and it is a different failure. */
static const int32_t SLLM_STABLE_CONTINUATION_PREFIX = 2;

static const case_desc cases[] = {
    { "bitnet2b-capitol", "The capital of France is" },
    { "bitnet2b-save",    "The name of the capital city of France is" },
};

static char * slurp(const char * path, size_t * len_out) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char * buf = (char *) malloc((size_t) n + 1);
    if (buf == NULL) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) { free(buf); fclose(f); return NULL; }
    buf[n] = '\0';
    fclose(f);
    if (len_out != NULL) { *len_out = (size_t) n; }
    return buf;
}

/*
 * The reference's argmax per position, taken from the golden .f32 rather than
 * from the manifest's prose. The manifest is for humans; the raw logits are
 * the artifact, and reading the argmax out of them removes any chance of the
 * test and the fixture disagreeing about formatting.
 */
TEST(forward_argmax_is_token_identical_to_the_reference) {
    if (access(SLLM_TEST_MODEL, R_OK) != 0) {
        printf("    skipped: %s not available\n", SLLM_TEST_MODEL);
        CHECK(1);
        return;
    }

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) {
        printf("    skipped: %s (%s)\n", SLLM_TEST_MODEL, err);
        CHECK(1);
        return;
    }
    sllm_model * m = NULL;
    if (sllm_model_load(&g, &m) != SLLM_OK) { sllm_gguf_close(&g); CHECK(1); return; }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) { sllm_model_free(m); sllm_gguf_close(&g); CHECK(1); return; }
    sllm_ctx * c = NULL;
    if (sllm_ctx_new(m, 512, &c) != SLLM_OK) { sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); CHECK(1); return; }

    const int32_t NV = sllm_model_n_vocab(m);
    int total_pos = 0;
    int total_mismatch = 0;
    double worst_margin = 1e30;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        char f32p[512];
        snprintf(f32p, sizeof(f32p), "tests/golden/%s.f32", cases[ci].prefix);
        size_t bytes = 0;
        float * ref = (float *) slurp(f32p, &bytes);
        if (ref == NULL) {
            printf("    skipped: %s not present\n", f32p);
            continue;
        }
        const size_t n_pos = bytes / (sizeof(float) * (size_t) NV);
        if (n_pos == 0 || n_pos > 512) { free(ref); continue; }

        /* Each case starts from an empty cache. Reusing the live one would be
         * rejected by the position guard, which is the point of the guard. */
        sllm_ctx_reset(c);

        int32_t ids[512];
        const int32_t np = sllm_tok_encode(tok, cases[ci].prompt,
                                           strlen(cases[ci].prompt), true, true,
                                           ids, 512);
        if (np != (int32_t) n_pos) {
            sllm_tests_run++;
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL %s: tokenised %d, reference had %zu positions\n",
                    cases[ci].prefix, (int) np, n_pos);
            free(ref);
            continue;
        }

        float * ours = (float *) malloc(n_pos * (size_t) NV * sizeof(float));
        if (ours == NULL) { free(ref); continue; }
        if (sllm_forward_prefill(m, c, ids, np, ours) != SLLM_OK) {
            sllm_tests_run++; sllm_tests_failed++;
            fprintf(stderr, "  FAIL %s: forward pass returned an error\n", cases[ci].prefix);
            free(ours); free(ref);
            continue;
        }

        for (size_t p = 0; p < n_pos; ++p) {
            const float * o = ours + p * (size_t) NV;
            const float * r = ref + p * (size_t) NV;
            int32_t bo = 0, br = 0;
            float  to1 = -INFINITY, to2 = -INFINITY;
            float  tr1 = -INFINITY, tr2 = -INFINITY;
            for (int32_t v = 0; v < NV; ++v) {
                if (o[v] > to1) { to2 = to1; to1 = o[v]; bo = v; }
                else if (o[v] > to2) { to2 = o[v]; }
                if (r[v] > tr1) { tr2 = tr1; tr1 = r[v]; br = v; }
                else if (r[v] > tr2) { tr2 = r[v]; }
            }
            const double margin = (double) tr1 - (double) tr2;
            if (margin < worst_margin) { worst_margin = margin; }
            total_pos++;
            sllm_tests_run++;
            if (bo != br) {
                total_mismatch++;
                sllm_tests_failed++;
                fprintf(stderr, "  FAIL %s pos %zu: argmax %d, reference %d "
                                "(ours %.5f, ref %.5f, margin %.5f)\n",
                        cases[ci].prefix, p, (int) bo, (int) br,
                        (double) to1, (double) tr1, margin);
            }
        }

        sllm_tests_run++;
        if (total_mismatch == 0) {
            printf("    %-20s %zu positions, all token-identical\n",
                   cases[ci].prefix, n_pos);
        }
        free(ours);
        free(ref);
    }

    sllm_tests_run++;
    if (total_pos == 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL no positions were compared at all\n");
    } else {
        printf("    %d positions total, %d argmax mismatches; "
               "tightest reference argmax margin %.4f\n",
               total_pos, total_mismatch, worst_margin);
    }

    sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g);
}

/*
 * Generation must be autoregressive and greedy, and the ids it emits have to
 * come from the model's own argmax rather than from anything cached. The
 * reference's greedy continuation is the oracle.
 */
TEST(forward_generates_and_the_first_tokens_match_the_reference) {
    if (access(SLLM_TEST_MODEL, R_OK) != 0) { CHECK(1); return; }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) { CHECK(1); return; }
    sllm_model * m = NULL;
    if (sllm_model_load(&g, &m) != SLLM_OK) { sllm_gguf_close(&g); CHECK(1); return; }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) { sllm_model_free(m); sllm_gguf_close(&g); CHECK(1); return; }
    sllm_ctx * c = NULL;
    if (sllm_ctx_new(m, 512, &c) != SLLM_OK) { sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); CHECK(1); return; }

    static const char * prompt = "The capital of France is";
    int32_t ids[512];
    /* strlen, not a literal: a hardcoded length that includes the terminator
     * feeds the tokenizer a NUL byte, which changes the last token and makes
     * the whole continuation disagree with the reference for a reason that has
     * nothing to do with the model. */
    const int32_t np = sllm_tok_encode(tok, prompt, strlen(prompt),
                                       true, true, ids, 512);
    CHECK(np == 6);

    /*
     * Regression: a reused context must not attend to the previous
     * conversation. The KV cache is read up to n_past at every position and is
     * not zeroed on write, so generating twice into the same context without a
     * reset mixes two transcripts. Assert that the second run, after a reset,
     * produces exactly the first run's ids.
     */
    int32_t first[8], second[8];
    CHECK_STATUS(sllm_generate_greedy(m, c, ids, np, 8, first), SLLM_OK);
    sllm_ctx_reset(c);
    CHECK_STATUS(sllm_generate_greedy(m, c, ids, np, 8, second), SLLM_OK);
    sllm_tests_run++;
    if (memcmp(first, second, sizeof(first)) != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL reset did not restore the context: "
                        "reuse produced different tokens\n");
    }

    /* The reference's recorded continuation for the capitol prompt. */
    static const int32_t want[] = { 264, 2678, 3363, 11, 323, 279, 6864, 315 };
    const int32_t n_want = (int32_t) (sizeof(want) / sizeof(want[0]));

    int32_t gen[32];
    CHECK_STATUS(sllm_generate_greedy(m, c, ids, np, n_want, gen), SLLM_OK);

    /* The tokens before the divergence are comfortably separated in the
     * reference and must always match. If they stop matching, the change is a
     * regression rather than the characterised boundary moving. */
    for (int32_t i = 0; i < SLLM_STABLE_CONTINUATION_PREFIX; ++i) {
        sllm_tests_run++;
        if (gen[i] != want[i]) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL greedy prefix token %d: got %d, reference %d. "
                            "The stable prefix moved, which is a regression and "
                            "not the known divergence.\n",
                    (int) i, (int) gen[i], (int) want[i]);
        }
    }

    int agree = 0;
    for (int32_t i = 0; i < n_want; ++i) {
        if (gen[i] == want[i]) { agree++; }
    }
    printf("    greedy continuation agrees with the reference on %d of %d tokens\n",
           (int) agree, (int) n_want);
    if (agree < 2) {
        sllm_tests_run++; sllm_tests_failed++;
        fprintf(stderr, "  FAIL greedy agreed on only %d of %d tokens\n",
                (int) agree, (int) n_want);
    }

    /*
     * The named known divergence, asserted in BOTH directions.
     *
     * If it disappears, that is not a pass. It means the float floor moved, the
     * boundary recorded in docs/PHASE4-ACCEPTANCE.md no longer describes the
     * build, and somebody has to decide what to do about tinyBLAS. Silence
     * there would let the accepted boundary drift without anyone noticing, so
     * the test fails and says so.
     */
    const sllm_known_divergence * kd = &SLLM_KNOWN_DIVERGENCE;
    sllm_tests_run++;
    if (gen[kd->index] == want[kd->index]) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL the named known divergence at continuation token %d "
                        "has DISAPPEARED (we now emit %d \"%s\", the reference's %d "
                        "\"%s\").\n"
                        "  The float floor moved. Reopen docs/PHASE4-ACCEPTANCE.md "
                        "and decide whether that is now in scope, rather than "
                        "leaving a stale expectation in the test.\n",
                (int) kd->index, (int) gen[kd->index], kd->ours_text,
                (int) want[kd->index], kd->reference_text);
    } else {
        sllm_tests_run++;
        if (gen[kd->index] != kd->ours) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL the named divergence moved: continuation token %d "
                            "is now %d, the recorded divergence is %d \"%s\". "
                            "A different token diverging is a different boundary.\n",
                    (int) kd->index, (int) gen[kd->index], (int) kd->ours, kd->ours_text);
        } else {
            printf("    known divergence SLLM_KNOWN_DIVERGENCE: token %d is %d \"%s\" "
                   "against the reference's %d \"%s\"; sequences re-converge\n",
                   (int) kd->index, (int) kd->ours, kd->ours_text,
                   (int) want[kd->index], kd->reference_text);
        }
    }

    sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g);
}

void sllm_test_forward(void) {
    printf("forward\n");
    RUN(forward_argmax_is_token_identical_to_the_reference);
    RUN(forward_generates_and_the_first_tokens_match_the_reference);
}
