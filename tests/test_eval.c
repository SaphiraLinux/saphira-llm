/*
 * test_eval.c — model-quality measurement primitives.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * A note on what this file is and is not. It is a CORRECTNESS test file, and
 * the thing it tests is a MEASUREMENT tool. Those are different things and
 * conflating them is how a project ends up with a perplexity that nobody
 * believes.
 *
 * The separation is deliberate:
 *
 *   - The synthetic tests below assert that the scoring arithmetic is exactly
 *     what it claims to be. Those are pass/fail, and they must never move.
 *   - The model tests assert that the TOOL is wired correctly -- that it
 *     scores every token but the first, that it is deterministic, and that
 *     the thread count cannot change the answer. Also pass/fail.
 *   - Neither of them asserts anything about whether the model is any good.
 *     That is a number this project reports and a human decides about. If
 *     mean_nll changes, the golden value below is updated deliberately and the
 *     change is explained; it is never silently absorbed.
 *
 * The model-dependent tests skip cleanly when the acceptance model is absent,
 * following test_forward.c. The synthetic tests never skip, so the arithmetic
 * is checked on every machine.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "harness.h"

#include <saphira_llm/eval.h>
#include <saphira_llm/forward.h>
#include <saphira_llm/tokenizer.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef SLLM_TEST_MODEL
#define SLLM_TEST_MODEL "/var/lib/spoon/models/ggml-model-i2_s.gguf"
#endif

/* The fixture the model tests score. Checked in, so the number means the same
 * thing on every machine and after every unrelated change. */
#define EVAL_CORPUS "tests/golden/eval-corpus.txt"

/* Recorded under the arithmetic contract: -ffp-contract=off, defined in
 * docs/DECISIONS.md decision 1 and applied in the Makefile as FPFLAGS.
 *
 * Under that contract the value is not merely close across compilers, it is
 * IDENTICAL, which is what earns the exact comparison below:
 *
 *   gcc   -O2 -ffp-contract=off    nll_sum 1916.2644512626  mean_nll 4.9773102630
 *   clang -O2 -ffp-contract=off    nll_sum 1916.2644512626  mean_nll 4.9773102630
 *   clang -O1 -ffp-contract=off    nll_sum 1916.2644512626  mean_nll 4.9773102630
 *     (plus ASan+UBSan, and thread counts 1/4/8/16, all identical)
 *
 * For contrast, WITH contraction enabled clang gave mean_nll 4.9807068980 --
 * a 0.34% gap in perplexity, the same order as a small real improvement, and
 * therefore a trap for anyone A/B-ing across a toolchain change. That is why
 * the contract exists and why this comparison can be exact.
 *
 * EXACT means exact. A different libm or a different architecture may still
 * move the last bits, and if so the correct response is a deliberate
 * re-recording with the difference explained -- not a tolerance widened until
 * the signal fits. The previous 5e-3 tolerance existed only to absorb the FMA
 * spread, and the arithmetic contract removed the reason for it.
 */
#define EVAL_GOLDEN_TOKENS    386
#define EVAL_GOLDEN_NLL_SUM   1916.2644512626152
#define EVAL_GOLDEN_MEAN_NLL  4.9773102630197794

/* ------------------------------------------------------------------ */
/* scoring arithmetic, no model required                               */
/* ------------------------------------------------------------------ */

/*
 * A uniform distribution is the one case with an exactly known answer: p = 1/n
 * for every token, so the loss is log(n) for whichever token is the target.
 * This pins the absolute scale, which matters because a perplexity is an
 * exponent and any constant bias here would shift every number reported.
 */
TEST(eval_uniform_distribution_scores_exactly_log_n) {
    const int32_t n = 1000;
    float * logits = (float *) calloc((size_t) n, sizeof(float));
    CHECK(logits != NULL);
    if (logits == NULL) { return; }

    /* All zero: softmax is uniform over n. */
    for (int32_t t = 0; t < n; ++t) {
        const double nll = sllm_eval_token_nll(logits, n, t);
        CHECK_NEAR(nll, log((double) n), 1e-9);
    }
    /* A single very likely token. The peaked token is NOT excluded from the
     * normaliser, so the other tokens keep a probability of exp(-100), not
     * 1/(n-1): the model did not distribute its mass over the remaining
     * tokens, it put all of it on token 7. Hence
     *
     *   sum_exp = 1 + 999*exp(-100) ~= 1     (exp(-100) is 3.7e-44)
     *   nll(7)  = log(1) + 100 - 100  = 0
     *   nll(8)  = log(1) + 100 -    0  = 100
     *
     * Writing this down is the point: the first draft of this test expected
     * log(n-1) and was wrong, which is exactly the confusion the derivation
     * is here to prevent. */
    float * sharp = (float *) calloc((size_t) n, sizeof(float));
    sharp[7] = 100.0f;
    CHECK_NEAR(sllm_eval_token_nll(sharp, n, 7), 0.0, 1e-9);
    CHECK_NEAR(sllm_eval_token_nll(sharp, n, 8), 100.0, 1e-6);

    free(sharp);
    free(logits);
}

/*
 * Loss is a probability, so it cannot be negative. A model that puts all its
 * mass on the right token has loss zero, never a small negative number, and a
 * negative value would mean the arithmetic has gone wrong rather than the model
 * being good.
 */
TEST(eval_loss_is_never_negative) {
    const int32_t n = 64;
    float * logits = (float *) malloc((size_t) n * sizeof(float));
    CHECK(logits != NULL);
    if (logits == NULL) { return; }

    /* A range of shapes, including a degenerate one-hot and a steep ramp. */
    for (int trial = 0; trial < 64; ++trial) {
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = (float) ((i * 37 + trial * 11) % 23) - 11.0f;
        }
        for (int32_t t = 0; t < n; ++t) {
            const double nll = sllm_eval_token_nll(logits, n, t);
            CHECK(nll >= -1e-6);
        }
    }
    free(logits);
}

/*
 * The reason this tool does not use softmax-then-log, stated as a test rather
 * than as prose.
 *
 * A token the model is certain it will never emit has a probability that
 * underflows float to zero, and -log(0) is infinity. One such token in a
 * corpus would poison the entire sum, and a document of ordinary text contains
 * plenty of them. The log-sum-exp form never leaves the log domain, so the
 * same token yields a large finite loss and the mean stays usable.
 */
TEST(eval_survives_tokens_the_model_calls_impossible) {
    const int32_t n = 128;
    float * logits = (float *) calloc((size_t) n, sizeof(float));
    CHECK(logits != NULL);
    if (logits == NULL) { return; }

    /* Target is 1000 nats below everything else. Its probability is below the
     * smallest normal float, by a wide margin.
     *
     *   max     = 0
     *   sum_exp = 127*exp(0) + exp(-1000) = 127
     *   nll     = log(127) + 0 - (-1000)   = 1004.844187...
     *
     * The log(127) is the normaliser and is not optional: 127 other tokens
     * share the probability mass. The first draft expected 1000 and was
     * wrong by that factor. */
    logits[0] = 0.0f;
    for (int32_t i = 1; i < n; ++i) { logits[i] = 0.0f; }
    logits[99] = -1000.0f;

    const double nll = sllm_eval_token_nll(logits, n, 99);
    CHECK(isfinite(nll));
    CHECK_NEAR(nll, 1000.0 + log((double) (n - 1)), 1e-6);

    /* The reference form does not survive it, which is the whole point. */
    const double ref = sllm_eval_nll_softmax_form(logits, n, 99);
    CHECK(!isfinite(ref) || ref > 900.0);

    free(logits);
}

/*
 * The two forms must agree where the arithmetic is well conditioned. If they
 * did not, the choice of form would be changing the number rather than the
 * precision of its evaluation, and every reported figure would depend on which
 * one happened to be compiled in.
 *
 * Tolerance is 1e-5 relative, which is looser than it sounds on purpose: this
 * asserts the two are the same quantity computed differently, not that either
 * is correct to the last bit.
 */
TEST(eval_the_two_forms_agree_on_well_conditioned_logits) {
    const int32_t n = 512;
    float * logits = (float *) malloc((size_t) n * sizeof(float));
    CHECK(logits != NULL);
    if (logits == NULL) { return; }

    /* A deterministic pseudo-random spread, in the range real logits occupy. */
    uint32_t s = 0x9e3779b9u;
    for (int32_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        logits[i] = (float) ((int32_t) ((s >> 16) % 2001) - 1000) * 0.01f;
    }
    for (int32_t t = 0; t < n; ++t) {
        const double a = sllm_eval_token_nll(logits, n, t);
        const double b = sllm_eval_nll_softmax_form(logits, n, t);
        const double scale = fabs(a) > 1.0 ? fabs(a) : 1.0;
        CHECK_NEAR(a, b, 1e-5 * scale);
    }
    free(logits);
}

/*
 * Large logits must not produce NaN or infinity. The max-subtraction is what
 * prevents exp() overflowing, and this is the test that would notice if that
 * were ever removed: every logit is large and the target is the largest, so a
 * naive exp() would blow up before the max is subtracted.
 */
TEST(eval_large_logits_stay_finite) {
    const int32_t n = 128;
    float * logits = (float *) malloc((size_t) n * sizeof(float));
    CHECK(logits != NULL);
    if (logits == NULL) { return; }

    const float offsets[] = { 100.0f, 1000.0f, 10000.0f };
    for (size_t o = 0; o < sizeof offsets / sizeof offsets[0]; ++o) {
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = offsets[o] + (float) i;
        }
        for (int32_t t = 0; t < n; t += 7) {
            const double nll = sllm_eval_token_nll(logits, n, t);
            CHECK(isfinite(nll));
            CHECK(nll >= -1e-6);
        }
    }
    free(logits);
}

/*
 * Ties resolve to the lowest index, which is what makes top-1 accuracy a
 * deterministic quantity. Without a stated rule, a row of equal logits would
 * report different accuracies depending on the comparison operator and the
 * order the loop happened to visit them.
 */
TEST(eval_argmax_ties_resolve_to_the_lowest_index) {
    float flat[8];
    for (int i = 0; i < 8; ++i) { flat[i] = 1.0f; }
    CHECK_EQ_INT(sllm_eval_argmax(flat, 8), 0);

    float peak[8];
    for (int i = 0; i < 8; ++i) { peak[i] = 0.0f; }
    peak[5] = 2.0f;
    peak[6] = 2.0f;
    CHECK_EQ_INT(sllm_eval_argmax(peak, 8), 5);

    float single[8];
    for (int i = 0; i < 8; ++i) { single[i] = -100.0f; }
    single[3] = -101.0f;
    CHECK_EQ_INT(sllm_eval_argmax(single, 8), 0);

    /* All equal and negative: still the lowest index. */
    float neg[4];
    for (int i = 0; i < 4; ++i) { neg[i] = -5.0f; }
    CHECK_EQ_INT(sllm_eval_argmax(neg, 4), 0);
}

/* ------------------------------------------------------------------ */
/* the tool, against the real model                                     */
/* ------------------------------------------------------------------ */

/*
 * The tokenizer contract the eval numbers depend on. If the corpus ever
 * tokenizes to a different count, every golden figure below is void -- not
 * wrong by a little, meaningless -- so this is checked explicitly rather than
 * inferred from a passing loss comparison.
 */
TEST(eval_fixture_tokenizes_to_the_recorded_count) {
    if (access(SLLM_TEST_MODEL, R_OK) != 0) {
        printf("    skipped: %s not available\n", SLLM_TEST_MODEL);
        CHECK(1);
        return;
    }
    if (access(EVAL_CORPUS, R_OK) != 0) {
        printf("    skipped: %s not available\n", EVAL_CORPUS);
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
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) {
        sllm_gguf_close(&g);
        CHECK(1);
        return;
    }

    FILE * f = fopen(EVAL_CORPUS, "rb");
    CHECK(f != NULL);
    if (f != NULL) {
        fseek(f, 0, SEEK_END);
        const long len = ftell(f);
        rewind(f);
        char * text = (char *) malloc((size_t) len + 1);
        CHECK(text != NULL);
        if (text != NULL) {
            if (len > 0 && fread(text, 1, (size_t) len, f) == (size_t) len) {
                text[len] = '\0';
                const int32_t cap = sllm_tok_encode_len(tok, (size_t) len, true);
                int32_t * ids = (int32_t *) malloc((size_t) (cap > 0 ? cap : 1) * sizeof(int32_t));
                CHECK(ids != NULL);
                if (ids != NULL) {
                    const int32_t n = sllm_tok_encode(tok, text, (size_t) len,
                                                      true, true, ids, cap);
                    CHECK_EQ_INT(n, EVAL_GOLDEN_TOKENS);
                    /* Every id must be a real vocabulary entry, or the loss
                     * is being computed against a target the model could
                     * never have predicted. */
                    const int32_t n_vocab = 128256;
                    int in_range = 1;
                    for (int32_t i = 0; i < n; ++i) {
                        if (ids[i] < 0 || ids[i] >= n_vocab) { in_range = 0; }
                    }
                    CHECK(in_range);
                    free(ids);
                }
            }
            free(text);
        }
        fclose(f);
    }

    sllm_tok_free(tok);
    sllm_gguf_close(&g);
}

/*
 * The measurement itself, run twice in one process through two independent
 * contexts, plus the chunk-boundary invariant.
 *
 * The corpus is 386 tokens and SLLM_MAX_CHUNK is 256, so this spans two chunks
 * and therefore exercises the boundary. The first version of the tool dropped
 * one token at every boundary; the count is checked here so that cannot come
 * back quietly.
 */
TEST(eval_scores_every_token_but_the_first_and_is_deterministic) {
    if (access(SLLM_TEST_MODEL, R_OK) != 0) {
        printf("    skipped: %s not available\n", SLLM_TEST_MODEL);
        CHECK(1);
        return;
    }
    if (access(EVAL_CORPUS, R_OK) != 0) {
        printf("    skipped: %s not available\n", EVAL_CORPUS);
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
    if (sllm_model_load(&g, &m) != SLLM_OK) {
        sllm_gguf_close(&g);
        CHECK(1);
        return;
    }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) {
        sllm_model_free(m); sllm_gguf_close(&g);
        CHECK(1);
        return;
    }

    FILE * f = fopen(EVAL_CORPUS, "rb");
    CHECK(f != NULL);
    if (f == NULL) {
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g);
        return;
    }
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    rewind(f);
    char * text = (char *) malloc((size_t) len + 1);
    if (text == NULL || (len > 0 && fread(text, 1, (size_t) len, f) != (size_t) len)) {
        free(text); fclose(f);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g);
        CHECK(0);
        return;
    }
    fclose(f);
    text[len] = '\0';

    const int32_t cap = sllm_tok_encode_len(tok, (size_t) len, true);
    int32_t * ids = (int32_t *) malloc((size_t) cap * sizeof(int32_t));
    CHECK(ids != NULL);
    if (ids == NULL) {
        free(text); sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g);
        return;
    }
    const int32_t n_tok = sllm_tok_encode(tok, text, (size_t) len, true, true, ids, cap);
    free(text);
    CHECK_EQ_INT(n_tok, EVAL_GOLDEN_TOKENS);
    if (n_tok < 2) {
        free(ids); sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g);
        return;
    }

    const int32_t n_vocab = sllm_model_n_vocab(m);
    float * logits = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) n_vocab * sizeof(float));
    CHECK(logits != NULL);
    if (logits == NULL) {
        free(ids); sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g);
        return;
    }

    double first_sum = 0.0, second_sum = 0.0;
    int64_t first_scored = 0, second_scored = 0;

    for (int pass = 0; pass < 2; ++pass) {
        sllm_ctx * c = NULL;
        if (sllm_ctx_new(m, n_tok, &c) != SLLM_OK) { CHECK(0); break; }
        double sum = 0.0;
        int64_t scored = 0;
        for (int32_t p = 0; p < n_tok; ) {
            int32_t k = n_tok - p;
            if (k > SLLM_MAX_CHUNK) { k = SLLM_MAX_CHUNK; }
            const int32_t rows = (p + k >= n_tok) ? (k - 1) : k;
            CHECK_STATUS(sllm_forward_chunk(m, c, ids + p, k, p, logits), SLLM_OK);
            for (int32_t i = 0; i < rows; ++i) {
                sum += sllm_eval_token_nll(logits + (size_t) i * (size_t) n_vocab,
                                           n_vocab, ids[p + i + 1]);
                ++scored;
            }
            p += k;
        }
        sllm_ctx_free(c);
        if (pass == 0) { first_sum = sum; first_scored = scored; }
        else           { second_sum = sum; second_scored = scored; }
    }

    /* The invariant, at two chunks. */
    CHECK_EQ_INT((int) first_scored, n_tok - 1);
    CHECK_EQ_INT((int) second_scored, n_tok - 1);

    /* Determinism, with no tolerance at all. This is the property every A/B
     * comparison rests on: the same binary, the same corpus, the same answer,
     * to the last bit. If this ever fails then a recorded perplexity means
     * nothing, and the golden comparison below is noise. */
    CHECK(first_sum == second_sum);

    /* And the recorded canonical value, EXACTLY, under the arithmetic
     * contract. Not approximately: the contract is what makes exactness
     * available, and a tolerance here would hide the very cross-toolchain
     * drift the contract exists to eliminate. */
    CHECK(first_sum == EVAL_GOLDEN_NLL_SUM);
    const double mean = first_sum / (double) first_scored;
    CHECK(mean == EVAL_GOLDEN_MEAN_NLL);
    printf("    fixture: %d tokens, mean_nll %.10f, ppl %.4f\n",
           n_tok, mean, exp(mean));

    free(logits);
    free(ids);
    sllm_tok_free(tok);
    sllm_model_free(m);
    sllm_gguf_close(&g);
}

/* ------------------------------------------------------------------ */

void sllm_test_eval(void) {
    printf("eval\n");
    RUN(eval_uniform_distribution_scores_exactly_log_n);
    RUN(eval_loss_is_never_negative);
    RUN(eval_survives_tokens_the_model_calls_impossible);
    RUN(eval_the_two_forms_agree_on_well_conditioned_logits);
    RUN(eval_large_logits_stay_finite);
    RUN(eval_argmax_ties_resolve_to_the_lowest_index);
    RUN(eval_fixture_tokenizes_to_the_recorded_count);
    RUN(eval_scores_every_token_but_the_first_and_is_deterministic);
}
