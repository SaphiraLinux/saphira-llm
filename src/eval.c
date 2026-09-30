/*
 * eval.c — model-quality measurement primitives.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Separate from src/eval_main.c on purpose. The arithmetic lives in the
 * library so the test binary can link it and assert on it directly; the
 * executable is a thin driver that tokenises a corpus, feeds it, and reduces.
 * A measurement whose only copy of its own definition sits inside the program
 * that prints the number cannot be checked by anything.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <saphira_llm/eval.h>

#include <math.h>

/*
 * The reference's own form, kept so the difference can be measured rather than
 * argued about. perplexity.cpp:60 computes a float exp accumulated into a
 * double, then takes log of it.
 */
double sllm_eval_nll_softmax_form(const float * logits, int32_t n_vocab, int32_t tok) {
    float max_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    double sum_exp = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum_exp += (double) expf(logits[i] - max_logit);
    }
    return -((double) logits[tok] - max_logit - log(sum_exp));
}

/*
 * The form this tool actually reports: log-sum-exp in double, never forming
 * the probability. Two reasons, both numerical rather than stylistic.
 *
 * softmax-then-log computes p = exp(L[t]-m)/sum, then takes -log(p). For a
 * token the model is certain about, p can underflow to zero and -log(0) is
 * infinity, so a single confident token poisons the whole sum. log-sum-exp
 * never leaves the log domain, so the same token yields a large finite loss
 * and a long document cannot be ruined by one of them.
 *
 * The vocabulary here is 128256, so the reduction is long enough that
 * accumulating it in float would itself lose several bits.
 */
double sllm_eval_token_nll(const float * logits, int32_t n_vocab, int32_t tok) {
    float max_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    double sum_exp = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) {
        sum_exp += exp((double) logits[i] - (double) max_logit);
    }
    /* log(sum) + max - L[tok], which is -log_softmax(L)[tok] with the
     * cancellation done before it can bite. */
    return log(sum_exp) + (double) max_logit - (double) logits[tok];
}

int32_t sllm_eval_argmax(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

