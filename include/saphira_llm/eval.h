/*
 * eval.h — model-quality measurement primitives.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * This is NOT a correctness gate. See the note at the top of src/eval_main.c.
 * Nothing here can fail a build; these functions report numbers.
 *
 * The scoring definitions are exposed rather than kept inside the executable
 * for one reason: the numerical choices below are claims, and claims need
 * tests. tests/test_eval.c uses them to compare the reported form against the
 * reference's own softmax-then-log form, so the difference between them is a
 * measured quantity instead of an assurance.
 */
#ifndef SAPHIRA_LLM_EVAL_H
#define SAPHIRA_LLM_EVAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Negative log likelihood of `tok` under one row of logits, in the form this
 * project reports: log-sum-exp in double, never forming a probability.
 *
 *   nll = log( sum_i exp(L[i] - max) ) + max - L[tok]
 *
 * `logits` is n_vocab f32 values as the forward pass produced them. `tok` must
 * be in [0, n_vocab).
 *
 * The returned value is in nats, and is non-negative up to floating-point
 * tolerance: a probability of 1 is log(1) = 0 loss, never negative loss.
 */
double sllm_eval_token_nll(const float * logits, int32_t n_vocab, int32_t tok);

/*
 * The reference's form, transcribed from the pinned upstream tree at
 * third_party/llama.cpp/tools/perplexity/perplexity.cpp:60 -- a float exp
 * accumulated into a double, then a log of the sum.
 *
 * Kept only so the two can be compared. It is not used to produce any reported
 * number, because softmax-then-log can underflow to -log(0) = infinity on a
 * token the model is confident it will never emit, and one such token would
 * poison an entire corpus sum.
 */
double sllm_eval_nll_softmax_form(const float * logits, int32_t n_vocab, int32_t tok);

/* Index of the largest logit. Ties resolve to the lowest index, which makes
 * top-1 accuracy a deterministic quantity rather than a platform-dependent
 * one. */
int32_t sllm_eval_argmax(const float * logits, int32_t n_vocab);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_EVAL_H */
