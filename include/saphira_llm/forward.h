/*
 * BitNet b1.58 forward pass.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The graph, the op contracts and the two places the reference is easy to
 * misread are all written down in docs/PHASE4-CONTRACT.md, with file and line
 * in the pinned tree. This file is the implementation of that document.
 */

#ifndef SAPHIRA_LLM_FORWARD_H
#define SAPHIRA_LLM_FORWARD_H

#include "saphira_llm/gguf.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct sllm_model sllm_model;
typedef struct sllm_ctx   sllm_ctx;

/*
 * Load a bitnet-b1.58 model. The tensor payloads are NOT copied: they stay
 * mapped in the sllm_gguf, so the sllm_model borrows from it and the gguf must
 * outlive the model. Frees with sllm_model_free, which does not touch the gguf.
 */
sllm_status sllm_model_load(const sllm_gguf * gguf, sllm_model ** out);
void        sllm_model_free(sllm_model * m);

int32_t sllm_model_n_vocab(const sllm_model * m);
int32_t sllm_model_n_embd(const sllm_model * m);

/*
 * A context holds the KV cache and the scratch buffers, so several can share one
 * model. n_ctx is the cache depth; the model's trained context is 4096 but the
 * frozen prompts are a few tokens, and allocating the full cache for those is
 * 600 MB of memory for nothing.
 */
sllm_status sllm_ctx_new(const sllm_model * m, int32_t n_ctx, sllm_ctx ** out);
void        sllm_ctx_free(sllm_ctx * c);

/* Drop the KV cache, keeping the buffers. */
void sllm_ctx_reset(sllm_ctx * c);

/*
 * Run one token at absolute position `pos` (0-based) and write n_vocab logits.
 * The caller supplies the logits buffer; it must be at least n_vocab floats.
 */
sllm_status sllm_forward(const sllm_model * m, sllm_ctx * c,
                         int32_t token, int32_t pos, float * logits);

/*
 * Run `n` tokens from position 0, writing n_vocab logits for each. This is the
 * prefill the golden-logit fixture was captured from: the reference dumps a
 * logit row per prompt position, so parity is per position, not just the last.
 */
sllm_status sllm_forward_prefill(const sllm_model * m, sllm_ctx * c,
                                 const int32_t * tokens, int32_t n,
                                 float * logits_out);

/*
 * Greedy generation at temperature 0: consume `n_prompt` tokens, then emit
 * `n_new` further tokens, always taking the argmax of the current position's
 * logits. This is the loop the tg128 benchmark measures and the loop the
 * token-parity gate is stated over: token-identical output means every emitted
 * id matches the reference's, so the whole continuation, not just the first
 * token, is covered.
 *
 * `out` receives n_new ids. The context's position advances, so a caller can
 * resume a conversation by seeding with the transcript.
 */
sllm_status sllm_generate_greedy(const sllm_model * m, sllm_ctx * c,
                                 const int32_t * prompt, int32_t n_prompt,
                                 int32_t n_new, int32_t * out);

#endif /* SAPHIRA_LLM_FORWARD_H */
