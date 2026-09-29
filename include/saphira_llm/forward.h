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
 * The logits of the most recent position, valid once anything has been
 * forwarded.
 *
 * This exists because a KV cache alone is not enough to resume. The cache holds
 * the keys and values; the next token needs the logit vector of the LAST
 * position, which is not in the cache and is not recoverable from it without
 * re-running the forward over that token. So a context that cannot report its
 * last logits can only be continued by recomputing, which defeats the point of
 * saving it.
 */
const float * sllm_ctx_last_logits(const sllm_ctx * c);

/*
 * Run one token at absolute position `pos` (0-based) and write n_vocab logits.
 * The caller supplies the logits buffer; it must be at least n_vocab floats.
 */
sllm_status sllm_forward(const sllm_model * m, sllm_ctx * c,
                         int32_t token, int32_t pos, float * logits);

/* The largest chunk a context will accept in one forward. Sized so the scratch
 * buffers are a fixed allocation rather than a realloc per call; chunking
 * exists for batching, not for memory, and a caller with a longer prompt
 * simply issues more chunks. */
#define SLLM_MAX_CHUNK 256

/*
 * Run `n` tokens as ONE chunk starting at absolute position `base`, writing
 * n_vocab logits for each.
 *
 * Chunking is a batching structure, not a different computation: a token at
 * absolute position p attends to every cached key at position q <= p, and the
 * diagonal is included. The reference expresses exactly that -- its mask fills
 * with -INFINITY and skips any key with p0 > p1 (llama-graph.cpp:450) -- so a
 * chunk boundary is not a semantic boundary and must not be observable.
 *
 * The per-query reduction order is deliberately identical to the single-token
 * path's. Phase 4 established that in this model a wider or differently
 * ordered accumulation moves the logits AWAY from the reference, so batching
 * here buys structure and locality, never arithmetic changes.
 */
sllm_status sllm_forward_chunk(const sllm_model * m, sllm_ctx * c,
                               const int32_t * tokens, int32_t n, int32_t base,
                               float * logits_out);

sllm_status sllm_forward_prefill_chunked(const sllm_model * m, sllm_ctx * c,
                                         const int32_t * tokens, int32_t n,
                                         int32_t chunk, float * logits_out);

/*
 * One token per forward. The shape the golden-logit fixture was captured
 * against and the shape the sealed Phase 4 gate is stated over: the reference
 * dumps a logit row per prompt position, so parity is per position and not
 * only at the last one.
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

/*
 * Save and restore the live KV cache and position.
 *
 * This is OUR format, not llama.cpp's. The reference's is a magic
 * (0xaf143cd8), a sequence id and the cache in its native type
 * (llama-context.cpp:2916); reproducing it byte for byte would be copying an
 * implementation, and nothing in the parity rule asks for it -- the gate is
 * that a restored context continues identically, which is a property of the
 * arithmetic rather than of a container.
 *
 * The file is versioned and bounds-checked on the way in. Every length is
 * validated against the model it is being loaded into, so a state file from a
 * different model, a different context depth, or a truncated write is refused
 * with a specific status instead of being read into a buffer.
 */
sllm_status sllm_state_save(const sllm_model * m, const sllm_ctx * c, const char * path);
sllm_status sllm_state_load(const sllm_model * m, sllm_ctx * c, const char * path);

#endif /* SAPHIRA_LLM_FORWARD_H */
