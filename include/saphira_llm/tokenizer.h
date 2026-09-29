/*
 * BPE tokenizer, matching the pinned reference exactly.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */

#ifndef SAPHIRA_LLM_TOKENIZER_H
#define SAPHIRA_LLM_TOKENIZER_H

#include "saphira_llm/gguf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct sllm_tok sllm_tok;

/*
 * Load the tokenizer from a parsed model's metadata. The returned tokenizer
 * borrows nothing from `gguf` except the key lookups done during load: the
 * vocabulary, merges and special-token strings are copied, so the tokenizer
 * outlives the sllm_gguf. That is deliberate, because a forward pass wants the
 * vocab and the model in the same struct and either one can be released first.
 *
 * Fails with a specific status when the model declares a tokenizer this build
 * does not implement, rather than producing plausible wrong tokens.
 */
sllm_status sllm_tok_load(const sllm_gguf * gguf, sllm_tok ** out);
void        sllm_tok_free(sllm_tok * tok);

/* Vocabulary size, and the special token ids. A negative id means the model
 * does not declare that token. */
uint32_t sllm_tok_n_vocab(const sllm_tok * tok);
int32_t  sllm_tok_bos(const sllm_tok * tok);
int32_t  sllm_tok_eos(const sllm_tok * tok);
bool     sllm_tok_add_bos(const sllm_tok * tok);
bool     sllm_tok_add_eos(const sllm_tok * tok);

/* True when the token is a control or user-defined special token, which decode
 * leaves out and which the pre-tokeniser must not merge. */
bool sllm_tok_is_special(const sllm_tok * tok, int32_t id);

/* The stored piece for a token: the vocab string, still byte-encoded, so
 * U+0120 for a space. NULL for an out-of-range id. */
const char * sllm_tok_piece(const sllm_tok * tok, int32_t id);

/*
 * Encode `text` into `out`, at most `cap` tokens.
 *
 * Returns the number of tokens written, or a negative sllm_status. Use
 * sllm_tok_encode_len for the two-pass form when the token count is unknown;
 * the count is the output size of the first pass, not a guess.
 *
 * `add_special` prepends BOS and appends EOS when the model asks for it.
 * `parse_special` recognises literal special-token text in the input and emits
 * it as a single token, which is how a prompt can contain one.
 */
int32_t sllm_tok_encode(const sllm_tok * tok, const char * text, size_t text_len,
                        bool add_special, bool parse_special,
                        int32_t * out, int32_t cap);

/* Upper bound on the token count for a given input, for sizing a buffer.
 * Exact when `parse_special` is false. */
int32_t sllm_tok_encode_len(const sllm_tok * tok, size_t text_len, bool add_special);

/*
 * Decode tokens back to text. `remove_special` drops control and user-defined
 * tokens, which is what a generation result needs; leaving it false reproduces
 * the reference's rendering of a prompt that contained specials.
 *
 * Returns bytes written excluding the NUL, or a negative sllm_status when the
 * buffer is too small, so the caller can size a retry. Always NUL-terminates
 * when cap > 0.
 */
int32_t sllm_tok_decode(const sllm_tok * tok, const int32_t * tokens, int32_t n,
                        bool remove_special, char * out, int32_t cap);

/*
 * Split `text` the way the pre-tokeniser does, without running BPE. Exposed
 * because the pre-tokeniser is where this diverges most from a textbook GPT-2
 * BPE, and it needs to be testable on its own: a golden vector over whole
 * token ids cannot tell you which of the four split passes was wrong.
 *
 * Each output is one word, byte-encoded, NUL-terminated, appended to `out` and
 * counted in *n_out. Words are separated by a single NUL, so the buffer is one
 * run of C strings; a NUL-separated run is returned rather than a pointer array
 * so this needs no allocation on the caller's side.
 */
sllm_status sllm_tok_pretokenize(const sllm_tok * tok, const char * text, size_t text_len,
                                 char * out, size_t cap, int32_t * n_out);

#endif /* SAPHIRA_LLM_TOKENIZER_H */
