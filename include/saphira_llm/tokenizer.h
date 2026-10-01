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
 * Which pre-tokeniser the model asked for.
 *
 * This is MODEL METADATA, not a property of the architecture and emphatically
 * not a property of BPE. It is read from `tokenizer.ggml.pre`, and when that
 * key is ABSENT the reference does not assume a canonical GPT-2 split: it falls
 * back to SLLM_PRE_DEFAULT, which is a different set of four passes, and logs
 * "GENERATION QUALITY WILL BE DEGRADED".
 *
 * That distinction is not academic. For the acceptance model it changes real
 * tokens: "they're" is `they` `'` `re` under DEFAULT and `they` `'re` under
 * GPT2, "1234567890" is `123` `456` `789` `0` under DEFAULT and not under GPT2,
 * and "%!" is one token under DEFAULT because of a pass that exists in no GPT-2
 * specification at all.
 *
 * So this is an enum resolved from the file, with an explicit fallback, and not
 * a constant in the tokenizer. The Phase 7 GGUF models will each carry their
 * own `pre` value, and a model whose `pre` names a pre-tokeniser we have not
 * implemented must be REJECTED with the offending string named, not silently
 * tokenised with whatever happens to be compiled in. The reference throws in
 * that situation; matching that is the difference between a specific error and
 * a wrong answer.
 */
typedef enum sllm_pre_type {
    SLLM_PRE_UNSET     = 0,  /* `pre` absent: the DEFAULT four-pass fallback */
    SLLM_PRE_GPT2      = 1,  /* `pre` = "gpt-2" and friends: one pass       */
    SLLM_PRE_UNSUPPORTED = 2, /* named in the file, not implemented here     */
    SLLM_PRE_QWEN2     = 3   /* `pre` = "qwen2" and friends: qwen2 pattern   */
} sllm_pre_type;

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

/* The pre-tokeniser this model resolved to, for reporting and for tests. */
sllm_pre_type sllm_tok_pre_type(const sllm_tok * tok);

/* A short name for a pre-type, for logs and diagnostics. */
const char * sllm_tok_pre_type_name(sllm_pre_type pre);

/*
 * The `tokenizer.ggml.pre` value the model declared, or "" when the key is
 * absent. The distinction matters: "" here means DEFAULT, which is NOT GPT-2.
 * Returns "" also for a NULL tokenizer.
 */
const char * sllm_tok_pre_declared(const sllm_tok * tok);

/* Whether the split pipeline for this model's pre-type is verified against
 * reference golden vectors. SLLM_PRE_GPT2 is implemented but has no fixture,
 * because no model carrying `pre` = "gpt-2" was available to capture one. */
bool sllm_tok_pre_is_verified(const sllm_tok * tok);

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

/*
 * Encode with an explicit pre-type rather than the model's own.
 *
 * Splitting depends on the pre-type and on nothing else -- no vocabulary, no
 * merges -- so this is the same function with the one model-specific input
 * exposed. Two reasons it is not a test-only back door:
 *
 *  - Phase 7's ordinary GGUF models each declare their own `pre`, and when one
 *    declares a value we map to a pre-type whose behaviour we want to compare
 *    against a neighbour, being able to ask "what would this vocabulary produce
 *    under the other one" is the fastest way to see which side is wrong.
 *
 *  - It is what makes the pre-type dispatch testable at all. The only model
 *    available declares no `pre`, so the alternative to this is a second
 *    gigabyte-scale fixture, and a gate that needs a gigabyte is a gate that
 *    does not run.
 *
 * SLLM_PRE_UNSUPPORTED is rejected rather than approximated.
 */
int32_t sllm_tok_encode_pre(const sllm_tok * tok, sllm_pre_type pre,
                            const char * text, size_t text_len,
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
 * Split `text` the way the given pre-tokeniser does, without running BPE.
 * Exposed because the pre-tokeniser is where this diverges most from a
 * textbook GPT-2 BPE, and it needs to be testable on its own: a golden vector
 * over whole token ids cannot tell you which of the split passes was wrong.
 *
 * Needs no tokenizer, because splitting does not consult the vocabulary.
 *
 * Each output is one word, byte-encoded, NUL-terminated, appended to `out` and
 * counted in *n_out. Words are separated by a single NUL, so the buffer is one
 * run of C strings; a NUL-separated run is returned rather than a pointer array
 * so this needs no allocation on the caller's side.
 */
sllm_status sllm_tok_pretokenize(sllm_pre_type pre, const char * text, size_t text_len,
                                 char * out, size_t cap, int32_t * n_out);

#endif /* SAPHIRA_LLM_TOKENIZER_H */
