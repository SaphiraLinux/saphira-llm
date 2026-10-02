/*
 * ops.h — vector operations for the transformer forward pass.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_OPS_H
#define SAPHIRA_LLM_OPS_H

#include <stddef.h>
#include <stdint.h>

#include <saphira_llm/gguf.h>
#include <saphira_llm/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Which RoPE layout. Both are used in the wild and they are not
 * interchangeable, so the caller states which one it has. */
typedef enum sllm_rope_type {
    SLLM_ROPE_NEOX = 0,  /* pairs the two halves: (i, i + n/2)          */
    SLLM_ROPE_NORMAL = 1,/* pairs adjacent dimensions: (2i, 2i+1)        */
    SLLM_ROPE_COUNT
} sllm_rope_type;

sllm_status sllm_rope_type_parse(const char * name, sllm_rope_type * out);

/*
 * RMSNorm over one vector of `n` elements, in place is not allowed: `x` is
 * read and `dst` written, so the caller can normalise into a scratch buffer
 * without a temporary. `eps` is added to the mean square before the root.
 */
void sllm_rms_norm(float * dst, const float * x, const float * weight,
                   size_t n, float eps);

/*
 * One row of an F16 table dotted with an F32 activation vector.
 *
 * This is the tied output projection: n_vocab rows of n_embd, once per token.
 * The scalar form is exposed alongside it so a test can hold the vectorised
 * path to the portable one; see src/ops.c for why this kernel is shaped the
 * way it is, and what the profile that motivated it actually said.
 */
float sllm_dot_f16_f32(const uint16_t * row, const float * x, size_t n);

/*
 * Step 2: matrix-vector product over stored rows of a (possibly quantised)
 * tensor payload, producing one f32 result per row.
 *
 *     out[r] = sum_i W[r][i] * x[i]
 *
 * `n` is the number of elements in one row (tensor ne[0]) and `n_rows` the number
 * of stored rows. `data` is the mapped tensor payload; this function performs no
 * offset arithmetic of its own, so the caller cannot be surprised by it.
 *
 * Deliberately a plain scalar loop. It is the reference against which any
 * optimised version must agree bit-for-bit, and it is where stride, index and
 * block-boundary mistakes are actually visible rather than merely plausible.
 *
 * Note this is a separate capability from decoding. A type may be decodable, or
 * have a known layout, or be computable, or be dispatchable; these are four
 * different questions and sllm_gguf_type_is_supported() answers only the first two.
 *
 * Returns SLLM_ERR_ARG when `n` is not a whole number of stored blocks, rather
 * than decoding a partial block.
 */
sllm_status sllm_gemv_f32(sllm_ggml_type type, const void * data, size_t n,
                          const float * x, size_t n_rows, float * out);
float sllm_dot_f16_f32_scalar(const uint16_t * row, const float * x, size_t n);

/*
 * Softmax over one row of `n` elements, overwriting the row with the
 * normalised values. The maximum is subtracted first, so this is safe on
 * f32 values that would otherwise overflow in exp.
 */
void sllm_softmax_inplace(float * x, size_t n);

/* x = x * sigmoid(x), elementwise. */
void sllm_silu_inplace(float * x, size_t n);

/* dst = a + b, elementwise. */
void sllm_add(float * dst, const float * a, const float * b, size_t n);

/* dst = a * b, elementwise. */
void sllm_mul(float * dst, const float * a, const float * b, size_t n);

/*
 * Rotary position embedding, in place on one head of `n_rot` dimensions.
 * `pos` is the absolute position, `theta` the base frequency and `freq_scale`
 * divides it (1.0 means unmodified).
 */
void sllm_rope_inplace(float * x, size_t n_rot, int32_t pos,
                       float theta, float freq_scale, sllm_rope_type type);

/*
 * Gather rows of a row-major f32 matrix: `dst[i] = src[idx[i] * row_bytes]`.
 * `row_bytes` is in floats, so it is `ne0` for a plain matrix. Out-of-range
 * indices are clamped to zero rather than dereferenced, because an index
 * coming from a model is input, not a promise.
 */
void sllm_get_rows(float * dst, const float * src, const int32_t * idx,
                   size_t n_rows, size_t row_floats);

/* expf without the libm call overhead in a hot loop. Public for testing. */
float sllm_fast_exp(float x);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_OPS_H */
