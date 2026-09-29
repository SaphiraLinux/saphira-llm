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
