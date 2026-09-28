/*
 * quant.h — tensor storage formats and dequantisation.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_QUANT_H
#define SAPHIRA_LLM_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include <saphira_llm/status.h>
#include <saphira_llm/gguf.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Block layouts, mirroring the reference exactly. Verified against
 * ggml_blck_size()/ggml_type_size() rather than transcribed. */
#define SLLM_QK_8_0 32
#define SLLM_QK_4_0 32
#define SLLM_QK_K  256
#define SLLM_I2_S_TAIL 32

/* ggml_half stored as a raw little-endian 16-bit value, as it is on disk. */
typedef uint16_t sllm_fp16;

/* IEEE 754 binary16 -> binary32. Exposed because the golden-vector tests need
 * it directly, and because a wrong conversion is a silent precision loss. */
float sllm_fp16_to_fp32(sllm_fp16 h);

/* ------------------------------------------------------------------ */
/* generic row dequantisation                                          */
/* ------------------------------------------------------------------ */

/*
 * Expand one row of `n` values of `type` at `src` into `n` floats at `dst`.
 *
 * Implemented now: F32, F16, BF16, Q8_0, Q4_0, and I2_S. The K-quants are
 * Phase 7 scope: they only appear in conventional GGUF models, and the plan
 * is BitNet first. Asking for one of them returns
 * SLLM_ERR_TYPE_UNSUPPORTED rather than a wrong answer.
 *
 * For I2_S the destination receives the *integer* ternary codes as floats,
 * because the scale is applied separately by the GEMM epilogue exactly as
 * upstream does. Folding it in here would diverge from the reference.
 */
sllm_status sllm_dequant_row(sllm_ggml_type type, const void * src,
                             float * dst, size_t n);

/* ------------------------------------------------------------------ */
/* I2_S, the BitNet ternary format                                     */
/* ------------------------------------------------------------------ */

/*
 * I2_S stores four weights per byte and is laid out in transposed tiles for
 * the BLAST kernels, so a linear weight index does NOT map to a linear byte
 * offset. Within each 1024-byte block, which holds 4096 weights:
 *
 *   byte_in_chunk = k % 32
 *   chunk         = (k / 128) % 32
 *   byte_offset   = chunk * 32 + byte_in_chunk
 *   field         = (k % 128) / 32        (which 2-bit field of that byte)
 *
 * so a byte holds four weights 32 apart rather than four consecutive ones.
 * Reading it as consecutive would produce plausible, entirely wrong weights,
 * which is the same failure mode as trusting the ggml type table for the size.
 */
uint8_t sllm_i2s_code(const uint8_t * packed, size_t k);

/* Codes are 0, 1, 2 for ternary -1, 0, +1. */
#define SLLM_I2S_BIAS 1

/*
 * The f32 scale sits at data + n_elements/4, immediately after the packed
 * weights. The remaining 28 bytes of the 32-byte tail are padding.
 */
float sllm_i2s_scale(const void * packed, size_t n_elements);

/* Expand one row of I2_S codes as floats, still without the scale. */
void sllm_i2s_row(const void * packed, float * dst, size_t n);

/* The scale for an I2_S row, given the row's element count. */
static inline float sllm_i2s_row_scale(const void * row, size_t n) {
    return sllm_i2s_scale(row, n);
}

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_QUANT_H */
