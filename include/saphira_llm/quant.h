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
 * Implemented now: F32, F16, BF16, Q8_0, Q4_0, I2_S, Q4_K and Q6_K. The
 * K-quants are transcribed from the vendored reference and gated against a
 * REFERENCE golden in tests/test_quant_k.c -- expected values captured by
 * driving the reference build's own type traits over a real Q4_K_M model, not by
 * recording our own output. A type we have not implemented still returns
 * SLLM_ERR_TYPE_UNSUPPORTED rather than a wrong answer, and a K-quant asked for a
 * length that is not a whole number of 256-element super-blocks returns
 * SLLM_ERR_ARG rather than decoding a partial block.
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
 *   field         = (k % 128) / 32
 *
 * so a byte holds four weights 32 apart rather than four consecutive ones, and
 * field a is at bit shift 6 - 2a, meaning field 0 is the TOP of the byte.
 * Both details were wrong in the first version of this reader, and the first
 * golden test could not catch either because it shared the same assumption.
 * Reading it as consecutive, or reading the fields bottom-up, produces
 * plausible entirely wrong weights -- the same failure mode as trusting the
 * ggml type table for the size.
 */
uint8_t sllm_i2s_code(const uint8_t * packed, size_t k);

/* Codes are 0, 1, 2 for ternary -1, 0, +1. Code 3 is unused and dequantises
 * to zero, matching the reference table. */
#define SLLM_I2S_BIAS 1

/*
 * The f32 scale sits at data + n_elements/4, immediately after the packed
 * weights. The remaining 28 bytes of the 32-byte tail are padding.
 */
float sllm_i2s_scale(const void * packed, size_t n_elements);

/* Expand one row of I2_S codes as floats {0,1,2}, still without the scale.
 * This is what the GEMM consumes. */
void sllm_i2s_row(const void * packed, float * dst, size_t n);

/*
 * Expand one row as the signed, scaled ternary value, using the reference's
 * own table {-1, 0, +1, 0} so that an out-of-range code 3 yields zero rather
 * than 2 * scale.
 *
 * This is what the reference's dequantize_row_i2_s returns, and the golden
 * vectors in tests/golden/i2s-reference.txt are that function's output on real
 * tensors. The two paths are kept separate on purpose: the kernel wants raw
 * codes, this wants the semantic value, and conflating them is how a sign
 * error becomes a plausible number.
 */
void sllm_i2s_dequant(const void * packed, float * dst, size_t n);

/* The scale for an I2_S row, given the row's element count. */
static inline float sllm_i2s_row_scale(const void * row, size_t n) {
    return sllm_i2s_scale(row, n);
}

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_QUANT_H */
