/*
 * i2s_gemm.h — the BitNet I2_S ternary matrix kernels.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_I2S_GEMM_H
#define SAPHIRA_LLM_I2S_GEMM_H

#include <stddef.h>
#include <stdint.h>

#include <saphira_llm/status.h>
#include <saphira_llm/isa.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The activation block the kernels walk. 128 weights share 128 int8
 * activations, split into four groups of 32 that pair with the four two-bit
 * fields of each packed byte. */
#define SLLM_I2S_QK 128

/*
 * Metadata for one quantised activation row.
 *
 * Deliberately does NOT contain the quantised data. The first version had a
 * fixed int8_t q[128] here and the quantiser wrote n bytes into it, which
 * smashed the stack on any row longer than one block. The scale is per ROW, not
 * per block, so the data has to be as long as the row and the caller has to
 * own it. A fixed buffer in a struct is the wrong shape for a variable-length
 * payload, and the compiler will not help you notice.
 */
typedef struct sllm_i2s_act {
    float   scale;   /* 127 / amax, or 0 when the row is all zero */
    int32_t sum;     /* sum of the quantised values               */
} sllm_i2s_act;

/*
 * Quantise one activation row of `n` values.
 *
 * An exact port of the reference's quantize_row_i8_s:
 *
 *   amax   = max(|x|)          (no epsilon floor: a zero row gives s = 0)
 *   s      = amax > 0 ? 127/amax : 0
 *   q[i]   = clamp((int)roundf(x[i] * s), -128, 127)
 *   sum    = sum of q
 *
 * Note `roundf` is round-half-away-from-zero, not banker's rounding, and that
 * the reference has no 1e-5 floor here. An earlier reading of a different
 * function in the same tree suggested a floor and a magic-constant rounding;
 * the function that actually runs is this one, and the two disagree.
 */
void sllm_i2s_quant_act(const float * x, size_t n, int8_t * q_out,
                        sllm_i2s_act * out);

/* The scalar oracle. Always compiled; never dispatched to except by tests. */
int32_t sllm_i2s_dot_scalar(const uint8_t * w, const int8_t * a, size_t n);

/*
 * One weight row against one quantised activation row.
 *
 * `w` points at the row's packed bytes; the 128-element block structure means
 * the row is walked block by block, and `w` advances by 32 bytes per block.
 *
 * Returns the raw integer dot product, not the scaled result, because that is
 * what the kernel computes and what the epilogue expects.
 */
int32_t sllm_i2s_dot(const uint8_t * w, const int8_t * q, size_t n);

/*
 * A matrix of `n_rows` weight rows against one activation row. `rows_out`
 * receives the raw integer dot products.
 *
 * This is the shape token generation uses: one activation vector, many weight
 * rows.
 */
void sllm_i2s_gemv(const uint8_t * w, size_t n_rows, size_t n,
                   const int8_t * q, int32_t * rows_out);

/*
 * The epilogue, reproduced verbatim from the reference:
 *
 *   dst = (dot - act_sum) / act_scale * w_scale
 *
 * The subtraction of act_sum is mathematically suspicious: it is only correct
 * if the ternary weights sum to zero over K, which is not generally true. The
 * reference applies it, so parity requires applying it. "Correcting" it would
 * be a divergence from the oracle, not an improvement, and would fail the
 * parity gate for reasons unrelated to the gate.
 */
float sllm_i2s_epilogue(int32_t dot, int32_t act_sum, float act_scale, float w_scale);

/*
 * The reference applies the epilogue in place over the GEMM's FLOAT output
 * buffer, not to a separate int32 accumulator array:
 *
 *   ggml/src/ggml-bitnet-compute.c:164
 *     tmp[row] = (tmp[row] - act_sums[i1]) / (act_scales[i1]) * (*scale);
 *
 * where tmp[row] holds an integral dot product stored as a float. There is
 * therefore no standalone "apply the epilogue to an int32 array" step to
 * mirror. An earlier version of this header declared exactly that, casting
 * each result back to int32_t and silently truncating every value to 0. The
 * function is gone; the forward pass applies sllm_i2s_epilogue in place over
 * the float GEMM output, where the reference does.
 */

/* Which ISA the installed kernels will use. For tests and --log debug. */
sllm_isa_level sllm_i2s_installed_isa(void);

/*
 * Choose the kernel set for this run. Called once, before any inference, with
 * the dispatch from sllm_isa_build(). Until it is called the v3 kernel is used
 * unconditionally, so the module is safe without any initialisation at all.
 */
void sllm_i2s_select_isa(const sllm_isa_dispatch * d);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_I2S_GEMM_H */
