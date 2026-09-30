/*
 * i2s_convert.h — native BF16 to I2_S conversion.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * This is the conversion direction, not a second inference path. It produces
 * the I2_S representation that saphira-llm already reads; it never loads a
 * model and never runs a forward pass.
 *
 * WHY THIS EXISTS. Producing an I2_S GGUF from BF16 masters currently needs the
 * upstream Python converter, gguf-py, numpy and torch. None of those are here
 * and the project policy is no Python in the build or the runtime. A C11
 * converter removes that dependency for the conversion step. The training step
 * needs the BF16 masters as input and nothing else from upstream.
 */
#ifndef SAPHIRA_LLM_I2S_CONVERT_H
#define SAPHIRA_LLM_I2S_CONVERT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Which upstream ternarisation rule to use.
 *
 * The pinned upstream tree contains TWO quantisers and they are not the same
 * function. This is not a subtlety invented here; it was found by
 * transcribing both and comparing them, and the difference is large:
 *
 *   SLLM_I2S_RULE_GGUF_PY
 *       third_party/BitNet/utils/convert-hf-to-gguf-bitnet.py:666
 *       t = w * inv_scale, then code 2 if t > 0.5, 0 if t < -0.5, else 1.
 *
 *   SLLM_I2S_RULE_BITNET_C
 *       third_party/BitNet/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c:1358
 *       |w| < 1e-6 gives code 1, otherwise the sign of w.
 *
 * Measured over 1000 values spanning 0.001..1.0 of the scale, they DISAGREE on
 * 500 of them: they part company exactly where |w/scale| < 0.5. On input that is
 * already ternary they agree completely.
 *
 * The default is the GGUF_PY rule, because that is the one that produced the
 * I2_S model this project ships, and a converted model should be comparable
 * with it. The C rule is available because bitnet.cpp uses it for any
 * on-the-fly quantisation, so a caller replicating bitnet.cpp's behaviour asks
 * for it explicitly.
 */
typedef enum sllm_i2s_rule {
    SLLM_I2S_RULE_GGUF_PY  = 0,
    SLLM_I2S_RULE_BITNET_C = 1,
    SLLM_I2S_RULE_COUNT    = 2
} sllm_i2s_rule;

/*
 * Bytes sllm_i2s_quantize will write for n elements: n/4 packed bytes plus the
 * 32-byte tail that holds the f32 scale. Returned rather than assumed, because
 * reading a scale from the wrong offset is the failure this project has already
 * made once (see the note at sllm_i2s_scale in src/quant.c).
 */
size_t sllm_i2s_packed_size(size_t n_elements);

/*
 * Quantise `n` floats into the I2_S representation.
 *
 * Layout, from the pinned reference and identical to what src/quant.c decodes:
 *
 *   - codes are the raw values {0, 1, 2} for ternary {-1, 0, +1}
 *   - 2 bits per element, 4 elements per byte
 *   - field 0 is the TOP of the byte: element 4j+0 sits at shift 6, 4j+1 at 4,
 *     4j+2 at 2, 4j+3 at 0
 *   - the f32 scale follows the packed data at offset n/4, in a tail that is
 *     zero-padded out to 32 bytes
 *
 * The scale is the FIRST nonzero magnitude, which is the BitNet convention and
 * is what both upstream implementations do. It is NOT the maximum: for already
 * -ternary input the two coincide, and for general input they do not. Callers
 * that know the true scale must use sllm_i2s_quantize_scaled.
 *
 * Returns 0 on success, non-zero on bad arguments. Writes exactly
 * sllm_i2s_packed_size(n) bytes to `dst`.
 */
int sllm_i2s_quantize(const float * src, size_t n, void * dst, sllm_i2s_rule rule);

/*
 * As above, but with the scale supplied rather than recovered. This is the
 * override_scale path of the Python converter, and it is the only way to make a
 * round trip byte-exact when the recovered scale would differ from the original
 * -- which is exactly the situation a dequantise-then-requantise creates.
 */
int sllm_i2s_quantize_scaled(const float * src, size_t n, void * dst,
                             float scale, sllm_i2s_rule rule);

/* The f32 scale stored in an I2_S block, read the way src/quant.c reads it. */
float sllm_i2s_block_scale(const void * packed, size_t n_elements);

/* BF16 <-> f32. BF16 is the top 16 bits of an f32, so these are exact and
 * symmetric for every finite input. */
float  sllm_bf16_to_f32(uint16_t h);
uint16_t sllm_f32_to_bf16(float f);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_I2S_CONVERT_H */
