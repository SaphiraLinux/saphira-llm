/*
 * i2s_convert.c — native BF16 to I2_S conversion.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Every semantic here is transcribed from the pinned upstream tree, not from
 * memory and not from this project's own decoder. The sources are named at each
 * decision, because a converter whose only authority is the thing it feeds
 * cannot be used to check that thing.
 *
 *   PYTHON  third_party/BitNet/utils/convert-hf-to-gguf-bitnet.py:666
 *           quantize_to_i2_s
 *   C       third_party/BitNet/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c:1358
 *           quantize_i2_s
 *   DEQ     third_party/BitNet/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c:1335
 *           dequantize_row_i2_s
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <saphira_llm/i2s_convert.h>

#include <math.h>
#include <string.h>

size_t sllm_i2s_packed_size(size_t n_elements) {
    /* n/4 packed bytes, then the 32-byte tail holding the f32 scale. Matches
     * the upstream return value n/4 + 32 exactly. */
    return n_elements / 4u + 32u;
}

float sllm_i2s_block_scale(const void * packed, size_t n_elements) {
    float s;
    /* Immediately after the packed data, which is where the upstream C writer
     * puts it: float * scale_ptr = (float *)(q + n / 4). */
    memcpy(&s, (const uint8_t *) packed + n_elements / 4u, sizeof s);
    return s;
}

/* BF16 is the top 16 bits of an f32. Both directions are exact for every
 * finite input, including subnormals, because the conversion only shifts. */
float sllm_bf16_to_f32(uint16_t h) {
    const uint32_t bits = (uint32_t) h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

uint16_t sllm_f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return (uint16_t) (bits >> 16);
}

/*
 * The scale. Both upstream implementations take the FIRST nonzero magnitude,
 * not the maximum, and the C one does it with a break on the first element
 * whose absolute value exceeds the running maximum -- which, starting from
 * zero, is the first nonzero element.
 *
 * The deadzone differs between them and is preserved exactly: the C version
 * treats anything under 1e-6 as zero, the Python version has no such deadzone
 * and instead divides by the scale and thresholds at +-0.5.
 */
static double first_nonzero_scale(const float * src, size_t n) {
    double max = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double v = fabs((double) src[i]);
        if (v > max) { max = v; break; }
    }
    return max;
}

static uint8_t code_of(float w, double scale, sllm_i2s_rule rule) {
    if (rule == SLLM_I2S_RULE_BITNET_C) {
        /* quants.c:1373-1377, verbatim in behaviour. */
        if (fabs((double) w) < 1e-6) { return 1; }
        return ((double) w * scale > 0.0) ? 2 : 0;
    }
    /*
     * convert-hf-to-gguf-bitnet.py:700-702, verbatim in behaviour.
     *
     * The reference writes this as
     *     q_float = np.round(w * inv_scale).clip(-1, 1)
     *     q[q_float >  0.5] = 2 ;  q[q_float < -0.5] = 0 ;  default 1
     * and both the round and the clip are REDUNDANT: the strict +-0.5
     * threshold decides every case, so the effective rule is just the
     * threshold. Established by transcription and sweep rather than by
     * inspection -- 56014 values around the thresholds and at exact half-steps,
     * zero mismatches (see docs/BITNET-LIFECYCLE.md).
     *
     * Writing it as a threshold also means there is no floating-point
     * rounding-mode subtlety to get wrong, which is the single most likely way
     * a C port would silently diverge from the Python one.
     */
    const double t = (double) w * (1.0 / scale);
    if (t >  0.5) { return 2; }
    if (t < -0.5) { return 0; }
    return 1;
}

int sllm_i2s_quantize_scaled(const float * src, size_t n, void * dst,
                             float scale, sllm_i2s_rule rule) {
    if (dst == NULL || (src == NULL && n > 0)) { return 1; }
    if (rule >= SLLM_I2S_RULE_COUNT) { return 1; }
    /*
     * n must be a multiple of 4, and this is a format property rather than a
     * convenience.
     *
     * Four elements share one byte and the f32 scale begins at offset n/4. If n
     * is not a multiple of 4, the packing loop writes past the packed region
     * and into the scale -- silently. The upstream C reference has exactly this
     * hazard: it memsets n/4 bytes and then writes q[i/4] for every i, so for
     * n = 1 it writes the code into the byte the scale is about to occupy, and
     * the scale then overwrites it. The upstream Python avoids it by padding the
     * input to a multiple of 128.
     *
     * Rejecting the case is better than either: a caller that gets an error
     * learns its tensor shape is wrong, rather than shipping a model with a
     * silently mangled scale.
     */
    if ((n % 4u) != 0u) { return 2; }

    uint8_t * q = (uint8_t *) dst;
    /* The packed region is zeroed, and the whole tail with it, because the
     * reference memsets n/4 bytes and only then writes the scale; the
     * remaining 28 tail bytes must still be deterministic rather than whatever
     * was in the buffer. */
    memset(q, 0, sllm_i2s_packed_size(n));

    /*
     * THE LAYOUT IS INTERLEAVED, and getting this wrong is the single easiest
     * way to produce a file that loads and computes nonsense.
     *
     * Within each 128-element group, the four fields of a byte are 32 weights
     * apart, not consecutive. Element i goes to
     *
     *     byte  = (i / 128) * 32 + (i % 32)
     *     field = (i % 128) / 32
     *
     * and field 0 is the TOP of the byte, at shift 6.
     *
     * Three independent sources agree on this and are the authority:
     *   - dequantize_row_i2_s, quants.c:1335, which reads
     *     y[done + o*32 + gp] from x[done/4 + gp] with field o,
     *   - the python packer, convert-hf-to-gguf-bitnet.py:709, which reshapes
     *     to (nblocks, 4, 32) and shifts q[:,0,:]<<6 | q[:,1,:]<<4 |
     *     q[:,2,:]<<2 | q[:,3,:],
     *   - this project's own decoder, src/quant.c, which is gated against
     *     tests/golden/i2s-reference.txt captured from the reference's own
     *     dequantiser.
     *
     * A FOURTH source disagrees: upstream's own C quantiser,
     * quantize_i2_s at quants.c:1358, packs flat as byte i/4 with field
     * i%4. That is a latent upstream bug -- its own dequantiser does not read
     * what it writes. It is never exercised in practice because bitnet.cpp
     * consumes weights that were already packed by the python converter, so
     * the buggy C quantiser is not on the path that matters. It is transcribed
     * faithfully in tests/golden/i2s-converter.txt only so the divergence is on
     * the record; packing that way would produce a model no reader can make
     * sense of.
     */
    for (size_t i = 0; i < n; ++i) {
        const uint8_t val = code_of(src[i], (double) scale, rule);
        const size_t b128 = i / 128u;
        const unsigned a   = (unsigned) ((i % 128u) / 32u);
        const size_t  bb  = i % 32u;
        q[b128 * 32u + bb] |= (uint8_t) (val << (6u - 2u * a));
    }

    memcpy(q + n / 4u, &scale, sizeof scale);
    return 0;
}

int sllm_i2s_quantize(const float * src, size_t n, void * dst, sllm_i2s_rule rule) {
    if (dst == NULL || (src == NULL && n > 0)) { return 1; }
    if (rule >= SLLM_I2S_RULE_COUNT) { return 1; }
    if ((n % 4u) != 0u) { return 2; }
    const double max = first_nonzero_scale(src, n);
    /* Both upstreams fall back rather than divide by zero. The Python one uses
     * 1e-5; the C one leaves max at 0.0 and then codes everything as 1 because
     * |w| < 1e-6, so it writes a scale of 0. Preserved per rule. */
    const float scale = (rule == SLLM_I2S_RULE_BITNET_C)
                            ? (float) max
                            : (max > 0.0 ? (float) max : 1e-5f);
    return sllm_i2s_quantize_scaled(src, n, dst, scale, rule);
}
