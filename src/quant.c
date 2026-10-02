/*
 * quant.c — tensor storage formats and dequantisation.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */

#include <saphira_llm/quant.h>
#include <saphira_llm/log.h>

#if defined(__x86_64__)
#  include <immintrin.h>
#  define SLLM_X86 1
#endif

#include <string.h>

/* ------------------------------------------------------------------ */
/* half precision                                                      */
/* ------------------------------------------------------------------ */

float sllm_fp16_to_fp32(sllm_fp16 h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t man  = h & 0x3ffu;

    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                       /* signed zero */
        } else {
            /* Subnormal: normalise it. */
            uint32_t e = 0;
            uint32_t m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++e;
            }
            m &= 0x3ffu;
            bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);   /* inf or NaN */
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* bfloat16 is the top half of a float, so it needs no arithmetic at all. */
static inline float sllm_bf16_to_fp32(uint16_t b) {
    const uint32_t bits = (uint32_t) b << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ------------------------------------------------------------------ */
/* I2_S                                                                */
/* ------------------------------------------------------------------ */

/*
 * 1024 bytes = 4096 weights per block. Within the block the layout is a
 * transposed tile: 32 chunks of 32 bytes, and within a chunk the 4 two-bit
 * fields of each byte are 32 weights apart rather than consecutive.
 */
#define I2S_BLOCK_BYTES 1024u
#define I2S_BLOCK_ELEMS 4096u
#define I2S_CHUNKS      32u
#define I2S_CHUNK_BYTES 32u

uint8_t sllm_i2s_code(const uint8_t * packed, size_t k) {
    const size_t block = k / I2S_BLOCK_ELEMS;
    const size_t inblk = k % I2S_BLOCK_ELEMS;
    const size_t chunk = (inblk / 128u) % I2S_CHUNKS;
    const size_t byte_in_chunk = inblk % 32u;
    const unsigned field = (unsigned) ((inblk % 128u) / 32u);

    /*
     * Field 0 is the TOP of the byte, not the bottom.
     *
     * The converter packs (q[:,0,:] << 6) | (q[:,1,:] << 4) | (q[:,2,:] << 2)
     * | q[:,3,:] over a (4, 32) reshape, so field a sits at shift 6 - 2a.
     * Reading it as shift 2a reverses the four fields in every byte.
     *
     * This was wrong in the first version, and the golden test did not catch
     * it, because that test built the packed buffer and read it back with the
     * same inverted assumption. A test that shares its assumption with the code
     * under test cannot detect that code is wrong. The permanent fix is a
     * golden vector captured from the reference's own dequantiser on a real
     * tensor -- see tests/golden/i2s-reference.txt.
     */
    const uint8_t byte = packed[block * I2S_BLOCK_BYTES + chunk * I2S_CHUNK_BYTES + byte_in_chunk];
    return (uint8_t) ((byte >> (6u - 2u * field)) & 0x3u);
}

float sllm_i2s_scale(const void * packed, size_t n_elements) {
    float s;
    memcpy(&s, (const uint8_t *) packed + n_elements / 4u, sizeof(s));
    return s;
}

void sllm_i2s_row(const void * packed, float * dst, size_t n) {
    const uint8_t * p = (const uint8_t *) packed;
    for (size_t k = 0; k < n; ++k) {
        /*
         * The raw code {0,1,2}, not the signed value.
         *
         * The kernel needs the codes, because dpbusd takes its first operand
         * unsigned and that is exactly what non-negative codes are for. The
         * sign and the scale live in the GEMM epilogue, which is where
         * upstream applies them, so folding them in here would diverge from
         * the reference the parity gate compares against.
         *
         * Use sllm_i2s_dequant for the signed, scaled value.
         */
        dst[k] = (float) sllm_i2s_code(p, k);
    }
}

/*
 * The reference's own map, verbatim:
 *   static const float map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };
 *
 * Code 3 is not -2 or +2: it maps to ZERO. The converter only ever emits
 * 0, 1 and 2, so this never fires on a well-formed model, which is why the
 * 46-million-element cross-check passed even though the first version of this
 * function used code - 1 and would have returned 2 * scale for a code of 3.
 *
 * Reproduced rather than tidied. A cleaner formula would be wrong on exactly
 * the inputs nobody tests by hand.
 */
static const float sllm_i2s_map[4] = { -1.0f, 0.0f, 1.0f, 0.0f };

void sllm_i2s_dequant(const void * packed, float * dst, size_t n) {
    const float scale = sllm_i2s_scale(packed, n);
    const uint8_t * p = (const uint8_t *) packed;
    for (size_t k = 0; k < n; ++k) {
        dst[k] = scale * sllm_i2s_map[sllm_i2s_code(p, k)];
    }
}

/* ------------------------------------------------------------------ */
/* K-quants: Q4_K and Q6_K                                            */
/* ------------------------------------------------------------------ */

/*
 * Transcribed from the vendored reference (ggml-common.h block layouts and
 * ggml-quants.c dequantize_row_q4_K / dequantize_row_q6_K) and gated against
 * tests/golden/mainstream-qwen3-dequant.txt, which was captured from the
 * reference build rather than from this code. If these two disagree with the
 * golden by more than f32 rounding, the gate fails -- the flag cannot be a
 * promise without a kernel.
 *
 * QK_K = 256 elements per super-block. K_SCALE_SIZE = 12 bytes carrying eight
 * 6-bit scales and eight 6-bit mins, interleaved. The unpack below is the part
 * that must match bit-for-bit: for j < 4 the low six bits live in q[j] and
 * q[j+4], and for j >= 4 two bits are borrowed from the top of the neighbouring
 * bytes. Reimplementing that layout "reasonably" rather than exactly is how a
 * dequantiser ends up subtly, plausibly wrong.
 */
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;    /* super-block scale for the quantized scales, f16 */
    uint16_t dmin; /* super-block scale for the quantized mins,  f16 */
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2]; /* 4-bit quants */
} sllm_block_q4_K;

typedef struct {
    uint8_t  ql[QK_K / 2];  /* quants, lower 4 bits */
    uint8_t  qh[QK_K / 4];  /* quants, upper 2 bits */
    int8_t   scales[QK_K / 16]; /* scales, quantized with 8 bits */
    uint16_t d;              /* super-block scale, f16 */
} sllm_block_q6_K;

/* Transcribed verbatim from the reference get_scale_min_k4. */
static inline void sllm_get_scale_min_k4(int j, const uint8_t * q,
                                         uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4));
    }
}

static void dequant_q4_K(const void * vx, float * y, size_t k) {
    const sllm_block_q4_K * x = (const sllm_block_q4_K *) vx;
    const size_t nb = k / QK_K;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        const float d    = sllm_fp16_to_fp32((sllm_fp16) x[i].d);
        const float dmin = sllm_fp16_to_fp32((sllm_fp16) x[i].dmin);
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            sllm_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            sllm_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l]  >> 4)  - m2;
            q += 32; is += 2;
        }
    }
}

static void dequant_q6_K(const void * vx, float * y, size_t k) {
    const sllm_block_q6_K * x = (const sllm_block_q6_K *) vx;
    const size_t nb = k / QK_K;
    for (size_t i = 0; i < nb; i++) {
        const float d = sllm_fp16_to_fp32((sllm_fp16) x[i].d);
        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)    | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

/* ------------------------------------------------------------------ */
/* generic row dequantisation                                          */
/* ------------------------------------------------------------------ */

#if defined(SLLM_X86)
/*
 * Eight int8 lanes in, eight floats out. _mm256_cvtepi8_epi32 takes a __m128i
 * holding exactly eight bytes, which is the detail that makes or breaks these
 * kernels: passing a 256-bit register does not compile, and widening the wrong
 * way silently reorders values.
 */
static inline __m256 i8x8_to_ps(__m128i v) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
}

static void dequant_q8_0(const void * src, float * dst, size_t n) {
    const uint8_t * s = (const uint8_t *) src;
    size_t i = 0;
    for (; i + SLLM_QK_8_0 <= n; i += SLLM_QK_8_0) {
        uint16_t h;
        memcpy(&h, s, 2);
        const __m256 vd = _mm256_set1_ps(sllm_fp16_to_fp32(h));
        const int8_t * q = (const int8_t *) (s + 2);
        for (int g = 0; g < 4; ++g) {
            const __m128i v = _mm_loadl_epi64((const __m128i *) (q + g * 8));
            _mm256_storeu_ps(dst + i + g * 8, _mm256_mul_ps(vd, i8x8_to_ps(v)));
        }
        s += SLLM_QK_8_0 + 2;
    }
    if (i < n) {
        /* A partial trailing block. Handled scalar so the tail is exact
         * rather than approximately right. */
        const uint16_t h = (uint16_t) (s[0] | ((uint16_t) s[1] << 8));
        const float d = sllm_fp16_to_fp32(h);
        const int8_t * q = (const int8_t *) (s + 2);
        for (size_t j = 0; i + j < n; ++j) {
            dst[i + j] = d * (float) q[j];
        }
    }
}

/*
 * Q4_0: sixteen bytes carry thirty-two values, low nibble first, so the
 * output order is low(0..7), high(0..7), low(8..15), high(8..15). Getting
 * that sequence wrong still produces 32 plausible numbers, just the wrong 32.
 */
static void dequant_q4_0(const void * src, float * dst, size_t n) {
    const uint8_t * s = (const uint8_t *) src;
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i bias = _mm_set1_epi8(8);

    size_t i = 0;
    for (; i + SLLM_QK_4_0 <= n; i += SLLM_QK_4_0) {
        uint16_t h;
        memcpy(&h, s, 2);
        const __m256 vd = _mm256_set1_ps(sllm_fp16_to_fp32(h));
        const __m128i packed = _mm_loadu_si128((const __m128i *) (s + 2));

        const __m128i lo = _mm_and_si128(packed, mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);

        /*
         * Interleave before converting.
         *
         * The output order is low(nibble of byte 0), high(byte 0),
         * low(byte 1), high(byte 1), ... so the two nibble planes have to be
         * woven back together. Grouping all the lows and then all the highs
         * produces thirty-two entirely plausible numbers in entirely the wrong
         * order, and only a comparison against the definition catches it.
         */
        const __m128i i0 = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), bias);
        const __m128i i1 = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), bias);

        _mm256_storeu_ps(dst + i +  0, _mm256_mul_ps(vd, i8x8_to_ps(_mm_loadl_epi64((const __m128i *) &i0))));
        _mm256_storeu_ps(dst + i +  8, _mm256_mul_ps(vd, i8x8_to_ps(_mm_srli_si128(i0, 8))));
        _mm256_storeu_ps(dst + i + 16, _mm256_mul_ps(vd, i8x8_to_ps(_mm_loadl_epi64((const __m128i *) &i1))));
        _mm256_storeu_ps(dst + i + 24, _mm256_mul_ps(vd, i8x8_to_ps(_mm_srli_si128(i1, 8))));

        s += SLLM_QK_4_0 / 2 + 2;
    }
    if (i < n) {
        const uint16_t h = (uint16_t) (s[0] | ((uint16_t) s[1] << 8));
        const float d = sllm_fp16_to_fp32(h);
        for (size_t j = 0; i + j < n; ++j) {
            const uint8_t byte = s[2 + j / 2];
            const int v = (j % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
            dst[i + j] = d * (float) (v - 8);
        }
    }
}

static void dequant_f16(const void * src, float * dst, size_t n) {
    const uint8_t * s = (const uint8_t *) src;
    size_t i = 0;
#if defined(SLLM_X86)
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *) (s + i * 2));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = sllm_fp16_to_fp32((sllm_fp16) (s[i * 2] | ((uint16_t) s[i * 2 + 1] << 8)));
    }
}
#endif /* SLLM_X86 */

sllm_status sllm_dequant_row(sllm_ggml_type type, const void * src,
                             float * dst, size_t n) {
    if (src == NULL || (dst == NULL && n > 0)) {
        return SLLM_ERR_ARG;
    }

    switch (type) {
        case SLLM_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            return SLLM_OK;

        case SLLM_TYPE_F16:
#if defined(SLLM_X86)
            dequant_f16(src, dst, n);
#else
            for (size_t i = 0; i < n; ++i) {
                const uint8_t * s = (const uint8_t *) src;
                dst[i] = sllm_fp16_to_fp32((sllm_fp16) (s[i * 2] | ((uint16_t) s[i * 2 + 1] << 8)));
            }
#endif
            return SLLM_OK;

        case SLLM_TYPE_BF16: {
            const uint8_t * s = (const uint8_t *) src;
            for (size_t i = 0; i < n; ++i) {
                dst[i] = sllm_bf16_to_fp32((uint16_t) (s[i * 2] | ((uint16_t) s[i * 2 + 1] << 8)));
            }
            return SLLM_OK;
        }

        case SLLM_TYPE_Q8_0: {
#if defined(SLLM_X86)
            dequant_q8_0(src, dst, n);
#else
            const uint8_t * s = (const uint8_t *) src;
            for (size_t i = 0; i < n; i += SLLM_QK_8_0) {
                const float d = sllm_fp16_to_fp32((sllm_fp16) (s[0] | ((uint16_t) s[1] << 8)));
                const int8_t * q = (const int8_t *) (s + 2);
                for (size_t j = 0; j < SLLM_QK_8_0 && i + j < n; ++j) {
                    dst[i + j] = d * (float) q[j];
                }
                s += SLLM_QK_8_0 + 2;
            }
#endif
            return SLLM_OK;
        }

        /* K-quants. These are the two types a Q4_K_M mainstream model actually
         * needs, and they are the first types implemented against a REFERENCE
         * golden rather than against Saphira's own output. */
        case SLLM_TYPE_Q4_K:
            if (n % QK_K != 0) { return SLLM_ERR_ARG; }
            dequant_q4_K(src, dst, n);
            return SLLM_OK;

        case SLLM_TYPE_Q6_K:
            if (n % QK_K != 0) { return SLLM_ERR_ARG; }
            dequant_q6_K(src, dst, n);
            return SLLM_OK;

        case SLLM_TYPE_Q4_0: {
#if defined(SLLM_X86)
            dequant_q4_0(src, dst, n);
#else
            const uint8_t * s = (const uint8_t *) src;
            for (size_t i = 0; i < n; i += SLLM_QK_4_0) {
                const float d = sllm_fp16_to_fp32((sllm_fp16) (s[0] | ((uint16_t) s[1] << 8)));
                const uint8_t * q = s + 2;
                for (size_t j = 0; j < SLLM_QK_4_0 && i + j < n; ++j) {
                    const uint8_t byte = q[j / 2];
                    const int v = (j % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
                    dst[i + j] = d * (float) (v - 8);
                }
                s += SLLM_QK_4_0 / 2 + 2;
            }
#endif
            return SLLM_OK;
        }

        case SLLM_TYPE_I2_S:
            sllm_i2s_row(src, dst, n);
            return SLLM_OK;

        default:
            /*
             * A valid GGUF type we have not implemented. Said plainly, so the
             * user learns which format is missing rather than that the file
             * looks wrong.
             */
            return SLLM_ERR_TYPE_UNSUPPORTED;
    }
}
