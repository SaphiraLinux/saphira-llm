/*
 * ops.c — vector operations for the transformer forward pass.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Every operation here is a real x86-64-v3 path: the baseline is not a
 * fallback tier, it is the thing that runs. Each has a plain scalar tail so
 * odd lengths are exact rather than approximately right.
 *
 * The scalar form of each kernel is also what the golden-vector tests compare
 * against, so the tests do not merely confirm that two SIMD kernels agree
 * with each other.
 */

#include <saphira_llm/ops.h>
#include <stdlib.h>
#include <saphira_llm/quant.h>
#include <saphira_llm/log.h>

#if defined(__x86_64__)
#  include <immintrin.h>
#  define SLLM_X86 1
#endif

#include <math.h>
#include <string.h>

sllm_status sllm_rope_type_parse(const char * name, sllm_rope_type * out) {
    if (name == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    if (strcmp(name, "neox") == 0) {
        *out = SLLM_ROPE_NEOX;
        return SLLM_OK;
    }
    if (strcmp(name, "normal") == 0) {
        *out = SLLM_ROPE_NORMAL;
        return SLLM_OK;
    }
    return SLLM_ERR_ARG;
}

/* ------------------------------------------------------------------ */
/* expf                                                                */
/* ------------------------------------------------------------------ */

float sllm_fast_exp(float x) {
    return expf(x);
}

/* ------------------------------------------------------------------ */
/* F16 weight against an F32 activation: the tied output projection     */
/* ------------------------------------------------------------------ */

/*
 * This kernel exists because a profile said it had to, and the profile was
 * not what the plan expected.
 *
 * The Phase 6 brief named the AVX-VNNI ternary GEMM as the primary kernel
 * work, on the reasonable grounds that it is the model's whole distinctive
 * arithmetic. Sampling the decode path said otherwise: the ternary dot product
 * was 12 percent of the time and this loop was 86 percent, one line of
 * src/forward.c, and the reason was not the multiply-accumulate at all. It was
 * the half-to-float conversion. `f16_to_f32` was a software function that
 * unpacks the exponent, the mantissa and the sign by hand and spins in a while
 * loop for subnormals, and it was being called 328 million times per token,
 * once per element of 128256 vocabulary rows by 2560 dimensions.
 *
 * The fix is not an invention. It is the reference's own kernel: ggml's
 * GGML_F16_VEC on x86 is eight floats wide, loaded with _mm256_cvtph_ps, and
 * ggml_vec_dot_f16_unroll runs it over GGML_F16_STEP (32) elements with
 * GGML_F16_ARR (4) accumulators, reduced by a binary tree. Those constants are
 * the reference's, and so is the reduction order, which is deliberate: the
 * Phase 4 contract already records that our lm_head reduction order differs
 * from upstream's and is governed by the margin gate rather than by bit
 * equality, so matching upstream's shape is the direction that can only help.
 *
 * What is deliberately NOT changed is the arithmetic. The reference narrows the
 * activation to F16 before this kernel runs (F16's vec_dot_type is F16); we do
 * not, and Phase 4 accepted that as-is. The product is still F16 weight times
 * F32 activation accumulated in F32. Only the order of the additions moves.
 */

#define SLLM_F16_STEP 32
#define SLLM_F16_ARR  4

/* Exposed so a test can hold the vectorised path to the portable one. */
float sllm_dot_f16_f32_scalar(const uint16_t * row, const float * x, size_t n) {
    float acc = 0.0f;
    for (size_t d = 0; d < n; ++d) {
        acc += sllm_fp16_to_fp32((sllm_fp16) row[d]) * x[d];
    }
    return acc;
}

float sllm_dot_f16_f32(const uint16_t * row, const float * x, size_t n) {
    if (row == NULL || x == NULL) {
        return 0.0f;
    }
    size_t d = 0;
#if defined(SLLM_X86)
    /*
     * Four independent accumulators. One would be simpler and would serialise
     * on a four-cycle FMA latency across all eight lanes, which is most of the
     * gain thrown away; four match the reference's GGML_F16_ARR.
     */
    __m256 acc[SLLM_F16_ARR];
    for (int k = 0; k < SLLM_F16_ARR; ++k) { acc[k] = _mm256_setzero_ps(); }

    if (n >= SLLM_F16_STEP) {
        const size_t np = n & ~(size_t) (SLLM_F16_STEP - 1);
        for (; d < np; d += SLLM_F16_STEP) {
            for (int j = 0; j < SLLM_F16_ARR; ++j) {
                const __m128i h = _mm_loadu_si128((const __m128i *) (row + d + (size_t) j * 8));
                const __m256  w = _mm256_cvtph_ps(h);
                const __m256  a = _mm256_loadu_ps(x + d + (size_t) j * 8);
                acc[j] = _mm256_fmadd_ps(w, a, acc[j]);
            }
        }
    }

    /* ggml's reduction: halve the accumulator count, adding pairs, then fold
     * the 256-bit lanes down. Same tree, same order, so the rounding matches
     * the reference rather than merely resembling it. */
    for (int off = SLLM_F16_ARR / 2; off >= 1; off >>= 1) {
        for (int i = 0; i < off; ++i) {
            acc[i] = _mm256_add_ps(acc[i], acc[off + i]);
        }
    }
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(acc[0]),
                                 _mm256_extractf128_ps(acc[0], 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    const float head = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

    /* Tail, in the portable order, added onto the reduced head. */
    float acc_scalar = head;
    for (; d < n; ++d) {
        acc_scalar += sllm_fp16_to_fp32((sllm_fp16) row[d]) * x[d];
    }
    return acc_scalar;
#else
    return sllm_dot_f16_f32_scalar(row, x, n);
#endif
}

/* ------------------------------------------------------------------ */
/* add / mul                                                           */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Step 2: matrix-vector over stored (quantised) rows                   */
/* ------------------------------------------------------------------ */

/*
 * The first f32 GEMV. It is deliberately BORING: a plain scalar loop, no intrinsics,
 * no blocking, no threading, no fused scale application. That is an intentional
 * choice and the reason is falsifiability.
 *
 * Everything optimisable in this function is also where a dequantised GEMV
 * silently goes wrong: a stride computed once and reused, an accumulator reset in
 * the wrong scope, a block boundary that happens to line up for the first block
 * and not the others. Those bugs are invisible when the loop is clever and obvious
 * when it is not. The optimised version comes after this one is gated against an
 * external witness, and it will be required to agree with THIS one bit-for-bit, so
 * a plain reference always exists to diff against.
 *
 * `n` is the number of elements in ONE row, i.e. tensor ne[0]; `n_rows` is the
 * count of stored rows. The caller supplies the mapped tensor payload and this
 * function knows nothing about GGUF or offsets, so it cannot get them wrong on the
 * caller's behalf.
 *
 * A row must be a whole number of stored blocks. That is CHECKED rather than
 * assumed: a length that is not a multiple of the block size cannot be decoded, and
 * truncating it would produce a plausible number from a tensor the caller did not
 * intend.
 */
sllm_status sllm_gemv_f32(sllm_ggml_type type, const void * data, size_t n,
                          const float * x, size_t n_rows, float * out) {
    if (data == NULL || x == NULL || out == NULL) { return SLLM_ERR_ARG; }
    if (n == 0 || n_rows == 0) { return SLLM_ERR_ARG; }

    uint32_t blck = 0, tsz = 0;
    const sllm_status ts = sllm_gguf_type_traits(type, &blck, &tsz);
    if (ts != SLLM_OK) { return ts; }
    if (blck == 0 || n % blck != 0) { return SLLM_ERR_ARG; }

    const size_t blocks_per_row = n / blck;
    const size_t row_bytes = (size_t) tsz * blocks_per_row;

    /* Scratch for one decoded block, allocated once and reused for every block of
     * every row, so the steady state allocates nothing. A stack buffer would be
     * marginally faster and would put a hidden size limit on the block size. */
    float * buf = (float *) malloc(sizeof(float) * blck);
    if (buf == NULL) { return SLLM_ERR_NOMEM; }

    for (size_t r = 0; r < n_rows; ++r) {
        const uint8_t * row = (const uint8_t *) data + r * row_bytes;
        float acc = 0.0f;
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const sllm_status ds =
                sllm_dequant_row(type, row + b * (size_t) tsz, buf, blck);
            if (ds != SLLM_OK) { free(buf); return ds; }
            for (size_t i = 0; i < blck; ++i) {
                acc += buf[i] * x[b * blck + i];
            }
        }
        out[r] = acc;
    }

    free(buf);
    return SLLM_OK;
}

void sllm_add(float * dst, const float * a, const float * b, size_t n) {
    size_t i = 0;
#if defined(SLLM_X86)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
}

void sllm_mul(float * dst, const float * a, const float * b, size_t n) {
    size_t i = 0;
#if defined(SLLM_X86)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
}

/* ------------------------------------------------------------------ */
/* RMSNorm                                                             */
/* ------------------------------------------------------------------ */

void sllm_rms_norm(float * dst, const float * x, const float * weight,
                   size_t n, float eps) {
    /* Sum of squares first. Accumulated in double because a 4096-element f32
     * sum of squares loses several bits, and the reference accumulates the same
     * way; matching the reference matters more here than the last bit. */
    double ss = 0.0;
    size_t i = 0;
#if defined(SLLM_X86)
    __m256d acc = _mm256_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        /*
         * Square in f32, widen to f64, accumulate in f64.
         *
         * The square is not optional. An earlier version of this loop
         * accumulated the raw values, so the "sum of squares" was a plain sum;
         * the mean could then come out near zero or negative, the square root
         * went imaginary, and the result was silently nonsense for large n
         * while looking fine for small n. The golden vector caught it at
         * n = 186.
         *
         * Accumulating in f64 matters at model scale: a 4096-element f32 sum
         * of squares has already lost several bits by the time it is scaled.
         */
        const __m256 sq = _mm256_mul_ps(v, v);
        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_castps256_ps128(sq)));
        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_extractf128_ps(sq, 1)));
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, acc);
    for (int k = 0; k < 4; ++k) {
        ss += tmp[k];
    }
#endif
    for (; i < n; ++i) {
        ss += (double) x[i] * (double) x[i];
    }

    /*
     * The narrowing order is taken from ggml_compute_forward_rms_norm_f32 and
     * is not negotiable:
     *
     *     const float mean  = sum/ne00;                    // narrows to f32
     *     const float scale = 1.0f/sqrtf(mean + eps);      // f32 sqrtf
     *
     * Computing sqrt in double and narrowing afterwards is arguably more
     * accurate and is off by about an ulp. That ulp is the difference between
     * a value landing either side of a rounding boundary when the next layer
     * quantises the activation to int8, so "more accurate" would be a parity
     * bug. The reference is the specification.
     */
    const float mean  = (float) (ss / (double) n);
    const float scale = 1.0f / sqrtf(mean + eps);

    i = 0;
#if defined(SLLM_X86)
    const __m256 vs = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(vs, _mm256_loadu_ps(x + i)));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = scale * x[i];
    }

    if (weight != NULL) {
        sllm_mul(dst, dst, weight, n);
    }
}

/*
 * expf, as the reference computes it.
 *
 * This is ggml_v_expf's AVX2 polynomial, transcribed. It is NOT libm's expf,
 * and the two differ by a couple of ulps per element.
 *
 * That matters more than it looks. expf feeds softmax and SiLU, both of which
 * feed the activation quantiser, and the model is thirty stages of int8
 * quantisation deep. A one-ulp difference in an exponential is a systematic
 * bias across every element of the softmax, not noise that averages out, and it
 * is amplified at every layer boundary. Using libm here was the largest
 * remaining contributor to the Phase 4 logit gap.
 *
 * The code is deliberately a transcription rather than a rewrite: the exact
 * sequence of fused multiply-adds is the specification, and reassociating any
 * of it changes the result.
 */
#if defined(SLLM_X86)
static inline __m256 ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}
#endif /* SLLM_X86 */

/* ------------------------------------------------------------------ */
/* softmax                                                             */
/* ------------------------------------------------------------------ */

void sllm_softmax_inplace(float * x, size_t n) {
    if (n == 0) {
        return;
    }
    float max = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    double sum = 0.0;
    size_t i = 0;
#if defined(SLLM_X86)
    /* The reference's exp, its 8-wide reduction, and the same double sum in the
     * same order. ggml_vec_soft_max_f32 adds the per-vector partial sums into a
     * ggml_float (double) accumulator, and the reciprocal is 1.0/sum in double
     * before being narrowed to float -- which is what happens here. */
    for (; i + 7 < n; i += 8) {
        const __m256 v = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                                   _mm256_set1_ps(max)));
        _mm256_storeu_ps(x + i, v);
        double t[4];
        _mm256_storeu_pd(t, _mm256_add_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(v)),
                                          _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1))));
        sum += t[0] + t[1] + t[2] + t[3];
    }
#endif
    for (; i < n; ++i) {
        const float e = expf(x[i] - max);
        x[i] = e;
        sum += (double) e;
    }
    const float inv = (float) (1.0 / sum);
    for (size_t i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

/* ------------------------------------------------------------------ */
/* SiLU                                                                */
/* ------------------------------------------------------------------ */

void sllm_silu_inplace(float * x, size_t n) {
    /*
     * x / (1 + exp(-x)), in both paths.
     *
     * The vector form here originally computed x * exp(x), which is not SiLU
     * at all: it is the product of the input with the exponential rather than
     * with the logistic. The golden-vector test caught it at the first
     * element, and the discrepancy was enormous (164.9 against 3.69) precisely
     * because the two formulas agree nowhere except near zero.
     *
     * The vector path uses the reference's polynomial exp, ggml_v_expf, and the
     * same x / (1 + exp(-x)) shape as ggml_v_silu. Using libm's expf here was
     * a systematic bias across all 6912 elements of every layer's FFN, which
     * the int8 activation quantiser then amplified at every layer boundary.
     * It is not the only thing that was wrong -- see PHASE4-GAP.md -- but it is
     * the largest single contributor, and it is free to fix.
     *
     * The loop bound is `i + 7 < n`, matching the reference's, so the same
     * elements go through the same code. A different bound would leave a tail
     * computed by a different expression, which is a subtle way to be wrong
     * only at the end of an array.
     */
    size_t i = 0;
#if defined(SLLM_X86)
    for (; i + 7 < n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 e = ggml_v_expf(_mm256_sub_ps(_mm256_setzero_ps(), v));
        _mm256_storeu_ps(x + i, _mm256_div_ps(v, _mm256_add_ps(_mm256_set1_ps(1.0f), e)));
    }
#endif
    for (; i < n; ++i) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* ------------------------------------------------------------------ */
/* RoPE                                                                */
/* ------------------------------------------------------------------ */

void sllm_rope_inplace(float * x, size_t n_rot, int32_t pos,
                       float theta, float freq_scale, sllm_rope_type type) {
    if (n_rot < 2) {
        return;
    }
    if (n_rot % 2 != 0) {
        /* An odd rotary dimension has no well-defined pairing. Half of it is
         * still rotatable and the reference leaves the rest alone. */
        n_rot -= 1;
    }

    /*
     * Two layouts, and they are not interchangeable:
     *
     *   NEOX   pairs the two halves:  (0, h), (1, h+1), ...
     *   NORMAL pairs adjacent pairs: (0,1), (2,3), (4,5), ...
     *
     * where h = n_rot/2. From ggml's rotate_pairs, called with
     * (n_dims, n_dims/2) for NEOX and (n_dims, 1, scale=1) for NORMAL:
     * NEOX reads src[ic] against src[ic + n/2] with ic = i0/2, NORMAL reads
     * src[i0] against src[i0+1] with i0 stepping by two.
     *
     * These two were swapped here, and the golden vector could not see it: the
     * test asserts the rotation preserves each pair's norm, and BOTH pairings
     * preserve their pairs' norms perfectly. It is a norm-preserving relabelling
     * of the whole vector, so every element looks right and the relative phase
     * between dimensions -- which is the entire content of the encoding -- is
     * destroyed. The only way to catch it is to compare against the reference
     * on real data, which is what the Phase 4 logit gate is for.
     *
     * Note the argument order. The reference computes theta_i = base^(-2i/n)
     * and then scales the position, so the equivalent form is to divide theta
     * by freq_scale. Reversing that is invisible at position 0 and wrong
     * everywhere else.
     */
    const size_t half = n_rot / 2;

    /*
     * The angle is built by REPEATED MULTIPLICATION, exactly as the reference
     * builds its rope cache, and not as pos * powf(theta, -2k/n).
     *
     *   ggml_rope_cache_init, ops.cpp:5804
     *       float theta = theta_base;              // theta_base is the position
     *       for (i0 = 0; i0 < ne0; i0 += 2) {
     *           rope_yarn(theta/ff, ...);          // theta = freq_scale * theta
     *           theta *= theta_scale;             // theta_scale = powf(freq_base, -2/n_dims)
     *       }
     *
     * So the k-th angle is pos * (powf(freq_base, -2/n))^k, accumulated one
     * multiplication at a time. Computing it directly as
     * pos * powf(freq_base, -2k/n) is a different float: same mathematics,
     * different rounding, and the difference is a systematic bias in every
     * rotated dimension rather than noise.
     *
     * That bias is what the Phase 4 logit gap turned out to be. Q and K come
     * out of the integer projections exactly, so a one-ulp angle error here is
     * the only error entering the residual stream, and the 30 stages of int8
     * quantisation downstream amplify it. Running the attention and lm_head in
     * double precision moved the worst-case logit deviation by 0.03 out of
     * 0.92, which is what identified this as a systematic error rather than
     * float reordering.
     */
    const float theta_scale = powf(theta, -2.0f / (float) n_rot);
    float theta_k = (float) pos;

    for (size_t k = 0; k < half; ++k) {
        const size_t a = (type == SLLM_ROPE_NEOX) ? k : (k * 2);
        const size_t b = (type == SLLM_ROPE_NEOX) ? (k + half) : (k * 2 + 1);
        if (b >= n_rot) {
            continue;
        }

        /* rope_yarn: theta = freq_scale * theta_extrap, and with no ext_factor
         * the ramp is skipped, so this is the whole transform. */
        const float f = freq_scale * theta_k;
        const float c = cosf(f);
        const float s = sinf(f);
        theta_k *= theta_scale;

        const float xa = x[a];
        const float xb = x[b];
        x[a] = xa * c - xb * s;
        x[b] = xa * s + xb * c;
    }
}

/* ------------------------------------------------------------------ */
/* get_rows                                                            */
/* ------------------------------------------------------------------ */

void sllm_get_rows(float * dst, const float * src, const int32_t * idx,
                   size_t n_rows, size_t row_floats) {
    for (size_t i = 0; i < n_rows; ++i) {
        const int32_t r = idx[i];
        if (r < 0) {
            /* An index from a model is input, not a promise. Clamping is
             * safer than reading wherever the number points. */
            sllm_log(SLLM_LOG_WARN, "get_rows: negative row index %d, clamped to 0", (int) r);
            memcpy(dst + i * row_floats, src, row_floats * sizeof(float));
            continue;
        }
        memcpy(dst + i * row_floats, src + (size_t) r * row_floats, row_floats * sizeof(float));
    }
}
