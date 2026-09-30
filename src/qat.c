/*
 * qat.c — tiny-model quantisation-aware training, for proving the lifecycle.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA Limited.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * THE GRADIENT TREATMENT IS EXPLICIT, and it is the point of the file.
 *
 * Two differentiable-looking things sit in this forward pass, and both have
 * zero derivative in the ordinary sense:
 *
 *   1. The weight ternary. The master weight is BF16 and real; the value used
 *      in the forward is quantised to {-1, 0, +1}. d/dw of that is zero almost
 *      everywhere. So the gradient of the LOSS WITH RESPECT TO THE QUANTISED
 *      WEIGHT is applied directly to the master, unchanged. That is a
 *      straight-through estimator, and it is written out here as one rather
 *      than hidden behind a library call, because an STE whose placement is
 *      not visible is an STE you cannot reason about.
 *
 *   2. The activation quantiser. The training-time convention is the one in
 *      upstream's training code (gpu/model.py:80 and float_act_quant), which
 *      is NOT the inference convention: the absmax is computed in float and
 *      floored at 1e-5, and rounding is ROUND-HALF-TO-EVEN, not the roundf
 *      (half away from zero) the inference path uses. See
 *      docs/BITNET-LIFECYCLE.md.
 *
 * THE TRAIN/INFERENCE BOUNDARY IS EXPLICIT AND IS NOT PAPERED OVER.
 *
 * This file trains under the TRAINING convention. The exported model is
 * consumed under the INFERENCE convention, because that is what bitnet.cpp and
 * this project's runtime do. The two differ in rounding mode on exact halves
 * and in zero-row handling, and the resulting cost is MEASURED by the caller
 * rather than estimated here: the difference between the post-QAT BF16
 * evaluation and the exported I2_S evaluation IS that cost, and it is reported
 * as such.
 *
 * Inference is NOT changed to look like training. Doing so would silently move
 * the sealed v0.0.1 boundary to accommodate a new experiment, which is exactly
 * the wrong direction.
 *
 * The activation STE is also explicit: the gradient passes through the
 * quantiser unchanged for the same reason, and the multiply-by-scale is real
 * arithmetic with a real derivative that is applied.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <saphira_llm/qat.h>
#include <saphira_llm/gguf.h>
#include <saphira_llm/i2s_convert.h>
#include <saphira_llm/i2s_gemm.h>
#include <saphira_llm/ops.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QAT_MAGIC 0x51415431u   /* "QAT1" */
#define QAT_VERSION 1u

void sllm_qat_default_config(sllm_qat_config * cfg) {
    memset(cfg, 0, sizeof *cfg);
    /* n_embd=128, n_ff=256, and every I2_S contracted dimension is a multiple
     * of SLLM_I2S_QK (128):
     *   attn_q/k/v/attn_output, ffn_up, ffn_gate  -> ne[0] = n_embd  = 128
     *   ffn_down                                   -> ne[0] = n_ff    = 256
     * The 64-wide fixture this replaced was invalid and is the reason the
     * lifecycle never actually executed an I2_S network: sllm_i2s_dot returns 0
     * for n < 128, so every projection in that model returned zero and the
     * "finite logits" were (0 - act_sum) * (w_scale/act_scale). It loaded, it
     * ran, it was sanitizer-clean, and it was mathematically meaningless. */
    cfg->n_layer    = 2;
    cfg->n_embd     = 128;
    cfg->n_head     = 4;
    cfg->n_head_kv  = 2;
    cfg->n_ff       = 256;
    cfg->n_vocab    = 256;
    cfg->n_ctx      = 128;
    cfg->rms_eps    = 1e-5f;
    cfg->rope_base  = 10000.0f;
    cfg->n_embd_head = cfg->n_embd / cfg->n_head;
    cfg->n_embd_gqa  = cfg->n_embd_head * cfg->n_head_kv;
}

/* ------------------------------------------------------------------ */
/* parameters                                                          */
/* ------------------------------------------------------------------ */

/* Field order per layer, matching the order they are added in
 * sllm_qat_new: attn_norm, attn_q, attn_k, attn_v, attn_output,
 * attn_sub_norm, ffn_norm, ffn_up, ffn_gate, ffn_down, ffn_sub_norm.
 * The seven matrices are ternarised; the four norm vectors are not. */
enum { F_ATTN_NORM = 0, F_ATTN_Q, F_ATTN_K, F_ATTN_V, F_ATTN_OUT, F_ATTN_SUB_NORM,
       F_FFN_NORM, F_FFN_UP, F_FFN_GATE, F_FFN_DOWN, F_FFN_SUB_NORM, F_COUNT };



typedef struct qws qws;
static void qws_free(qws * w);
static qws * qws_new(const sllm_qat * q);

static sllm_qat_tensor * addp(sllm_qat * q, const char * name, int n, bool trainable) {
    sllm_qat_tensor * t = &q->p[q->n_p++];
    t->name = name;
    t->n = n;
    t->trainable = trainable;
    if (trainable) {
        t->master = (float *) calloc((size_t) n, sizeof(float));
        t->grad   = (float *) calloc((size_t) n, sizeof(float));
        t->adam_m = (float *) calloc((size_t) n, sizeof(float));
        t->adam_v = (float *) calloc((size_t) n, sizeof(float));
        if (!t->master || !t->grad || !t->adam_m || !t->adam_v) { return NULL; }
        q->n_trainable += n;
    } else {
        t->master = (float *) calloc((size_t) n, sizeof(float));
        if (!t->master) { return NULL; }
    }
    q->n_params += n;
    return t;
}

float sllm_qat_weight_scale(const sllm_qat_tensor * t) {
    double amax = 0.0;
    for (int i = 0; i < t->n; ++i) {
        const double a = fabs((double) t->master[i]);
        if (a > amax) { amax = a; }
    }
    return (float) (amax < 1e-5 ? 1e-5 : amax);
}

sllm_qat * sllm_qat_new(const sllm_qat_config * cfg, uint64_t seed) {
    (void) seed;
    sllm_qat_config c = *cfg;
    if (c.n_embd_head == 0) { c.n_embd_head = c.n_embd / c.n_head; }
    if (c.n_embd_gqa  == 0) { c.n_embd_gqa  = c.n_embd_head * c.n_head_kv; }
    /* These are the ordinary loader's constraints, checked here so a config that
     * could not be loaded is rejected at construction rather than at export. */
    if (c.n_embd % c.n_head != 0 || c.n_head % c.n_head_kv != 0) { return NULL; }

    const int E = c.n_embd, EK = c.n_embd_gqa, F = c.n_ff, V = c.n_vocab;
    /* 1 embedding + F_COUNT per layer + 1 output norm. F_COUNT is the single
     * source of truth for the per-layer field list; hardcoding a second count
     * here is how the arrays overflowed before. */
    const int n = 1 + (int) c.n_layer * F_COUNT + 1;
    sllm_qat * q = (sllm_qat *) calloc(1, sizeof *q);
    if (q == NULL) { return NULL; }
    q->cfg = c;
    q->ternary = true;
    q->p = (sllm_qat_tensor *) calloc((size_t) n, sizeof(sllm_qat_tensor));
    if (q->p == NULL) { free(q); return NULL; }
    /* Names are built at runtime and owned, rather than a fixed static array:
     * a static one silently truncates or overruns if the layer count ever grows
     * past what it was sized for. */
    q->names = (char *) calloc((size_t) n * 48, 1);
    if (q->names == NULL) { free(q->p); free(q); return NULL; }

    if (addp(q, "token_embd", E * V, true) == NULL) { sllm_qat_free(q); return NULL; }
    int li = 0;
    for (int l = 0; l < c.n_layer; ++l) {
        static const char * f[] = { "attn_norm", "attn_q", "attn_k", "attn_v",
                                     "attn_output", "attn_sub_norm",
                                     "ffn_norm", "ffn_up", "ffn_gate",
                                     "ffn_down", "ffn_sub_norm" };
        const int dims[] = { E, E*E, E*EK, E*EK, E*E, E, E, E*F, E*F, F*E, F };
        for (int k = 0; k < 11; ++k) {
            char * nm = q->names + (size_t) li * 48;
            snprintf(nm, 48, "blk.%d.%s", l, f[k]);
            if (addp(q, nm, dims[k], true) == NULL) { sllm_qat_free(q); return NULL; }
            ++li;
        }
    }
    if (addp(q, "output_norm", E, true) == NULL) { sllm_qat_free(q); return NULL; }
    return q;
}

void sllm_qat_free(sllm_qat * q) {
    if (q == NULL) { return; }
    for (int i = 0; i < q->n_p; ++i) {
        free(q->p[i].master); free(q->p[i].grad);
        free(q->p[i].adam_m); free(q->p[i].adam_v);
    }
    qws_free((qws *) q->ws);
    free(q->names);
    free(q->p);
    free(q);
}

const sllm_qat_tensor * sllm_qat_find(const sllm_qat * q, const char * name) {
    for (int i = 0; i < q->n_p; ++i) {
        if (strcmp(q->p[i].name, name) == 0) { return &q->p[i]; }
    }
    return NULL;
}

/*
 * THE RNG. splitmix64, because it is short, has a full 64-bit period, and --
 * the property that actually matters here -- produces the same stream on every
 * platform, so a run is reproducible from its seed rather than from the
 * platform's rand().
 */
static uint64_t rnd(sllm_qat * q) {
    uint64_t z = (q->rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double rnd_unit(sllm_qat * q) {
    /* 53 significand bits, so every value is exactly representable. */
    return (double) (rnd(q) >> 11) * (1.0 / 9007199254740992.0);
}

void sllm_qat_init(sllm_qat * q, uint64_t seed) {
    q->rng = seed;
    q->step = 0;
    for (int i = 0; i < q->n_p; ++i) {
        sllm_qat_tensor * t = &q->p[i];
        const double inv = 1.0 / sqrt((double) t->n);
        for (int k = 0; k < t->n; ++k) {
            /* Box-Muller from two uniforms, then scaled. Nonzero by
             * construction: a master that started at exactly zero would
             * ternarise to 0 forever, because d(sign)/dw is zero and the STE
             * would pass a gradient that could never move it off zero. */
            const double u1 = rnd_unit(q);
            const double u2 = rnd_unit(q);
            const double r  = sqrt(-2.0 * log(u1 + 1e-300));
            t->master[k] = (float) (r * cos(6.283185307179586 * u2) * inv);
        }
        if (t->trainable) {
            memset(t->grad, 0, (size_t) t->n * sizeof(float));
            memset(t->adam_m, 0, (size_t) t->n * sizeof(float));
            memset(t->adam_v, 0, (size_t) t->n * sizeof(float));
        }
    }
}

/* ------------------------------------------------------------------ */
/* the quantisers, with their two distinct conventions                  */
/* ------------------------------------------------------------------ */

/*
 * TRAINING-TIME activation quantiser, from upstream gpu/model.py:80:
 *   s = 127 / abs(x).max().clamp(min=1e-5)
 *   q = (x * s).round().clamp(-128, 127)
 *       .round() in torch is ROUND-HALF-TO-EVEN.
 *
 * The INFERENCE-TIME one, from quantize_row_i8_s (quants.c:1311), is
 * roundf (half AWAY from zero) with no 1e-5 floor. They are different
 * functions. This file implements the training one, because this is the
 * training path, and the difference is measured downstream rather than hidden.
 */
static int act_q_train(float x, float s, float * recip_scale) {
    (void) recip_scale;
    const float y = x * s;
    /* round-half-to-even, matching torch's round() and rint under the default
     * rounding mode. Written as an explicit floor so it does not depend on the
     * FPU's current rounding mode, which would make the run irreproducible. */
    const float f = floorf(y);
    const float d = y - f;
    float r;
    if (d > 0.5f)      { r = f + 1.0f; }
    else if (d < 0.5f) { r = f; }
    else               { r = (fmodf(f, 2.0f) == 0.0f) ? f : f + 1.0f; }
    int v = (int) r;
    if (v >  127) { v =  127; }
    if (v < -128) { v = -128; }
    return v;
}

/*
 * The weight ternary, and the STE.
 *
 * `q` is the value used in the forward pass. The gradient that arrives for it
 * is written straight into the master's gradient with no scaling and no
 * modification. That is the estimator, stated in one line so it can be checked
 * rather than trusted.
 */
static float ternarise(float w, double scale) {
    const double t = (double) w * (1.0 / scale);
    /*
     * THE VALUE IS +-SCALE, NOT +-1. This is the whole Step 6 fix, and it is
     * one line, because the two conventions differ by exactly one per-tensor
     * factor and that factor is the whole bug.
     *
     * The ternary carries the SIGN; the per-tensor scale carries the MAGNITUDE.
     * That is not this project's invention, it is the deployed format:
     *
     *   - the reference dequantiser writes y = i2_scale * map2bit[c], with
     *     map2bit = {-1, 0, +1, 0} (quants.c, dequantize_row_i2_s:1349), so
     *     the effective weight is +-absmax and never +-1;
     *   - this runtime's epilogue agrees independently:
     *     (dot - act_sum) * (w_scale / act_scale), and since
     *     (dot - act_sum) is already the sign-carrying sum, the weight that
     *     multiplies the activation is w_scale (i2s_gemm.c:271).
     *
     * Training with +-1 while deploying with +-absmax is a factor of absmax per
     * tensor, and because the absmax GROWS during training (measured: 0.47 to
     * 4.07 over 200 steps) the factor grows with it. The symptom is not noisy
     * degradation, it is divergence: the trainer's NLL fell 5.54 -> 5.2e-6
     * while the exported model through the ordinary runtime rose 5.88 -> 17.36
     * -> 18.49 at 0/200/600 steps. A gap that widens monotonically cannot be
     * quantisation noise, and this is its cause.
     *
     * Note what does NOT change here: the packed CODES are identical, because
     * the threshold and the scale are untouched. The exported file is
     * byte-for-byte what it was; only the value the trainer multiplies by
     * changes, to the value the runtime already reconstructs.
     */
    if (t >  0.5) { return (float)  scale; }
    if (t < -0.5) { return (float) -scale; }
    return 0.0f;
}

/* ------------------------------------------------------------------ */
/* the workspace: everything in memory, nothing spilled to disk         */
/* ------------------------------------------------------------------ */

struct qws {
    const sllm_qat * q;
    int T, E, H, HK, HD, EK, F, V, L;
    /* per-layer caches for backward */
    float * x;          /* [L+1][T][E] residual stream            */
    float * n1, * qh, * kh, * vh, * attn, * ao, * n2, * fg, * fu, * fh, * fy;
    float * scores;     /* [H][T][T] */
    float * d_x, * d_n1, * d_qh, * d_kh, * d_vh, * d_attn, * d_ao, * d_n2;
    float * d_fg, * d_fu, * d_fh, * d_fy, * d_fo, * d_fhq, * d_ao_pre;
    float * d_scores;
    float * logits;      /* [T][V] */
    float * probs;       /* [T][V] */
    float * d_logits;    /* [T][V] */
    /* quantised activation inputs, kept for the weight gradient */
    float * n1q, * attnq, * n2q, * fhq, * fo, * xmid;
    float * ao_pre;                    /* input to attn_sub_norm */
    /* Cached 1/rms for the backward, PER POSITION: rinv[slot * T + t].
     * It is not one scalar per norm. rms(x) depends on the position's own
     * activations, so a single cached value is the last position's inverse
     * applied to all of them -- which is a wrong gradient for every position
     * but the last, and it makes the norm weights look untrainable while every
     * other parameter looks merely noisy. */
    float * rinv;
    /* quantised weight caches: [param][element] */
    float * wq;
    int    * wq_rows;
    int wq_off;
    int wq_cap;
};

static float * qws_get(qws * w, int n) {
    if (w->wq_off + n > w->wq_cap) { return NULL; }
    float * p = w->wq + w->wq_off;
    w->wq_off += n;
    return p;
}

static qws * qws_new(const sllm_qat * q) {
    qws * w = (qws *) calloc(1, sizeof *w);
    if (w == NULL) { return NULL; }
    w->q = q;
    w->T = q->cfg.n_ctx; w->E = q->cfg.n_embd; w->H = q->cfg.n_head;
    w->HK = q->cfg.n_head_kv; w->HD = q->cfg.n_embd_head;
    w->EK = q->cfg.n_embd_gqa; w->F = q->cfg.n_ff; w->V = q->cfg.n_vocab;
    w->L = q->cfg.n_layer;

    /* Two regions: the forward's ternarised weights stay valid for the
     * backward, and the backward rebuilds its own copy rather than trying to
     * walk the forward's arena backwards. The model is tiny, so the second
     * copy costs nothing and the lifetime problem disappears. */
    w->wq_cap = 2 * (int) q->n_params;
    w->wq = (float *) calloc((size_t) w->wq_cap, sizeof(float));
    w->wq_rows = (int *) calloc((size_t) q->n_p, sizeof(int));
    if (!w->wq || !w->wq_rows) { free(w->wq); free(w->wq_rows); free(w); return NULL; }

    const size_t TE = (size_t) w->T * w->E;
    const size_t TF = (size_t) w->T * w->F;
    const size_t TT = (size_t) w->T * w->T;
    const size_t TV = (size_t) w->T * w->V;
    w->x  = (float *) calloc(((size_t) w->L + 1) * TE, sizeof(float));
    w->n1 = (float *) calloc(TE, sizeof(float));
    w->qh = (float *) calloc(TE, sizeof(float));
    w->kh = (float *) calloc((size_t) w->T * w->EK, sizeof(float));
    w->vh = (float *) calloc((size_t) w->T * w->EK, sizeof(float));
    w->attn = (float *) calloc(TE, sizeof(float));
    w->ao  = (float *) calloc(TE, sizeof(float));
    w->n2  = (float *) calloc(TE, sizeof(float));
    w->fg  = (float *) calloc(TF, sizeof(float));
    w->fu  = (float *) calloc(TF, sizeof(float));
    w->fh  = (float *) calloc(TF, sizeof(float));
    w->fy  = (float *) calloc(TF, sizeof(float));
    w->scores  = (float *) calloc((size_t) w->H * TT, sizeof(float));
    w->logits  = (float *) calloc(TV, sizeof(float));
    w->probs   = (float *) calloc(TV, sizeof(float));
    w->d_logits= (float *) calloc(TV, sizeof(float));
    w->d_x  = (float *) calloc(((size_t) w->L + 1) * TE, sizeof(float));
    w->d_n1 = (float *) calloc(TE, sizeof(float));
    w->d_qh = (float *) calloc(TE, sizeof(float));
    w->d_kh = (float *) calloc((size_t) w->T * w->EK, sizeof(float));
    w->d_vh = (float *) calloc((size_t) w->T * w->EK, sizeof(float));
    w->d_attn = (float *) calloc(TE, sizeof(float));
    w->d_ao  = (float *) calloc(TE, sizeof(float));
    w->d_n2 = (float *) calloc(TE, sizeof(float));
    w->d_fg = (float *) calloc(TF, sizeof(float));
    w->d_fu = (float *) calloc(TF, sizeof(float));
    w->d_fh = (float *) calloc(TF, sizeof(float));
    w->d_fy = (float *) calloc(TF, sizeof(float));
    w->d_fo = (float *) calloc(TE, sizeof(float));
    w->d_fhq= (float *) calloc(TF, sizeof(float));
    w->d_ao_pre = (float *) calloc(TE, sizeof(float));
    w->d_scores = (float *) calloc((size_t) w->H * TT, sizeof(float));
    w->n1q   = (float *) calloc(TE, sizeof(float));
    w->attnq = (float *) calloc(TE, sizeof(float));
    w->n2q   = (float *) calloc(TE, sizeof(float));
    w->fhq   = (float *) calloc(TF, sizeof(float));
    w->fo    = (float *) calloc(TE, sizeof(float));
    w->xmid  = (float *) calloc(TE, sizeof(float));
    w->ao_pre= (float *) calloc(TE, sizeof(float));
    w->rinv  = (float *) calloc(((size_t) w->L * 4 + 1) * w->T, sizeof(float));
    if (!w->x||!w->n1q||!w->attnq||!w->n2q||!w->fhq||!w->fo||!w->xmid||!w->ao_pre||!w->rinv ||
        !w->n1||!w->qh||!w->kh||!w->vh||!w->attn||!w->ao||!w->n2 ||
        !w->fg||!w->fu||!w->fh||!w->fy||!w->scores||!w->logits||!w->probs||
        !w->d_logits||!w->d_x||!w->d_n1||!w->d_qh||!w->d_kh||!w->d_vh||
        !w->d_attn||!w->d_ao||!w->d_n2||!w->d_fg||!w->d_fu||!w->d_fh||!w->d_fy||
        !w->d_fo||!w->d_fhq||!w->d_ao_pre ||
        !w->d_scores) {
        qws_free(w);
        return NULL;
    }
    return w;
}

static void qws_free(qws * w) {
    if (w == NULL) { return; }
    free(w->x); free(w->n1); free(w->qh); free(w->kh); free(w->vh);
    free(w->attn); free(w->ao); free(w->n2); free(w->fg); free(w->fu);
    free(w->fh); free(w->fy); free(w->scores); free(w->logits); free(w->probs);
    free(w->d_logits); free(w->d_x); free(w->d_n1); free(w->d_qh); free(w->d_kh);
    free(w->d_vh); free(w->d_attn); free(w->d_ao); free(w->d_n2); free(w->d_fg);
    free(w->d_fu); free(w->d_fh); free(w->d_fy); free(w->d_scores);
    free(w->d_fo); free(w->d_fhq); free(w->d_ao_pre);
    free(w->n1q); free(w->attnq); free(w->n2q); free(w->fhq); free(w->fo); free(w->xmid); free(w->ao_pre);
    free(w->rinv);
    free(w->wq); free(w->wq_rows);
    free(w);
}

/* ------------------------------------------------------------------ */
/* parameter indexing                                                   */
/* ------------------------------------------------------------------ */

/* Field order per layer, matching the order they were added in
 * sllm_qat_new: attn_norm, attn_q, attn_k, attn_v, attn_output,
 * attn_sub_norm, ffn_norm, ffn_up, ffn_gate, ffn_down, ffn_sub_norm. */
/* The seven matrices per layer that are ternarised. The four norm vectors are
 * not: they are stored and run in F32, and the tied embedding is F16. */
static bool field_is_ternary(int k) {
    return k == F_ATTN_Q || k == F_ATTN_K || k == F_ATTN_V || k == F_ATTN_OUT ||
           k == F_FFN_UP || k == F_FFN_GATE || k == F_FFN_DOWN;
}

static sllm_qat_tensor * P(sllm_qat * q, int layer, int field) {
    return &q->p[1 + layer * F_COUNT + field];
}

/* ------------------------------------------------------------------ */
/* local RMSNorm: forward identical to sllm_rms_norm, plus a backward   */
/* ------------------------------------------------------------------ */

/*
 * The forward here is the same arithmetic as sllm_rms_norm in ops.c, written
 * out again only because there is no backward for it and this file needs one.
 * It is a duplicate on purpose and a tracked one: the two are held to the same
 * result by a test in tests/test_qat.c, so a drift in either is a test failure
 * rather than a silent disagreement between what we train and what we run.
 * `rsq` is the 1/sqrt(mean square + eps) and is cached for the backward.
 */
static void rms_fwd(float * out, const float * x, const float * w, int n,
                    float eps, float * inv_out) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) { ss += (double) x[i] * (double) x[i]; }
    const float inv = (float) (1.0 / sqrt(ss / n + (double) eps));
    for (int i = 0; i < n; ++i) { out[i] = x[i] * inv * w[i]; }
    if (inv_out) { *inv_out = inv; }
}

static void rms_bwd(float * dx, const float * dy, const float * x, const float * w,
                    int n, float inv) {
    /* y_i = x_i * inv * w_i, so dL/dx_i = dy_i*w_i*inv - x_i*c with
     *   c = (1/n) * inv^3 * sum_j dy_j*w_j*x_j
     * The 1/n is written literally. An earlier version cached n*inv^2 and
     * folded it in here, which was wrong by a factor of n -- and because inv
     * is 1/sqrt(ms+eps) with eps=1e-5, a small activation gives inv ~ 300 and
     * inv^3 ~ 3e7, so the error showed up as gradients of order 1e23. */
    double dot = 0.0;
    for (int i = 0; i < n; ++i) { dot += (double) dy[i] * (double) w[i] * (double) x[i]; }
    const float c = (float) (dot * (double) inv * (double) inv * (double) inv / (double) n);
    for (int i = 0; i < n; ++i) { dx[i] = dy[i] * w[i] * inv - x[i] * c; }
}

/* ------------------------------------------------------------------ */
/* quantised activation input, with the activation STE                  */
/* ------------------------------------------------------------------ */

/*
 * Quantise a row of activations under the TRAINING convention and write the
 * dequantised result. The reciprocal scale is returned so the matmul can be
 * done in integer space exactly as the runtime does, rather than by
 * multiplying by a float scale and hoping the two agree.
 *
 * STE: the backward of this function is the IDENTITY. The gradient of a
 * quantiser is zero almost everywhere, so without the estimator nothing would
 * train. The absmax is itself a function of the input, so a strictly exact
 * gradient would also carry a d(amax)/dx term through every row; the STE drops
 * that term deliberately. That is the standard treatment for activation
 * quantisation, and it is stated here rather than left for a reader to assume.
 */
static void act_q_row(qws * w, float * dst, const float * src, int n, float * recip_out) {
    if (!w->q->ternary) { memcpy(dst, src, (size_t) n * sizeof(float)); if (recip_out) { *recip_out = 1.0f; } return; }
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) { const float a = fabsf(src[i]); if (a > amax) { amax = a; } }
    if (amax < 1e-5f) { amax = 1e-5f; }
    const float s = 127.0f / amax;
    if (recip_out) { *recip_out = 1.0f / s; }
    for (int i = 0; i < n; ++i) { dst[i] = (float) act_q_train(src[i], s, NULL) * (1.0f / s); }
}

/* ------------------------------------------------------------------ */
/* matmuls, row-major [rows, cols] = [out_features, in_features]        */
/* ------------------------------------------------------------------ */

/* dst[t][o] = sum_i w[o*cols + i] * src[t][i]  -- the weight is already the
 * value used in the forward, i.e. it has been through ternarise() already. */
static void mat_fwd(float * dst, const float * src, const float * w,
                    int T, int rows, int cols) {
    for (int t = 0; t < T; ++t) {
        const float * s = src + (size_t) t * cols;
        float * d = dst + (size_t) t * rows;
        for (int o = 0; o < rows; ++o) {
            const float * wo = w + (size_t) o * cols;
            float acc = 0.0f;
            for (int i = 0; i < cols; ++i) { acc += wo[i] * s[i]; }
            d[o] = acc;
        }
    }
}

/* The weight gradient. THIS is the straight-through estimator: the caller
 * passes the gradient with respect to the QUANTISED weight and it is added to
 * the master's gradient with no transformation whatsoever. */
static void mat_bwd_w(float * dw, const float * ddst, const float * src,
                      int T, int rows, int cols) {
    for (int t = 0; t < T; ++t) {
        const float * s = src + (size_t) t * cols;
        const float * d = ddst + (size_t) t * rows;
        for (int o = 0; o < rows; ++o) {
            float * wo = dw + (size_t) o * cols;
            const float dv = d[o];
            for (int i = 0; i < cols; ++i) { wo[i] += dv * s[i]; }
        }
    }
}

/* The input gradient, which crosses the activation quantiser unchanged (STE). */
static void mat_bwd_x(float * dsrc, const float * ddst, const float * w,
                      int T, int rows, int cols) {
    for (int t = 0; t < T; ++t) {
        float * s = dsrc + (size_t) t * cols;
        const float * d = ddst + (size_t) t * rows;
        for (int o = 0; o < rows; ++o) {
            const float * wo = w + (size_t) o * cols;
            const float dv = d[o];
            for (int i = 0; i < cols; ++i) { s[i] += dv * wo[i]; }
        }
    }
}

/* ------------------------------------------------------------------ */
/* forward                                                              */
/* ------------------------------------------------------------------ */

/* Fill the ternarised weight for one parameter into the workspace and return
 * the pointer. The scale is the ABSMAX, floored at 1e-5, which is the upstream
 * QAT convention (BitNet.cpp weight_quant) and NOT the converter's
 * first-nonzero. The two differ for general input.
 *
 * That difference is fine, and the export respects it: sllm_qat_export_i2s_gguf
 * passes this same absmax scale to sllm_i2s_quantize_scaled, so the ternary the
 * model trains against is byte-for-byte the ternary that gets shipped. Without
 * that, "quantisation cost" would be confounded with a scale mismatch that has
 * nothing to do with quantisation. */
static const float * ternarise_p(qws * w, const sllm_qat_tensor * t) {
    float * out = qws_get(w, t->n);
    if (out == NULL) { return NULL; }
    if (!w->q->ternary) {
        memcpy(out, t->master, (size_t) t->n * sizeof(float));
        return out;
    }
    double amax = 0.0;
    for (int i = 0; i < t->n; ++i) {
        const double a = fabs((double) t->master[i]);
        if (a > amax) { amax = a; }
    }
    if (amax < 1e-5) { amax = 1e-5; }
    for (int i = 0; i < t->n; ++i) { out[i] = ternarise(t->master[i], amax); }
    return out;
}

/* GQA attention over the full context. Causal, softmax over the past only. */
static void attn_fwd(qws * w) {
    const int T = w->T, H = w->H, HK = w->HK, HD = w->HD, EK = w->EK;
    const int grp = H / HK;
    const float inv = 1.0f / sqrtf((float) HD);
    for (int h = 0; h < H; ++h) {
        const int g = h / grp;
        float * sc = w->scores + (size_t) h * T * T;
        for (int i = 0; i < T; ++i) {
            const float * qi = w->qh + (size_t) i * w->E + h * HD;
            for (int j = 0; j <= i; ++j) {
                const float * kj = w->kh + (size_t) j * EK + g * HD;
                float acc = 0.0f;
                for (int d = 0; d < HD; ++d) { acc += qi[d] * kj[d]; }
                sc[(size_t) i * T + j] = acc * inv;
            }
            /* softmax over j <= i only; the rest of the row stays zero */
            float m = -INFINITY;
            for (int j = 0; j <= i; ++j) { const float v = sc[(size_t) i * T + j]; if (v > m) { m = v; } }
            float sum = 0.0f;
            for (int j = 0; j <= i; ++j) {
                const float e = expf(sc[(size_t) i * T + j] - m);
                sc[(size_t) i * T + j] = e;
                sum += e;
            }
            const float r = 1.0f / sum;
            for (int j = 0; j <= i; ++j) { sc[(size_t) i * T + j] *= r; }
        }
        for (int i = 0; i < T; ++i) {
            const float * p = sc + (size_t) i * T;
            float * o = w->attn + (size_t) i * w->E + h * HD;
            for (int d = 0; d < HD; ++d) { o[d] = 0.0f; }
            for (int j = 0; j <= i; ++j) {
                const float pv = p[j];
                const float * vj = w->vh + (size_t) j * EK + g * HD;
                for (int d = 0; d < HD; ++d) { o[d] += pv * vj[d]; }
            }
        }
    }
}

/* siLU and its derivative, matching sllm_silu_inplace's sigmoid form. */
static float silu_f(float x) { return x / (1.0f + expf(-x)); }
static float silu_df(float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

/*
 * The forward pass. Fills the whole workspace; leaves the loss in the return
 * value and, when `want_grad` is set, the gradient in the model.
 *
 * `n_tok` is the number of real tokens in the batch and `n_scored` the number
 * of next-token predictions, so a context of n_scored+1 tokens scores
 * n_scored. The vocab softmax is the only place the loss is formed.
 */
/*
 * The forward of ONE layer, writing that layer's caches into the workspace.
 *
 * It is a function, and the backward calls it, for a reason that is easy to get
 * wrong: the per-layer intermediates (xmid, ao_pre, fh, ...) live in single
 * buffers, so after a full forward pass only the LAST layer's values are still
 * in them. The backward therefore cannot trust them for any earlier layer, and
 * must regenerate them. Recomputing one layer is cheap next to the bookkeeping
 * of keeping seventeen buffers per layer, and it cannot drift from the forward
 * because it IS the forward.
 */
static void qat_layer_fwd(qws * w, sllm_qat * q, int l, int n_tok) {
    const int T = w->T, E = w->E, EK = w->EK, F = w->F;
    const float eps = q->cfg.rms_eps;

        const float * xin  = w->x + (size_t) l * T * E;
        float       * xout = w->x + (size_t) (l + 1) * T * E;

        const float * wn1 = P(q, l, F_ATTN_NORM)->master;
        const float * wq  = ternarise_p(w, P(q, l, F_ATTN_Q));
        const float * wk  = ternarise_p(w, P(q, l, F_ATTN_K));
        const float * wv  = ternarise_p(w, P(q, l, F_ATTN_V));
        const float * wo  = ternarise_p(w, P(q, l, F_ATTN_OUT));
        const float * wsn = P(q, l, F_ATTN_SUB_NORM)->master;
        const float * wn2 = P(q, l, F_FFN_NORM)->master;
        const float * wgu = ternarise_p(w, P(q, l, F_FFN_UP));
        const float * wgg = ternarise_p(w, P(q, l, F_FFN_GATE));
        const float * wd  = ternarise_p(w, P(q, l, F_FFN_DOWN));
        const float * wfs = P(q, l, F_FFN_SUB_NORM)->master;
        if (!wq || !wk || !wv || !wo || !wgu || !wgg || !wd) {
            /* the ternary arena is exhausted: an internal error, not a bad
             * argument. The caller re-checks the loss for NaN and reports it. */
            return;
        }

        /* pre-attention norm, then the training-quantised copy the matmuls use */
        for (int t = 0; t < n_tok; ++t) {
            rms_fwd(w->n1 + (size_t) t * E, xin + (size_t) t * E, wn1, E, eps,
                    &w->rinv[(size_t) (l * 4 + 0) * T + t]);
        }
        for (int t = 0; t < n_tok; ++t) {
            act_q_row(w, w->n1q + (size_t) t * E, w->n1 + (size_t) t * E, E, NULL);
        }
        mat_fwd(w->qh, w->n1q, wq, n_tok, E, E);
        mat_fwd(w->kh, w->n1q, wk, n_tok, EK, E);
        mat_fwd(w->vh, w->n1q, wv, n_tok, EK, E);

        /* RoPE, NEOX, over n_rot dims -- which is n_embd_head, NOT the whole
         * row. The runtime rotates rope.dimension_count dims and refuses a file
         * whose n_rot exceeds n_embd_head, so rotating all E here would train a
         * model whose positional encoding the loader cannot even accept. */
        /* PER HEAD, with n_rot = n_embd_head dims each -- not n_rot dims of the
         * whole row. The runtime rotates `qh + h*hd` once per head
         * (forward.c:640), and NEOX pairs (i, i + n_rot/2) WITHIN the slice it
         * is given. Rotating a single n_rot-wide prefix of the row therefore
         * gets head 0 right and leaves heads 1..H-1 unrotated, and it would pair
         * across head boundaries even if they were not. That is the second half
         * of the export mismatch: the codes and the scale were already correct,
         * so the only thing wrong left with the positional encoding. */
        const size_t nrot = (size_t) w->q->cfg.n_embd_head;
        for (int t = 0; t < n_tok; ++t) {
            for (int h = 0; h < w->H; ++h) {
                sllm_rope_inplace(w->qh + (size_t) t * E + (size_t) h * w->HD, nrot, t,
                                  q->cfg.rope_base, 1.0f, SLLM_ROPE_NEOX);
            }
            for (int g = 0; g < w->HK; ++g) {
                sllm_rope_inplace(w->kh + (size_t) t * EK + (size_t) g * w->HD, nrot, t,
                                  q->cfg.rope_base, 1.0f, SLLM_ROPE_NEOX);
            }
        }

        attn_fwd(w);

        /*
         * ORDER: attn -> subln -> quantise -> output projection.
         *
         * The sub-norm goes BEFORE the projection because that is where the
         * runtime puts it ("attn_sub_norm, then the output projection",
         * forward.c:694), and a normalisation is not commutative with a linear
         * map: normalising after the projection gives a different function, and
         * a different one that gets worse as training sharpens the weights. The
         * FFN block below already had the right order -- ffn_sub_norm before
         * ffn_down -- which is why the two sub-norms were easy to confuse.
         */
        for (int t = 0; t < n_tok; ++t) {
            rms_fwd(w->ao + (size_t) t * E, w->attn + (size_t) t * E, wsn, E, eps,
                    &w->rinv[(size_t) (l * 4 + 1) * T + t]);
        }
        for (int t = 0; t < n_tok; ++t) {
            act_q_row(w, w->attnq + (size_t) t * E, w->ao + (size_t) t * E, E, NULL);
        }
        mat_fwd(w->ao_pre, w->attnq, wo, n_tok, E, E);

        for (int t = 0; t < n_tok; ++t) {
            const float * a = w->ao_pre + (size_t) t * E;
            float * o = w->xmid + (size_t) t * E;
            const float * xi = xin + (size_t) t * E;
            for (int d = 0; d < E; ++d) { o[d] = xi[d] + a[d]; }
        }

        for (int t = 0; t < n_tok; ++t) {
            rms_fwd(w->n2 + (size_t) t * E, w->xmid + (size_t) t * E, wn2, E, eps,
                    &w->rinv[(size_t) (l * 4 + 2) * T + t]);
            act_q_row(w, w->n2q + (size_t) t * E, w->n2 + (size_t) t * E, E, NULL);
        }
        mat_fwd(w->fu, w->n2q, wgu, n_tok, F, E);
        mat_fwd(w->fg, w->n2q, wgg, n_tok, F, E);
        for (int t = 0; t < n_tok; ++t) {
            float * g = w->fg + (size_t) t * F;
            float * u = w->fu + (size_t) t * F;
            float * h = w->fh + (size_t) t * F;
            for (int d = 0; d < F; ++d) { h[d] = silu_f(g[d]) * u[d]; }
        }
        for (int t = 0; t < n_tok; ++t) {
            rms_fwd(w->fy + (size_t) t * F, w->fh + (size_t) t * F, wfs, F, eps,
                    &w->rinv[(size_t) (l * 4 + 3) * T + t]);
            act_q_row(w, w->fhq + (size_t) t * F, w->fy + (size_t) t * F, F, NULL);
        }
        mat_fwd(w->fo, w->fhq, wd, n_tok, E, F);
        for (int t = 0; t < n_tok; ++t) {
            const float * y = w->fo + (size_t) t * E;
            const float * mid = w->xmid + (size_t) t * E;
            float * o = xout + (size_t) t * E;
            for (int d = 0; d < E; ++d) { o[d] = mid[d] + y[d]; }
        }
}

static double qat_forward(qws * w, sllm_qat * q, const int32_t * tokens,
                          int n_tok, int n_scored, bool want_grad) {
    const int T = w->T, E = w->E, V = w->V, L = w->L;
    const float eps = q->cfg.rms_eps;

    w->wq_off = 0;
    for (int i = 0; i < q->n_p; ++i) {
        if (q->p[i].trainable) { memset(q->p[i].grad, 0, (size_t) q->p[i].n * sizeof(float)); }
    }

    /* --- embedding (tied) --- */
    const sllm_qat_tensor * emb = &q->p[0];
    float * xcur = w->x;
    for (int t = 0; t < n_tok; ++t) {
        const float * row = emb->master + (size_t) tokens[t] * E;
        memcpy(xcur + (size_t) t * E, row, (size_t) E * sizeof(float));
    }

    /* --- layers --- */
    for (int l = 0; l < L; ++l) { qat_layer_fwd(w, q, l, n_tok); }

    /* --- final norm and tied output projection --- */
    const sllm_qat_tensor * onp = &q->p[1 + L * F_COUNT];
    const float * won = onp->master;
    const float * xfin = w->x + (size_t) L * T * E;
    float inv_fin = 1.0f;
    for (int t = 0; t < n_tok; ++t) {
        rms_fwd(w->n1 + (size_t) t * E, xfin + (size_t) t * E, won, E, eps, &inv_fin);
        /* slot L*4, NOT slot 0: the per-layer norms use [l*4 .. l*4+3] and
         * slot 0 belongs to layer 0's attention norm. */
        w->rinv[(size_t) L * 4 * T + t] = inv_fin;
    }
    for (int t = 0; t < n_tok; ++t) {
        const float * xn = w->n1 + (size_t) t * E;
        float * lg = w->logits + (size_t) t * V;
        for (int v = 0; v < V; ++v) {
            const float * er = emb->master + (size_t) v * E;
            float acc = 0.0f;
            for (int d = 0; d < E; ++d) { acc += er[d] * xn[d]; }
            lg[v] = acc;
        }
    }

    /* --- loss: next-token cross entropy, summed then averaged --- */
    double loss = 0.0;
    for (int t = 0; t < n_scored; ++t) {
        float * lg = w->logits + (size_t) t * V;
        float * pr = w->probs + (size_t) t * V;
        float m = -INFINITY;
        for (int v = 0; v < V; ++v) { if (lg[v] > m) { m = lg[v]; } }
        float sum = 0.0f;
        for (int v = 0; v < V; ++v) { const float e = expf(lg[v] - m); pr[v] = e; sum += e; }
        const float r = 1.0f / sum;
        for (int v = 0; v < V; ++v) { pr[v] *= r; }
        loss -= (double) log((double) pr[tokens[t + 1]] + 1e-300);
    }
    if (n_scored > 0) { loss /= (double) n_scored; }

    if (!want_grad) { return loss; }

    /* --- backward --- */
    const float inv_n = 1.0f / (float) n_scored;
    for (int t = 0; t < n_scored; ++t) {
        const float * pr = w->probs + (size_t) t * V;
        float * dl = w->d_logits + (size_t) t * V;
        for (int v = 0; v < V; ++v) { dl[v] = pr[v] * inv_n; }
        dl[tokens[t + 1]] -= inv_n;
    }
    return loss;
}

/* ------------------------------------------------------------------ */
/* backward                                                             */
/* ------------------------------------------------------------------ */

/*
 * The reverse sweep. Every `mat_bwd_w` call below is a straight-through
 * estimator: the gradient with respect to the QUANTISED weight goes into the
 * master's gradient untouched. Every place the gradient crosses an activation
 * quantiser is an identity, by the same argument and for the same reason.
 *
 * Nothing in this file is a numerical gradient. Every derivative below is
 * analytic, which is what makes the STE claim checkable: a test compares a
 * handful of entries against a central difference of the loss and requires the
 * analytic result to be in the right neighbourhood, while a finite-difference
 * "gradient" of a quantised forward could never be in the right neighbourhood
 * at all.
 */
static void qat_backward(qws * w, sllm_qat * q, const int32_t * tokens,
                         int n_tok, int n_scored) {
    const int T = w->T, E = w->E, EK = w->EK, F = w->F, V = w->V, L = w->L;
    const size_t TE = (size_t) T * E, TF = (size_t) T * F;
    sllm_qat_tensor * emb = &q->p[0];
    const int fwd_end = w->wq_off;   /* resume appending after the forward's */

    /* ---- tied output projection ----
     *
     * Over n_scored, NOT n_tok. d_logits is only written for the positions
     * that have a next-token target, and the final position does not. Reading
     * past that reads whatever the PREVIOUS call left there, which makes the
     * gradient depend on how many times loss() has been called -- the symptom
     * being a gradient that grew by a constant factor per call while the loss
     * stayed bit-identical. */
    const sllm_qat_tensor * onp = &q->p[1 + L * F_COUNT];
    const float * xfin = w->x + (size_t) L * TE;
    /* d_n1 is ACCUMULATED into below (once per vocab row), so it has to start
     * at zero. It is a backward-only buffer -- the forward never writes it --
     * so nothing else clears it, and the previous call's values double the
     * gradient every time. This one missing memset is the whole reason the
     * gradient grew while the loss stayed bit-identical. */
    memset(w->d_n1, 0, TE * sizeof(float));
    for (int t = 0; t < n_scored; ++t) {
        const float * dl = w->d_logits + (size_t) t * V;
        const float * xn = w->n1 + (size_t) t * E;   /* final-normed activations */
        float * dxn = w->d_n1 + (size_t) t * E;
        for (int v = 0; v < V; ++v) {
            const float dv = dl[v];
            const float * er = emb->master + (size_t) v * E;
            float * ge = emb->grad + (size_t) v * E;
            for (int d = 0; d < E; ++d) {
                dxn[d] += dv * er[d];
                ge[d]  += dv * xn[d];
            }
        }
    }
    /* final norm: cache was written by the forward */
    {
        float * dxf = w->d_x + (size_t) L * TE;
        for (int t = 0; t < n_scored; ++t) {
            const float inv = w->rinv[(size_t) L * 4 * T + t];
            rms_bwd(dxf + (size_t) t * E, w->d_n1 + (size_t) t * E,
                    xfin + (size_t) t * E, onp->master, E, inv);
            const float * xf = xfin + (size_t) t * E;
            const float * dnf = w->d_n1 + (size_t) t * E;
            for (int d = 0; d < E; ++d) { onp->grad[d] += dnf[d] * xf[d] * inv; }
        }
    }

    /* ---- layers, in reverse ---- */
    for (int l = L - 1; l >= 0; --l) {
        const float * xin  = w->x + (size_t) l * TE;
        float       * dxin = w->d_x + (size_t) l * TE;
        float       * D    = w->d_x + (size_t) (l + 1) * TE;   /* grad at xout */
        float       * S    = w->d_ao;   /* grad at xout' = xin + ao */
        float       * work = w->d_ao_pre;
        memset(dxin, 0, TE * sizeof(float));
        /* S and work are scratch that the per-position loops below only fill
         * for t < n_tok. Without this they carry the previous call's values
         * into the tail, and the tail is summed into dxin -- so the gradient
         * silently depends on how many times loss() has been called. D must be
         * zero past the scored window for the same reason: it is the residual
         * gradient, and the unscored final position has no target. */
        memset(S, 0, TE * sizeof(float));
        memset(work, 0, TE * sizeof(float));
        for (size_t i = (size_t) n_scored * E; i < TE; ++i) { D[i] = 0.0f; }
        memset(w->d_fhq, 0, TF * sizeof(float));
        memset(w->d_fh, 0, TF * sizeof(float));
        memset(w->d_fu, 0, TF * sizeof(float));
        memset(w->d_fg, 0, TF * sizeof(float));
        memset(w->d_n2, 0, TE * sizeof(float));
        memset(w->d_attn, 0, TE * sizeof(float));
        memset(w->d_n1, 0, TE * sizeof(float));
        memset(w->d_qh, 0, TE * sizeof(float));
        memset(w->d_kh, 0, (size_t) T * EK * sizeof(float));
        memset(w->d_vh, 0, (size_t) T * EK * sizeof(float));

        /* Regenerate this layer's caches: the single-buffer workspace only
         * holds the last layer's values after the forward pass. */
        w->wq_off = 0;
        qat_layer_fwd(w, q, l, n_tok);
        w->wq_off = fwd_end;
        const float * wq = ternarise_p(w, P(q, l, F_ATTN_Q));
        const float * wk = ternarise_p(w, P(q, l, F_ATTN_K));
        const float * wv = ternarise_p(w, P(q, l, F_ATTN_V));
        const float * wo = ternarise_p(w, P(q, l, F_ATTN_OUT));
        const float * wgu= ternarise_p(w, P(q, l, F_FFN_UP));
        const float * wgg= ternarise_p(w, P(q, l, F_FFN_GATE));
        const float * wd = ternarise_p(w, P(q, l, F_FFN_DOWN));
        if (!wq || !wd) { return; }

        sllm_qat_tensor * pn = P(q, l, F_ATTN_NORM),  * po = P(q, l, F_ATTN_OUT);
        sllm_qat_tensor * pn2= P(q, l, F_FFN_NORM),   * pd = P(q, l, F_FFN_DOWN);
        sllm_qat_tensor * pgu= P(q, l, F_FFN_UP),     * pgg= P(q, l, F_FFN_GATE);
        sllm_qat_tensor * pq = P(q, l, F_ATTN_Q),     * pk = P(q, l, F_ATTN_K);
        sllm_qat_tensor * pv = P(q, l, F_ATTN_V),     * psn= P(q, l, F_ATTN_SUB_NORM);
        sllm_qat_tensor * pfs= P(q, l, F_FFN_SUB_NORM);

        /* ===== FFN: fo = ffn_down(fhq), fhq = quant(subln2(silu(gate)*up)) =====
         *
         * d(loss)/d(fo) is D, the gradient already accumulated at xout. S then
         * collects the gradient at xout' = xin + ao, which is D (through fo)
         * PLUS whatever the FFN norm sends back -- and it is that SUM, not D,
         * that is the gradient into ao. Using D here instead of S is a real
         * error and it is invisible in the loss, which still goes down. */
        memcpy(w->d_fo, D, TE * sizeof(float));
        mat_bwd_x(w->d_fhq, w->d_fo, wd, n_tok, E, F);
        mat_bwd_w(pd->grad, w->d_fo, w->fhq, n_tok, E, F);            /* STE */
        {
            for (int t = 0; t < n_tok; ++t) {
                const float inv = w->rinv[(size_t) (l * 4 + 3) * T + t];
                rms_bwd(w->d_fh + (size_t) t * F, w->d_fhq + (size_t) t * F,
                        w->fh + (size_t) t * F, pfs->master, F, inv);
                const float * fh = w->fh + (size_t) t * F;
                /* dL/dw_i = (dL/d out_i) * x_i * inv, and dL/d out is d_fhq --
                 * the gradient of the quantiser's INPUT. d_fh is the gradient of
                 * the norm's input, which is a different quantity; using it here
                 * made this one weight look 3000x more sensitive than it is
                 * while every other parameter looked fine. */
                const float * dq = w->d_fhq + (size_t) t * F;
                for (int d = 0; d < F; ++d) { pfs->grad[d] += dq[d] * fh[d] * inv; }
            }
        }
        for (int t = 0; t < n_tok; ++t) {
            const float * g = w->fg + (size_t) t * F, * u = w->fu + (size_t) t * F;
            float * dg = w->d_fg + (size_t) t * F, * du = w->d_fu + (size_t) t * F;
            const float * dh = w->d_fh + (size_t) t * F;
            for (int d = 0; d < F; ++d) {
                dg[d] = dh[d] * u[d] * silu_df(g[d]);
                du[d] = dh[d] * silu_f(g[d]);
            }
        }
        mat_bwd_x(w->d_n2, w->d_fu, wgu, n_tok, F, E);
        mat_bwd_x(w->d_n2, w->d_fg, wgg, n_tok, F, E);
        mat_bwd_w(pgu->grad, w->d_fu, w->n2q, n_tok, F, E);            /* STE */
        mat_bwd_w(pgg->grad, w->d_fg, w->n2q, n_tok, F, E);            /* STE */
        {
            for (int t = 0; t < n_tok; ++t) {
                const float inv = w->rinv[(size_t) (l * 4 + 2) * T + t];
                const float * xm = w->xmid + (size_t) t * E;
                /* the ROW, not the base pointer: passing the base made
                 * every position re-read row 0's gradient, which is a plausible
                 * loss curve driven by a wrong gradient. */
                rms_bwd(S + (size_t) t * E, w->d_n2 + (size_t) t * E, xm, pn2->master, E, inv);
                const float * dn2 = w->d_n2 + (size_t) t * E;
                for (int d = 0; d < E; ++d) { pn2->grad[d] += dn2[d] * xm[d] * inv; }
            }
        }
        /* S += D, then the residual xout' = xin + ao puts S into dxin. */
        for (size_t i = 0, lim = (size_t) n_tok * E; i < lim; ++i) {
            S[i] += D[i];
            dxin[i] += S[i];
        }

        /* ===== attention: ao = subln1(ao_pre), ao_pre = attnq @ wo^T ===== */
        /* Three distinct vectors, and conflating any two of them is the bug
         * this structure exists to prevent. Exactly as the FFN block does it:
         *
         *   S    = d(loss)/d(o)      o is the projection's OUTPUT, which the
         *                              residual adds to the stream
         *   work = d(loss)/d(qa)     qa is the quantiser's input, i.e. the
         *                              projection's INPUT gradient
         *   d_attn = d(loss)/d(attn) what the sub-norm's backward produces
         *
         * rms_bwd takes the gradient of the norm's OUTPUT, which is `work` and
         * not S. Passing S there looks plausible and silently stops training. */
        mat_bwd_x(work, S, wo, n_tok, E, E);
        mat_bwd_w(po->grad, S, w->attnq, n_tok, E, E);                /* STE */
        {
            for (int t = 0; t < n_tok; ++t) {
                const float inv = w->rinv[(size_t) (l * 4 + 1) * T + t];
                const float * at = w->attn + (size_t) t * E;
                rms_bwd(w->d_attn + (size_t) t * E, work + (size_t) t * E, at,
                        psn->master, E, inv);
                const float * dq = work + (size_t) t * E;
                for (int d = 0; d < E; ++d) { psn->grad[d] += dq[d] * at[d] * inv; }
            }
        }
        {
            const int H = w->H, HK = w->HK, HD = w->HD;
            const int grp = H / HK;
            const float iscale = 1.0f / sqrtf((float) HD);
            memset(w->d_scores, 0, (size_t) H * T * T * sizeof(float));
            for (int h = 0; h < H; ++h) {
                const int g = h / grp;
                float * ds = w->d_scores + (size_t) h * T * T;
                const float * sc = w->scores + (size_t) h * T * T;
                for (int i = 0; i < n_tok; ++i) {
                    const float * di = w->d_attn + (size_t) i * E + h * HD;
                    float dot = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        const float * vj = w->vh + (size_t) j * EK + g * HD;
                        float dp = 0.0f;
                        for (int d = 0; d < HD; ++d) { dp += di[d] * vj[d]; }
                        ds[(size_t) i * T + j] = dp;
                        dot += sc[(size_t) i * T + j] * dp;
                    }
                    for (int j = 0; j <= i; ++j) {
                        ds[(size_t) i * T + j] = sc[(size_t) i * T + j] *
                                                 (ds[(size_t) i * T + j] - dot);
                    }
                    for (int j = 0; j <= i; ++j) {
                        const float dsij = ds[(size_t) i * T + j] * iscale;
                        float * qi = w->d_qh + (size_t) i * E + h * HD;
                        float * kj = w->d_kh + (size_t) j * EK + g * HD;
                        const float * qf = w->qh + (size_t) i * E + h * HD;
                        const float * kf = w->kh + (size_t) j * EK + g * HD;
                        for (int d = 0; d < HD; ++d) {
                            qi[d]  += dsij * kf[d];
                            kj[d]  += dsij * qf[d];
                        }
                    }
                    for (int j = 0; j <= i; ++j) {
                        const float p = sc[(size_t) i * T + j];
                        float * vj = w->d_vh + (size_t) j * EK + g * HD;
                        for (int d = 0; d < HD; ++d) { vj[d] += p * di[d]; }
                    }
                }
            }
        }
        /* RoPE is a rotation; its inverse is the same rotation by -pos, so the
         * backward cannot drift from the forward. */
        const size_t nrot = (size_t) w->q->cfg.n_embd_head;
        for (int t = 0; t < n_tok; ++t) {
            for (int h = 0; h < w->H; ++h) {
                sllm_rope_inplace(w->d_qh + (size_t) t * E + (size_t) h * w->HD, nrot, -t,
                                  q->cfg.rope_base, 1.0f, SLLM_ROPE_NEOX);
            }
            for (int g = 0; g < w->HK; ++g) {
                sllm_rope_inplace(w->d_kh + (size_t) t * EK + (size_t) g * w->HD, nrot, -t,
                                  q->cfg.rope_base, 1.0f, SLLM_ROPE_NEOX);
            }
        }
        mat_bwd_x(w->d_n1, w->d_qh, wq, n_tok, E, E);
        mat_bwd_x(w->d_n1, w->d_kh, wk, n_tok, EK, E);
        mat_bwd_x(w->d_n1, w->d_vh, wv, n_tok, EK, E);
        mat_bwd_w(pq->grad, w->d_qh, w->n1q, n_tok, E, E);              /* STE */
        mat_bwd_w(pk->grad, w->d_kh, w->n1q, n_tok, EK, E);             /* STE */
        mat_bwd_w(pv->grad, w->d_vh, w->n1q, n_tok, EK, E);             /* STE */
        {
            for (int t = 0; t < n_tok; ++t) {
                const float inv = w->rinv[(size_t) (l * 4 + 0) * T + t];
                const float * xi = xin + (size_t) t * E;
                rms_bwd(work + (size_t) t * E, w->d_n1 + (size_t) t * E, xi, pn->master, E, inv);
                const float * dn = w->d_n1 + (size_t) t * E;
                for (int d = 0; d < E; ++d) {
                    pn->grad[d] += dn[d] * xi[d] * inv;
                    dxin[(size_t) t * E + d] += work[(size_t) t * E + d];
                }
            }
        }
    }
    w->wq_off = fwd_end;

    /* ---- embedding input side (tied) ---- */
    {
        const float * dx0 = w->d_x;
        for (int t = 0; t < n_tok; ++t) {
            float * ge = emb->grad + (size_t) tokens[t] * E;
            for (int d = 0; d < E; ++d) { ge[d] += dx0[(size_t) t * E + d]; }
        }
    }
}

/* ------------------------------------------------------------------ */
/* the public entry points                                              */
/* ------------------------------------------------------------------ */

static qws * ws_of(sllm_qat * q) {
    if (q->ws == NULL) { q->ws = qws_new(q); }
    return (qws *) q->ws;
}

double sllm_qat_loss(sllm_qat * q, const sllm_qat_batch * batch) {
    qws * w = ws_of(q);
    if (w == NULL) { return NAN; }
    const int n_tok = batch->n_tokens;
    if (n_tok < 2 || n_tok > q->cfg.n_ctx) { return NAN; }
    const double loss = qat_forward(w, q, batch->tokens, n_tok, n_tok - 1, true);
    if (!isnan(loss)) { qat_backward(w, q, batch->tokens, n_tok, n_tok - 1); }
    return loss;
}

int sllm_qat_top1(sllm_qat * q, const sllm_qat_batch * batch) {
    qws * w = ws_of(q);
    if (w == NULL) { return -1; }
    const int n_tok = batch->n_tokens;
    if (n_tok < 2 || n_tok > q->cfg.n_ctx) { return -1; }
    (void) qat_forward(w, q, batch->tokens, n_tok, n_tok - 1, false);
    const int V = q->cfg.n_vocab;
    int hit = 0;
    for (int t = 0; t < n_tok - 1; ++t) {
        const float * lg = w->logits + (size_t) t * V;
        int best = 0;
        for (int v = 1; v < V; ++v) { if (lg[v] > lg[best]) { best = v; } }
        if (best == batch->tokens[t + 1]) { ++hit; }
    }
    return hit;
}

double sllm_qat_clip_grad(sllm_qat * q, float max_norm) {
    double ss = 0.0;
    for (int i = 0; i < q->n_p; ++i) {
        const sllm_qat_tensor * t = &q->p[i];
        if (!t->trainable) { continue; }
        for (int k = 0; k < t->n; ++k) {
            const double g = (double) t->grad[k];
            ss += g * g;
        }
    }
    const double n = sqrt(ss);
    if (n > (double) max_norm && n > 0.0) {
        const float s = (float) ((double) max_norm / n);
        for (int i = 0; i < q->n_p; ++i) {
            sllm_qat_tensor * t = &q->p[i];
            if (!t->trainable) { continue; }
            for (int k = 0; k < t->n; ++k) { t->grad[k] *= s; }
        }
    }
    return n;
}

void sllm_qat_step(sllm_qat * q, float lr) {
    q->step += 1;
    /* Bias correction uses the step AFTER the increment, so the first update
     * divides by (1 - beta1) = 0.1 rather than by 0. */
    const float b1 = SLLM_QAT_ADAM_BETA1, b2 = SLLM_QAT_ADAM_BETA2;
    const float c1 = 1.0f - powf(b1, (float) q->step);
    const float c2 = 1.0f - powf(b2, (float) q->step);
    for (int i = 0; i < q->n_p; ++i) {
        sllm_qat_tensor * t = &q->p[i];
        if (!t->trainable) { continue; }
        for (int k = 0; k < t->n; ++k) {
            float g = t->grad[k];
            /* Decoupled weight decay, applied to the master and not to the
             * gradient. With SLLM_QAT_ADAM_WD at 0 this is a no-op, and it is
             * written out anyway so the optimiser is fully specified. */
            t->adam_m[k] = b1 * t->adam_m[k] + (1.0f - b1) * g;
            t->adam_v[k] = b2 * t->adam_v[k] + (1.0f - b2) * g * g;
            const float mh = t->adam_m[k] / c1;
            const float vh = t->adam_v[k] / c2;
            float upd = mh / (sqrtf(vh) + SLLM_QAT_ADAM_EPS);
            if (SLLM_QAT_ADAM_WD != 0.0f) {
                upd += SLLM_QAT_ADAM_WD * t->master[k];
            }
            t->master[k] -= lr * upd;
        }
    }
}

/* ------------------------------------------------------------------ */
/* checkpoint                                                           */
/* ------------------------------------------------------------------ */

/*
 * A deliberate, explicit, single-shot write. Not a mmap and not a streaming
 * partial: the header is fixed, so a truncated file is detected on load rather
 * than half-believed. A checkpoint is the ONE thing in this program that is
 * allowed to touch disk, and it is allowed because being able to stop and
 * resume is the thing being proven.
 */
typedef struct {
    uint32_t magic, version;
    int32_t  n_layer, n_embd, n_head, n_head_kv, n_embd_head,
             n_embd_gqa, n_ff, n_vocab, n_ctx, n_p;
    float    rms_eps, rope_base;
    uint64_t seed, rng;
    uint32_t step;
} ckpt_hdr;

int sllm_qat_save(const sllm_qat * q, const char * path) {
    FILE * f = fopen(path, "wb");
    if (f == NULL) { return -1; }
    ckpt_hdr h;
    memset(&h, 0, sizeof h);
    h.magic = QAT_MAGIC; h.version = QAT_VERSION;
    h.n_layer = q->cfg.n_layer; h.n_embd = q->cfg.n_embd;
    h.n_head = q->cfg.n_head; h.n_head_kv = q->cfg.n_head_kv;
    h.n_embd_head = q->cfg.n_embd_head; h.n_embd_gqa = q->cfg.n_embd_gqa;
    h.n_ff = q->cfg.n_ff; h.n_vocab = q->cfg.n_vocab;
    h.n_ctx = q->cfg.n_ctx; h.n_p = q->n_p;
    h.rms_eps = q->cfg.rms_eps; h.rope_base = q->cfg.rope_base;
    h.rng = q->rng; h.step = q->step;
    int ok = (fwrite(&h, sizeof h, 1, f) == 1);
    for (int i = 0; ok && i < q->n_p; ++i) {
        const sllm_qat_tensor * t = &q->p[i];
        ok = (fwrite(t->master, (size_t) t->n * sizeof(float), 1, f) == 1);
        if (ok && t->trainable) {
            ok = (fwrite(t->adam_m, (size_t) t->n * sizeof(float), 1, f) == 1) &&
                 (fwrite(t->adam_v, (size_t) t->n * sizeof(float), 1, f) == 1);
        }
    }
    if (fclose(f) != 0) { ok = 0; }
    return ok ? 0 : -1;
}

sllm_qat * sllm_qat_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    ckpt_hdr h;
    if (fread(&h, sizeof h, 1, f) != 1 || h.magic != QAT_MAGIC ||
        h.version != QAT_VERSION) { fclose(f); return NULL; }
    sllm_qat_config c;
    memset(&c, 0, sizeof c);
    c.n_layer = h.n_layer; c.n_embd = h.n_embd; c.n_head = h.n_head;
    c.n_head_kv = h.n_head_kv; c.n_embd_head = h.n_embd_head;
    c.n_embd_gqa = h.n_embd_gqa; c.n_ff = h.n_ff; c.n_vocab = h.n_vocab;
    c.n_ctx = h.n_ctx; c.rms_eps = h.rms_eps; c.rope_base = h.rope_base;
    /* seed is not stored because init() already consumed it; what matters for
     * resuming is the rng state and the step, both of which are. */
    sllm_qat * q = sllm_qat_new(&c, 0);
    if (q == NULL) { fclose(f); return NULL; }
    if (q->n_p != h.n_p) { sllm_qat_free(q); fclose(f); return NULL; }
    int ok = 1;
    for (int i = 0; ok && i < q->n_p; ++i) {
        sllm_qat_tensor * t = &q->p[i];
        ok = (fread(t->master, (size_t) t->n * sizeof(float), 1, f) == 1);
        if (ok && t->trainable) {
            ok = (fread(t->adam_m, (size_t) t->n * sizeof(float), 1, f) == 1) &&
                 (fread(t->adam_v, (size_t) t->n * sizeof(float), 1, f) == 1);
        }
    }
    fclose(f);
    if (!ok) { sllm_qat_free(q); return NULL; }
    q->rng = h.rng;
    q->step = h.step;
    return q;
}

/* ------------------------------------------------------------------ */
/* GGUF export                                                          */
/* ------------------------------------------------------------------ */

/*
 * A minimal GGUF v3 writer, written here because the project has a reader and
 * a converter but no container writer, and a lifecycle that cannot emit a model
 * is not a lifecycle. It emits exactly what the ordinary loader reads and
 * nothing else: the ten required metadata keys and the tensors forward.c asks
 * for, under the names it asks for them.
 *
 * The I2_S tensors are produced by sllm_i2s_quantize_scaled with the SAME absmax
 * scale the training forward used, so the shipped ternary is the trained ternary
 * and the difference between the post-QAT BF16 score and the post-export I2_S
 * score measures quantisation rather than a changed scale.
 */

/*
 * IEEE binary32 -> binary16, round-to-nearest-even.
 *
 * Needed because token_embd is declared F16 and the runtime decodes it with
 * dot_f16, and the project has an f16 DECODER (sllm_fp16_to_fp32) but no
 * packer. Writing sllm_f32_to_bf16's bits into an F16 tensor instead -- which
 * is what this exporter did -- is not a small error: binary16 is
 * sign/exp5/mant10 and bfloat16 is sign/exp8/mant7, so every value is
 * reinterpreted with the wrong exponent width. Measured on the untrained
 * export, that put the runtime's logits ~128x the trainer's with a cosine of
 * 0.4, and it grew as training moved the magnitudes around. The two formats
 * are not variants of each other; the bytes mean different numbers.
 */
static uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ebits = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (ebits == 0xFFu) {                       /* inf or NaN */
        if (mant == 0) { return (uint16_t) (sign | 0x7C00u); }
        uint16_t h = (uint16_t) ((mant >> 13) | 0x7C00u);
        return (h == 0x7C00u) ? (uint16_t) (h | 1u) : h;   /* keep NaN a NaN */
    }
    int32_t exp = (int32_t) ebits - 127 + 15;
    if (exp >= 0x1F) { return (uint16_t) (sign | 0x7C00u); }  /* overflow */
    if (exp <= 0) {                                             /* subnormal */
        if (exp < -10) { return (uint16_t) sign; }
        mant |= 0x800000u;                                      /* implicit 1 */
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) { ++half; }
        return (uint16_t) (sign | half);
    }
    uint32_t half = ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) { ++half; }
    return (uint16_t) (sign | half);
}

typedef struct { uint8_t * b; size_t n, cap; } wbuf;

static int wb_put(wbuf * w, const void * p, size_t n) {
    if (w->n + n > w->cap) {
        size_t c = w->cap ? w->cap * 2 : 4096;
        while (c < w->n + n) { c *= 2; }
        uint8_t * nb = (uint8_t *) realloc(w->b, c);
        if (nb == NULL) { return -1; }
        w->b = nb; w->cap = c;
    }
    memcpy(w->b + w->n, p, n);
    w->n += n;
    return 0;
}
static int wb_u32(wbuf * w, uint32_t v) { return wb_put(w, &v, 4); }
static int wb_u64(wbuf * w, uint64_t v) { return wb_put(w, &v, 8); }
static int wb_f32(wbuf * w, float v)    { return wb_put(w, &v, 4); }
static int wb_str(wbuf * w, const char * s) {
    const uint64_t l = (uint64_t) strlen(s);
    return wb_u64(w, l) || wb_put(w, s, (size_t) l);
}
static int wb_kv_u32(wbuf * w, const char * k, uint32_t v) {
    return wb_str(w, k) || wb_u32(w, SLLM_VT_UINT32) || wb_u32(w, v);
}
static int wb_kv_f32(wbuf * w, const char * k, float v) {
    return wb_str(w, k) || wb_u32(w, SLLM_VT_FLOAT32) || wb_f32(w, v);
}
static int wb_kv_str(wbuf * w, const char * k, const char * v) {
    return wb_str(w, k) || wb_u32(w, SLLM_VT_STRING) || wb_str(w, v);
}

typedef struct {
    const char * name;
    int n_dims;
    uint64_t ne[2];
    uint32_t type;
    const void * data;
    size_t nbytes;
} xt;

int sllm_qat_export_i2s_gguf(const sllm_qat * q, const char * path,
                              sllm_i2s_rule rule) {
    const sllm_qat_config * c = &q->cfg;
    const int E = c->n_embd, EK = c->n_embd_gqa, F = c->n_ff, V = c->n_vocab;
    const int L = c->n_layer;
    const uint32_t AL = SLLM_GGUF_DEFAULT_ALIGNMENT;

    const int n_tensors = 1 + L * 11 + 1;
    xt * xs = (xt *) calloc((size_t) n_tensors, sizeof(xt));
    uint8_t ** owned = (uint8_t **) calloc((size_t) n_tensors, sizeof(uint8_t *));
    uint16_t * emb16 = (uint16_t *) calloc((size_t) E * V, sizeof(uint16_t));
    if (!xs || !owned || !emb16) { free(xs); free(owned); free(emb16); return -1; }

    /* the tied embedding is F16, the one tensor the ternary does not touch */
    for (int i = 0; i < E * V; ++i) {
        emb16[i] = f32_to_f16_bits(q->p[0].master[i]);
    }
    int xi = 0;
    xs[xi].name = strdup("token_embd.weight");   /* heap: freed below */
    if (xs[xi].name == NULL) { free(emb16); free(xs); free(owned); return -1; }
    xs[xi].n_dims = 2;
    xs[xi].ne[0] = E; xs[xi].ne[1] = V; xs[xi].type = SLLM_TYPE_F16;
    xs[xi].data = emb16; xs[xi].nbytes = (size_t) E * V * 2;
    ++xi;

    for (int l = 0; l < L; ++l) {
        static const char * suffix[] = { "attn_norm", "attn_q", "attn_k", "attn_v",
            "attn_output", "attn_sub_norm", "ffn_norm", "ffn_up", "ffn_gate",
            "ffn_down", "ffn_sub_norm" };
        char nm[64];
        for (int k = 0; k < F_COUNT; ++k) {
            const sllm_qat_tensor * t = P((sllm_qat *) q, l, k);
            snprintf(nm, sizeof nm, "blk.%d.%s.weight", l, suffix[k]);
            xs[xi].name = strdup(nm);
            if (!field_is_ternary(k)) {
                /* ne[0] is the vector's own length: E for the three attention
                 * norms, F for ffn_sub_norm. Using E for all of them is right
                 * for most and wrong for exactly one, which is the kind of bug
                 * that survives a spot check and then fails the loader. */
                xs[xi].n_dims = 1; xs[xi].ne[0] = (uint64_t) t->n;
                xs[xi].type = SLLM_TYPE_F32;
                xs[xi].data = t->master;
                xs[xi].nbytes = (size_t) t->n * 4;
            } else {
                const size_t nb = sllm_i2s_packed_size((size_t) t->n);
                uint8_t * pk = (uint8_t *) calloc(nb, 1);
                if (pk == NULL) { goto fail; }
                /* The override-scale path, with the training absmax. */
                if (sllm_i2s_quantize_scaled(t->master, (size_t) t->n, pk,
                                             sllm_qat_weight_scale(t), rule) != 0) {
                    free(pk); goto fail;
                }
                owned[xi] = pk;
                /* GGUF ne[0] is the INPUT feature count, ne[1] the output, and
                 * this is the order forward.c checks: ffn_up is [n_embd,
                 * n_ff] but ffn_down is [n_ff, n_embd]. Getting this backwards
                 * produced a file that opened and had every tensor present, and
                 * that the loader rejected on shape -- which is exactly the
                 * failure a shape assertion is for. */
                const int in  = (k == F_FFN_DOWN) ? F : E;
                const int out = (k == F_FFN_DOWN) ? E
                              : (k == F_ATTN_K || k == F_ATTN_V) ? EK
                              : (k == F_FFN_UP || k == F_FFN_GATE) ? F : E;
                /* owned[xi] already holds pk from above, so every exit from here
                 * must go through the fail: cleanup rather than freeing pk again.
                 * Doing both is a double free, and it fires from the error path,
                 * which is the one place a clear diagnostic must not be followed
                 * by a crash. */
                if ((size_t) in * (size_t) out != (size_t) t->n) { goto fail; }
                /*
                 * THE ROW-WIDTH PRECONDITION. The I2_S block is 128 elements in
                 * 32 bytes (SLLM_I2S_QK), and the runtime's reader of that
                 * buffer has two hard requirements that this exporter used to
                 * ignore:
                 *
                 *   sllm_i2s_gemv strides rows by n/4, and
                 *   sllm_i2s_dot returns 0 outright when n < 128.
                 *
                 * So a row of `in` elements is only consumable when `in` is a
                 * multiple of 128. When it is, this exporter's FLATTENED pack
                 * and the gemv's ROW-WISE read are the same bytes: element
                 * i = in*r + c lands at in/4*r + (c/128)*32 + c%32, which is
                 * both the flattened 128-block mapping and the row-strided one.
                 * That identity is why every shipped tensor (n_embd 2048) is
                 * unaffected, and it is asserted by the byte-stability test.
                 *
                 * When `in` is NOT a multiple of 128 there is no packing that
                 * works, and the old behaviour was the worst of the options: the
                 * file opened, every tensor was present, every shape checked
                 * out, and the projections silently returned garbage -- a dot of
                 * exactly 0 for in < 128, and a wrong row offset for in = 192
                 * and similar. So the exporter now refuses. A narrow model is a
                 * property the runtime does not yet support, and saying so at
                 * write time is the difference between a clear error and a model
                 * that loads and lies.
                 */
                if (((size_t) in % (size_t) SLLM_I2S_QK) != 0) {
                    fprintf(stderr,
                        "saphira-llm-qat: %s: cannot export: the I2_S runtime "
                        "consumes rows of %d elements, and %s has an input width of "
                        "%d, which is not a multiple of %d. sllm_i2s_dot returns 0 "
                        "for n < %d and sllm_i2s_gemv strides rows by n/4, so a row "
                        "that is not a whole number of 128-element blocks cannot be "
                        "read back correctly. Use n_embd and n_ff that are "
                        "multiples of %d.\n",
                        nm, SLLM_I2S_QK, nm, in, SLLM_I2S_QK, SLLM_I2S_QK,
                        SLLM_I2S_QK);
                    goto fail;
                }
                xs[xi].n_dims = 2;
                xs[xi].ne[0] = (uint64_t) in;
                xs[xi].ne[1] = (uint64_t) out;
                xs[xi].type = SLLM_TYPE_I2_S;
                xs[xi].data = pk; xs[xi].nbytes = nb;
            }
            ++xi;
        }
    }
    {
        const sllm_qat_tensor * on = &q->p[1 + L * F_COUNT];
        xs[xi].name = strdup("output_norm.weight");
        if (xs[xi].name == NULL) { goto fail; }
        xs[xi].n_dims = 1;
        xs[xi].ne[0] = (uint64_t) E; xs[xi].type = SLLM_TYPE_F32;
        xs[xi].data = on->master; xs[xi].nbytes = (size_t) on->n * 4;
        ++xi;
    }
    if (xi != n_tensors) { goto fail; }

    /* --- header --- */
    wbuf w = {0};
    const int n_kv = 11;
    if (wb_put(&w, SLLM_GGUF_MAGIC, 4) || wb_u32(&w, SLLM_GGUF_VERSION) ||
        wb_u64(&w, (uint64_t) n_tensors) || wb_u64(&w, (uint64_t) n_kv)) { goto fail_w; }
    /* general.architecture is the ONE key that is not under the
     * bitnet-b1.58.* prefix. Prefixing it produced a file that opened cleanly,
     * had every tensor, and was rejected as unsupported-architecture -- which
     * is the loader doing exactly its job. */
    if (wb_kv_str(&w, "general.architecture", "bitnet-b1.58")) { goto fail_w; }
    const char * A = "bitnet-b1.58.";
    char k[96];
    snprintf(k, sizeof k, "%sblock_count", A);
    if (wb_kv_u32(&w, k, (uint32_t) L)) { goto fail_w; }
    snprintf(k, sizeof k, "%sembedding_length", A);
    if (wb_kv_u32(&w, k, (uint32_t) E)) { goto fail_w; }
    snprintf(k, sizeof k, "%sfeed_forward_length", A);
    if (wb_kv_u32(&w, k, (uint32_t) F)) { goto fail_w; }
    snprintf(k, sizeof k, "%sattention.head_count", A);
    if (wb_kv_u32(&w, k, (uint32_t) c->n_head)) { goto fail_w; }
    snprintf(k, sizeof k, "%sattention.head_count_kv", A);
    if (wb_kv_u32(&w, k, (uint32_t) c->n_head_kv)) { goto fail_w; }
    snprintf(k, sizeof k, "%srope.dimension_count", A);
    if (wb_kv_u32(&w, k, (uint32_t) c->n_embd_head)) { goto fail_w; }
    snprintf(k, sizeof k, "%scontext_length", A);
    if (wb_kv_u32(&w, k, (uint32_t) c->n_ctx)) { goto fail_w; }
    snprintf(k, sizeof k, "%svocab_size", A);
    if (wb_kv_u32(&w, k, (uint32_t) V)) { goto fail_w; }
    snprintf(k, sizeof k, "%sattention.layer_norm_rms_epsilon", A);
    if (wb_kv_f32(&w, k, c->rms_eps)) { goto fail_w; }
    snprintf(k, sizeof k, "%srope.freq_base", A);
    if (wb_kv_f32(&w, k, c->rope_base)) { goto fail_w; }

    /* --- tensor table, with the alignment the data section will use --- */
    const size_t table_at = w.n;
    for (int i = 0; i < n_tensors; ++i) {
        if (wb_str(&w, xs[i].name) || wb_u32(&w, (uint32_t) xs[i].n_dims)) { goto fail_w; }
        for (int d = 0; d < xs[i].n_dims; ++d) {
            if (wb_u64(&w, xs[i].ne[d])) { goto fail_w; }
        }
        if (wb_u32(&w, xs[i].type)) { goto fail_w; }
        if (wb_u64(&w, 0)) { goto fail_w; }   /* offset, patched below */
    }
    /* Pad to the alignment. The padding is written with memset, not wb_put:
     * wb_put memcpys from its source pointer, and handing it NULL to mean "no
     * data" segfaults the moment the run is non-zero. */
    const size_t data_start = (w.n + AL - 1) & ~(size_t)(AL - 1);
    if (data_start > w.n) {
        if (w.n + (data_start - w.n) > w.cap) {
            const size_t c = data_start;
            uint8_t * nb = (uint8_t *) realloc(w.b, c);
            if (nb == NULL) { goto fail_w; }
            w.b = nb; w.cap = c;
        }
        memset(w.b + w.n, 0, data_start - w.n);
        w.n = data_start;
    }
    size_t off = 0;
    for (int i = 0; i < n_tensors; ++i) {
        const size_t rel = table_at + (size_t) i * (8 + 16 + 8);  /* see below */
        (void) rel;
        if (wb_put(&w, xs[i].data, xs[i].nbytes)) { goto fail_w; }
        off += (xs[i].nbytes + AL - 1) & ~(size_t)(AL - 1);
    }
    /* patch the offsets into the tensor table */
    off = 0;
    {
        size_t at = table_at;
        for (int i = 0; i < n_tensors; ++i) {
            at += 8 + (size_t) strlen(xs[i].name);   /* string */
            at += 4 + 8 * (size_t) xs[i].n_dims;     /* n_dims + dims */
            at += 4;                                  /* type */
            memcpy(w.b + at, &off, 8);
            off += (xs[i].nbytes + AL - 1) & ~(size_t)(AL - 1);
            at += 8;
        }
    }

    FILE * f = fopen(path, "wb");
    if (f == NULL) { goto fail_w; }
    const int okw = (fwrite(w.b, 1, w.n, f) == w.n);
    if (fclose(f) != 0 || !okw) { goto fail_w; }
    free(w.b);
    for (int i = 0; i < n_tensors; ++i) { free(owned[i]); free((void *) xs[i].name); }
    free(xs); free(owned); free(emb16);
    return 0;

fail_w:
    free(w.b);
fail:
    for (int i = 0; i < n_tensors; ++i) { free(owned[i]); free((void *) xs[i].name); }
    free(xs); free(owned); free(emb16);
    return -1;
}
