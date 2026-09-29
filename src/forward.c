/*
 * BitNet b1.58 forward pass.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * This is the implementation of docs/PHASE4-CONTRACT.md. That document has the
 * graph, the hyperparameter values, the op contracts and the file-and-line for
 * every place the reference is easy to misread. Read it before changing this.
 *
 * The shape of the arithmetic, which is the whole reason the model is tractable
 * to get exactly right:
 *
 *   210 of the 211 weight tensors are I2_S, so every per-layer projection is
 *   integer arithmetic -- an int8-quantised activation dotted against ternary
 *   codes, accumulated in int32, then one float epilogue per column. There is
 *   no float error in them at all.
 *
 *   The one F16 tensor is token_embd, which is also the lm_head, and the 121
 *   F32 tensors are the norms. So the only float accumulation in the entire
 *   model is the 2560-wide lm_head reduction per position, plus the attention
 *   QK and softmax-V products.
 *
 * That is why the hard parity gate is the greedy token sequence and the
 * measured one is the logits, and it is why "token-identical" is a realistic
 * ask here even though our reduction order is our own.
 */

#include "saphira_llm/forward.h"
#include "saphira_llm/i2s_gemm.h"
#include "saphira_llm/ops.h"
#include "saphira_llm/quant.h"
#include "saphira_llm/status.h"
#include "saphira_llm/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SLLM_LAYER_MAX 64

typedef struct {
    const uint8_t  * wq, *wk, *wv, *wo;
    const uint8_t  * ffn_up, *ffn_gate, *ffn_down;
    const float    * attn_norm, * attn_sub_norm;
    const float    * ffn_norm, * ffn_sub_norm;
    size_t          wq_rows, wk_rows, wv_rows, wo_rows;
    size_t          up_rows, down_rows;
} sllm_layer;

struct sllm_model {
    const sllm_gguf * gguf;      /* borrowed; tensor payloads point into it */

    int32_t  n_layer;
    int32_t  n_embd;
    int32_t  n_ff;
    int32_t  n_head;
    int32_t  n_head_kv;
    int32_t  n_embd_head;
    int32_t  n_embd_gqa;
    int32_t  n_vocab;
    int32_t  n_ctx_train;
    int32_t  n_rot;
    float    f_rms_eps;
    float    rope_freq_base;

    sllm_layer layer[SLLM_LAYER_MAX];

    const uint16_t * tok_embd;   /* F16, [n_embd, n_vocab] */
    const float    * output_norm;
};

struct sllm_ctx {
    int32_t  n_ctx;
    int32_t  n_past;     /* how much of the KV cache is live */
    /* KV cache: [layer][head_kv][pos][n_embd_head] */
    float   * k_cache;
    float   * v_cache;

    /* scratch, sized once */
    float   * x;        /* the running residual, n_embd            */
    float   * xn;       /* normalised copy, n_embd                 */
    float   * ffn_inp;  /* n_embd                                   */
    float   * q;        /* n_embd                                   */
    float   * k;        /* n_embd_gqa                               */
    float   * v;        /* n_embd_gqa                               */
    float   * attn;     /* n_embd, attention output                */
    float   * proj;     /* n_embd, a projection result              */
    float   * ffn_gate_buf; /* n_ff                                  */
    float   * ffn_up_buf;   /* n_ff                                  */
    float   * ffn_h;    /* n_ff                                     */
    float   * scores;   /* n_ctx                                    */
    int8_t  * q_act;    /* n_embd                                   */
    int8_t  * q_act_ffn;/* n_ff                                     */
    int32_t * dots;     /* max(n_embd, n_ff)                        */

    size_t    kv_head_stride;  /* n_ctx * n_embd_head, per kv head    */
};

/* ------------------------------------------------------------------ */
/* f16                                                                  */
/* ------------------------------------------------------------------ */

static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            /* Subnormal: normalise it. */
            uint32_t e = 0;
            uint32_t m = man;
            while ((m & 0x400u) == 0) { m <<= 1; ++e; }
            m &= 0x3FFu;
            bits = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ------------------------------------------------------------------ */
/* model                                                                */
/* ------------------------------------------------------------------ */

static const sllm_gguf_tensor * need(const sllm_gguf * g, const char * name,
                                     sllm_ggml_type type, int32_t n_dims) {
    const sllm_gguf_tensor * t = sllm_gguf_find_tensor(g, name);
    if (t == NULL) {
        return NULL;
    }
    if (type != SLLM_TYPE_COUNT && (sllm_ggml_type) t->type != type) {
        return NULL;
    }
    if (n_dims > 0 && (int32_t) t->n_dims != n_dims) {
        return NULL;
    }
    return t;
}

sllm_status sllm_model_load(const sllm_gguf * g, sllm_model ** out) {
    if (g == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    *out = NULL;

    const char * arch = NULL;
    if (sllm_gguf_kv_str(g, "general.architecture", &arch) != SLLM_OK ||
        strcmp(arch, "bitnet-b1.58") != 0) {
        return SLLM_ERR_ARCH_UNSUPPORTED;
    }

    sllm_model * m = (sllm_model *) calloc(1, sizeof(sllm_model));
    if (m == NULL) {
        return SLLM_ERR_NOMEM;
    }
    m->gguf = g;

    uint32_t u = 0;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.block_count", &u) != SLLM_OK) { goto bad_kv; }
    m->n_layer = (int32_t) u;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.embedding_length", &u) != SLLM_OK) { goto bad_kv; }
    m->n_embd = (int32_t) u;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.feed_forward_length", &u) != SLLM_OK) { goto bad_kv; }
    m->n_ff = (int32_t) u;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.attention.head_count", &u) != SLLM_OK) { goto bad_kv; }
    m->n_head = (int32_t) u;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.attention.head_count_kv", &u) != SLLM_OK) { goto bad_kv; }
    m->n_head_kv = (int32_t) u;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.rope.dimension_count", &u) != SLLM_OK) { goto bad_kv; }
    m->n_rot = (int32_t) u;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.context_length", &u) != SLLM_OK) { goto bad_kv; }
    m->n_ctx_train = (int32_t) u;
    if (sllm_gguf_kv_u32(g, "bitnet-b1.58.vocab_size", &u) != SLLM_OK) { goto bad_kv; }
    m->n_vocab = (int32_t) u;
    if (sllm_gguf_kv_f32(g, "bitnet-b1.58.attention.layer_norm_rms_epsilon", &m->f_rms_eps) != SLLM_OK) { goto bad_kv; }
    if (sllm_gguf_kv_f32(g, "bitnet-b1.58.rope.freq_base", &m->rope_freq_base) != SLLM_OK) { goto bad_kv; }

    if (m->n_layer <= 0 || m->n_layer > SLLM_LAYER_MAX ||
        m->n_embd <= 0 || m->n_head <= 0 || m->n_head_kv <= 0 ||
        m->n_head % m->n_head_kv != 0 || m->n_ff <= 0 || m->n_vocab <= 0) {
        goto bad_shape;
    }
    m->n_embd_head = m->n_embd / m->n_head;
    m->n_embd_gqa  = m->n_embd_head * m->n_head_kv;
    if (m->n_embd % m->n_head != 0) { goto bad_shape; }
    if (m->n_rot > m->n_embd_head) { goto bad_shape; }

    {
        const sllm_gguf_tensor * t = need(g, "token_embd.weight", SLLM_TYPE_F16, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd || (int64_t) t->ne[1] != m->n_vocab) {
            goto bad_shape;
        }
        m->tok_embd = (const uint16_t *) t->data;
    }
    {
        const sllm_gguf_tensor * t = need(g, "output_norm.weight", SLLM_TYPE_F32, 1);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd) { goto bad_shape; }
        m->output_norm = (const float *) t->data;
    }

    for (int32_t il = 0; il < m->n_layer; ++il) {
        char nm[128];
        sllm_layer * L = &m->layer[il];

#define LNAME(field, il) (snprintf(nm, sizeof(nm), "blk.%d.%s.weight", (il), (field)), nm)

        const sllm_gguf_tensor * t = need(g, LNAME("attn_q", il), SLLM_TYPE_I2_S, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd || (int64_t) t->ne[1] != m->n_embd) { goto bad_shape; }
        L->wq = (const uint8_t *) t->data; L->wq_rows = (size_t) m->n_embd;

        t = need(g, LNAME("attn_k", il), SLLM_TYPE_I2_S, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd || (int64_t) t->ne[1] != m->n_embd_gqa) { goto bad_shape; }
        L->wk = (const uint8_t *) t->data; L->wk_rows = (size_t) m->n_embd_gqa;

        t = need(g, LNAME("attn_v", il), SLLM_TYPE_I2_S, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd || (int64_t) t->ne[1] != m->n_embd_gqa) { goto bad_shape; }
        L->wv = (const uint8_t *) t->data; L->wv_rows = (size_t) m->n_embd_gqa;

        t = need(g, LNAME("attn_output", il), SLLM_TYPE_I2_S, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd || (int64_t) t->ne[1] != m->n_embd) { goto bad_shape; }
        L->wo = (const uint8_t *) t->data; L->wo_rows = (size_t) m->n_embd;

        t = need(g, LNAME("ffn_up", il), SLLM_TYPE_I2_S, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd || (int64_t) t->ne[1] != m->n_ff) { goto bad_shape; }
        L->ffn_up = (const uint8_t *) t->data; L->up_rows = (size_t) m->n_ff;

        t = need(g, LNAME("ffn_gate", il), SLLM_TYPE_I2_S, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd || (int64_t) t->ne[1] != m->n_ff) { goto bad_shape; }
        L->ffn_gate = (const uint8_t *) t->data;

        t = need(g, LNAME("ffn_down", il), SLLM_TYPE_I2_S, 2);
        if (t == NULL || (int64_t) t->ne[0] != m->n_ff || (int64_t) t->ne[1] != m->n_embd) { goto bad_shape; }
        L->ffn_down = (const uint8_t *) t->data; L->down_rows = (size_t) m->n_embd;

        t = need(g, LNAME("attn_norm", il), SLLM_TYPE_F32, 1);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd) { goto bad_shape; }
        L->attn_norm = (const float *) t->data;

        t = need(g, LNAME("attn_sub_norm", il), SLLM_TYPE_F32, 1);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd) { goto bad_shape; }
        L->attn_sub_norm = (const float *) t->data;

        t = need(g, LNAME("ffn_norm", il), SLLM_TYPE_F32, 1);
        if (t == NULL || (int64_t) t->ne[0] != m->n_embd) { goto bad_shape; }
        L->ffn_norm = (const float *) t->data;

        /* ffn_sub_norm is the FFN's HIDDEN width, not n_embd. See
         * PHASE4-CONTRACT.md: this is one of the three easy mistakes. */
        t = need(g, LNAME("ffn_sub_norm", il), SLLM_TYPE_F32, 1);
        if (t == NULL || (int64_t) t->ne[0] != m->n_ff) { goto bad_shape; }
        L->ffn_sub_norm = (const float *) t->data;

#undef LNAME
    }

    *out = m;
    return SLLM_OK;

bad_kv:
    free(m);
    return SLLM_ERR_KV_MISSING;
bad_shape:
    free(m);
    return SLLM_ERR_TENSOR_SHAPE;
}

void sllm_model_free(sllm_model * m) { free(m); }

int32_t sllm_model_n_vocab(const sllm_model * m) { return m != NULL ? m->n_vocab : 0; }
int32_t sllm_model_n_embd(const sllm_model * m)  { return m != NULL ? m->n_embd : 0; }

/* ------------------------------------------------------------------ */
/* context                                                              */
/* ------------------------------------------------------------------ */

sllm_status sllm_ctx_new(const sllm_model * m, int32_t n_ctx, sllm_ctx ** out) {
    if (m == NULL || out == NULL || n_ctx <= 0) {
        return SLLM_ERR_ARG;
    }
    *out = NULL;

    sllm_ctx * c = (sllm_ctx *) calloc(1, sizeof(sllm_ctx));
    if (c == NULL) {
        return SLLM_ERR_NOMEM;
    }
    c->n_ctx = n_ctx;
    c->n_past = 0;
    c->kv_head_stride = (size_t) m->n_head_kv * (size_t) n_ctx * (size_t) m->n_embd_head;

    const size_t kv_per_layer = (size_t) m->n_head_kv * (size_t) n_ctx * (size_t) m->n_embd_head;
    const size_t kv_total = kv_per_layer * (size_t) m->n_layer;
    const size_t big = (size_t) (m->n_embd > m->n_ff ? m->n_embd : m->n_ff);

    c->k_cache = (float *) calloc(kv_total, sizeof(float));
    c->v_cache = (float *) calloc(kv_total, sizeof(float));
    c->x        = (float *) malloc((size_t) m->n_embd * sizeof(float));
    c->xn       = (float *) malloc((size_t) m->n_embd * sizeof(float));
    c->ffn_inp  = (float *) malloc((size_t) m->n_embd * sizeof(float));
    c->q        = (float *) malloc((size_t) m->n_embd * sizeof(float));
    c->k        = (float *) malloc((size_t) m->n_embd_gqa * sizeof(float));
    c->v        = (float *) malloc((size_t) m->n_embd_gqa * sizeof(float));
    c->attn     = (float *) malloc((size_t) m->n_embd * sizeof(float));
    c->proj     = (float *) malloc((size_t) m->n_embd * sizeof(float));
    c->ffn_gate_buf = (float *) malloc((size_t) m->n_ff * sizeof(float));
    c->ffn_up_buf   = (float *) malloc((size_t) m->n_ff * sizeof(float));
    c->ffn_h    = (float *) malloc((size_t) m->n_ff * sizeof(float));
    c->scores   = (float *) malloc((size_t) n_ctx * sizeof(float));
    c->q_act    = (int8_t *) malloc((size_t) big);
    c->q_act_ffn= (int8_t *) malloc((size_t) big);
    c->dots     = (int32_t *) malloc(big * sizeof(int32_t));

    if (c->k_cache == NULL || c->v_cache == NULL || c->x == NULL || c->xn == NULL ||
        c->ffn_inp == NULL || c->q == NULL || c->k == NULL || c->v == NULL ||
        c->attn == NULL || c->proj == NULL || c->ffn_gate_buf == NULL ||
        c->ffn_up_buf == NULL || c->ffn_h == NULL || c->scores == NULL ||
        c->q_act == NULL || c->q_act_ffn == NULL || c->dots == NULL) {
        sllm_ctx_free(c);
        return SLLM_ERR_NOMEM;
    }

    *out = c;
    return SLLM_OK;
}

void sllm_ctx_free(sllm_ctx * c) {
    if (c == NULL) { return; }
    free(c->k_cache); free(c->v_cache);
    free(c->x); free(c->xn); free(c->ffn_inp);
    free(c->q); free(c->k); free(c->v); free(c->attn); free(c->proj);
    free(c->ffn_gate_buf); free(c->ffn_up_buf); free(c->ffn_h);
    free(c->scores); free(c->q_act); free(c->q_act_ffn); free(c->dots);
    free(c);
}

void sllm_ctx_reset(sllm_ctx * c) {
    if (c == NULL) { return; }
    /*
     * Clear the live prefix, not the whole cache: a 4096-deep cache is 600 MB
     * and clearing all of it would dominate a short generation.
     *
     * This has to actually clear. The KV cache is read up to `n_past` at every
     * position and never zeroed on write, so a context reused without a reset
     * attends to the previous conversation's keys and values. That is not a
     * subtle numerical difference: it is a different model output, and it
     * reproduces only when the caller happens to reuse a context, which is
     * exactly the case a test is least likely to cover.
     */
    if (c->k_cache != NULL && c->n_past > 0) {
        const size_t live = (size_t) c->n_past * (size_t) c->kv_head_stride;
        memset(c->k_cache, 0, live * sizeof(float));
        memset(c->v_cache, 0, live * sizeof(float));
    }
    c->n_past = 0;
}

/* ------------------------------------------------------------------ */
/* the pieces of one layer                                              */
/* ------------------------------------------------------------------ */

/*
 * An I2_S projection: quantise the activation, take the integer dot against
 * every output row, then the epilogue.
 *
 * The epilogue is ggml-cpu.c's, not ggml-bitnet-compute.c's: the division
 * happens once per column and the per-element work is a subtraction and a
 * multiply. See PHASE4-CONTRACT.md. The whole result is deterministic integer
 * arithmetic plus one multiply per element, so this is exact against the
 * reference for the same weights and activation.
 */
static void i2s_project(const uint8_t * w, size_t n_rows, size_t n, float w_scale,
                        const float * x, int8_t * q, int32_t * dots,
                        float * out) {
    sllm_i2s_act act;
    sllm_i2s_quant_act(x, n, q, &act);
    sllm_i2s_gemv(w, n_rows, n, q, dots);
    const float post = w_scale / act.scale;
    for (size_t r = 0; r < n_rows; ++r) {
        out[r] = ((float) (dots[r] - act.sum)) * post;
    }
}

static float i2s_scale_of(const uint8_t * w, size_t n_rows, size_t n) {
    return sllm_i2s_scale(w, n_rows * n);
}

sllm_status sllm_forward(const sllm_model * m, sllm_ctx * c,
                         int32_t token, int32_t pos, float * logits) {
    if (m == NULL || c == NULL || logits == NULL) {
        return SLLM_ERR_ARG;
    }
    if (pos < 0 || pos >= c->n_ctx) {
        return SLLM_ERR_ARG;
    }
    if (token < 0 || token >= m->n_vocab) {
        return SLLM_ERR_ARG;
    }
    if (pos < c->n_past) {
        /* Rewriting a cached position is not something the caller can do by
         * accident, and silently accepting it would interleave two different
         * conversations in one cache. */
        return SLLM_ERR_ARG;
    }
    if (pos + 1 > c->n_past) {
        c->n_past = pos + 1;
    }

    const int32_t n_embd = m->n_embd;
    const int32_t n_ff   = m->n_ff;
    const int32_t hd     = m->n_embd_head;
    const int32_t hkv    = m->n_head_kv;
    const int32_t g      = m->n_head / hkv;      /* query heads per kv head */
    const float   attn_scale = 1.0f / sqrtf((float) hd);

    /* token_embd: the embedding lookup is a row of the tied F16 matrix. */
    for (int32_t d = 0; d < n_embd; ++d) {
        c->x[d] = f16_to_f32(m->tok_embd[(size_t) token * n_embd + d]);
    }

    const size_t kv_per_layer = (size_t) hkv * (size_t) c->n_ctx * (size_t) hd;

    for (int32_t il = 0; il < m->n_layer; ++il) {
        const sllm_layer * L = &m->layer[il];

        /* attn_norm */
        sllm_rms_norm(c->xn, c->x, L->attn_norm, (size_t) n_embd, m->f_rms_eps);

        /* Q, K, V */
        i2s_project(L->wq, (size_t) n_embd, (size_t) n_embd,
                    i2s_scale_of(L->wq, (size_t) n_embd, (size_t) n_embd),
                    c->xn, c->q_act, c->dots, c->q);
        i2s_project(L->wk, (size_t) m->n_embd_gqa, (size_t) n_embd,
                    i2s_scale_of(L->wk, (size_t) m->n_embd_gqa, (size_t) n_embd),
                    c->xn, c->q_act, c->dots, c->k);
        i2s_project(L->wv, (size_t) m->n_embd_gqa, (size_t) n_embd,
                    i2s_scale_of(L->wv, (size_t) m->n_embd_gqa, (size_t) n_embd),
                    c->xn, c->q_act, c->dots, c->v);

        /* RoPE, NeoX, on Q and K only. */
        for (int32_t h = 0; h < m->n_head; ++h) {
            sllm_rope_inplace(c->q + h * hd, m->n_rot, pos,
                              m->rope_freq_base, 1.0f, SLLM_ROPE_NEOX);
        }
        for (int32_t h = 0; h < hkv; ++h) {
            sllm_rope_inplace(c->k + h * hd, m->n_rot, pos,
                              m->rope_freq_base, 1.0f, SLLM_ROPE_NEOX);
        }

        /* Store K and V for this position, per kv head. */
        float * kbase = c->k_cache + (size_t) il * kv_per_layer + (size_t) pos * hd;
        float * vbase = c->v_cache + (size_t) il * kv_per_layer + (size_t) pos * hd;
        for (int32_t h = 0; h < hkv; ++h) {
            memcpy(kbase + (size_t) h * c->n_ctx * hd, c->k + h * hd, (size_t) hd * sizeof(float));
            memcpy(vbase + (size_t) h * c->n_ctx * hd, c->v + h * hd, (size_t) hd * sizeof(float));
        }

        /* Causal attention over positions 0..pos. */
        const int32_t npast = pos + 1;
        for (int32_t h = 0; h < m->n_head; ++h) {
            const int32_t kvh = h / g;
            const float * qh = c->q + h * hd;
            const float * kcache = c->k_cache + (size_t) il * kv_per_layer +
                                   (size_t) kvh * c->n_ctx * hd;
            const float * vcache = c->v_cache + (size_t) il * kv_per_layer +
                                   (size_t) kvh * c->n_ctx * hd;

            for (int32_t s = 0; s < npast; ++s) {
                const float * ks = kcache + (size_t) s * hd;
                float acc = 0.0f;
                for (int32_t d = 0; d < hd; ++d) { acc += qh[d] * ks[d]; }
                c->scores[s] = acc * attn_scale;
            }
            sllm_softmax_inplace(c->scores, (size_t) npast);

            float * oh = c->attn + h * hd;
            for (int32_t d = 0; d < hd; ++d) { oh[d] = 0.0f; }
            for (int32_t s = 0; s < npast; ++s) {
                const float w = c->scores[s];
                const float * vs = vcache + (size_t) s * hd;
                for (int32_t d = 0; d < hd; ++d) { oh[d] += w * vs[d]; }
            }
        }

        /* attn_sub_norm, then the output projection. */
        sllm_rms_norm(c->xn, c->attn, L->attn_sub_norm, (size_t) n_embd, m->f_rms_eps);
        i2s_project(L->wo, (size_t) n_embd, (size_t) n_embd,
                    i2s_scale_of(L->wo, (size_t) n_embd, (size_t) n_embd),
                    c->xn, c->q_act, c->dots, c->proj);

        /* ffn_inp = attn_out + inpSA */
        for (int32_t d = 0; d < n_embd; ++d) { c->ffn_inp[d] = c->proj[d] + c->x[d]; }

        /* FFN: gate and up are PARALLEL, both from ffn_norm(x). */
        sllm_rms_norm(c->xn, c->ffn_inp, L->ffn_norm, (size_t) n_embd, m->f_rms_eps);
        i2s_project(L->ffn_gate, (size_t) n_ff, (size_t) n_embd,
                    i2s_scale_of(L->ffn_gate, (size_t) n_ff, (size_t) n_embd),
                    c->xn, c->q_act_ffn, c->dots, c->ffn_gate_buf);
        i2s_project(L->ffn_up, (size_t) n_ff, (size_t) n_embd,
                    i2s_scale_of(L->ffn_up, (size_t) n_ff, (size_t) n_embd),
                    c->xn, c->q_act_ffn, c->dots, c->ffn_up_buf);

        /* silu(gate) * up */
        sllm_silu_inplace(c->ffn_gate_buf, (size_t) n_ff);
        for (int32_t i = 0; i < n_ff; ++i) {
            c->ffn_h[i] = c->ffn_gate_buf[i] * c->ffn_up_buf[i];
        }

        /* ffn_sub_norm is over n_ff, then down to n_embd. */
        sllm_rms_norm(c->ffn_h, c->ffn_h, L->ffn_sub_norm, (size_t) n_ff, m->f_rms_eps);
        i2s_project(L->ffn_down, (size_t) n_embd, (size_t) n_ff,
                    i2s_scale_of(L->ffn_down, (size_t) n_embd, (size_t) n_ff),
                    c->ffn_h, c->q_act_ffn, c->dots, c->proj);

        /* l_out = ffn_down + ffn_inp */
        for (int32_t d = 0; d < n_embd; ++d) { c->x[d] = c->proj[d] + c->ffn_inp[d]; }
    }

    /* output_norm, then the tied lm_head. */
    sllm_rms_norm(c->xn, c->x, m->output_norm, (size_t) n_embd, m->f_rms_eps);
    for (int32_t v = 0; v < m->n_vocab; ++v) {
        const uint16_t * row = m->tok_embd + (size_t) v * n_embd;
        float acc = 0.0f;
        for (int32_t d = 0; d < n_embd; ++d) { acc += f16_to_f32(row[d]) * c->xn[d]; }
        logits[v] = acc;
    }

    return SLLM_OK;
}

int32_t sllm_argmax(const float * logits, int32_t n) {
    int32_t best = 0;
    float bv = logits[0];
    for (int32_t i = 1; i < n; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

sllm_status sllm_generate_greedy(const sllm_model * m, sllm_ctx * c,
                                 const int32_t * prompt, int32_t n_prompt,
                                 int32_t n_new, int32_t * out) {
    if (m == NULL || c == NULL || out == NULL || n_prompt < 0 || n_new < 0) {
        return SLLM_ERR_ARG;
    }

    if (n_prompt + n_new + c->n_past > c->n_ctx) {
        return SLLM_ERR_TOO_LARGE;
    }

    float * logits = (float *) malloc((size_t) m->n_vocab * sizeof(float));
    if (logits == NULL) {
        return SLLM_ERR_NOMEM;
    }

    /* Prefill, then take the argmax of the final prompt position. */
    const int32_t base = c->n_past;
    int32_t next = 128000;   /* the model's BOS, used only when n_prompt == 0 */
    for (int32_t i = 0; i < n_prompt; ++i) {
        const sllm_status rc = sllm_forward(m, c, prompt[i], base + i, logits);
        if (rc != SLLM_OK) { free(logits); return rc; }
    }
    if (n_prompt > 0) {
        next = sllm_argmax(logits, m->n_vocab);
    }

    for (int32_t i = 0; i < n_new; ++i) {
        out[i] = next;
        const sllm_status rc = sllm_forward(m, c, next, base + n_prompt + i, logits);
        if (rc != SLLM_OK) { free(logits); return rc; }
        next = sllm_argmax(logits, m->n_vocab);
    }

    free(logits);
    return SLLM_OK;
}

sllm_status sllm_forward_prefill(const sllm_model * m, sllm_ctx * c,
                                 const int32_t * tokens, int32_t n,
                                 float * logits_out) {
    if (m == NULL || c == NULL || tokens == NULL || logits_out == NULL || n < 0) {
        return SLLM_ERR_ARG;
    }
    if (n > c->n_ctx) {
        return SLLM_ERR_TOO_LARGE;
    }
    for (int32_t i = 0; i < n; ++i) {
        const sllm_status rc = sllm_forward(m, c, tokens[i], i,
                                            logits_out + (size_t) i * m->n_vocab);
        if (rc != SLLM_OK) {
            return rc;
        }
    }
    return SLLM_OK;
}
