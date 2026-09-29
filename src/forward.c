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


    /* Chunk scratch. Every vector buffer holds `chunk` columns, laid out
     * [n * chunk] so column t's vector starts at t*n. Keeping the column
     * stride explicit is what lets one token and a full chunk share this code
     * without a special case. */
    float   * cx;       /* the running residual,     chunk * n_embd */
    float   * cxn;      /* normalised copy,          chunk * n_embd */
    float   * cffn_inp; /*                            chunk * n_embd */
    float   * cq;       /*                            chunk * n_embd */
    float   * ck;       /*                            chunk * n_embd_gqa */
    float   * cv;       /*                            chunk * n_embd_gqa */
    float   * cattn;    /* attention output,         chunk * n_embd */
    float   * cproj;    /* a projection result,      chunk * n_embd */
    float   * cffn_g;   /*                            chunk * n_ff   */
    float   * cffn_u;   /*                            chunk * n_ff   */
    float   * cffn_h;   /*                            chunk * n_ff   */
    int8_t  * q_act;      /* SLLM_MAX_CHUNK * n_embd   */
    int8_t  * q_act_ffn;  /* SLLM_MAX_CHUNK * n_ff     */
    int32_t * dots;       /* SLLM_MAX_CHUNK * max(n_embd, n_ff) */
    float   * cscores;    /* n_ctx                      */
    float   * last_logits; /* n_vocab, from the most recent position */
    int32_t   chunk_cap;/* the largest chunk these buffers hold      */
    int32_t   n_layer;  /* the model this cache was sized for       */

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
    c->n_layer = m->n_layer;
    c->kv_head_stride = (size_t) m->n_head_kv * (size_t) n_ctx * (size_t) m->n_embd_head;

    const size_t kv_per_layer = (size_t) m->n_head_kv * (size_t) n_ctx * (size_t) m->n_embd_head;
    const size_t kv_total = kv_per_layer * (size_t) m->n_layer;
    const size_t big = (size_t) (m->n_embd > m->n_ff ? m->n_embd : m->n_ff);

    c->k_cache = (float *) calloc(kv_total, sizeof(float));
    c->v_cache = (float *) calloc(kv_total, sizeof(float));
    c->cscores  = (float *) malloc((size_t) n_ctx * sizeof(float));
    c->last_logits = (float *) malloc((size_t) m->n_vocab * sizeof(float));
    c->chunk_cap = SLLM_MAX_CHUNK;
    c->cx     = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd    * sizeof(float));
    c->cxn    = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd    * sizeof(float));
    c->cffn_inp=(float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd   * sizeof(float));
    c->cq     = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd    * sizeof(float));
    c->ck     = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd_gqa* sizeof(float));
    c->cv     = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd_gqa* sizeof(float));
    c->cattn  = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd    * sizeof(float));
    c->cproj  = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_embd    * sizeof(float));
    c->cffn_g = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_ff      * sizeof(float));
    c->cffn_u = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_ff      * sizeof(float));
    c->cffn_h = (float *) malloc((size_t) SLLM_MAX_CHUNK * (size_t) m->n_ff      * sizeof(float));
    c->q_act    = (int8_t *)  malloc((size_t) SLLM_MAX_CHUNK * (size_t) big);
    c->q_act_ffn= (int8_t *)  malloc((size_t) SLLM_MAX_CHUNK * (size_t) big);
    c->dots     = (int32_t *) malloc((size_t) SLLM_MAX_CHUNK * big * sizeof(int32_t));

    if (c->k_cache == NULL || c->v_cache == NULL ||
        c->cx == NULL || c->cxn == NULL || c->cffn_inp == NULL || c->cq == NULL ||
        c->ck == NULL || c->cv == NULL || c->cattn == NULL || c->cproj == NULL ||
        c->cffn_g == NULL || c->cffn_u == NULL || c->cffn_h == NULL ||
        c->cscores == NULL || c->last_logits == NULL || c->q_act == NULL || c->q_act_ffn == NULL ||
        c->dots == NULL) {
        sllm_ctx_free(c);
        return SLLM_ERR_NOMEM;
    }

    *out = c;
    return SLLM_OK;
}

void sllm_ctx_free(sllm_ctx * c) {
    if (c == NULL) { return; }
    free(c->k_cache); free(c->v_cache);
    free(c->cscores); free(c->last_logits); free(c->q_act); free(c->q_act_ffn); free(c->dots);
    free(c->cx); free(c->cxn); free(c->cffn_inp); free(c->cq); free(c->ck);
    free(c->cv); free(c->cattn); free(c->cproj);
    free(c->cffn_g); free(c->cffn_u); free(c->cffn_h);
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
     *
     * Every layer is cleared. The cache is laid out
     * [layer][head_kv][pos][n_embd_head], so one layer's live prefix is
     * n_past * n_head_kv * n_embd_head and the layers follow contiguously;
     * clearing n_past * kv_head_stride would zero layer 0 only.
     *
     * Be clear about how much this is worth. Setting n_past = 0 is what makes
     * the reset correct: a following prefill reads only what it just wrote, so
     * the memset is defence in depth against a future path that reads the cache
     * before writing it, not a fix for anything observable today. It is kept
     * correct rather than cheap, because the day it matters it will be in a
     * layer 29 key that nobody thought to look at.
     */
    if (c->k_cache != NULL && c->n_past > 0 && c->n_layer > 0) {
        const size_t live_per_layer = (size_t) c->n_past * c->kv_head_stride / (size_t) c->n_ctx;
        const size_t live = live_per_layer * (size_t) c->n_layer;
        memset(c->k_cache, 0, live * sizeof(float));
        memset(c->v_cache, 0, live * sizeof(float));
    }
    c->n_past = 0;
}

/* ------------------------------------------------------------------ */
/* the pieces of one layer                                              */
/* ------------------------------------------------------------------ */

static float i2s_scale_of(const uint8_t * w, size_t n_rows, size_t n) {
    return sllm_i2s_scale(w, n_rows * n);
}

/*
 * One chunk: `n` tokens at absolute positions [base, base+n).
 *
 * Every vector buffer holds `n` columns of width w, so column t starts at
 * t*w. The inner loops are written once and used for both a single token and a
 * full chunk, which is what makes the two paths identical by construction
 * rather than by agreement.
 *
 * Attention is the only place a chunk could go wrong, and the rule is the
 * reference's: a query at absolute position p sees every cached key at
 * position q <= p, and the diagonal is included. Its mask fills with
 * -INFINITY and skips any key with p0 > p1 (llama-graph.cpp:450), so the
 * comparison is on absolute positions and not on indices within the batch.
 * That is why a chunk boundary is not a semantic boundary: key indices below
 * the chunk are cached and have smaller positions, and keys above the chunk
 * have not been written yet.
 */
sllm_status sllm_forward_chunk(const sllm_model * m, sllm_ctx * c,
                               const int32_t * tokens, int32_t n, int32_t base,
                               float * logits_out) {
    if (m == NULL || c == NULL || tokens == NULL || logits_out == NULL) {
        return SLLM_ERR_ARG;
    }
    if (n < 1 || n > c->chunk_cap) {
        return SLLM_ERR_ARG;
    }
    if (base < c->n_past || base + n > c->n_ctx) {
        /* Rewinding is refused: the cache is not zeroed on write, so a token
         * placed behind n_past would be read as if it belonged to whatever
         * wrote that position before. */
        return SLLM_ERR_ARG;
    }
    for (int32_t t = 0; t < n; ++t) {
        if (tokens[t] < 0 || tokens[t] >= m->n_vocab) {
            return SLLM_ERR_ARG;
        }
    }
    c->n_past = base + n;

    const int32_t n_embd = m->n_embd;
    const int32_t n_ff   = m->n_ff;
    const int32_t hd     = m->n_embd_head;
    const int32_t hkv    = m->n_head_kv;
    const int32_t g      = m->n_head / hkv;   /* query heads per kv head */
    const float   attn_scale = 1.0f / sqrtf((float) hd);

    const size_t kv_per_layer = (size_t) hkv * (size_t) c->n_ctx * (size_t) hd;
    const size_t big = (size_t) (n_embd > n_ff ? n_embd : n_ff);

    for (int32_t t = 0; t < n; ++t) {
        for (int32_t d = 0; d < n_embd; ++d) {
            c->cx[(size_t) t * n_embd + d] =
                f16_to_f32(m->tok_embd[(size_t) tokens[t] * n_embd + d]);
        }
    }

    for (int32_t il = 0; il < m->n_layer; ++il) {
        const sllm_layer * L = &m->layer[il];

        /* attn_norm, per column. */
        for (int32_t t = 0; t < n; ++t) {
            sllm_rms_norm(c->cxn + (size_t) t * n_embd,
                          c->cx   + (size_t) t * n_embd,
                          L->attn_norm, (size_t) n_embd, m->f_rms_eps);
        }

        /* Q, K, V: each column is quantised on its own, because the
         * activation scale and sum are per column and the epilogue divides
         * once per column. Quantising the chunk as a block would be a
         * different model. */
        const float ws_q = i2s_scale_of(L->wq, (size_t) n_embd, (size_t) n_embd);
        const float ws_k = i2s_scale_of(L->wk, (size_t) m->n_embd_gqa, (size_t) n_embd);
        const float ws_v = i2s_scale_of(L->wv, (size_t) m->n_embd_gqa, (size_t) n_embd);
        for (int32_t t = 0; t < n; ++t) {
            const float * src = c->cxn + (size_t) t * n_embd;
            int8_t * qbuf = c->q_act + (size_t) t * big;
            int32_t * dbuf = c->dots + (size_t) t * big;

            sllm_i2s_act act;
            sllm_i2s_quant_act(src, (size_t) n_embd, qbuf, &act);
            sllm_i2s_gemv(L->wq, (size_t) n_embd, (size_t) n_embd, qbuf, dbuf);
            const float pq = ws_q / act.scale;
            float * o = c->cq + (size_t) t * n_embd;
            for (int32_t r = 0; r < n_embd; ++r) { o[r] = ((float) (dbuf[r] - act.sum)) * pq; }

            sllm_i2s_gemv(L->wk, (size_t) m->n_embd_gqa, (size_t) n_embd, qbuf, dbuf);
            const float pk = ws_k / act.scale;
            o = c->ck + (size_t) t * m->n_embd_gqa;
            for (int32_t r = 0; r < m->n_embd_gqa; ++r) { o[r] = ((float) (dbuf[r] - act.sum)) * pk; }

            sllm_i2s_gemv(L->wv, (size_t) m->n_embd_gqa, (size_t) n_embd, qbuf, dbuf);
            const float pv = ws_v / act.scale;
            o = c->cv + (size_t) t * m->n_embd_gqa;
            for (int32_t r = 0; r < m->n_embd_gqa; ++r) { o[r] = ((float) (dbuf[r] - act.sum)) * pv; }
        }

        /* RoPE, NeoX, on Q and K only, per token and per head. */
        for (int32_t t = 0; t < n; ++t) {
            const int32_t pos = base + t;
            float * qh = c->cq + (size_t) t * n_embd;
            float * kh = c->ck + (size_t) t * m->n_embd_gqa;
            for (int32_t h = 0; h < m->n_head; ++h) {
                sllm_rope_inplace(qh + h * hd, m->n_rot, pos,
                                  m->rope_freq_base, 1.0f, SLLM_ROPE_NEOX);
            }
            for (int32_t h = 0; h < hkv; ++h) {
                sllm_rope_inplace(kh + h * hd, m->n_rot, pos,
                                  m->rope_freq_base, 1.0f, SLLM_ROPE_NEOX);
            }
        }

        /* Store K and V for the whole chunk. */
        float * kbase = c->k_cache + (size_t) il * kv_per_layer;
        float * vbase = c->v_cache + (size_t) il * kv_per_layer;
        for (int32_t t = 0; t < n; ++t) {
            const int32_t pos = base + t;
            for (int32_t h = 0; h < hkv; ++h) {
                memcpy(kbase + (size_t) h * c->n_ctx * hd + (size_t) pos * hd,
                       c->ck + (size_t) t * m->n_embd_gqa + (size_t) h * hd,
                       (size_t) hd * sizeof(float));
                memcpy(vbase + (size_t) h * c->n_ctx * hd + (size_t) pos * hd,
                       c->cv + (size_t) t * m->n_embd_gqa + (size_t) h * hd,
                       (size_t) hd * sizeof(float));
            }
        }

        /* Causal attention. Query t sits at position base+t and sees keys
         * 0..base+t inclusive, so the diagonal is in and everything above it
         * is out. Keys inside the chunk are already stored above. */
        for (int32_t t = 0; t < n; ++t) {
            const int32_t qpos = base + t;
            const int32_t npast = qpos + 1;
            for (int32_t h = 0; h < m->n_head; ++h) {
                const int32_t kvh = h / g;
                const float * qh = c->cq + (size_t) t * n_embd + (size_t) h * hd;
                const float * kcache = kbase + (size_t) kvh * c->n_ctx * hd;
                const float * vcache = vbase + (size_t) kvh * c->n_ctx * hd;

                for (int32_t s = 0; s < npast; ++s) {
                    const float * ks = kcache + (size_t) s * hd;
                    float acc = 0.0f;
                    for (int32_t d = 0; d < hd; ++d) { acc += qh[d] * ks[d]; }
                    c->cscores[s] = acc * attn_scale;
                }
                sllm_softmax_inplace(c->cscores, (size_t) npast);

                float * oh = c->cattn + (size_t) t * n_embd + (size_t) h * hd;
                for (int32_t d = 0; d < hd; ++d) { oh[d] = 0.0f; }
                for (int32_t s = 0; s < npast; ++s) {
                    const float w = c->cscores[s];
                    const float * vs = vcache + (size_t) s * hd;
                    for (int32_t d = 0; d < hd; ++d) { oh[d] += w * vs[d]; }
                }
            }
        }

        /* attn_sub_norm, then the output projection. */
        const float ws_o = i2s_scale_of(L->wo, (size_t) n_embd, (size_t) n_embd);
        for (int32_t t = 0; t < n; ++t) {
            sllm_rms_norm(c->cxn + (size_t) t * n_embd,
                          c->cattn + (size_t) t * n_embd,
                          L->attn_sub_norm, (size_t) n_embd, m->f_rms_eps);
            int8_t * qbuf = c->q_act + (size_t) t * big;
            int32_t * dbuf = c->dots + (size_t) t * big;
            sllm_i2s_act act;
            sllm_i2s_quant_act(c->cxn + (size_t) t * n_embd, (size_t) n_embd, qbuf, &act);
            sllm_i2s_gemv(L->wo, (size_t) n_embd, (size_t) n_embd, qbuf, dbuf);
            const float po = ws_o / act.scale;
            float * o = c->cproj + (size_t) t * n_embd;
            for (int32_t r = 0; r < n_embd; ++r) { o[r] = ((float) (dbuf[r] - act.sum)) * po; }
        }

        /* ffn_inp = attn_out + inpSA */
        for (int32_t t = 0; t < n; ++t) {
            float * fi = c->cffn_inp + (size_t) t * n_embd;
            for (int32_t d = 0; d < n_embd; ++d) {
                fi[d] = c->cproj[(size_t) t * n_embd + d] + c->cx[(size_t) t * n_embd + d];
            }
        }

        /* FFN: gate and up are PARALLEL, both from ffn_norm's output. */
        const float ws_g = i2s_scale_of(L->ffn_gate, (size_t) n_ff, (size_t) n_embd);
        const float ws_u = i2s_scale_of(L->ffn_up,   (size_t) n_ff, (size_t) n_embd);
        const float ws_d = i2s_scale_of(L->ffn_down, (size_t) n_embd, (size_t) n_ff);
        for (int32_t t = 0; t < n; ++t) {
            sllm_rms_norm(c->cxn + (size_t) t * n_embd,
                          c->cffn_inp + (size_t) t * n_embd,
                          L->ffn_norm, (size_t) n_embd, m->f_rms_eps);
            int8_t * qbuf = c->q_act_ffn + (size_t) t * big;
            int32_t * dbuf = c->dots + (size_t) t * big;
            sllm_i2s_act act;
            sllm_i2s_quant_act(c->cxn + (size_t) t * n_embd, (size_t) n_embd, qbuf, &act);

            sllm_i2s_gemv(L->ffn_gate, (size_t) n_ff, (size_t) n_embd, qbuf, dbuf);
            const float pg = ws_g / act.scale;
            float * og = c->cffn_g + (size_t) t * n_ff;
            for (int32_t r = 0; r < n_ff; ++r) { og[r] = ((float) (dbuf[r] - act.sum)) * pg; }

            sllm_i2s_gemv(L->ffn_up, (size_t) n_ff, (size_t) n_embd, qbuf, dbuf);
            const float pu = ws_u / act.scale;
            float * ou = c->cffn_u + (size_t) t * n_ff;
            for (int32_t r = 0; r < n_ff; ++r) { ou[r] = ((float) (dbuf[r] - act.sum)) * pu; }

            float * oh = c->cffn_h + (size_t) t * n_ff;
            for (int32_t r = 0; r < n_ff; ++r) { oh[r] = og[r] * ou[r]; }
            sllm_silu_inplace(og, (size_t) n_ff);
            for (int32_t r = 0; r < n_ff; ++r) { oh[r] = og[r] * ou[r]; }
        }

        /* ffn_sub_norm is over n_ff, then down to n_embd. */
        for (int32_t t = 0; t < n; ++t) {
            float * hcol = c->cffn_h + (size_t) t * n_ff;
            sllm_rms_norm(hcol, hcol, L->ffn_sub_norm, (size_t) n_ff, m->f_rms_eps);
            int8_t * qbuf = c->q_act_ffn + (size_t) t * big;
            int32_t * dbuf = c->dots + (size_t) t * big;
            sllm_i2s_act act;
            sllm_i2s_quant_act(hcol, (size_t) n_ff, qbuf, &act);
            sllm_i2s_gemv(L->ffn_down, (size_t) n_embd, (size_t) n_ff, qbuf, dbuf);
            const float pd = ws_d / act.scale;
            float * o = c->cproj + (size_t) t * n_embd;
            for (int32_t r = 0; r < n_embd; ++r) { o[r] = ((float) (dbuf[r] - act.sum)) * pd; }
        }

        /* l_out = ffn_down + ffn_inp */
        for (int32_t t = 0; t < n; ++t) {
            for (int32_t d = 0; d < n_embd; ++d) {
                c->cx[(size_t) t * n_embd + d] =
                    c->cproj[(size_t) t * n_embd + d] + c->cffn_inp[(size_t) t * n_embd + d];
            }
        }
    }

    /* output_norm, then the tied lm_head, per column. */
    for (int32_t t = 0; t < n; ++t) {
        float * xcol = c->cx + (size_t) t * n_embd;
        sllm_rms_norm(c->cxn + (size_t) t * n_embd, xcol,
                      m->output_norm, (size_t) n_embd, m->f_rms_eps);
        const float * xn = c->cxn + (size_t) t * n_embd;
        float * out = logits_out + (size_t) t * m->n_vocab;
        for (int32_t v = 0; v < m->n_vocab; ++v) {
            const uint16_t * row = m->tok_embd + (size_t) v * n_embd;
            float acc = 0.0f;
            for (int32_t d = 0; d < n_embd; ++d) { acc += f16_to_f32(row[d]) * xn[d]; }
            out[v] = acc;
        }
    }

    /* Keep the final position's logits so a context can be continued or
     * restored without re-running it. See sllm_ctx_last_logits. */
    memcpy(c->last_logits, logits_out + (size_t) (n - 1) * m->n_vocab,
           (size_t) m->n_vocab * sizeof(float));

    return SLLM_OK;
}

const float * sllm_ctx_last_logits(const sllm_ctx * c) {
    return (c != NULL && c->last_logits != NULL) ? c->last_logits : NULL;
}

sllm_status sllm_forward(const sllm_model * m, sllm_ctx * c,
                         int32_t token, int32_t pos, float * logits) {
    return sllm_forward_chunk(m, c, &token, 1, pos, logits);
}

sllm_status sllm_forward_prefill_chunked(const sllm_model * m, sllm_ctx * c,
                                         const int32_t * tokens, int32_t n,
                                         int32_t chunk, float * logits_out) {
    if (m == NULL || c == NULL || tokens == NULL || logits_out == NULL || n < 0) {
        return SLLM_ERR_ARG;
    }
    if (chunk < 1 || chunk > c->chunk_cap) {
        return SLLM_ERR_ARG;
    }
    if (n > c->n_ctx) {
        return SLLM_ERR_TOO_LARGE;
    }
    for (int32_t done = 0; done < n; ) {
        const int32_t take = (n - done < chunk) ? (n - done) : chunk;
        const sllm_status rc = sllm_forward_chunk(m, c, tokens + done, take, done,
                                                  logits_out + (size_t) done * m->n_vocab);
        if (rc != SLLM_OK) {
            return rc;
        }
        done += take;
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
    /*
     * With no prompt there is nothing to run, so the next token comes from
     * whatever the context already knows -- which is how a restored context
     * continues. A context that has never been forwarded has no last logits,
     * and the model's BOS is the only defensible fallback.
     */
    int32_t next = 128000;
    if (n_prompt == 0 && base > 0) {
        next = sllm_argmax(c->last_logits, m->n_vocab);
        for (int32_t i = 0; i < n_new; ++i) {
            out[i] = next;
            const sllm_status rc = sllm_forward(m, c, next, base + i, logits);
            if (rc != SLLM_OK) { free(logits); return rc; }
            next = sllm_argmax(logits, m->n_vocab);
        }
        free(logits);
        return SLLM_OK;
    }
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

/* ------------------------------------------------------------------ */
/* state save and restore                                               */
/* ------------------------------------------------------------------ */

/*
 * Our own format, not llama.cpp's. The reference writes a magic (0xaf143cd8),
 * a sequence id and the cache in its native type (llama-context.cpp:2916).
 * Reproducing that byte for byte would be copying an implementation, and the
 * parity rule does not ask for it: what has to hold is that a restored context
 * continues identically, which is a property of the arithmetic.
 *
 *   magic     u32   'S''L''K''V'
 *   version   u32   format version
 *   n_layer   i32
 *   n_ctx     i32
 *   n_head_kv i32
 *   n_embd_head i32
 *   n_past    i32
 *   n_vocab   i32
 *   data      n_past * n_layer * n_head_kv * n_embd_head floats, K then V,
 *             followed by n_vocab floats: the last position's logits
 *
 * Every geometry field is checked against the model being loaded into, so a
 * state file from a different model or a different context depth is refused
 * with a specific status rather than read into a buffer. The length is also
 * checked against the file size, so a truncated write is a truncation error
 * and not a short read of somebody else's memory.
 */
#define SLLM_STATE_MAGIC   0x564B4C53u  /* "SLKV" little-endian */
#define SLLM_STATE_VERSION 1u

static bool wu32(FILE * f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool wi32(FILE * f, int32_t v)  { return fwrite(&v, 4, 1, f) == 1; }
static bool ru32(FILE * f, uint32_t * v) { return fread(v, 4, 1, f) == 1; }
static bool ri32(FILE * f, int32_t * v)  { return fread(v, 4, 1, f) == 1; }

sllm_status sllm_state_save(const sllm_model * m, const sllm_ctx * c, const char * path) {
    if (m == NULL || c == NULL || path == NULL) {
        return SLLM_ERR_ARG;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        return SLLM_ERR_IO;
    }

    bool ok = wu32(f, SLLM_STATE_MAGIC) && wu32(f, SLLM_STATE_VERSION) &&
              wi32(f, m->n_layer) && wi32(f, c->n_ctx) &&
              wi32(f, m->n_head_kv) && wi32(f, m->n_embd_head) &&
              wi32(f, c->n_past) && wi32(f, m->n_vocab);

    /* The live prefix of each layer's cache, in cache order, K then V. */
    for (int32_t il = 0; ok && il < m->n_layer; ++il) {
        const size_t per = c->kv_head_stride;
        const float * k = c->k_cache + (size_t) il * per;
        const float * v = c->v_cache + (size_t) il * per;
        for (int32_t h = 0; ok && h < m->n_head_kv; ++h) {
            const size_t want = (size_t) c->n_past * (size_t) m->n_embd_head;
            const float * krow = k + (size_t) h * (size_t) c->n_ctx * m->n_embd_head;
            const float * vrow = v + (size_t) h * (size_t) c->n_ctx * m->n_embd_head;
            if (want > 0 && fwrite(krow, sizeof(float), want, f) != want) { ok = false; }
            if (ok && want > 0 && fwrite(vrow, sizeof(float), want, f) != want) { ok = false; }
        }
    }

    if (ok && fwrite(c->last_logits, sizeof(float), (size_t) m->n_vocab, f)
              != (size_t) m->n_vocab) {
        ok = false;
    }

    if (fclose(f) != 0) {
        ok = false;
    }
    return ok ? SLLM_OK : SLLM_ERR_IO;
}

sllm_status sllm_state_load(const sllm_model * m, sllm_ctx * c, const char * path) {
    if (m == NULL || c == NULL || path == NULL) {
        return SLLM_ERR_ARG;
    }
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        return SLLM_ERR_IO;
    }

    uint32_t magic = 0, version = 0;
    int32_t n_layer = 0, n_ctx = 0, n_head_kv = 0, n_embd_head = 0, n_past = 0, n_vocab = 0;

    if (!ru32(f, &magic) || magic != SLLM_STATE_MAGIC) {
        fclose(f);
        return SLLM_ERR_KV_TYPE;          /* not one of ours */
    }
    if (!ru32(f, &version) || version != SLLM_STATE_VERSION) {
        fclose(f);
        return SLLM_ERR_KV_TYPE;          /* a version we do not read */
    }
    if (!ri32(f, &n_layer) || !ri32(f, &n_ctx) || !ri32(f, &n_head_kv) ||
        !ri32(f, &n_embd_head) || !ri32(f, &n_past) || !ri32(f, &n_vocab)) {
        fclose(f);
        return SLLM_ERR_GGUF_TRUNCATED;
    }

    /* Geometry must match the model, or the bytes mean something else. */
    if (n_layer != m->n_layer || n_head_kv != m->n_head_kv ||
        n_embd_head != m->n_embd_head || n_ctx != c->n_ctx ||
        n_vocab != m->n_vocab) {
        fclose(f);
        return SLLM_ERR_TENSOR_SHAPE;
    }
    if (n_past < 0 || n_past > c->n_ctx) {
        fclose(f);
        return SLLM_ERR_TENSOR_SHAPE;
    }

    const size_t want = (size_t) n_past * (size_t) n_embd_head;
    for (int32_t il = 0; il < n_layer; ++il) {
        float * k = c->k_cache + (size_t) il * c->kv_head_stride;
        float * v = c->v_cache + (size_t) il * c->kv_head_stride;
        for (int32_t h = 0; h < n_head_kv; ++h) {
            float * krow = k + (size_t) h * (size_t) c->n_ctx * n_embd_head;
            float * vrow = v + (size_t) h * (size_t) c->n_ctx * n_embd_head;
            if (want > 0 && fread(krow, sizeof(float), want, f) != want) {
                fclose(f);
                return SLLM_ERR_GGUF_TRUNCATED;
            }
            if (want > 0 && fread(vrow, sizeof(float), want, f) != want) {
                fclose(f);
                return SLLM_ERR_GGUF_TRUNCATED;
            }
        }
    }
    /* The last position's logits travel with the cache: without them a
     * restored context has no way to name the token that was current when it
     * was saved, and would have to re-run the prompt to find out. */
    if (fread(c->last_logits, sizeof(float), (size_t) m->n_vocab, f) != (size_t) m->n_vocab) {
        fclose(f);
        return SLLM_ERR_GGUF_TRUNCATED;
    }
    fclose(f);

    /*
     * Only the live prefix was written, so positions past n_past still hold
     * whatever the destination context had in them. They must read as absent.
     *
     * The cache is [layer][head][pos][dim], so within a layer the position is
     * NOT the outermost index -- head is. Zeroing one contiguous range from
     * n_past*dim to the end of the layer would clear the wrong bytes: it would
     * run across head boundaries and leave the tail of every earlier head
     * intact. The live-to-live range has to be taken per head.
     */
    for (int32_t il = 0; il < n_layer; ++il) {
        const size_t per_head = (size_t) c->n_ctx * (size_t) n_embd_head;
        for (int32_t h = 0; h < n_head_kv; ++h) {
            const size_t off = (size_t) il * c->kv_head_stride + (size_t) h * per_head;
            const size_t from = (size_t) n_past * (size_t) n_embd_head;
            if (from < (size_t) c->n_ctx * (size_t) n_embd_head) {
                const size_t cnt = (size_t) c->n_ctx * (size_t) n_embd_head - from;
                memset(c->k_cache + off + from, 0, cnt * sizeof(float));
                memset(c->v_cache + off + from, 0, cnt * sizeof(float));
            }
        }
    }

    c->n_past = n_past;
    return SLLM_OK;
}
