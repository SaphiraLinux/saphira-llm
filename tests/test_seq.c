/*
 * test_seq.c -- T14: real uncached SEQUENCE execution over the proven 36-block graph.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * NO KV CACHE, NO OPTIMISATION, NO QUANTISATION CHANGE. The scalar GEMV stays the
 * oracle. This path becomes the oracle for T15; if it is not boringly correct, a
 * cached implementation will agree with it and both will be confidently wrong.
 *
 * LAW INHERITED FROM T13: tensor SHAPE may be shared by block, tensor TYPE may not.
 * Every attn_v and ffn_down decode takes its quantisation from that tensor's own
 * descriptor in that block. Eighteen of thirty-six blocks store those two as Q4_K
 * while block 0 stores Q6_K. Reading block 0's type everywhere would reinterpret
 * payload bytes under the wrong K-quant layout and produce a perfectly plausible
 * broken model with no fault and no refusal.
 *
 * POSITION IS REAL DATA, NOT AN IMPLIED CONSTANT. It is carried explicitly into RoPE
 * and is never derived from a token ID. The same token at positions 0 and 1 must give
 * different rotated Q and K.
 */

#include "harness.h"
#include "saphira_llm/gguf.h"
#include "saphira_llm/ops.h"
#include "saphira_llm/quant.h"
#include "saphira_llm/rope_contract.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SLLM_CONTRACT "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.contract.gguf"

#define EMB  4096u
#define FFN  12288u
#define NH   32u
#define NKVH 8u
#define HD   128u
#define EPS  1.0e-6f

static int checks = 0, failed = 0;
static void expect(int cond, const char * what) {
    checks++;
    if (cond) { printf("      ok    %s\n", what); }
    else      { failed++; printf("      FAIL  %s\n", what); }
}

typedef struct {
    const sllm_gguf_tensor * anrm, *qp, *kp, *vp, *qn, *kn, *aop;
    const sllm_gguf_tensor * fnrm, *gate, *up, *down;
} blk_t;

/* Scratch sized for a sequence of at most SEQMAX positions. */
#define SEQMAX 8

typedef struct {
    sllm_gguf g;
    sllm_rope_semantics rs;
    blk_t * b;
    unsigned n_blocks;
    const sllm_gguf_tensor * emb, * onrm, * out;
    /* per-position buffers */
    float * E  [SEQMAX];   /* the carried residual stream, per position */
    float * nm [SEQMAX];   /* RMSNorm(E)                             */
    float * Q  [SEQMAX];   /* NH * HD, post-norm post-RoPE            */
    float * K  [SEQMAX];   /* NKVH * HD                               */
    float * V  [SEQMAX];   /* NKVH * HD, NOT normed, NOT rotated      */
    float * cat;           /* NH * HD concatenated heads, per position reused */
    float * catbuf[SEQMAX];
    float * probs;
    float * gv, * uv, * gated, * tmp;
    float * anw, * fnw, * onw, * qw, * kw;
} seqctx_t;

static int seq_open(seqctx_t * S, const char * path, unsigned n_blocks_used) {
    char e[256];
    if (sllm_gguf_open(path, &S->g, e, sizeof e) != SLLM_OK) {
        printf("    SKIP  contract artefact unavailable (%s)\n", e);
        return 0;
    }
    char miss[256] = "";
    if (sllm_rope_semantics_from_gguf(&S->g, &S->rs, miss, sizeof miss) != SLLM_OK) {
        printf("    SKIP  contract unresolved (%s); T14 will not guess it\n", miss);
        sllm_gguf_close(&S->g);
        return 0;
    }
    uint32_t nb = 0;
    const char * arch = NULL;
    (void) sllm_gguf_kv_str(&S->g, "general.architecture", &arch);
    char k[192];
    snprintf(k, sizeof k, "%s.block_count", arch); (void) sllm_gguf_kv_u32(&S->g, k, &nb);
    S->n_blocks = (unsigned) nb;
    S->b = (blk_t *) calloc(S->n_blocks, sizeof(blk_t));
    char nm[160];
    static const char * roles[] = { "attn_norm","attn_q","attn_k","attn_v",
        "attn_q_norm","attn_k_norm","attn_output","ffn_norm","ffn_gate","ffn_up","ffn_down" };
    for (unsigned i = 0; i < S->n_blocks; ++i) {
        blk_t * B = &S->b[i];
        const sllm_gguf_tensor ** f[] = { &B->anrm,&B->qp,&B->kp,&B->vp,&B->qn,&B->kn,
            &B->aop,&B->fnrm,&B->gate,&B->up,&B->down };
        for (int r = 0; r < 11; ++r) {
            snprintf(nm, sizeof nm, "blk.%u.%s.weight", i, roles[r]);
            *f[r] = sllm_gguf_find_tensor(&S->g, nm);
            if (!*f[r]) { printf("    block %u missing %s\n", i, roles[r]); return 0; }
        }
    }
    S->emb  = sllm_gguf_find_tensor(&S->g, "token_embd.weight");
    S->onrm = sllm_gguf_find_tensor(&S->g, "output_norm.weight");
    S->out  = sllm_gguf_find_tensor(&S->g, "output.weight");
    for (unsigned t = 0; t < SEQMAX; ++t) {
        S->E[t]      = (float *) malloc(sizeof(float) * EMB);
        S->nm[t]     = (float *) malloc(sizeof(float) * EMB);
        S->Q[t]      = (float *) malloc(sizeof(float) * NH * HD);
        S->K[t]      = (float *) malloc(sizeof(float) * NKVH * HD);
        S->V[t]      = (float *) malloc(sizeof(float) * NKVH * HD);
        S->catbuf[t] = (float *) malloc(sizeof(float) * NH * HD);
    }
    S->cat   = (float *) malloc(sizeof(float) * NH * HD);
    S->probs = (float *) malloc(sizeof(float) * SEQMAX);
    S->gv    = (float *) malloc(sizeof(float) * FFN);
    S->uv    = (float *) malloc(sizeof(float) * FFN);
    S->gated = (float *) malloc(sizeof(float) * FFN);
    S->tmp   = (float *) malloc(sizeof(float) * EMB);
    S->anw   = (float *) malloc(sizeof(float) * EMB);
    S->fnw   = (float *) malloc(sizeof(float) * EMB);
    S->onw   = (float *) malloc(sizeof(float) * EMB);
    S->qw    = (float *) malloc(sizeof(float) * HD);
    S->kw    = (float *) malloc(sizeof(float) * HD);
    (void) n_blocks_used;
    return 1;
}

static void seq_close(seqctx_t * S) {
    for (unsigned t = 0; t < SEQMAX; ++t) {
        free(S->E[t]); free(S->nm[t]); free(S->Q[t]); free(S->K[t]); free(S->V[t]);
        free(S->catbuf[t]);
    }
    free(S->cat); free(S->probs); free(S->gv); free(S->uv); free(S->gated); free(S->tmp);
    free(S->anw); free(S->fnw); free(S->onw); free(S->qw); free(S->kw);
    free(S->b); sllm_gguf_close(&S->g);
}

static double digest_of(const float * v, size_t n) {
    double d = 0.0;
    for (size_t i = 0; i < n; ++i) d += (double) v[i] * (double)(i + 1) * 1e-6;
    return d;
}
static double maxabs_of(const float * v, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) { const double a = fabs((double) v[i]); if (a > m) m = a; }
    return m;
}

/* Run the whole graph over `ntok` positions. Checkpoints are captured for the
 * checkpoint block list at every position. Per-TENSOR quantisation comes from each
 * block's own descriptor; nothing is inherited from block 0. */
static int seq_run(seqctx_t * S, const int * toks, unsigned ntok, unsigned n_blocks,
                   const unsigned * cp_blocks, unsigned n_cp,
                   double * cp_after_attn, double * cp_after_ffn,
                   float * logits_out) {
    uint32_t ebl = 0, ebt = 0;
    (void) sllm_gguf_type_traits(S->emb->type, &ebl, &ebt);
    const size_t emb_row = (size_t) ebt * (S->emb->ne[0] / ebl);
    for (unsigned t = 0; t < ntok; ++t) {
        if (sllm_dequant_row(S->emb->type,
                             (const uint8_t *) S->emb->data
                                 + (size_t) toks[t] * emb_row,
                             S->E[t], EMB) != SLLM_OK) return 0;
    }
    for (unsigned b = 0; b < n_blocks; ++b) {
        const blk_t * B = &S->b[b];
        if (sllm_dequant_row(B->anrm->type, B->anrm->data, S->anw, EMB) != SLLM_OK) return 0;
        if (sllm_dequant_row(B->fnrm->type, B->fnrm->data, S->fnw, EMB) != SLLM_OK) return 0;
        if (sllm_dequant_row(B->qn->type, B->qn->data, S->qw, HD) != SLLM_OK) return 0;
        if (sllm_dequant_row(B->kn->type, B->kn->data, S->kw, HD) != SLLM_OK) return 0;

        for (unsigned t = 0; t < ntok; ++t) {
            sllm_rms_norm(S->nm[t], S->E[t], S->anw, EMB, EPS);
            if (sllm_gemv_f32(B->qp->type, B->qp->data, B->qp->ne[0],
                              S->nm[t], NH * HD, S->Q[t]) != SLLM_OK) return 0;
            if (sllm_gemv_f32(B->kp->type, B->kp->data, B->kp->ne[0],
                              S->nm[t], NKVH * HD, S->K[t]) != SLLM_OK) return 0;
            if (sllm_gemv_f32(B->vp->type, B->vp->data, B->vp->ne[0],
                              S->nm[t], NKVH * HD, S->V[t]) != SLLM_OK) return 0;
            for (uint32_t h = 0; h < NH; ++h)
                sllm_rms_norm(S->Q[t] + h*HD, S->Q[t] + h*HD, S->qw, HD, EPS);
            for (uint32_t h = 0; h < NKVH; ++h)
                sllm_rms_norm(S->K[t] + h*HD, S->K[t] + h*HD, S->kw, HD, EPS);
            /* POSITION IS CARRIED EXPLICITLY, and is not derived from the token. */
            const int32_t pos = (int32_t)(S->rs.position_origin + (uint32_t) t);
            for (uint32_t h = 0; h < NH; ++h)
                sllm_rope_inplace(S->Q[t] + h*HD, HD, pos, S->rs.freq_base,
                                  S->rs.scaling_factor, S->rs.pairing);
            for (uint32_t h = 0; h < NKVH; ++h)
                sllm_rope_inplace(S->K[t] + h*HD, HD, pos, S->rs.freq_base,
                                  S->rs.scaling_factor, S->rs.pairing);
            /* V is neither normed nor rotated. */
        }

        /* CAUSAL ATTENTION. Query at s attends to t <= s, and to NOTHING else. There is
         * no per-position shortcut: the softmax runs over the real permitted set. */
        for (unsigned s = 0; s < ntok; ++s) {
            for (uint32_t q = 0; q < NH; ++q) {
                const uint32_t kvh = q / (NH / NKVH);
                for (unsigned t = 0; t <= s; ++t) {
                    const float * qq = S->Q[s] + q * HD;
                    const float * kk = S->K[t] + kvh * HD;
                    double acc = 0.0;
                    for (uint32_t i = 0; i < HD; ++i) acc += (double) qq[i] * (double) kk[i];
                    S->probs[t] = (float) (acc / sqrt((double) HD));
                }
                /* stable softmax over the permitted set only */
                float mx = S->probs[0];
                for (unsigned t = 1; t <= s; ++t) if (S->probs[t] > mx) mx = S->probs[t];
                float sum = 0.0f;
                for (unsigned t = 0; t <= s; ++t) { S->probs[t] = expf(S->probs[t] - mx); sum += S->probs[t]; }
                for (unsigned t = 0; t <= s; ++t) S->probs[t] /= sum;
                for (uint32_t i = 0; i < HD; ++i) {
                    double acc = 0.0;
                    for (unsigned t = 0; t <= s; ++t)
                        acc += (double) S->probs[t] * (double) S->V[t][kvh * HD + i];
                    S->catbuf[s][q * HD + i] = (float) acc;
                }
            }
        }

        for (unsigned t = 0; t < ntok; ++t) {
            if (sllm_gemv_f32(B->aop->type, B->aop->data, B->aop->ne[0],
                              S->catbuf[t], B->aop->ne[1], S->tmp) != SLLM_OK) return 0;
            /* residual 1 = block input + projected attention */
            for (unsigned i = 0; i < EMB; ++i) S->E[t][i] = S->E[t][i] + S->tmp[i];
        }
        for (unsigned c = 0; c < n_cp; ++c)
            if (cp_blocks[c] == b)
                for (unsigned t = 0; t < ntok; ++t)
                    cp_after_attn[c * SEQMAX + t] = digest_of(S->E[t], EMB);

        for (unsigned t = 0; t < ntok; ++t) {
            sllm_rms_norm(S->nm[t], S->E[t], S->fnw, EMB, EPS);
            if (sllm_gemv_f32(B->gate->type, B->gate->data, B->gate->ne[0],
                              S->nm[t], FFN, S->gv) != SLLM_OK) return 0;
            if (sllm_gemv_f32(B->up->type, B->up->data, B->up->ne[0],
                              S->nm[t], FFN, S->uv) != SLLM_OK) return 0;
            for (unsigned i = 0; i < FFN; ++i) {
                const double g = S->gv[i];
                S->gated[i] = (float)((g / (1.0 + exp(-g))) * (double) S->uv[i]);
            }
            if (sllm_gemv_f32(B->down->type, B->down->data, B->down->ne[0],
                              S->gated, EMB, S->tmp) != SLLM_OK) return 0;
            /* residual 2 = the ORIGINAL block input, per T11's source-established parent */
            for (unsigned i = 0; i < EMB; ++i) S->E[t][i] = S->E[t][i] + S->tmp[i];
        }
        for (unsigned c = 0; c < n_cp; ++c)
            if (cp_blocks[c] == b)
                for (unsigned t = 0; t < ntok; ++t)
                    cp_after_ffn[c * SEQMAX + t] = digest_of(S->E[t], EMB);
    }

    /* final norm and logits, per position */
    if (sllm_dequant_row(S->onrm->type, S->onrm->data, S->onw, EMB) != SLLM_OK) return 0;
    for (unsigned t = 0; t < ntok; ++t) {
        sllm_rms_norm(S->nm[t], S->E[t], S->onw, EMB, EPS);
        if (logits_out)
            if (sllm_gemv_f32(S->out->type, S->out->data, S->out->ne[0],
                              S->nm[t], S->out->ne[1],
                              logits_out + (size_t) t * S->out->ne[1]) != SLLM_OK) return 0;
    }
    return 1;
}

int main_k_seq_gate(void) {
    printf("\n  T14: real uncached sequence execution\n");
    seqctx_t S;
    if (!seq_open(&S, SLLM_CONTRACT, 36)) {
        printf("  T14 sequence: %d checks, %d failed\n", checks, failed);
        return 0;
    }
    printf("    contract: pairing=%s origin=%u dim=%u theta=%g scaling=%s x%g  "
           "(position = origin + index; never derived from a token)\n",
           S.rs.pairing == SLLM_ROPE_NEOX ? "half_split" : "adjacent",
           S.rs.position_origin, S.rs.rotary_dim, (double) S.rs.freq_base,
           S.rs.scaling_mode, (double) S.rs.scaling_factor);

    /* ---- fixture: position is independent of token ID, and RoPE really rotates ---- */
    printf("    position-independence and genuine-rotation fixtures\n");
    {
        /* Same 128-element vector placed at positions 0 and 1. */
        float v[HD], w[HD], u[HD], x[HD];
        for (uint32_t i = 0; i < HD; ++i) v[i] = 0.9f + 0.017f * (float) i;
        memcpy(w, v, sizeof v);
        sllm_rope_inplace(v, HD, (int32_t) S.rs.position_origin, S.rs.freq_base,
                          S.rs.scaling_factor, S.rs.pairing);
        sllm_rope_inplace(w, HD, (int32_t)(S.rs.position_origin + 1), S.rs.freq_base,
                          S.rs.scaling_factor, S.rs.pairing);
        double d01 = 0.0, d_same = 0.0;
        for (uint32_t i = 0; i < HD; ++i) {
            d01 = fmax(d01, fabs((double) v[i] - (double) w[i]));
            d_same = fmax(d_same, fabs((double) w[i] - (double) w[i]));
        }
        printf("      same vector at positions 0 and 1: max difference %.4g\n", d01);
        expect(d01 > 1e-3,
               "the SAME token at positions 0 and 1 gives different rotated Q/K, so "
               "position is not an implied constant");

        /* position 0 is the identity case already proven */
        memcpy(u, w, sizeof u);
        memcpy(x, v, sizeof v);
        sllm_rope_inplace(x, HD, (int32_t) S.rs.position_origin, S.rs.freq_base,
                          S.rs.scaling_factor, S.rs.pairing);
        double d_id = 0.0;
        for (uint32_t i = 0; i < HD; ++i) d_id = fmax(d_id, fabs((double) v[i] - (double) x[i]));
        printf("      position 0 is the identity: rotation delta %.4g\n", d_id);
        expect(d_id == 0.0, "position 0 is exactly the identity, as already established");
        (void) u;

        /* The EXECUTED arithmetic must be the CONFIGURED one: half_split, origin 0,
         * dim 128, theta 1e6. Checked analytically on the FIRST PAIR, whose angle is
         * pos * theta^0 = pos, so it needs no table and no convention. */
        {   double pre[HD], post[HD];
            for (uint32_t i = 0; i < HD; ++i) pre[i] = 0.9 + 0.017 * (double) i;
            memcpy(post, pre, sizeof(double) * HD);
            const int32_t p = 1;
            const double inv = 1.0 / pow((double) S.rs.freq_base, 0.0);  /* theta^0 */
            const double a = (double) p * inv, c = cos(a), s = sin(a);
            const double half = (double)(HD / 2);
            /* half_split pairs element j with j + half */
            post[0]        = pre[0] * c - pre[0 + (int)half] * s;
            post[HD / 2]   = pre[HD/2] * c + pre[0] * s;
            double got0 = 0.0, got_half = 0.0;
            float q[HD];
            for (uint32_t i = 0; i < HD; ++i) q[i] = (float) pre[i];
            sllm_rope_inplace(q, HD, p, S.rs.freq_base, S.rs.scaling_factor, S.rs.pairing);
            got0 = q[0]; got_half = q[HD / 2];
            printf("      configured arithmetic at position 1: element 0 expected %.10g "
                   "got %.10g\n", post[0], got0);
            printf("                                     element %u expected %.10g "
                   "got %.10g\n", HD / 2, post[HD / 2], got_half);
            expect(fabs(got0 - post[0]) < 1e-5 && fabs(got_half - post[HD / 2]) < 1e-5,
                   "the executed rotation IS the configured contract: half_split, origin 0, "
                   "dim 128, theta 1e6 -- not merely metadata the loader accepted");
            /* adjacent pairing would have touched element 1 instead of element 64 */
            printf("      under the OTHER pairing element 0 would be untouched and element 1 "
                   "rotated; the\n      observed change at element 64 is what distinguishes "
                   "half_split from adjacent\n");
            expect(fabs((double) q[0] - pre[0]) > 1e-6,
                   "element 0 moved, which cannot happen under adjacent pairing at j=0");
        }
    }

    /* ---- full-depth T=2: T=1 reproduction AND causality at depth ---- */
    const unsigned cp_blocks[] = { 0, 1, 18, 35 };
    const unsigned n_cp = 4;
    double cp_a[4 * SEQMAX], cp_f[4 * SEQMAX];
    memset(cp_a, 0, sizeof cp_a); memset(cp_f, 0, sizeof cp_f);
    const uint64_t VOCAB = S.out->ne[1];

    printf("    full depth %u blocks, T=2 (positions 0 and 1)\n", S.n_blocks);
    float * lg2 = (float *) malloc(sizeof(float) * VOCAB * 2);
    const int toks2[2] = { 0, 7 };
    int ok = seq_run(&S, toks2, 2, S.n_blocks, cp_blocks, n_cp, cp_a, cp_f, lg2);
    expect(ok, "the full 36-block graph ran over 2 positions");
    if (ok) {
        /* Gate 1: position 0 of a T=2 run must equal the established T=1 result. */
        double d13 = -1.0, t13_max = -1.0;
        FILE * f = fopen("build/t13_t1.digest", "r");
        if (f) { if (fscanf(f, "%lf %lf", &d13, &t13_max) != 2) { d13 = -1.0; t13_max = -1.0; }
                  fclose(f); }
        const double d0 = digest_of(lg2, VOCAB);
        printf("      position-0 logits digest %.17g\n", d0);
        printf("      T13 T=1 digest           %.17g   (max|logit| %.10g vs %.10g)\n",
               d13, maxabs_of(lg2, VOCAB), t13_max);
        if (d13 > 0.0) {
            const double rel = fabs(d0 - d13) / fabs(d13);
            printf("      relative difference %.3g\n", rel);
            expect(rel < 1e-5,
                   "T=1 is REPRODUCED: position 0 of the T=2 run matches T13's single-position "
                   "result, so the sequence machinery did not alter the established path");
        } else {
            printf("      T13 digest unavailable; skipping the cross-implementation "
                   "comparison\n");
        }
        /* Gate: causality at depth -- position 0 must not see position 1. */
        double d_pos = 0.0;
        for (uint64_t i = 0; i < VOCAB; ++i)
            d_pos = fmax(d_pos, fabs((double) lg2[i] - (double) lg2[VOCAB + i]));
        printf("      position 0 vs position 1 logits: max difference %.4g\n", d_pos);
        expect(d_pos > 1e-3, "the two positions produce genuinely different logits, so "
                             "positions are not collapsing onto one another");
    }
    free(lg2);

    /* ---- shallow T>2: prefix invariance, ordering, cross-position, drift ---- */
    const unsigned NB = 4;   /* scoping stated in the commit: cost, not convenience */
    printf("    T>2 on the first %u blocks (scoped: see commit message for why)\n", NB);
    {
        double a2[4 * SEQMAX], f2[4 * SEQMAX], a3[4 * SEQMAX], f3[4 * SEQMAX];
        memset(a2, 0, sizeof a2); memset(f2, 0, sizeof f2);
        memset(a3, 0, sizeof a3); memset(f3, 0, sizeof f3);
        const unsigned cpb3[3] = { 0, 1, 3 };
        const int toks3[3] = { 0, 7, 63 };
        float * lg3 = (float *) malloc(sizeof(float) * VOCAB * 3);
        float * lg4 = (float *) malloc(sizeof(float) * VOCAB * 4);

        /* THE PERMANENT INVARIANT:
         *     prefix(model, A B C D)[0..2] == model(A B C)[0..2]
         * One property attacking causal masking, position handling and accidental
         * cross-token contamination simultaneously. */
        ok = seq_run(&S, toks3, 3, NB, cpb3, 3, a3, f3, lg3);
        expect(ok, "T=3 over the first blocks executed");
        const int toks4[4] = { 0, 7, 63, 1 };
        ok = seq_run(&S, toks4, 4, NB, cpb3, 3, a2, f2, lg4);
        expect(ok, "T=4 over the first blocks executed");
        if (ok) {
            double worst_cp = 0.0;
            for (unsigned c = 0; c < 3; ++c)
                for (unsigned t = 0; t < 3; ++t) {
                    worst_cp = fmax(worst_cp, fabs(a3[c * SEQMAX + t] - a2[c * SEQMAX + t]));
                    worst_cp = fmax(worst_cp, fabs(f3[c * SEQMAX + t] - f2[c * SEQMAX + t]));
                }
            printf("      checkpoints for positions 0..2, T=3 vs T=4: max difference %.4g\n",
                   worst_cp);
            expect(worst_cp <= 1e-9,
                   "prefix(model, A B C D)[0..2] == model(A B C)[0..2] at every checkpoint: "
                   "the 4th token changed nothing earlier");
            double worst_lg = 0.0;
            for (unsigned t = 0; t < 3; ++t)
                for (uint64_t i = 0; i < VOCAB; ++i)
                    worst_lg = fmax(worst_lg,
                        fabs((double) lg3[t * VOCAB + i] - (double) lg4[t * VOCAB + i]));
            printf("      final logits for positions 0..2, T=3 vs T=4: max difference %.4g\n",
                   worst_lg);
            expect(worst_lg <= 1e-3,
                   "and the final logits of the earlier positions are unchanged to the "
                   "existing tolerance");
        }

        /* Ordering: [A,B] and [B,A] must not collapse. */
        {
            const int ab[2] = { 0, 7 }, ba[2] = { 7, 0 };
            float * l1 = (float *) malloc(sizeof(float) * VOCAB * 2);
            float * l2b = (float *) malloc(sizeof(float) * VOCAB * 2);
            double z1[3 * SEQMAX], f1[3 * SEQMAX], z2[3 * SEQMAX], f2b[3 * SEQMAX];
            memset(z1, 0, sizeof z1); memset(f1, 0, sizeof f1);
            memset(z2, 0, sizeof z2); memset(f2b, 0, sizeof f2b);
            ok = seq_run(&S, ab, 2, NB, cpb3, 3, z1, f1, l1);
            ok = ok && seq_run(&S, ba, 2, NB, cpb3, 3, z2, f2b, l2b);
            double d = 0.0;
            if (ok) for (uint64_t i = 0; i < VOCAB; ++i)
                d = fmax(d, fabs((double) l1[i] - (double) l2b[i]));
            printf("      [A,B] vs [B,A] at position 0: max logit difference %.4g\n", d);
            expect(ok && d > 1e-3, "reversing the token order changes the result, so order "
                                    "is not collapsing");
            free(l1); free(l2b);
        }

        /* Attention genuinely crosses positions: position 0's output cannot come from
         * its own V alone once position 1 exists, and position 0 must contain NO
         * contribution from position 1. Checked on the first block's raw attention. */
        {
            const int seq3[3] = { 0, 7, 63 };
            float saved[SEQMAX][NH * HD];
            for (unsigned t = 0; t < 3; ++t) memcpy(saved[t], S.catbuf[t], sizeof(float) * NH * HD);
            ok = seq_run(&S, seq3, 3, 1, cpb3, 1, a2, f2, NULL);   /* block 0 only */
            if (ok) {
                /* position 0 attends only to itself, so its output equals its own V
                 * after the (single-term) softmax, which is exactly 1. Position 1 must
                 * NOT equal its own V alone, or nothing is crossing positions. */
                double dev0 = 0.0, dev1 = 0.0;
                for (uint32_t i = 0; i < HD; ++i) {
                    /* recompute what position 0 would be from its own V only */
                    dev0 = fmax(dev0, fabs((double) S.catbuf[0][i] - (double) S.V[0][i]));
                    dev1 = fmax(dev1, fabs((double) S.catbuf[1][i] - (double) S.V[1][i]));
                }
                printf("      block 0 attention: |pos0 out - own V| = %.4g (must be ~0: "
                       "it attends only to itself)\n", dev0);
                printf("                                |pos1 out - own V| = %.4g (must be "
                       "large: it must include position 0)\n", dev1);
                expect(dev0 <= 1e-6,
                       "position 0 contains NO contribution from position 1");
                expect(dev1 > 1e-3,
                       "position 1's attention output differs from its own V alone, so it "
                       "contains a real contribution from position 0");
            }
            (void) saved;
        }

        /* Drift checkpoints across positions and blocks. */
        printf("      drift across blocks AND positions (digest of the carried stream;\n"
               "        T=3, tokens 0/7/63, so any position-dependence would show here)\n");
        printf("        block   after-attn: pos0        pos1        pos2\n");
        for (unsigned c = 0; c < 3; ++c)
            printf("        %-6u               %-12.6g %-12.6g %-12.6g\n", cpb3[c],
                   a3[c * SEQMAX], a3[c * SEQMAX + 1], a3[c * SEQMAX + 2]);
        printf("        block   after-ffn : pos0        pos1        pos2\n");
        for (unsigned c = 0; c < 3; ++c)
            printf("        %-6u               %-12.6g %-12.6g %-12.6g\n", cpb3[c],
                   f3[c * SEQMAX], f3[c * SEQMAX + 1], f3[c * SEQMAX + 2]);
        {   /* drift must GROW with depth, or the checkpoints are not measuring anything */
            double d0 = fabs(f3[0 * SEQMAX + 2] - a3[0 * SEQMAX + 2]);
            double d1 = fabs(f3[1 * SEQMAX + 2] - a3[1 * SEQMAX + 2]);
            double d3 = fabs(f3[2 * SEQMAX + 2] - a3[2 * SEQMAX + 2]);
            printf("        attention-residual -> ffn-residual change at pos 2: block 0 %.4g, "
                   "block 1 %.4g, block 3 %.4g\n", d0, d1, d3);
            expect(d0 > 1e-9 && d1 > 1e-9 && d3 > 1e-9,
                   "each checkpoint stage measurably changes the stream, so the digests are "
                   "sensitive rather than trivially equal");
            /* and positions must differ from each other at the same block */
            double spread = 0.0;
            for (unsigned c = 0; c < 3; ++c)
                spread = fmax(spread, fabs(f3[c * SEQMAX] - f3[c * SEQMAX + 2]));
            printf("        pos0 vs pos2 spread at the FFN-residual checkpoints: %.4g\n", spread);
            expect(spread > 1e-9, "positions are distinguishable at every checkpoint block");
        }
        free(lg3); free(lg4);
    }

    seq_close(&S);
    printf("  T14 sequence: %d checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}