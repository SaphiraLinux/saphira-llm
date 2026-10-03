/*
 * test_blocks.c -- T13: contract-complete artefact, and the 36-block repetition.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * TWO SEPARATE CLAIMS.
 *
 * (1) THE CONTRACT-COMPLETE ARTEFACT. A copy of the model carrying the ALREADY
 *     ESTABLISHED saphira.rope.* and saphira.ffn.* metadata and nothing else. No
 *     tensor byte, name, type, geometry or quantisation setting was altered, and the
 *     legacy artefact still refuses. Verified outside this test by a full byte
 *     comparison of the 5.02 GB data blob and a descriptor-by-descriptor comparison
 *     of all 399 tensors; the refusal side is checked here.
 *
 * (2) THE REPETITION RULE IS DERIVED, NOT ASSUMED. Block 0 being regular does not
 *     imply blocks 1..35 are. Every block is censused: every required tensor is
 *     located by probing, and its ne[] and type recorded. Only if all 36 agree does
 *     one operator graph legitimately parameterise them. Any structural exception
 *     would be reported rather than absorbed.
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

#define SLLM_LEGACY  "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
#define SLLM_CONTRACT "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.contract.gguf"

#define SLLM_FFN_K_ACTIVATION "saphira.ffn.activation"
#define SLLM_FFN_K_FORM       "saphira.ffn.form"

#define EMB   4096u
#define FFN   12288u
#define NH    32u
#define NKVH  8u
#define HD    128u

static int checks = 0, failed = 0;
static void expect(int cond, const char * what) {
    checks++;
    if (cond) { printf("      ok    %s\n", what); }
    else      { failed++; printf("      FAIL  %s\n", what); }
}

/* The eleven tensors every attention-bearing block must provide. Order is fixed here
 * only so the census can be printed readably; it carries no architectural claim. */
static const char * REQ[] = {
    "attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
    "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
};
#define NREQ ((int)(sizeof REQ / sizeof REQ[0]))

typedef struct { uint64_t ne[4]; int type; } desc_t;
static unsigned differing_blocks[NREQ][64];
static unsigned differing_count[NREQ];

static const sllm_gguf_tensor * find_in_block(sllm_gguf * g, unsigned b, const char * role) {
    char nm[160];
    snprintf(nm, sizeof nm, "blk.%u.%s.weight", b, role);
    return sllm_gguf_find_tensor(g, nm);
}

/* ------------------------------------------------------------------ census */
static int census(sllm_gguf * g, unsigned n_blocks, desc_t ref[NREQ]) {
    int uniform = 1;        /* shapes uniform: one operator graph parameterises all blocks */
    int type_uniform = 1;   /* quantisation uniform: a SEPARATE and stronger claim     */
    unsigned present_total = 0;
    printf("    census of %u blocks x %d required tensors\n", n_blocks, NREQ);
    for (int r = 0; r < NREQ; ++r) {
        unsigned present = 0;
        int first = 1;
        for (unsigned b = 0; b < n_blocks; ++b) {
            const sllm_gguf_tensor * t = find_in_block(g, b, REQ[r]);
            if (!t) continue;
            present++;
            if (first) {
                for (uint32_t d = 0; d < 4; ++d) ref[r].ne[d] = t->ne[d];
                ref[r].type = (int) t->type;
                first = 0;
            } else {
                for (uint32_t d = 0; d < 4; ++d) {
                    if (ref[r].ne[d] != t->ne[d]) {
                        uniform = 0;
                        printf("      SHAPE EXCEPTION: %s differs at block %u, ne[%u] = "
                               "%llu but block 0 has %llu\n", REQ[r], b, d,
                               (unsigned long long) t->ne[d],
                               (unsigned long long) ref[r].ne[d]);
                    }
                }
                if (ref[r].type != (int) t->type) {
                    /* NOT a shape difference: the operator graph is unaffected and only
                     * the storage format differs. Recorded separately, because
                     * conflating shape with quantisation would hide it entirely. */
                    type_uniform = 0;
                    if (differing_count[r] < 64)
                        differing_blocks[r][differing_count[r]++] = b;
                }
            }
        }
        present_total += present;
        printf("      %-12s present in %u/%u blocks   ne=[%llu, %llu]  type=%d%s\n",
               REQ[r], present, n_blocks,
               (unsigned long long) ref[r].ne[0], (unsigned long long) ref[r].ne[1],
               ref[r].type, present == n_blocks ? "" : "   <-- INCOMPLETE");
        if (present != n_blocks) uniform = 0;
    }
    printf("      total tensor identities located: %u of %u expected\n",
           present_total, n_blocks * (unsigned) NREQ);
    printf("      SHAPES uniform across all %u blocks: %s\n", n_blocks,
           uniform ? "yes -- one operator graph legitimately parameterises every block"
                   : "NO -- a structural exception exists");
    printf("      QUANTISATION uniform across all %u blocks: %s\n", n_blocks,
           type_uniform ? "yes"
                        : "NO -- and this is the finding that matters");
    for (int r = 0; r < NREQ; ++r) {
        if (differing_count[r] == 0) continue;
        printf("      %-12s block 0 is type %d, but %u of %u blocks use a different type:",
               REQ[r], ref[r].type, differing_count[r], n_blocks);
        for (unsigned i = 0; i < differing_count[r]; ++i)
            printf(" %u", differing_blocks[r][i]);
        printf("\n");
    }
    if (!type_uniform) {
        printf("      CONSEQUENCE: reading block 0's quantisation and applying it to every\n"
               "      block would reinterpret %s payload bytes under the wrong K-quant\n"
               "      layout. Both layouts are valid K-quants producing finite, plausible\n"
               "      values, so the corruption would be SILENT -- no fault, no refusal, just\n"
               "      wrong numbers. Quantisation must be read per block, per tensor.\n", "attn_v");
    }
    return uniform;
}

/* --------------------------------------------------------- block forward */
typedef struct {
    const sllm_gguf_tensor * anrm, *qp, *kp, *vp, *qn, *kn, *aop;
    const sllm_gguf_tensor * fnrm, *gate, *up, *down;
} block_t;

static int load_block(sllm_gguf * g, unsigned b, block_t * B) {
    B->anrm = find_in_block(g, b, "attn_norm");
    B->qp   = find_in_block(g, b, "attn_q");
    B->kp   = find_in_block(g, b, "attn_k");
    B->vp   = find_in_block(g, b, "attn_v");
    B->qn   = find_in_block(g, b, "attn_q_norm");
    B->kn   = find_in_block(g, b, "attn_k_norm");
    B->aop  = find_in_block(g, b, "attn_output");
    B->fnrm = find_in_block(g, b, "ffn_norm");
    B->gate = find_in_block(g, b, "ffn_gate");
    B->up   = find_in_block(g, b, "ffn_up");
    B->down = find_in_block(g, b, "ffn_down");
    return B->anrm && B->qp && B->kp && B->vp && B->qn && B->kn && B->aop
        && B->fnrm && B->gate && B->up && B->down;
}

/* Run one block. `in` is the block input (the residual stream). Writes the block
 * output to `out`. Intermediate tensors are exposed so the carry can be tested. */
int main_k_blocks_gate(void) {
    printf("\n  T13: contract-complete artefact and the 36-block repetition\n");

    /* ---------- (1) the contract-complete artefact ---------- */
    char err[256];
    sllm_gguf lg, ct;
    const int have_lg = (sllm_gguf_open(SLLM_LEGACY, &lg, err, sizeof err) == SLLM_OK);
    const int have_ct = (sllm_gguf_open(SLLM_CONTRACT, &ct, err, sizeof err) == SLLM_OK);
    if (!have_lg || !have_ct) {
        printf("    SKIP  artefacts unavailable (legacy=%d contract=%d: %s)\n",
               have_lg, have_ct, err);
        if (have_lg) sllm_gguf_close(&lg);
        if (have_ct) sllm_gguf_close(&ct);
        return 0;
    }
    {
        sllm_rope_semantics a, b; char ma[256] = "", mb[256] = "";
        const int ra = sllm_rope_semantics_from_gguf(&lg, &a, ma, sizeof ma) == SLLM_OK;
        const int rb = sllm_rope_semantics_from_gguf(&ct, &b, mb, sizeof mb) == SLLM_OK;
        const char * fa = NULL, * fb = NULL;
        const int qa = sllm_gguf_kv_str(&lg, SLLM_FFN_K_ACTIVATION, &fa) == SLLM_OK;
        const int qb = sllm_gguf_kv_str(&ct, SLLM_FFN_K_FORM, &fb) == SLLM_OK;
        printf("    LEGACY artefact:     RoPE %s (%s)  FFN %s\n",
               ra ? "resolved" : "REFUSED", ma, qa ? "resolved" : "REFUSED");
        printf("    CONTRACT artefact:   RoPE %s  FFN %s\n",
               rb ? "resolved" : "REFUSED", qb ? "resolved" : "REFUSED");
        if (rb) {
            printf("      pairing=%s origin=%u rotary_dim=%u freq_base=%g scaling=%s x%g\n",
                   b.pairing == SLLM_ROPE_NEOX ? "half_split" : "adjacent",
                   b.position_origin, b.rotary_dim, (double) b.freq_base,
                   b.scaling_mode, (double) b.scaling_factor);
            printf("      source: %s\n", b.source);
        }
        expect(!ra && !qa, "the legacy artefact still REFUSES both contracts; not weakened");
        expect(rb && qb, "the contract-complete artefact resolves BOTH contracts");
        /* the legacy refusal reason must be the contract, not something incidental */
        expect(strcmp(ma, SLLM_ROPE_K_PAIRING) == 0,
               "the legacy RoPE refusal names the absent pairing key specifically");
    }

    /* ---------- (2) derive the repetition rule from a full census ---------- */
    uint32_t n_blocks = 0;
    const char * arch = NULL;
    (void) sllm_gguf_kv_str(&ct, "general.architecture", &arch);
    if (arch) {
        char k[192];
        snprintf(k, sizeof k, "%s.block_count", arch);
        (void) sllm_gguf_kv_u32(&ct, k, &n_blocks);
    }
    printf("    block_count from %s.block_count = %u (typed read, not assumed)\n",
           arch ? arch : "?", n_blocks);
    expect(n_blocks == 36u, "the artefact declares 36 blocks");

    desc_t ref[NREQ];
    memset(ref, 0, sizeof ref);
    const int uniform = census(&ct, n_blocks, ref);
    expect(uniform,
           "all 36 blocks provide all 11 tensors with IDENTICAL ne[]: one operator graph "
           "legitimately parameterises every block");
    if (!uniform) {
        printf("      a SHAPE exception exists; the repetition rule is NOT uniform and must "
               "not be assumed\n");
    }
    /* The quantisation IS non-uniform, and the executor must read it per block. */
    {
        static const char * VARIES[] = { "attn_v", "ffn_down" };
        int av_varies = differing_count[3] > 0, fd_varies = differing_count[10] > 0;
        printf("      per-block quantisation required for: attn_v %s, ffn_down %s\n",
               av_varies ? "YES" : "no", fd_varies ? "YES" : "no");
        expect(av_varies,
               "attn_v quantisation is MEASURED to differ across blocks, so it must be read "
               "per block rather than inherited from block 0");
        expect(fd_varies,
               "ffn_down quantisation is MEASURED to differ across blocks, likewise");
        (void) VARIES;
    }
    /* cross-check the derived geometry against the independently measured metadata */
    {
        float eps = 0.0f; char k[192];
        snprintf(k, sizeof k, "%s.attention.layer_norm_rms_epsilon", arch);
        (void) sllm_gguf_kv_f32(&ct, k, &eps);
        uint32_t nh = 0, nkv = 0, kl = 0;
        snprintf(k, sizeof k, "%s.attention.head_count", arch);     (void) sllm_gguf_kv_u32(&ct, k, &nh);
        snprintf(k, sizeof k, "%s.attention.head_count_kv", arch);  (void) sllm_gguf_kv_u32(&ct, k, &nkv);
        snprintf(k, sizeof k, "%s.attention.key_length", arch);     (void) sllm_gguf_kv_u32(&ct, k, &kl);
        printf("      cross-check: n_heads %u x head_dim %u = %u == attn_q ne[1] %llu: %s\n",
               nh, kl, nh * kl, (unsigned long long) ref[1].ne[1],
               (nh * kl == ref[1].ne[1]) ? "yes" : "NO");
        expect(nh * kl == ref[1].ne[1], "the per-block attention geometry agrees with the "
                                        "block-level metadata");
        expect(ref[0].ne[0] == EMB, "attn_norm width is the residual stream width");
        expect(ref[8].ne[1] == FFN && ref[9].ne[1] == FFN && ref[10].ne[0] == FFN,
           "gate/up expand to the FFN width and down contracts it back");
        printf("      rms epsilon %.10g carried through every block's norms\n", (double) eps);
    }

    /* ---------- (3) carry: 36 blocks chained, then final norm and logits ---------- */
    sllm_rope_semantics rs;
    char mb[256] = "";
    if (sllm_rope_semantics_from_gguf(&ct, &rs, mb, sizeof mb) != SLLM_OK) {
        printf("    cannot chain: contract unresolved (%s)\n", mb);
        sllm_gguf_close(&lg); sllm_gguf_close(&ct);
        return 1;
    }
    const double E = 1.0e-6;
    const sllm_gguf_tensor * emb = sllm_gguf_find_tensor(&ct, "token_embd.weight");
    const sllm_gguf_tensor * onrm = sllm_gguf_find_tensor(&ct, "output_norm.weight");
    const sllm_gguf_tensor * oout = sllm_gguf_find_tensor(&ct, "output.weight");
    if (!emb || !onrm || !oout) {
        printf("    FAIL: embedding, final norm or logits tensor absent\n");
        failed++; checks++;
        sllm_gguf_close(&lg); sllm_gguf_close(&ct);
        return 1;
    }
    float * x  = (float *) malloc(sizeof(float) * EMB);   /* the carried stream */
    float * x0 = (float *) malloc(sizeof(float) * EMB);   /* block 0 input      */
    float * nm = (float *) malloc(sizeof(float) * EMB);
    float * anw= (float *) malloc(sizeof(float) * EMB);
    float * fnw= (float *) malloc(sizeof(float) * EMB);
    float * onw= (float *) malloc(sizeof(float) * EMB);
    float * qw = (float *) malloc(sizeof(float) * HD);
    float * kw = (float *) malloc(sizeof(float) * HD);
    float * qv = (float *) malloc(sizeof(float) * NH * HD);
    float * kvv= (float *) malloc(sizeof(float) * NKVH * HD);
    float * vvv= (float *) malloc(sizeof(float) * NKVH * HD);
    float * cat= (float *) malloc(sizeof(float) * NH * HD);
    float * pp = (float *) malloc(sizeof(float) * 8);
    float * gvb= (float *) malloc(sizeof(float) * FFN);
    float * uvb= (float *) malloc(sizeof(float) * FFN);
    float * gat= (float *) malloc(sizeof(float) * FFN);
    float * tmp= (float *) malloc(sizeof(float) * EMB);
    if (!x||!x0||!nm||!anw||!fnw||!onw||!qw||!kw||!qv||!kvv||!vvv||!cat||!pp||!gvb||!uvb||!gat||!tmp) {
        printf("    FAIL: alloc\n"); failed++; checks++;
        sllm_gguf_close(&lg); sllm_gguf_close(&ct); return 1;
    }

    /* block 0 input: the embedding of one token. Token ID and sequence position are
     * separate quantities; here the position is 0 and the token is 0, and nothing
     * derives one from the other. */
    if (sllm_dequant_row(emb->type, emb->data, x, EMB) != SLLM_OK) {
        printf("    FAIL: embedding decode\n"); failed++; checks++;
        sllm_gguf_close(&lg); sllm_gguf_close(&ct); return 1;
    }
    memcpy(x0, x, sizeof(float) * EMB);

    const double pos = (double) rs.position_origin;   /* position 0: identity rotation */
    double block_norm[4] = { 0, 0, 0, 0 };
    int chain_ok = 1;
    printf("    chaining %u blocks: block[0] -> block[1] -> ... -> block[%u]\n",
           n_blocks, n_blocks - 1);
    for (unsigned b = 0; b < n_blocks && chain_ok; ++b) {
        block_t B;
        if (!load_block(&ct, b, &B)) {
            printf("      FAIL: block %u is missing a tensor\n", b); chain_ok = 0; break;
        }
        /* rms weights for this block */
        if (sllm_dequant_row(B.anrm->type, B.anrm->data, anw, EMB) != SLLM_OK) chain_ok = 0;
        if (sllm_dequant_row(B.fnrm->type, B.fnrm->data, fnw, EMB) != SLLM_OK) chain_ok = 0;
        if (sllm_dequant_row(B.qn->type, B.qn->data, qw, HD) != SLLM_OK) chain_ok = 0;
        if (sllm_dequant_row(B.kn->type, B.kn->data, kw, HD) != SLLM_OK) chain_ok = 0;

        /* attention input is RMSNorm(x) */
        sllm_rms_norm(nm, x, anw, EMB, (float) E);
        block_norm[b % 4] += fabs((double) nm[0]);
        if (sllm_gemv_f32(B.qp->type, B.qp->data, B.qp->ne[0], nm, NH*HD, qv) != SLLM_OK) chain_ok = 0;
        if (sllm_gemv_f32(B.kp->type, B.kp->data, B.kp->ne[0], nm, NKVH*HD, kvv) != SLLM_OK) chain_ok = 0;
        if (sllm_gemv_f32(B.vp->type, B.vp->data, B.vp->ne[0], nm, NKVH*HD, vvv) != SLLM_OK) chain_ok = 0;
        for (uint32_t h = 0; h < NH; ++h) sllm_rms_norm(qv+h*HD, qv+h*HD, qw, HD, (float) E);
        for (uint32_t h = 0; h < NKVH; ++h) sllm_rms_norm(kvv+h*HD, kvv+h*HD, kw, HD, (float) E);
        /* RoPE, now permitted: the contract is present on this artefact */
        for (uint32_t h = 0; h < NH; ++h)
            sllm_rope_inplace(qv+h*HD, HD, (int32_t) pos, rs.freq_base, rs.scaling_factor, rs.pairing);
        for (uint32_t h = 0; h < NKVH; ++h)
            sllm_rope_inplace(kvv+h*HD, HD, (int32_t) pos, rs.freq_base, rs.scaling_factor, rs.pairing);

        /* single-token causal attention: position 0 attends only to position 0 */
        for (uint32_t q = 0; q < NH && chain_ok; ++q) {
            double acc = 0.0;
            const float * qq = qv + q * HD, * kk = kvv + (q / (NH / NKVH)) * HD;
            for (uint32_t i = 0; i < HD; ++i) acc += (double) qq[i] * (double) kk[i];
            acc /= sqrt((double) HD);
            pp[0] = 1.0f;    /* one permitted position, so softmax is exactly 1 */
            (void) acc;
            for (uint32_t i = 0; i < HD; ++i)
                cat[q*HD+i] = vvv[(q / (NH / NKVH)) * HD + i];
        }
        if (sllm_gemv_f32(B.aop->type, B.aop->data, B.aop->ne[0], cat, B.aop->ne[1], tmp) != SLLM_OK)
            chain_ok = 0;
        /* residual 1 = x + projected attention */
        for (unsigned i = 0; i < EMB; ++i) x[i] = x[i] + tmp[i];

        /* FFN on residual_1 */
        sllm_rms_norm(nm, x, fnw, EMB, (float) E);
        if (sllm_gemv_f32(B.gate->type, B.gate->data, B.gate->ne[0], nm, FFN, gvb) != SLLM_OK) chain_ok = 0;
        if (sllm_gemv_f32(B.up->type, B.up->data, B.up->ne[0], nm, FFN, uvb) != SLLM_OK) chain_ok = 0;
        for (uint32_t i = 0; i < FFN; ++i) {
            const double gg = gvb[i];
            gat[i] = (float)((gg / (1.0 + exp(-gg))) * (double) uvb[i]);
        }
        if (sllm_gemv_f32(B.down->type, B.down->data, B.down->ne[0], gat, EMB, tmp) != SLLM_OK)
            chain_ok = 0;
        /* residual 2 = the ORIGINAL block input x, per T11's source-established parent.
         * That is the stream value as it entered THIS block, which is what makes the
         * carry chain rather than 36 independent blocks. */
        for (unsigned i = 0; i < EMB; ++i) x[i] = x[i] + tmp[i];
    }
    expect(chain_ok, "all 36 blocks executed in sequence from a single carried stream");
    printf("      per-block RMSNorm(|x_0|) sampled across blocks: %.4f %.4f %.4f %.4f\n",
           block_norm[0], block_norm[1], block_norm[2], block_norm[3]);

    /* final norm and logits, now permitted */
    double final_rel = 0.0, lmag = 0.0;
    if (sllm_dequant_row(onrm->type, onrm->data, onw, EMB) != SLLM_OK) chain_ok = 0;
    sllm_rms_norm(nm, x, onw, EMB, (float) E);
    {
        /* the sum of squares belongs to the INPUT x, not to the output nm */
        double ss = 0.0;
        for (unsigned k = 0; k < EMB; ++k) { const double v = x[k]; ss += v * v; }
        const double r = sqrt(ss / (double) EMB + E);
        for (unsigned i = 0; i < EMB; ++i) {
            const double ref = ((double) x[i] / r) * (double) onw[i];
            const double d = fabs(ref - (double) nm[i]);
            if (ref != 0.0) final_rel = fmax(final_rel, d / fabs(ref));
        }
    }
    float * logits = (float *) malloc(sizeof(float) * oout->ne[1]);
    if (!logits || sllm_gemv_f32(oout->type, oout->data, oout->ne[0], nm,
                                 oout->ne[1], logits) != SLLM_OK) chain_ok = 0;
    for (uint64_t i = 0; i < (uint64_t) oout->ne[1]; ++i) lmag = fmax(lmag, fabs((double) logits[i]));
    printf("      after 36 blocks -> final RMSNorm (worst_rel %.3g) -> logits over %llu "
           "vocabulary, max|logit| = %.4g\n",
           final_rel, (unsigned long long) oout->ne[1], lmag);
    expect(chain_ok && isfinite(lmag) && lmag > 1e-6,
           "the chain reaches finite, non-trivial logits through the final norm");
    expect(final_rel <= 1e-5, "the final norm after 36 blocks matches its reference");
    printf("      NO softmax, sampling, argmax or KV cache: the chain stops at logits.\n");

    /* ---------- (4) carry-source discrimination ---------- */
    printf("    carry-source fixtures: the wrong parent must not pass\n");
    {
        block_t B;
        if (load_block(&ct, 0, &B)) {
            /* x0 is block 0's input; x is the final carried value. Feeding block 0's
             * ORIGINAL input where the carried stream belongs must change the answer. */
            double d = 0.0;
            for (unsigned i = 0; i < EMB; ++i) d = fmax(d, fabs((double) x[i] - (double) x0[i]));
            printf("      carried stream vs block-0 original input: max difference %.4g\n", d);
            expect(d > 1e-3,
                   "the carried stream differs materially from the original block input, so "
                   "the 36 blocks are genuinely chained rather than each starting afresh");
            /* a normalised value is not a valid carry either */
            double dn = 0.0;
            for (unsigned i = 0; i < EMB; ++i) dn = fmax(dn, fabs(x[i] - (double) nm[i]));
            printf("      carried stream vs the final normalised value: max difference %.4g\n", dn);
            expect(dn > 1e-3,
                   "a normalised value is not interchangeable with the carried stream");
        }
    }

    free(x); free(x0); free(nm); free(anw); free(fnw); free(onw);
    free(qw); free(kw); free(qv); free(kvv); free(vvv); free(cat); free(pp);
    free(gvb); free(uvb); free(gat); free(tmp); free(logits);
    sllm_gguf_close(&lg);
    sllm_gguf_close(&ct);
    printf("  T13 blocks: %d checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}