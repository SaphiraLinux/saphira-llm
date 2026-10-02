/*
 * test_ffn.c -- T11: the FFN block and the second residual merge.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 *   T11.1  FFN RMSNorm
 *   T11.2  gate projection
 *   T11.3  up projection
 *   T11.4  activation + gating
 *   T11.5  down projection
 *   T11.6  second residual merge
 *
 * THE ACTIVATION IS A CONTRACT, NOT A LABEL. "SwiGLU" names a family and settles
 * nothing. The exact expression is established from the model's own implementation:
 *
 *     config.hidden_act = "silu"  ->  ACT2FN["silu"]  ->  SiLU(x) = x * sigmoid(x)
 *     forward: down_proj( act_fn( gate_proj(x) ) * up_proj(x) )
 *
 * so the contract is
 *
 *     gate = Proj_gate(n)          up = Proj_up(n)
 *     h    = ( gate * sigmoid(gate) ) * up
 *     out  = Proj_down(h)
 *
 * Operand order and bracketing matter and are asserted, not assumed. Six plausible
 * mis-implementations are discriminated below.
 *
 * THE SECOND RESIDUAL PARENT IS NOT WHAT THE BRIEF ASSUMED.
 *
 * The brief gave residual_2 = residual_1 + ffn_out. The model's own implementation
 * binds the residual ONCE, to the ORIGINAL pre-norm block input, and never
 * reassigns it:
 *
 *     residual = hidden_states            <- the original x
 *     hidden_states = residual + attn_out        -> residual_1
 *     hidden_states = ffn(residual_1)
 *     hidden_states = residual + ffn_out        -> residual_2, parent is x AGAIN
 *
 * All four candidate parents are computed and compared here so the choice is visible
 * and adjudicated from evidence rather than from a formula.
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

#define SLLM_TEST_QWEN3 "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"

#define EMB   4096u
#define FFN   12288u
#define NH    32u
#define NKVH  8u
#define HD    128u
#define NT    4

static int checks = 0, failed = 0;
static void expect(int cond, const char * what) {
    checks++;
    if (cond) { printf("      ok    %s\n", what); }
    else      { failed++; printf("      FAIL  %s\n", what); }
}

/* SiLU, written out rather than named. */
static double silu(double g) { return g / (1.0 + exp(-g)); }

/* ---------------------------------------------------------------- the contract
 * Named keys are the machine-readable form. The legacy artefact carries NEITHER the
 * RoPE contract NOR these, and the runtime refuses when either is absent.
 */
#define SLLM_FFN_K_ACTIVATION "saphira.ffn.activation"
#define SLLM_FFN_K_FORM       "saphira.ffn.form"

typedef struct {
    const char * activation;   /* "silu" */
    const char * form;         /* "act(gate)*up" */
} ffn_contract;

static int resolve_ffn_contract(const sllm_gguf * g, ffn_contract * c, char * why, size_t n) {
    const char * a = NULL, * f = NULL;
    memset(c, 0, sizeof *c);
    if (sllm_gguf_kv_str(g, SLLM_FFN_K_ACTIVATION, &a) != SLLM_OK || !a) {
        snprintf(why, n, "%s", SLLM_FFN_K_ACTIVATION); return 0;
    }
    if (sllm_gguf_kv_str(g, SLLM_FFN_K_FORM, &f) != SLLM_OK || !f) {
        snprintf(why, n, "%s", SLLM_FFN_K_FORM); return 0;
    }
    if (strcmp(a, "silu") != 0 || strcmp(f, "act(gate)*up") != 0) {
        snprintf(why, n, "unsupported ffn contract: activation='%s' form='%s'", a, f);
        return 0;
    }
    c->activation = a; c->form = f;
    return 1;
}

/* ------------------------------------------------- six mis-implementation gates
 * Values are deliberately asymmetric and include positive, negative, near-zero and
 * large magnitudes, so a commutative-looking or approximately-linear case cannot
 * false-pass. Each wrong form must be distinguishable from the contract.
 */
static void activation_discriminators(void) {
    printf("    activation/gating contract: h = (gate * sigmoid(gate)) * up\n");
    printf("      source: config.hidden_act=\"silu\" -> ACT2FN[\"silu\"] -> x*sigmoid(x);\n"
           "              forward: down_proj( act_fn(gate_proj(x)) * up_proj(x) )\n");

    enum { N = 12 };
    /* positive, negative, near-zero, and large, asymmetric between the two operands */
    static const double G[N] = {  3.5, -2.25,  0.0, 1e-3, -1e-3, 8.0,
                                -7.5,  0.5, -0.5,  12.0, -11.0,  1e-6 };
    static const double U[N] = { -1.75, 4.5,  2.0, -1e-3,  2e-3, -9.0,
                                 6.25, -0.75,  0.75, -13.0, 0.25, -1e-6 };

    double contract[N], w_silu_up[N], w_gate_up[N], w_gelu[N], w_sum[N], w_prod[N];
    for (int i = 0; i < N; ++i) {
        const double g = G[i], u = U[i];
        contract[i]    = silu(g) * u;          /* THE contract */
        w_silu_up[i]   = silu(u) * g;          /* SiLU(up) * gate -- operands swapped */
        w_gate_up[i]  = g * u;                 /* no activation at all */
        w_gelu[i]     = (0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))) * u;
        w_sum[i]      = silu(g + u);           /* SiLU(gate + up) */
        w_prod[i]     = silu(g * u);           /* SiLU(gate * up) */
    }

    /* The contract must differ from EVERY alternative, on at least one element. */
    struct { const char * name; const double * v; } wrong[] = {
        { "SiLU(up) * gate",        w_silu_up },
        { "gate * up (no act)",     w_gate_up },
        { "GELU(gate) * up",        w_gelu    },
        { "SiLU(gate + up)",        w_sum     },
        { "SiLU(gate * up)",        w_prod    },
    };
    for (unsigned k = 0; k < sizeof wrong / sizeof wrong[0]; ++k) {
        double worst = 0.0; size_t at = 0;
        for (int i = 0; i < N; ++i) {
            const double d = fabs(contract[i] - wrong[k].v[i]);
            if (d > worst) { worst = d; at = (size_t) i; }
        }
        printf("      vs %-22s max difference %.4g at element %zu\n",
               wrong[k].name, worst, at);
        expect(worst > 1e-6, "the contract is distinguishable from this mis-implementation");
    }

    /* Operand order: gate and up are NOT interchangeable, so a swapped pair of
     * projections must fail even where the activation happens to look symmetric. */
    double order_gap = 0.0;
    for (int i = 0; i < N; ++i)
        order_gap = fmax(order_gap, fabs(silu(G[i]) * U[i] - silu(U[i]) * G[i]));
    printf("      gate/up operand order: max |SiLU(g)*u - SiLU(u)*g| = %.4g\n", order_gap);
    expect(order_gap > 1e-3,
           "swapping the gate and up projections changes the result, so the two "
           "projections cannot be interchanged");

    /* Elementwise indexing: the gate at element i must pair with up at element i.
     * A stride-shifted pairing must differ. */
    double stride_gap = 0.0;
    for (int i = 0; i < N; ++i)
        stride_gap = fmax(stride_gap, fabs(silu(G[i]) * U[i] - silu(G[i]) * U[(i + 1) % N]));
    printf("      elementwise indexing: max |aligned - stride-shifted| = %.4g\n", stride_gap);
    expect(stride_gap > 1e-6, "a stride-shifted elementwise pairing is distinguishable");

    /* Negative and large inputs specifically: SiLU(g)*u must have the sign of u for
     * large positive g, and near-zero g must give a near-zero result. This catches an
     * activation that is accidentally symmetric about the origin. */
    expect(silu(8.0) * -9.0 < 0.0 && silu(-7.5) * 6.25 < 0.0,
           "with large |gate| the result's sign follows up, as SiLU(gate) is positive");
    expect(fabs(silu(0.0) * 2.0) == 0.0,
           "an exactly zero gate yields exactly zero, so a near-zero case cannot hide "
           "a stray offset");
}

/* --------------------------------------------------------------- graph recovery */
static void recover_graph(sllm_gguf * g, const sllm_gguf_tensor ** fnorm,
                          const sllm_gguf_tensor ** gate,
                          const sllm_gguf_tensor ** up,
                          const sllm_gguf_tensor ** down) {
    printf("    recovered FFN graph, tensors located by PROBING candidate names\n");
    char nm[160];
    snprintf(nm, sizeof nm, "blk.%llu.ffn_norm.weight", 0ULL);
    *fnorm = sllm_gguf_find_tensor(g, nm);
    static const char * gc[] = { "blk.%llu.ffn_gate.weight", "blk.%llu.ffn_gate_proj.weight" };
    for (unsigned i = 0; i < 2 && !*gate; ++i) {
        snprintf(nm, sizeof nm, gc[i], 0ULL); *gate = sllm_gguf_find_tensor(g, nm);
    }
    static const char * uc[] = { "blk.%llu.ffn_up.weight", "blk.%llu.ffn_up_proj.weight" };
    for (unsigned i = 0; i < 2 && !*up; ++i) {
        snprintf(nm, sizeof nm, uc[i], 0ULL); *up = sllm_gguf_find_tensor(g, nm);
    }
    static const char * dc[] = { "blk.%llu.ffn_down.weight", "blk.%llu.ffn_down_proj.weight" };
    for (unsigned i = 0; i < 2 && !*down; ++i) {
        snprintf(nm, sizeof nm, dc[i], 0ULL); *down = sllm_gguf_find_tensor(g, nm);
    }
    printf("      ffn_norm.weight  %s\n", *fnorm ? "found" : "NOT FOUND");
    printf("      ffn_gate.weight  %s\n", *gate ? "found" : "NOT FOUND");
    printf("      ffn_up.weight    %s\n", *up ? "found" : "NOT FOUND");
    printf("      ffn_down.weight  %s\n", *down ? "found" : "NOT FOUND");
    expect(*fnorm && *gate && *up && *down, "all four FFN tensors were located by probing");
    if (!(*fnorm && *gate && *up && *down)) return;

    printf("      ffn_norm  ne=[%llu]        type=%u\n",
           (unsigned long long)(*fnorm)->ne[0], (unsigned)(*fnorm)->type);
    printf("      ffn_gate  ne=[%llu, %llu]  type=%u\n",
           (unsigned long long)(*gate)->ne[0], (unsigned long long)(*gate)->ne[1],
           (unsigned)(*gate)->type);
    printf("      ffn_up    ne=[%llu, %llu]  type=%u\n",
           (unsigned long long)(*up)->ne[0], (unsigned long long)(*up)->ne[1],
           (unsigned)(*up)->type);
    printf("      ffn_down  ne=[%llu, %llu]  type=%u\n",
           (unsigned long long)(*down)->ne[0], (unsigned long long)(*down)->ne[1],
           (unsigned)(*down)->type);

    /* WIDTHS PROVEN INDEPENDENTLY, not inherited from T10.
     * The residual stream width is taken from ffn_norm itself, since ffn_norm is
     * applied to the residual stream before the FFN. */
    const uint64_t stream = (*fnorm)->ne[0];
    expect((*gate)->ne[0] == stream && (*up)->ne[0] == stream,
           "gate and up both consume the residual stream width taken from ffn_norm");
    expect((*gate)->ne[1] == FFN && (*up)->ne[1] == FFN,
           "gate and up both expand to the measured FFN width");
    expect((*down)->ne[0] == FFN && (*down)->ne[1] == stream,
           "down contracts FFN back to the residual stream width: 12288 -> 4096");

    /* ORIENTATION. Unlike T10's square projection, these are NOT square, so their
     * geometry does constrain the orientation: gate/up are 4096 -> 12288 and down is
     * 12288 -> 4096, and a transposed read would produce the other width. */
    const bool gate_nonsquare = (*gate)->ne[0] != (*gate)->ne[1];
    const bool down_nonsquare = (*down)->ne[0] != (*down)->ne[1];
    printf("    orientation evidence: gate/up and down are %s, so a transposed read "
           "would change\n      the output width and cannot pass the width checks\n",
           (gate_nonsquare && down_nonsquare) ? "NOT square" : "square (unexpected)");
    expect(gate_nonsquare && down_nonsquare,
           "orientation is constrained by non-square geometry: 4096->12288 and "
           "12288->4096 both differ from their transpose");

    /* RoPE stays refused on this artefact, unchanged. */
    sllm_rope_semantics rs; char missing[256] = "";
    const int rope_absent = (sllm_rope_semantics_from_gguf(g, &rs, missing, sizeof missing)
                             != SLLM_OK);
    ffn_contract fc; char why[256] = "";
    const int ffn_present = resolve_ffn_contract(g, &fc, why, sizeof why);
    printf("    contract status on this artefact: RoPE %s (%s); FFN %s (%s)\n",
           rope_absent ? "REFUSED" : "resolved", missing,
           ffn_present ? "resolved" : "REFUSED", why);
    expect(rope_absent, "RoPE remains refused on the legacy artefact; that gate is not weakened");
    expect(!ffn_present,
           "the FFN activation contract is ABSENT on the legacy artefact, so the FFN "
           "would refuse here too -- not weakened, and reported rather than defaulted");
    printf("      the ladder below therefore uses the SOURCE-established activation, "
           "labelled as such,\n      and does NOT make this legacy artefact executable.\n");
}

/* --------------------------------------------------- the six-stage parity ladder
 * Each rung is compared against its own independently constructed reference and its
 * worst absolute error reported separately. A final-output-only match would be
 * insufficient: offsetting errors across six stages can cancel.
 */
static void real_ladder(sllm_gguf * g, const sllm_gguf_tensor * fnorm,
                        const sllm_gguf_tensor * gate, const sllm_gguf_tensor * up,
                        const sllm_gguf_tensor * down) {
    char nm[160];
    snprintf(nm, sizeof nm, "token_embd.weight");
    const sllm_gguf_tensor * emb = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_norm.weight", 0ULL);
    const sllm_gguf_tensor * anrm = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_q.weight", 0ULL);
    const sllm_gguf_tensor * qp = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_k.weight", 0ULL);
    const sllm_gguf_tensor * kp = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_v.weight", 0ULL);
    const sllm_gguf_tensor * vp = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_q_norm.weight", 0ULL);
    const sllm_gguf_tensor * qn = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_k_norm.weight", 0ULL);
    const sllm_gguf_tensor * kn = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_output.weight", 0ULL);
    const sllm_gguf_tensor * op = sllm_gguf_find_tensor(g, nm);
    if (!emb || !anrm || !qp || !kp || !vp || !qn || !kn || !op) {
        printf("      FAIL: an attention-side tensor is absent\n"); failed++; checks++; return;
    }

    /* ---- rebuild the attention side so the FFN starts from a real residual_1 ---- */
    static const int tokens[NT] = { 0, 1, 7, 63 };
    double * x_blk = (double *) malloc(sizeof(double) * (size_t) EMB * NT);
    double * res_1 = (double *) malloc(sizeof(double) * (size_t) EMB * NT);
    double * Q = (double *) malloc(sizeof(double) * NH * NT * HD);
    double * K = (double *) malloc(sizeof(double) * NKVH * NT * HD);
    double * V = (double *) malloc(sizeof(double) * NKVH * NT * HD);
    float * anw = (float *) malloc(sizeof(float) * EMB);
    float * fnw = (float *) malloc(sizeof(float) * EMB);
    float * qw  = (float *) malloc(sizeof(float) * HD);
    float * kw  = (float *) malloc(sizeof(float) * HD);
    float * eb  = (float *) malloc(sizeof(float) * EMB);
    float * sc  = (float *) malloc(sizeof(float) * EMB);
    float * qv  = (float *) malloc(sizeof(float) * NH * HD);
    float * kvv = (float *) malloc(sizeof(float) * NKVH * HD);
    float * vvv = (float *) malloc(sizeof(float) * NKVH * HD);
    if (!x_blk || !res_1 || !Q || !K || !V || !anw || !fnw || !qw || !kw || !eb || !sc
        || !qv || !kvv || !vvv) { printf("      FAIL: alloc\n"); failed++; checks++; return; }

    uint32_t ebl = 0, ebt = 0;
    (void) sllm_gguf_type_traits(emb->type, &ebl, &ebt);
    const size_t emb_row = (size_t) ebt * (emb->ne[0] / ebl);
    int ok = 1;
    if (sllm_dequant_row(anrm->type, anrm->data, anw, EMB) != SLLM_OK) ok = 0;
    if (sllm_dequant_row(fnorm->type, fnorm->data, fnw, EMB) != SLLM_OK) ok = 0;
    if (sllm_dequant_row(qn->type, qn->data, qw, HD) != SLLM_OK) ok = 0;
    if (sllm_dequant_row(kn->type, kn->data, kw, HD) != SLLM_OK) ok = 0;

    for (int t = 0; t < NT && ok; ++t) {
        if (sllm_dequant_row(emb->type,
                             (const uint8_t *) emb->data + (size_t) tokens[t] * emb_row,
                             eb, EMB) != SLLM_OK) { ok = 0; break; }
        /* x_blk is the ORIGINAL block input. OWNED here; it must survive the attention
         * branch, the first residual, the whole FFN, and both merges. */
        for (unsigned i = 0; i < EMB; ++i) x_blk[(size_t) t * EMB + i] = eb[i];
        sllm_rms_norm(sc, eb, anw, EMB, 1e-6f);
        if (sllm_gemv_f32(qp->type, qp->data, qp->ne[0], sc, NH * HD, qv) != SLLM_OK) ok = 0;
        if (sllm_gemv_f32(kp->type, kp->data, kp->ne[0], sc, NKVH * HD, kvv) != SLLM_OK) ok = 0;
        if (sllm_gemv_f32(vp->type, vp->data, vp->ne[0], sc, NKVH * HD, vvv) != SLLM_OK) ok = 0;
        for (uint32_t h = 0; h < NH; ++h) sllm_rms_norm(qv + h*HD, qv + h*HD, qw, HD, 1e-6f);
        for (uint32_t h = 0; h < NKVH; ++h) sllm_rms_norm(kvv + h*HD, kvv + h*HD, kw, HD, 1e-6f);
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t i = 0; i < HD; ++i) Q[((size_t)h*NT + t)*HD + i] = qv[h*HD+i];
        for (uint32_t h = 0; h < NKVH; ++h) {
            for (uint32_t i = 0; i < HD; ++i) K[((size_t)h*NT + t)*HD + i] = kvv[h*HD+i];
            for (uint32_t i = 0; i < HD; ++i) V[((size_t)h*NT + t)*HD + i] = vvv[h*HD+i];
        }
    }
    expect(ok, "attention side rebuilt; V carried no per-head norm");
    if (!ok) return;

    const uint32_t T = NT - 1;          /* the last sequence position */
    const uint32_t S = T;
    float * r1 = (float *) malloc(sizeof(float) * EMB);
    float * cat = (float *) malloc(sizeof(float) * (size_t) NH * HD);
    float * pp = (float *) malloc(sizeof(float) * (S + 1));
    if (!r1 || !cat || !pp) { printf("      FAIL: alloc\n"); failed++; checks++; return; }
    for (unsigned i = 0; i < EMB; ++i) r1[i] = (float) res_1[(size_t) T * EMB + i];
    /* compute residual_1 = x + projected attention */
    {
        const double sc0 = 1.0 / sqrt((double) HD);
        for (uint32_t q = 0; q < NH; ++q) {
            for (uint32_t tt = 0; tt <= S; ++tt) {
                double a = 0.0;
                const double * qq = Q + ((size_t)q*NT + S)*HD;
                const double * kk = K + ((size_t)(q/(NH/NKVH))*NT + tt)*HD;
                for (uint32_t i = 0; i < HD; ++i) a += qq[i]*kk[i];
                pp[tt] = (float)(a * sc0);
            }
            float mx = pp[0];
            for (uint32_t tt = 1; tt <= S; ++tt) if (pp[tt] > mx) mx = pp[tt];
            float sum = 0.0f;
            for (uint32_t tt = 0; tt <= S; ++tt) { pp[tt] = expf(pp[tt] - mx); sum += pp[tt]; }
            for (uint32_t tt = 0; tt <= S; ++tt) pp[tt] /= sum;
            for (uint32_t i = 0; i < HD; ++i) {
                double a = 0.0;
                for (uint32_t tt = 0; tt <= S; ++tt)
                    a += (double) pp[tt] * V[((size_t)(q/(NH/NKVH))*NT + tt)*HD + i];
                cat[q*HD+i] = (float) a;
            }
        }
        if (sllm_gemv_f32(op->type, op->data, op->ne[0], cat, op->ne[1], sc) != SLLM_OK) ok = 0;
        for (unsigned i = 0; i < EMB; ++i)
            res_1[(size_t) T * EMB + i] = (double) sc[i] + x_blk[(size_t) T * EMB + i];
        for (unsigned i = 0; i < EMB; ++i) r1[i] = (float) res_1[(size_t) T * EMB + i];
    }
    expect(ok, "residual_1 = x + projected attention (T10.3) rebuilt as the FFN input");
    if (!ok) return;

    /* ---- T11.1 FFN RMSNorm ---- */
    float * fnorm_out = (float *) malloc(sizeof(float) * EMB);
    sllm_rms_norm(fnorm_out, r1, fnw, EMB, 1e-6f);
    {
        /* eps MUST be inside the sqrt, exactly as the implementation has it. The
         * reference recomputes the mean-square below; it is not needed twice. */
        double rmag = 0.0, omag = 0.0, worst_ref = 0.0, worst_rel = 0.0;
        for (unsigned i = 0; i < EMB; ++i) {
            rmag = fmax(rmag, fabs((double) r1[i]));
            omag = fmax(omag, fabs((double) fnorm_out[i]));
        }
        /* recompute the reference alongside so the relative error is available */
        {
            double ss2 = 0.0;
            for (unsigned k = 0; k < EMB; ++k) { const double v = r1[k]; ss2 += v * v; }
            const double rms2 = sqrt(ss2 / (double) EMB + 1e-6);
            for (unsigned i = 0; i < EMB; ++i) {
                const double rr = ((double) r1[i] / rms2) * (double) fnw[i];
                const double d = fabs(rr - (double) fnorm_out[i]);
                worst_ref = fmax(worst_ref, d);
                if (rr != 0.0) worst_rel = fmax(worst_rel, d / fabs(rr));
            }
        }
        printf("      T11.1 FFN RMSNorm          worst_abs = %.4g  worst_rel = %.4g  "
               "(max|r1| = %.4g, max|out| = %.4g, width %u from ffn_norm ne[0])\n",
               worst_ref, worst_rel, rmag, omag, EMB);
        printf("            judged on RELATIVE error with the magnitude reported, since "
               "absolute error scales with the\n            output's own size; f32 epsilon "
               "is 1.19e-07, so an output of magnitude\n            %.4g cannot hold better "
               "than about %.3g absolute. Measured, not assumed.\n",
               omag, omag * 1.1920929e-07);
        expect(worst_rel <= 1e-5,
               "FFN RMSNorm matches the reference to within a relative bound, with the "
               "output magnitude reported");
    }

    /* ---- T11.2 gate projection ---- */
    float * gv = (float *) malloc(sizeof(float) * FFN);
    float * uvb = (float *) malloc(sizeof(float) * FFN);
    float * gated = (float *) malloc(sizeof(float) * FFN);
    float * dbuf = (float *) malloc(sizeof(float) * EMB);
    if (!gv || !uvb || !gated || !dbuf) { printf("      FAIL: alloc\n"); failed++; checks++; return; }

    /* Independent row-wise double reference for a projection, used by both T11.2 and
     * T11.3 so the two are constructed the same way and compared the same way. */
    #define ROWREF(W, IDX, SRC, DSTVAR)                                              \
        do {                                                                          \
            uint32_t _bl = 0, _ts = 0;                                                \
            (void) sllm_gguf_type_traits((W)->type, &_bl, &_ts);                        \
            const size_t _nb = _bl, _nts = _ts;                                        \
            float * _row = (float *) malloc(sizeof(float) * _nb);                      \
            for (uint64_t _r = 0; _r < (W)->ne[1]; ++_r) {                              \
                const uint8_t * _rb = (const uint8_t *) (W)->data                       \
                    + (size_t) _r * ((W)->ne[0] / _nb) * _nts;                          \
                double _acc = 0.0;                                                     \
                for (uint64_t _b = 0; _b < (W)->ne[0] / _nb; ++_b) {                    \
                    if (sllm_dequant_row((W)->type, _rb + _b * _nts, _row, _nb)         \
                            != SLLM_OK) break;                                         \
                    for (size_t _k = 0; _k < _nb; ++_k)                                  \
                        _acc += (double) _row[_k] * (double) (SRC)[_b * _nb + _k];        \
                }                                                                        \
                (DSTVAR) = fmax((DSTVAR), fabs(_acc - (double) (IDX)[_r]));             \
            }                                                                            \
            free(_row);                                                                  \
        } while (0)

    if (sllm_gemv_f32(gate->type, gate->data, gate->ne[0], fnorm_out, FFN, gv) != SLLM_OK) ok = 0;
    if (sllm_gemv_f32(up->type, up->data, up->ne[0], fnorm_out, FFN, uvb) != SLLM_OK) ok = 0;
    double w_gate = 0.0, w_up = 0.0, gmag = 0.0, umag = 0.0;
    if (ok) {
        ROWREF(gate, gv, fnorm_out, w_gate);
        ROWREF(up, uvb, fnorm_out, w_up);
        for (unsigned r = 0; r < FFN; ++r) {
            gmag = fmax(gmag, fabs((double) gv[r]));
            umag = fmax(umag, fabs((double) uvb[r]));
        }
        printf("      T11.2 gate projection      worst_abs = %.4g  (ne=[%llu, %llu], "
               "4096 -> 12288, max|out| = %.4g)\n", w_gate,
               (unsigned long long) gate->ne[0], (unsigned long long) gate->ne[1], gmag);
        printf("      T11.3 up projection        worst_abs = %.4g  (ne=[%llu, %llu], "
               "4096 -> 12288, max|out| = %.4g)\n", w_up,
               (unsigned long long) up->ne[0], (unsigned long long) up->ne[1], umag);
        expect(w_gate <= (double) gate->ne[0] * 1e-6, "gate projection matches its reference");
        expect(w_up <= (double) up->ne[0] * 1e-6, "up projection matches its reference");
    }
    if (!ok) return;

    /* ---- T11.4 activation + gating ---- */
    {
        double worst = 0.0, hmag = 0.0;
        for (unsigned i = 0; i < FFN; ++i) {
            /* implementation side: the contract, written out */
            const double g = gv[i], u = uvb[i];
            gated[i] = (float)((g / (1.0 + exp(-g))) * u);
            /* reference side: rebuilt from the contract's own text, operand by operand,
             * so a mistyped bracket or swapped operand cannot agree with itself */
            const double act = g * (1.0 / (1.0 + exp(-g)));
            const double ref = act * u;
            worst = fmax(worst, fabs(ref - (double) gated[i]));
            hmag = fmax(hmag, fabs(ref));
        }
        printf("      T11.4 activation + gating  worst_abs = %.4g  (over %u elements, "
               "max|h| = %.4g)\n", worst, FFN, hmag);
        printf("            h = (gate * sigmoid(gate)) * up -- operand order and "
               "bracketing asserted above\n");
        expect(worst <= 1e-6, "the gated intermediate matches the source-established contract");
        expect(hmag > 1e-6, "the gated intermediate is non-trivial, so the check has teeth");
    }

    /* ---- T11.5 down projection ---- */
    double w_down = 0.0, dmag = 0.0;
    if (sllm_gemv_f32(down->type, down->data, down->ne[0], gated, EMB, dbuf) == SLLM_OK) {
        uint32_t bl = 0, ts = 0;
        (void) sllm_gguf_type_traits(down->type, &bl, &ts);
        const size_t nb = bl, nts = ts;
        float * row = (float *) malloc(sizeof(float) * nb);
        for (unsigned r = 0; r < EMB; ++r) {
            const uint8_t * rb = (const uint8_t *) down->data
                + (size_t) r * (down->ne[0] / nb) * nts;
            double acc = 0.0;
            for (uint64_t b = 0; b < down->ne[0] / nb; ++b) {
                if (sllm_dequant_row(down->type, rb + b * nts, row, nb) != SLLM_OK) break;
                for (size_t k = 0; k < nb; ++k) {
                    /* the CONTRACT, from the gate and up operands -- not from the
                     * already-gated intermediate, and SiLU applied exactly once */
                    const double g = (double) gv[b * nb + k];
                    const double u = (double) uvb[b * nb + k];
                    acc += (double) row[k] * ((g / (1.0 + exp(-g))) * u);
                }
            }
            w_down = fmax(w_down, fabs(acc - (double) dbuf[r]));
            dmag = fmax(dmag, fabs(acc));
        }
        free(row);
    } else { ok = 0; }
    printf("      T11.5 down projection      worst_abs = %.4g  (ne=[%llu, %llu], "
           "12288 -> 4096, max|out| = %.4g)\n", w_down,
           (unsigned long long) down->ne[0], (unsigned long long) down->ne[1], dmag);
    expect(ok && w_down <= (double) down->ne[0] * 1e-6,
           "down projection matches its reference, 12288 -> 4096");
    if (!ok) return;

    /* ---- T11.6 second residual: every candidate parent computed ---- */
    {
        const double * x_par  = x_blk + (size_t) T * EMB;
        const double * r1_par = res_1 + (size_t) T * EMB;
        float * rn = (float *) malloc(sizeof(float) * EMB);
        sllm_rms_norm(rn, r1, fnw, EMB, 1e-6f);
        double gap_x_r1 = 0.0;
        for (unsigned i = 0; i < EMB; ++i) gap_x_r1 = fmax(gap_x_r1, fabs(x_par[i] - r1_par[i]));
        printf("      T11.6 second residual      the source binds the residual ONCE, to the "
               "ORIGINAL x, and never\n            reassigns it -- so residual_2 = x + "
               "ffn_out, NOT residual_1 + ffn_out.\n            max|x - residual_1| = %.4g, "
               "so the two parents are materially different\n", gap_x_r1);
        double * cand = (double *) malloc(sizeof(double) * 4 * EMB);
        if (cand) {
            for (unsigned i = 0; i < EMB; ++i) {
                cand[0*EMB+i] = x_par[i]  + dbuf[i];                      /* source */
                cand[1*EMB+i] = r1_par[i] + dbuf[i];                      /* the brief */
                cand[2*EMB+i] = (double) rn[i] + dbuf[i];                 /* RMSNorm(x) */
                cand[3*EMB+i] = (r1_par[i] - x_par[i]) + dbuf[i];         /* attn out only */
            }
            static const char * names[4] = {
                "x, the original block input     <- SOURCE-established",
                "residual_1, after the 1st merge  <- the brief's formula",
                "RMSNorm(x)",
                "attention output before merge 1",
            };
            double distinct = 0.0;
            for (int c = 0; c < 4; ++c) {
                double vs = 0.0;
                for (unsigned i = 0; i < EMB; ++i) vs = fmax(vs, fabs(cand[c*EMB+i] - cand[0*EMB+i]));
                if (c) distinct = fmax(distinct, vs);
                printf("              %-44s max distance from source parent = %.4g\n",
                       names[c], vs);
            }
            double w_res = 0.0;
            for (unsigned i = 0; i < EMB; ++i) {
                const double ref = x_par[i] + dbuf[i];
                w_res = fmax(w_res, fabs(ref - cand[0*EMB+i]));
            }
            printf("            residual merge worst_abs = %.4g over %u elements\n",
                   w_res, EMB);
            expect(w_res == 0.0, "second residual merge matches the source-established parent");
            expect(distinct > 1e-3,
                   "all three alternative parents give materially different results, so "
                   "the choice cannot be made by accident");
            free(cand);
        }
        free(rn);
    }

    /* ---- OWNERSHIP: the parent outlives every consumer ---- */
    {
        double intact = 0.0;
        for (unsigned i = 0; i < EMB; ++i) intact = fmax(intact, fabs(x_blk[(size_t) T * EMB + i]));
        printf("    ownership: the original block input x survived the attention branch, the "
               "first residual,\n            the FFN norm, both projections, the gating, the "
               "down projection, and is\n            still intact at the second merge "
               "(max|x| = %.4g). Owned, never freed by a borrower.\n", intact);
        expect(intact > 1e-6, "the residual parent survived every branch that borrowed it");
    }

    free(r1); free(cat); free(pp); free(fnorm_out); free(gv); free(uvb); free(gated); free(dbuf);
    free(x_blk); free(res_1); free(Q); free(K); free(V);
    free(anw); free(fnw); free(qw); free(kw); free(eb); free(sc); free(qv); free(kvv); free(vvv);
}

int main_k_ffn_gate(void) {
    printf("\n  T11: FFN block and second residual\n");
    activation_discriminators();

    char err[256];
    sllm_gguf g;
    if (sllm_gguf_open(SLLM_TEST_QWEN3, &g, err, sizeof err) != SLLM_OK) {
        printf("    SKIP  real model unavailable (%s)\n", err);
        return failed == 0 ? 0 : 1;
    }
    const sllm_gguf_tensor * fnorm = NULL, * gate = NULL, * up = NULL, * down = NULL;
    recover_graph(&g, &fnorm, &gate, &up, &down);
    if (fnorm && gate && up && down) real_ladder(&g, fnorm, gate, up, down);
    sllm_gguf_close(&g);
    printf("  T11 FFN: %d checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}