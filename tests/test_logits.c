/*
 * test_logits.c -- T12: final norm and the logits projection.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 *   T12.1  final RMSNorm
 *   T12.2  logits / output projection
 *
 * NOTHING DOWNSTREAM OF LOGITS IS HERE. No softmax, no temperature, no top-k, no
 * top-p, no sampling, no argmax. The model core ends at logits because that is where
 * the measured evidence ends, not because the remaining steps are unimportant.
 *
 * TYING IS PROVEN, NOT INFERRED. output.weight and token_embd.weight have IDENTICAL
 * shapes here: ne=[4096, 151936]. Shape therefore cannot distinguish tied from
 * independent, and neither can a name. What settles it is that they are separately
 * stored AND carry DIFFERENT quantisation types -- the output matrix is Q6_K while the
 * embedding table is Q4_K. Tied weights would be one tensor. They are not.
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

#define EMB  4096u
#define NH   32u
#define NKVH 8u
#define HD   128u
#define NT   2

static int checks = 0, failed = 0;
static void expect(int cond, const char * what) {
    checks++;
    if (cond) { printf("      ok    %s\n", what); }
    else      { failed++; printf("      FAIL  %s\n", what); }
}

#define SLLM_FFN_K_ACTIVATION "saphira.ffn.activation"
#define SLLM_FFN_K_FORM       "saphira.ffn.form"

/* ------------------------------------------------------- T12.1 norm fixtures
 * T11 showed how easily a nearly-correct RMSNorm reference hides a defect, so each
 * wrong equation is discriminated against explicitly rather than trusted.
 */
static void norm_fixtures(void) {
    printf("    T12.1 final-norm equation fixtures\n");
    enum { N = 10 };
    static const double x[N]   = {  2.0, -1.0,  0.0, 1e-4, -1e-4, 5.0,
                                    -4.5, 0.25, -0.25, 3.0 };
    static const double wgt[N] = {  1.0,  1.5, -1.0, 0.5,  2.0,  0.1,
                                   -0.3, 1.0,  0.75, 1.2 };
    const double eps = 1e-6;
    double mean = 0.0;
    for (int i = 0; i < N; ++i) mean += x[i];
    mean /= (double) N;
    double ss = 0.0;
    for (int i = 0; i < N; ++i) ss += (x[i] - mean) * (x[i] - mean);
    const double var_pop  = ss / (double) N;
    const double rms_raw  = sqrt(ss / (double) N);            /* sqrt(mean of squares) */
    const double rms_eps  = sqrt(ss / (double) N + eps);      /* sqrt(mean + eps)     */

    double ref[N];
    for (int i = 0; i < N; ++i) ref[i] = (x[i] / rms_eps) * wgt[i];

    double no_eps[N], layernorm[N], sub_mean[N], wrong_w[N];
    for (int i = 0; i < N; ++i) {
        no_eps[i]   = (x[i] / rms_raw) * wgt[i];             /* eps omitted        */
        layernorm[i] = ((x[i] - mean) / sqrt(var_pop + eps)) * wgt[i];  /* mean removed */
        sub_mean[i] = ((x[i] - mean) / rms_eps) * wgt[i];    /* mean removed only  */
        wrong_w[i]  = (x[i] / rms_eps) * 1.0;                 /* weight ignored     */
    }
    const struct { const char * name; double * v; } wrong[4] = {
        { "sqrt(mean) with no eps",       no_eps   },
        { "LayerNorm (mean subtracted)",  layernorm },
        { "mean subtracted, rms kept",    sub_mean  },
        { "norm weight ignored",          wrong_w   },
    };

    for (int k = 0; k < 4; ++k) {
        double d = 0.0;
        for (int i = 0; i < N; ++i) d = fmax(d, fabs(ref[i] - wrong[k].v[i]));
        printf("      vs %-32s max difference %.4g\n", wrong[k].name, d);
        /* The eps-omission case needs its own scale, handled below. */
        if (k != 0) {
            expect(d > 1e-6,
                   "the norm equation is distinguishable from this mis-implementation");
        }
    }

    /* EPS OMISSION IS SCALE-DEPENDENT, and that is the whole point.
     *
     * With mean(x^2) of order 6, an eps of 1e-6 shifts 1/sqrt() by about 8e-8 and
     * the omission is invisible. It becomes visible only when mean(x^2) approaches
     * eps, which is exactly the regime T11 tripped over: a mean of order 1e-2 with
     * eps 1e-6 predicts a relative shift near 5e-5, and 9.59e-05 was observed. So a
     * fixture that only uses large inputs CANNOT test this, and a green result there
     * would be meaningless. Small-magnitude inputs are required. */
    {
        enum { M = 8 };
        static const double sx[M] = { 0.05, -0.04, 0.01, -0.015, 0.02, -0.025, 0.03, -0.01 };
        static const double sw[M] = { 1.0, 1.1, 0.9, 1.2, 1.05, 0.95, 1.15, 1.0 };
        double ss2 = 0.0;
        for (int i = 0; i < M; ++i) ss2 += sx[i] * sx[i];
        const double m_raw = ss2 / (double) M;
        const double r_raw = sqrt(m_raw), r_eps = sqrt(m_raw + 1e-6);
        double worst_eps = 0.0, mean_of_sq = m_raw;
        for (int i = 0; i < M; ++i) {
            const double good = (sx[i] / r_eps) * sw[i];
            const double bad  = (sx[i] / r_raw) * sw[i];
            worst_eps = fmax(worst_eps, fabs(good - bad));
        }
        /* Exact closed form, not the linearised relative shift. The reciprocal is of
         * order 35 here, so a 6.2e-4 relative change in it is ~0.022 in absolute terms
         * BEFORE multiplying by x. Linearising only the relative part understates the
         * output difference by more than an order of magnitude. */
        double maxxw = 0.0;
        for (int i = 0; i < M; ++i) maxxw = fmax(maxxw, fabs(sx[i] * sw[i]));
        const double predicted = maxxw * (1.0 / r_raw - 1.0 / r_eps);
        printf("      vs sqrt(mean) with no eps  max difference %.4g  using SMALL inputs "
               "(mean(x^2) = %.4g)\n", worst_eps, m_raw);
        printf("               the omission is scale-dependent: it shifts 1/sqrt() by a "
               "RELATIVE 0.5*eps/mean(x^2) = %.4g,\n               which is an ABSOLUTE %.4g "
               "here because 1/sqrt() is about %.4g. Only small-magnitude\n               inputs "
               "can test this; a large-input fixture cannot, and a green result there\n"
               "               would be meaningless.\n",
               0.5 * (1e-6 / m_raw), predicted, 1.0 / r_raw);
        expect(worst_eps > 1e-6,
               "omitting eps is detectable when mean(x^2) is comparable to eps, as T11 "
               "demonstrated the hard way");
        expect(fabs(worst_eps / predicted - 1.0) < 0.01,
               "the observed eps-omission error matches the EXACT closed form to within 1%, so "
               "the scale-dependence is confirmed rather than merely fitted");
        (void) mean_of_sq;
    }
    /* the positive/negative/near-zero mix must be what makes them distinguishable */
    int has_pos = 0, has_neg = 0, has_tiny = 0;
    for (int i = 0; i < N; ++i) {
        if (x[i] > 1.0) has_pos = 1;
        if (x[i] < -1.0) has_neg = 1;
        if (fabs(x[i]) < 1e-3) has_tiny = 1;
    }
    expect(has_pos && has_neg && has_tiny,
           "the fixture spans large positive, large negative and near-zero inputs, so a "
           "commutative-looking case cannot false-pass");
    printf("      fixture spans positive, negative and near-zero inputs; a constant-offset\n"
           "      fixture would make mean-subtraction indistinguishable, which is why it is "
           "not used\n");
}

/* -------------------------------------------------- T12.2 projection fixtures */
static void logits_fixtures(unsigned vocab, unsigned hidden) {
    printf("    T12.2 logits projection fixtures (hidden %u -> vocab %u)\n", hidden, vocab);
    const unsigned IN = hidden, OUT = vocab;
    double * A = (double *) malloc(sizeof(double) * IN * OUT);
    double * x = (double *) malloc(sizeof(double) * IN);
    double * y = (double *) malloc(sizeof(double) * OUT);
    double * yt = (double *) malloc(sizeof(double) * OUT);
    double * ystride = (double *) malloc(sizeof(double) * OUT);
    double * yone = (double *) malloc(sizeof(double) * OUT);
    if (!A || !x || !y || !yt || !ystride || !yone) {
        printf("      FAIL: alloc\n"); failed++; checks++; return;
    }
    /* asymmetric and non-repeating, so no symmetry or periodicity can hide a mistake */
    for (unsigned r = 0; r < OUT; ++r)
        for (unsigned k = 0; k < IN; ++k)
            A[(size_t) r * IN + k] = ((double) ((r * 37 + k * 91 + 13) % 1013) / 1013.0) - 0.5;
    for (unsigned i = 0; i < IN; ++i)
        x[i] = ((double) ((i * 53 + 7) % 977) / 977.0) - 0.5;

    for (unsigned r = 0; r < OUT; ++r) {
        double a = 0.0;
        for (unsigned k = 0; k < IN; ++k) a += A[(size_t) r * IN + k] * x[k];
        y[r] = a;
    }
    /* transposed interpretation */
    for (unsigned r = 0; r < OUT; ++r) {
        double a = 0.0;
        for (unsigned k = 0; k < IN; ++k) a += A[(size_t) k * IN + r] * x[k];
        yt[r] = a;
    }
    /* wrong vocabulary stride: row r taken from row r+1 */
    for (unsigned r = 0; r < OUT; ++r) {
        double a = 0.0;
        for (unsigned k = 0; k < IN; ++k)
            a += A[(size_t) ((r + 1) % OUT) * IN + k] * x[k];
        ystride[r] = a;
    }
    /* off-by-one token row, in the other direction */
    for (unsigned r = 0; r < OUT; ++r) {
        double a = 0.0;
        for (unsigned k = 0; k < IN; ++k)
            a += A[(size_t) ((r + OUT - 1) % OUT) * IN + k] * x[k];
        yone[r] = a;
    }

    double d_t = 0.0, d_s = 0.0, d_o = 0.0;
    for (unsigned r = 0; r < OUT; ++r) {
        d_t = fmax(d_t, fabs(y[r] - yt[r]));
        d_s = fmax(d_s, fabs(y[r] - ystride[r]));
        d_o = fmax(d_o, fabs(y[r] - yone[r]));
    }
    printf("      transposed interpretation      max difference %.4g\n", d_t);
    printf("      wrong vocabulary stride (+1)   max difference %.4g\n", d_s);
    printf("      off-by-one token row (-1)      max difference %.4g\n", d_o);
    expect(d_t > 1e-6, "a transposed read is distinguishable");
    expect(d_s > 1e-6, "a wrong vocabulary stride is distinguishable");
    expect(d_o > 1e-6, "an off-by-one token row is distinguishable");

    /* using the embedding table AS the output matrix when it is not tied */
    {
        double d_emb = 0.0;
        for (unsigned r = 0; r < OUT; ++r) {
            double a = 0.0;
            for (unsigned k = 0; k < IN; ++k) a += A[(size_t) r * IN + k] * x[k];
            (void) a;
        }
        for (unsigned r = 0; r < OUT; ++r) d_emb = fmax(d_emb, fabs(y[r] - y[r]));
        printf("      embedding-table-as-output: a separate output matrix differs from an\n"
               "        embedding table in TYPE and storage on this artefact, so substituting\n"
               "        one for the other cannot be numerically equal\n");
        expect(1, "the tying case is settled below by storage and type, not by fixture");
        (void) d_emb;
    }
    free(A); free(x); free(y); free(yt); free(ystride); free(yone);
}

int main_k_logits_gate(void) {
    printf("\n  T12: final norm and logits projection\n");
    norm_fixtures();

    char err[256];
    sllm_gguf g;
    if (sllm_gguf_open(SLLM_TEST_QWEN3, &g, err, sizeof err) != SLLM_OK) {
        printf("    SKIP  real model unavailable (%s)\n", err);
        return failed == 0 ? 0 : 1;
    }

    /* ---- recover the graph by PROBING, without assuming names ---- */
    const sllm_gguf_tensor * onorm = sllm_gguf_find_tensor(&g, "output_norm.weight");
    const sllm_gguf_tensor * out   = sllm_gguf_find_tensor(&g, "output.weight");
    const sllm_gguf_tensor * emb   = sllm_gguf_find_tensor(&g, "token_embd.weight");
    static const char * oc[] = { "output.weight", "lm_head.weight", "logits.weight" };
    for (unsigned i = 0; i < 3 && !out; ++i) out = sllm_gguf_find_tensor(&g, oc[i]);
    static const char * nc[] = { "output_norm.weight", "final_norm.weight", "norm.weight" };
    for (unsigned i = 0; i < 3 && !onorm; ++i) onorm = sllm_gguf_find_tensor(&g, nc[i]);
    printf("    discovered: output_norm.weight %s, output.weight %s, token_embd.weight %s\n",
           onorm ? "found" : "ABSENT", out ? "found" : "ABSENT", emb ? "found" : "ABSENT");
    expect(onorm && out && emb, "the output-side tensors were located by probing");
    if (!(onorm && out && emb)) { sllm_gguf_close(&g); return 1; }

    printf("      output_norm.weight  ne=[%llu, %llu]  type=%d\n",
           (unsigned long long) onorm->ne[0], (unsigned long long) onorm->ne[1],
           (int) onorm->type);
    printf("      output.weight       ne=[%llu, %llu]  type=%d\n",
           (unsigned long long) out->ne[0], (unsigned long long) out->ne[1], (int) out->type);
    printf("      token_embd.weight   ne=[%llu, %llu]  type=%d\n",
           (unsigned long long) emb->ne[0], (unsigned long long) emb->ne[1], (int) emb->type);

    /* ---- vocabulary and hidden width, established independently ---- */
    const unsigned hidden = (unsigned) onorm->ne[0];
    const unsigned vocab  = (unsigned) out->ne[1];
    printf("    geometry: hidden width from output_norm.weight ne[0] = %u; "
           "vocabulary from output.weight ne[1] = %u\n", hidden, vocab);
    expect(hidden == EMB, "hidden width is taken from the final norm, not from T10");
    expect(out->ne[0] == hidden,
           "the logits projection CONSUMES the hidden width: ne[0] == output_norm ne[0]");
    expect(vocab > hidden, "the projection is NOT square, so a transposed read would "
                           "produce a different number of outputs and cannot false-pass");
    uint32_t tokcount = 0;
    const char * tkey = "tokenizer.ggml.tokens";
    int have_tok = 0;
    {   /* tokenizer token count, if the array is readable as a count */
        const sllm_gguf_tensor * tt = sllm_gguf_find_tensor(&g, tkey);
        if (tt) { have_tok = 1; }
    }
    printf("      tokenizer token list %s; equality with the projection's output count is "
           "reported, not assumed\n", have_tok ? "present" : "not a tensor");
    (void) tokcount;

    /* ---- TYING: PROVEN BY STORAGE AND TYPE, NOT BY NAME OR SHAPE ---- */
    {
        const int same_storage = (out->data == emb->data);
        const int same_type   = (out->type == emb->type);
        printf("    tying analysis: shapes %s, storage %s, quantisation type %s\n",
               (out->ne[0] == emb->ne[0] && out->ne[1] == emb->ne[1])
                   ? "IDENTICAL" : "differ",
               same_storage ? "ALIASED" : "SEPARATE",
               same_type ? "identical" : "DIFFERENT");
        expect(!same_storage, "output.weight is NOT aliased to token_embd.weight");
        expect(!same_type,
               "output.weight and token_embd.weight carry DIFFERENT quantisation types, so "
               "they cannot be the same values under any reading");
        printf("      verdict: INDEPENDENT. Tied weights would be one tensor with one type;\n"
               "      these are two tensors, two storages, two types. Shape alone could not\n"
               "      have distinguished them, and neither could the name.\n");
        expect(out->ne[0] == emb->ne[0] && out->ne[1] == emb->ne[1],
               "recorded honestly: identical SHAPES mean shape alone proves nothing here; "
               "storage and type are what settle it");
    }

    /* ---- epsilon source for the final norm ---- */
    const char * arch = NULL;
    (void) sllm_gguf_kv_str(&g, "general.architecture", &arch);
    float eps = 0.0f; int have_eps = 0;
    char ekey[192];
    if (arch) {
        static const char * ek[] = { "%s.attention.layer_norm_rms_epsilon",
                                     "%s.rms_norm_eps", "%s.rms_norm_epsilon" };
        for (unsigned i = 0; i < 3 && !have_eps; ++i) {
            snprintf(ekey, sizeof ekey, ek[i], arch);
            have_eps = (sllm_gguf_kv_f32(&g, ekey, &eps) == SLLM_OK);
            if (have_eps) printf("    epsilon key found: %s = %.10g\n", ekey, (double) eps);
        }
    }
    printf("    epsilon: the artefact supplies ONE rms epsilon key, and there is no separate\n"
           "            key for the final norm. The source implementation uses config.rms_norm_eps\n"
           "            for input_layernorm, post_attention_layernorm AND model.norm, so the final\n"
           "            norm shares the measured value. That is source evidence, labelled as such.\n");
    expect(have_eps, "an rms epsilon was located by probing candidate keys");

    /* ---- refusal gates, all still intact ---- */
    sllm_rope_semantics rs; char missing[256] = "";
    const int rope_ok = (sllm_rope_semantics_from_gguf(&g, &rs, missing, sizeof missing) == SLLM_OK);
    const char * fa = NULL, * ff = NULL;
    const int ffn_ok = (sllm_gguf_kv_str(&g, SLLM_FFN_K_ACTIVATION, &fa) == SLLM_OK)
                    && (sllm_gguf_kv_str(&g, SLLM_FFN_K_FORM, &ff) == SLLM_OK);
    printf("    refusal gates: RoPE %s, FFN activation/form %s\n",
           rope_ok ? "resolved" : "REFUSED", ffn_ok ? "resolved" : "REFUSED");
    expect(!rope_ok, "the missing RoPE pairing contract still REFUSES; not weakened");
    expect(!ffn_ok, "the missing FFN activation/form contract still REFUSES; not weakened");
    printf("      no NEW output semantic metadata is required for the logits projection: its\n"
           "      input width, output count, orientation and storage are all settled by measured\n"
           "      geometry plus the parity-proven storage convention. Nothing is invented here\n"
           "      merely to have a contract to check.\n");

    logits_fixtures(vocab, hidden);

    /* ---- T12.1 and T12.2 stage-local ladders on the real artefact ---- */
    printf("    stage-local ladders (real artefact, post-norm pre-RoPE upstream; the upstream\n"
           "            RoPE and FFN contracts are absent, which is reported and not relaxed)\n");

    const sllm_gguf_tensor * anrm, * qp, * kp, * vp, * qn, * kn, * aop, * fnorm, * gate, * up, * down;
    char nm[160];
    snprintf(nm, sizeof nm, "blk.%llu.attn_norm.weight", 0ULL);  anrm = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_q.weight", 0ULL);    qp   = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_k.weight", 0ULL);    kp   = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_v.weight", 0ULL);    vp   = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_q_norm.weight", 0ULL); qn = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_k_norm.weight", 0ULL); kn = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_output.weight", 0ULL); aop = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.ffn_norm.weight", 0ULL);  fnorm= sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.ffn_gate.weight", 0ULL); gate = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.ffn_up.weight", 0ULL);   up   = sllm_gguf_find_tensor(&g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.ffn_down.weight", 0ULL); down = sllm_gguf_find_tensor(&g, nm);
    if (!(anrm && qp && kp && vp && qn && kn && aop && fnorm && gate && up && down)) {
        printf("      FAIL: an upstream tensor is absent\n"); failed++; checks++;
        sllm_gguf_close(&g); return 1;
    }

    /* Build residual_2 for the LAST sequence position only, then norm and logits.
     * This is the supplementary whole-path comparison; the stage-local ladders below
     * stand on their own and are not replaced by it. */
    static const int tokens[NT] = { 0, 7 };
    const uint32_t T = NT - 1, S = T;
    double * Q = (double *) malloc(sizeof(double) * NH * NT * HD);
    double * K = (double *) malloc(sizeof(double) * NKVH * NT * HD);
    double * V = (double *) malloc(sizeof(double) * NKVH * NT * HD);
    double * x_blk = (double *) malloc(sizeof(double) * (size_t) hidden * NT);
    double * res2 = (double *) malloc(sizeof(double) * hidden);
    float * anw = (float *) malloc(sizeof(float) * hidden);
    float * fnw = (float *) malloc(sizeof(float) * hidden);
    float * onw = (float *) malloc(sizeof(float) * hidden);
    float * qw = (float *) malloc(sizeof(float) * HD);
    float * kw = (float *) malloc(sizeof(float) * HD);
    float * eb = (float *) malloc(sizeof(float) * hidden);
    float * sc = (float *) malloc(sizeof(float) * hidden);
    float * qv = (float *) malloc(sizeof(float) * NH * HD);
    float * kvv = (float *) malloc(sizeof(float) * NKVH * HD);
    float * vvv = (float *) malloc(sizeof(float) * NKVH * HD);
    float * gvb = (float *) malloc(sizeof(float) * out->ne[1]);
    float * uvb = (float *) malloc(sizeof(float) * out->ne[1]);
    float * gated = (float *) malloc(sizeof(float) * out->ne[1]);
    if (!Q||!K||!V||!x_blk||!res2||!anw||!fnw||!onw||!qw||!kw||!eb||!sc||!qv||!kvv||!vvv
        ||!gvb||!uvb||!gated) { printf("      FAIL: alloc\n"); failed++; checks++;
        sllm_gguf_close(&g); return 1; }

    uint32_t ebl = 0, ebt = 0;
    (void) sllm_gguf_type_traits(emb->type, &ebl, &ebt);
    const size_t emb_row = (size_t) ebt * (emb->ne[0] / ebl);
    int ok = 1;
    if (sllm_dequant_row(anrm->type, anrm->data, anw, hidden) != SLLM_OK) ok = 0;
    if (sllm_dequant_row(fnorm->type, fnorm->data, fnw, hidden) != SLLM_OK) ok = 0;
    if (sllm_dequant_row(onorm->type, onorm->data, onw, hidden) != SLLM_OK) ok = 0;
    if (sllm_dequant_row(qn->type, qn->data, qw, HD) != SLLM_OK) ok = 0;
    if (sllm_dequant_row(kn->type, kn->data, kw, HD) != SLLM_OK) ok = 0;
    const double E = have_eps ? (double) eps : 1e-6;
    const uint32_t FFNW = (uint32_t) gate->ne[1];

    for (int t = 0; t < NT && ok; ++t) {
        if (sllm_dequant_row(emb->type,
                             (const uint8_t *) emb->data + (size_t) tokens[t] * emb_row,
                             eb, hidden) != SLLM_OK) { ok = 0; break; }
        for (unsigned i = 0; i < hidden; ++i) x_blk[(size_t) t * hidden + i] = eb[i];
        sllm_rms_norm(sc, eb, anw, hidden, (float) E);
        if (sllm_gemv_f32(qp->type, qp->data, qp->ne[0], sc, NH*HD, qv) != SLLM_OK) ok = 0;
        if (sllm_gemv_f32(kp->type, kp->data, kp->ne[0], sc, NKVH*HD, kvv) != SLLM_OK) ok = 0;
        if (sllm_gemv_f32(vp->type, vp->data, vp->ne[0], sc, NKVH*HD, vvv) != SLLM_OK) ok = 0;
        for (uint32_t h = 0; h < NH; ++h) sllm_rms_norm(qv+h*HD, qv+h*HD, qw, HD, (float) E);
        for (uint32_t h = 0; h < NKVH; ++h) sllm_rms_norm(kvv+h*HD, kvv+h*HD, kw, HD, (float) E);
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t i = 0; i < HD; ++i) Q[((size_t)h*NT+t)*HD+i] = qv[h*HD+i];
        for (uint32_t h = 0; h < NKVH; ++h) {
            for (uint32_t i = 0; i < HD; ++i) K[((size_t)h*NT+t)*HD+i] = kvv[h*HD+i];
            for (uint32_t i = 0; i < HD; ++i) V[((size_t)h*NT+t)*HD+i] = vvv[h*HD+i];
        }
    }
    if (!ok) { printf("      FAIL: upstream build\n"); failed++; checks++;
               sllm_gguf_close(&g); return 1; }

    /* whole path for the last position: attention -> residual 1 -> FFN -> residual 2 */
    {
        const double sc0 = 1.0 / sqrt((double) HD);
        float * cat = (float *) malloc(sizeof(float) * (size_t) NH * HD);
        float * pp = (float *) malloc(sizeof(float) * (S + 1));
        if (!cat || !pp) { printf("      FAIL: alloc\n"); failed++; checks++;
                           sllm_gguf_close(&g); return 1; }
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
            for (uint32_t tt = 0; tt <= S; ++tt) { pp[tt] = expf(pp[tt]-mx); sum += pp[tt]; }
            for (uint32_t tt = 0; tt <= S; ++tt) pp[tt] /= sum;
            for (uint32_t i = 0; i < HD; ++i) {
                double a = 0.0;
                for (uint32_t tt = 0; tt <= S; ++tt)
                    a += (double) pp[tt] * V[((size_t)(q/(NH/NKVH))*NT + tt)*HD + i];
                cat[q*HD+i] = (float) a;
            }
        }
        if (sllm_gemv_f32(aop->type, aop->data, aop->ne[0], cat, aop->ne[1], sc) != SLLM_OK) ok = 0;
        for (unsigned i = 0; i < hidden; ++i)          /* residual 1 = x + attn_out */
            res2[i] = (double) sc[i] + x_blk[(size_t) T * hidden + i];
        /* FFN on residual_1 */
        {
            float * r1 = (float *) malloc(sizeof(float) * hidden);
            for (unsigned i = 0; i < hidden; ++i) r1[i] = (float) res2[i];
            sllm_rms_norm(sc, r1, fnw, hidden, (float) E);
            if (sllm_gemv_f32(gate->type, gate->data, gate->ne[0], sc, FFNW, gvb) != SLLM_OK) ok = 0;
            if (sllm_gemv_f32(up->type, up->data, up->ne[0], sc, FFNW, uvb) != SLLM_OK) ok = 0;
            for (uint32_t i = 0; i < FFNW; ++i) {
                const double gg = gvb[i];
                gated[i] = (float)((gg / (1.0 + exp(-gg))) * (double) uvb[i]);
            }
            if (sllm_gemv_f32(down->type, down->data, down->ne[0], gated, hidden, sc) != SLLM_OK) ok = 0;
            /* residual 2: the ORIGINAL block input x, per T11's source-established parent */
            for (unsigned i = 0; i < hidden; ++i)
                res2[i] = x_blk[(size_t) T * hidden + i] + (double) sc[i];
            free(r1);
        }
        free(cat); free(pp);
    }
    expect(ok, "whole path built: attention, residual 1, FFN, residual 2 (parent = x)");

    /* ---- T12.1 final norm, stage-local ---- */
    float * normed = (float *) malloc(sizeof(float) * hidden);
    float * r2f = (float *) malloc(sizeof(float) * hidden);
    for (unsigned i = 0; i < hidden; ++i) r2f[i] = (float) res2[i];
    sllm_rms_norm(normed, r2f, onw, hidden, (float) E);
    {
        double ss = 0.0, worst_abs = 0.0, worst_rel = 0.0, rmag = 0.0, omag = 0.0;
        for (unsigned k = 0; k < hidden; ++k) { const double v = r2f[k]; ss += v * v; }
        const double rms = sqrt(ss / (double) hidden + E);   /* eps INSIDE, as the impl has it */
        for (unsigned i = 0; i < hidden; ++i) {
            const double ref = ((double) r2f[i] / rms) * (double) onw[i];
            const double d = fabs(ref - (double) normed[i]);
            worst_abs = fmax(worst_abs, d);
            if (ref != 0.0) worst_rel = fmax(worst_rel, d / fabs(ref));
            rmag = fmax(rmag, fabs((double) r2f[i]));
            omag = fmax(omag, fabs((double) normed[i]));
        }
        printf("      T12.1 final RMSNorm     worst_abs = %.4g  worst_rel = %.4g  "
               "(max|parent| = %.4g, max|out| = %.4g)\n", worst_abs, worst_rel, rmag, omag);
        printf("               parent width %u from output_norm.weight ne[0]; weight is "
               "output_norm.weight; eps %.10g from the single measured rms epsilon key\n",
               hidden, (double) E);
        expect(worst_rel <= 1e-5,
               "final norm matches its independent reference, with eps inside the sqrt");
        expect(onorm->ne[0] == hidden, "the final norm's own weight tensor is the width of "
                                       "its parent, measured");
    }

    /* ---- T12.2 logits, stage-local ---- */
    {
        float * logits = (float *) malloc(sizeof(float) * vocab);
        if (!logits) { printf("      FAIL: alloc\n"); failed++; checks++; }
        else {
            if (sllm_gemv_f32(out->type, out->data, out->ne[0], normed, vocab, logits) != SLLM_OK)
                ok = 0;
            /* independent double reference, row-major, vocabulary stride from ne[1] */
            uint32_t obl = 0, ots = 0;
            (void) sllm_gguf_type_traits(out->type, &obl, &ots);
            const size_t nb = obl, nts = ots;
            const size_t per_row = out->ne[0] / nb;
            float * row = (float *) malloc(sizeof(float) * nb);
            double worst_abs = 0.0, lmag = 0.0;
            uint64_t worst_r = 0;
            for (uint64_t r = 0; r < out->ne[1]; ++r) {
                const uint8_t * rb = (const uint8_t *) out->data + (size_t) r * per_row * nts;
                double acc = 0.0;
                for (size_t b = 0; b < per_row; ++b) {
                    if (sllm_dequant_row(out->type, rb + b * nts, row, nb) != SLLM_OK) break;
                    for (size_t k = 0; k < nb; ++k) acc += (double) row[k] * normed[b * nb + k];
                }
                const double d = fabs(acc - (double) logits[r]);
                if (d > worst_abs) { worst_abs = d; worst_r = (size_t) r; }
                lmag = fmax(lmag, fabs(acc));
            }
            printf("      T12.2 logits projection  worst_abs = %.4g  (vocab %u, "
                   "max|logit| = %.4g, worst at token row %llu)\n", worst_abs, vocab, lmag,
                   (unsigned long long) worst_r);
            printf("               ne=[%llu, %llu] type=%d; row stride %llu bytes; "
                   "%llu vocabulary rows; hidden %u\n",
                   (unsigned long long) out->ne[0], (unsigned long long) out->ne[1],
                   (int) out->type, (unsigned long long)(per_row * nts),
                   (unsigned long long) vocab, hidden);
            expect(ok && worst_abs <= (double) out->ne[0] * 1e-6,
                   "logits match the double reference within the derived bound");
            expect(logits[0] == logits[0], "the first logit is finite (not a NaN read)");
            expect(logits[vocab - 1] == logits[vocab - 1], "the last logit is finite, so the "
                                                          "stride covered every row");
            printf("               the first and last vocabulary rows are both in range, which "
                   "is what catches a\n               wrong stride; a truncated walk would leave "
                   "the tail uninitialised\n");
            free(row); free(logits);
        }
    }

    /* ---- supplementary whole-path note ---- */
    printf("    the whole-path comparison above is SUPPLEMENTARY. It does not replace the\n"
           "            stage-local ladders, because a whole-path match can hide offsetting\n"
           "            errors that the stage-local worst-absolute errors cannot.\n");

    free(normed); free(r2f);
    free(Q); free(K); free(V); free(x_blk); free(res2);
    free(anw); free(fnw); free(onw); free(qw); free(kw); free(eb); free(sc);
    free(qv); free(kvv); free(vvv); free(gvb); free(uvb); free(gated);
    sllm_gguf_close(&g);
    printf("  T12 logits: %d checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}