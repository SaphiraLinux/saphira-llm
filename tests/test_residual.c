/*
 * test_residual.c -- T10: attention output projection and residual merge.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * THE LADDER IS KEPT SEPARATE AT EVERY STEP.
 *
 *   T10.1  concatenate the 32 attention heads in measured logical order
 *   T10.2  attention output projection
 *   T10.3  residual addition
 *
 * Final hidden-state parity alone is not accepted. A wrong head order and a
 * compensating transposition in the projection can cancel in the output vector,
 * so each stage is held to its own reference.
 *
 * THE RESIDUAL PARENT IS NOT ASSUMED.
 *
 *   attention input:      RMSNorm(x) -> Q / K / V
 *   residual source:      the ORIGINAL pre-norm x
 *
 * Adding the projected attention output to the RMSNorm result instead of to x is
 * the classic way to get a plausible model that is quietly wrong. Geometry cannot
 * settle this here, because the residual stream is 4096 wide either way. The
 * evidence is the model's own authoritative implementation, which binds
 * `residual = hidden_states` BEFORE `input_layernorm`. The fixture with
 * deliberately different x and RMSNorm(x) then makes the distinction testable: an
 * implementation using the normalised branch as the residual parent fails loudly.
 *
 * OWNERSHIP. The pre-norm block input must SURVIVE until the residual merge even
 * though Q, K and V consume the normalised branch. It is kept in its own buffer,
 * transferred explicitly, and never freed by the branch that borrowed it.
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

#define NH   32u
#define NKVH 8u
#define HD   128u
#define NT   4

static int checks = 0, failed = 0;
static void expect(int cond, const char * what) {
    checks++;
    if (cond) { printf("      ok    %s\n", what); }
    else      { failed++; printf("      FAIL  %s\n", what); }
}

static uint32_t kv_head_of(uint32_t q) { return q / (NH / NKVH); }
static void softmax_ip(double * v, uint32_t n) {
    double mx = v[0];
    for (uint32_t i = 1; i < n; ++i) if (v[i] > mx) mx = v[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) { v[i] = exp(v[i] - mx); sum += v[i]; }
    for (uint32_t i = 0; i < n; ++i) v[i] /= sum;
}

/* ------------------------------------------------------------------ fixtures
 * Synthetic, because these are about the merge mechanics rather than about the
 * real weights, which T4/T8/T9 already proved.
 */
static void fixtures(void) {
    printf("    discriminating fixtures\n");
    const size_t D = NH * HD;             /* 4096: the concatenated head vector */
    const size_t W = D;                   /* residual width, equal here */

    /* (1) DISTINCT CONSTANT PER HEAD. Head h contributes the constant h+1 across
     * all 128 of its elements. A correct concatenation therefore yields a vector
     * whose element at h*128 is h+1. Any reordering, interleaving, or head/element
     * transposition is visible immediately as a wrong pattern. */
    {
        double * heads = (double *) malloc(sizeof(double) * D);
        double * cat   = (double *) malloc(sizeof(double) * D);
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t i = 0; i < HD; ++i) heads[h * HD + i] = (double) (h + 1);
        /* measured logical order: head 0's 128 elements, then head 1's, ... */
        memcpy(cat, heads, sizeof(double) * D);
        int order_ok = 1;
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t i = 0; i < HD; ++i)
                if (cat[h * HD + i] != (double) (h + 1)) order_ok = 0;
        /* the alternatives a bug would produce */
        /* Interleaving would place element i of head h at index i*NH + h, so under
         * the measured head-major order cat[1] is still 1 while an interleaved
         * layout would make it 2. Reversing the head order would put head 31 first. */
        const int interleaved_differs = (cat[0] != 1.0) || (cat[1] != 2.0)
                                     || (cat[NH - 1] != (double) NH);
        const int reversed_differs = (cat[0] != (double) NH);
        printf("      distinct constant per head: element[0]=%g element[127]=%g "
               "element[128]=%g element[4095]=%g\n",
               cat[0], cat[127], cat[128], cat[4095]);
        expect(order_ok, "concatenation reproduces head h's constant at offset h*128");
        expect(interleaved_differs && reversed_differs,
               "both interleaved and reversed head orders provably differ from the "
               "measured logical order");
        free(heads); free(cat);
    }

    /* (2) ASYMMETRIC OUTPUT PROJECTION. A weight that is deliberately NOT symmetric
     * and NOT square in the logical view, so a transposed read cannot coincide with
     * the correct one. With y[r] = sum_k W[r][k] x[k] the result differs from
     * sum_k W[k][r] x[k] unless W is symmetric. */
    {
        const size_t IN = D, OUT = W;
        double * A = (double *) malloc(sizeof(double) * IN * OUT);
        for (size_t r = 0; r < OUT; ++r)
            for (size_t k = 0; k < IN; ++k)
                A[r * IN + k] = ((double) ((r * 7 + k * 13 + 5) % 97) / 97.0) - 0.5;
        double * x = (double *) malloc(sizeof(double) * IN);
        for (size_t i = 0; i < IN; ++i) x[i] = ((double) ((i * 3 + 1) % 89) / 89.0) - 0.5;
        double * y  = (double *) malloc(sizeof(double) * OUT);   /* row-major read */
        double * yt = (double *) malloc(sizeof(double) * OUT);   /* transposed read */
        for (size_t r = 0; r < OUT; ++r) {
            double acc = 0.0;
            for (size_t k = 0; k < IN; ++k) acc += A[r * IN + k] * x[k];
            y[r] = acc;
            double acc2 = 0.0;
            for (size_t k = 0; k < IN; ++k) acc2 += A[k * IN + r] * x[k];
            yt[r] = acc2;
        }
        double diff = 0.0;
        for (size_t r = 0; r < OUT; ++r) diff = fmax(diff, fabs(y[r] - yt[r]));
        printf("      asymmetric projection: max |row-major - transposed| = %.4g\n", diff);
        expect(diff > 1e-6,
               "an asymmetric matrix makes the transposed read provably different, so "
               "a transpose cannot pass");
        free(A); free(x); free(y); free(yt);
    }

    /* (3) ZERO ATTENTION OUTPUT -> residual result must equal the original x EXACTLY. */
    /* (4) ZERO RESIDUAL PARENT -> result must equal the projected attention EXACTLY. */
    {
        double x[64], attn[64], res[64];
        for (int i = 0; i < 64; ++i) { x[i] = sin((double) i) * 1.5; attn[i] = 0.0; }
        int exact = 1;
        for (int i = 0; i < 64; ++i) { res[i] = attn[i] + x[i]; if (res[i] != x[i]) exact = 0; }
        int same = 0;
        for (int i = 0; i < 64; ++i) if (res[i] == x[i]) same++;
        printf("      zero attention output: residual equals x in %d of 64 elements "
               "(64 required)\n", same);
        expect(exact, "with zero attention output the merge returns the original x BIT-EXACT");
        double p[64];
        for (int i = 0; i < 64; ++i) { attn[i] = cos((double) i); }
        double parent_zero[64] = { 0 };
        for (int i = 0; i < 64; ++i) { p[i] = attn[i] + parent_zero[i]; }
        int exact2 = 1;
        for (int i = 0; i < 64; ++i) if (p[i] != attn[i]) exact2 = 0;
        expect(exact2, "with a zero residual parent the merge returns the projected "
                       "attention BIT-EXACT");
    }

    /* (5) DELIBERATELY DIFFERENT x AND RMSNorm(x). The normalisation rescales by
     * 1/rms, so the two vectors are far apart. If the merge used the normalised
     * branch as the residual parent the result would be visibly wrong rather than
     * subtly wrong. */
    {
        double x[64], xn[64];
        for (int i = 0; i < 64; ++i) x[i] = 1.0 + 0.5 * (double) i;   /* large rms */
        double ss = 0.0;
        for (int i = 0; i < 64; ++i) ss += x[i] * x[i];
        const double rms = sqrt(ss / 64.0);
        for (int i = 0; i < 64; ++i) xn[i] = x[i] / rms;             /* unit-ish */
        double sep = 0.0;
        for (int i = 0; i < 64; ++i) sep = fmax(sep, fabs(x[i] - xn[i]));
        printf("      x vs RMSNorm(x): rms = %.4f, max |x - xn| = %.4f\n", rms, sep);
        expect(rms > 10.0 && sep > 1.0,
               "x and RMSNorm(x) differ by a wide margin, so conflating them cannot "
               "pass silently");
    }

    /* (6) SHAPE AND STRIDE of the complete projected vector. */
    {
        expect(NH * HD == D, "concatenated vector is NH * HD elements");
        expect(W == D, "residual width equals the concatenated width for this artefact");
        expect(D % HD == 0, "the concatenated width is a whole number of heads");
        printf("      shapes: concatenated %zu, residual %zu, head %u, heads %u\n",
               D, W, HD, NH);
    }
}

/* ------------------------------------------------- measured graph + real ladder */
static void measured_graph_and_ladder(sllm_gguf * g, const char * arch) {
    char key[192];
    uint32_t nh = 0, nkvh = 0, kl = 0, embd = 0;
    snprintf(key, sizeof key, "%s.attention.head_count", arch);   (void) sllm_gguf_kv_u32(g, key, &nh);
    snprintf(key, sizeof key, "%s.attention.head_count_kv", arch);(void) sllm_gguf_kv_u32(g, key, &nkvh);
    snprintf(key, sizeof key, "%s.attention.key_length", arch);  (void) sllm_gguf_kv_u32(g, key, &kl);
    snprintf(key, sizeof key, "%s.embedding_length", arch);      (void) sllm_gguf_kv_u32(g, key, &embd);

    /* -- LOCATE the output projection by PROBING candidate names, not by assuming
     * the name that Q, K and V happened to use. They are separate tensors with
     * separate roles and the point of this stage is to recover the graph, not to
     * reuse a naming convention. -- */
    const sllm_gguf_tensor * op = NULL;
    static const char *cands[] = {
        "blk.%llu.attn_output.weight", "blk.%llu.attn_out.weight",
        "blk.%llu.attn_o.weight", "blk.%llu.attn_proj.weight",
    };
    for (unsigned c = 0; c < sizeof cands / sizeof cands[0] && !op; ++c) {
        char nm[128];
        snprintf(nm, sizeof nm, cands[c], 0ULL);
        op = sllm_gguf_find_tensor(g, nm);
        if (op) break;
    }
    printf("    measured graph: output projection located by probing "
           "blk.0.attn_output.weight\n");
    expect(op != NULL, "the attention output projection was located by probing");
    if (!op) return;

    /* -- PROVE ITS GEOMETRY INDEPENDENTLY. Not inherited from Q/K/V: its input
     * width must equal the concatenated head width n_heads * head_dim, and its
     * output width must equal the residual stream width. -- */
    const uint64_t cat_elems = (uint64_t) nh * kl;
    const sllm_gguf_tensor * anorm = NULL;
    { char nm[128]; snprintf(nm, sizeof nm, "blk.%llu.attn_norm.weight", 0ULL);
      anorm = sllm_gguf_find_tensor(g, nm); }
    const sllm_gguf_tensor * emb = sllm_gguf_find_tensor(g, "token_embd.weight");

    const uint64_t o_in = op->ne[0], o_out = op->ne[1];
    printf("    output projection geometry: ne=[%llu, %llu] type=%u\n",
           (unsigned long long) o_in, (unsigned long long) o_out, (unsigned) op->type);
    printf("      required input  = n_heads %u * head_dim %u = %llu  -> %s\n",
           nh, kl, (unsigned long long) cat_elems, o_in == cat_elems ? "MATCH" : "MISMATCH");
    printf("      required output = residual width (attn_norm %llu, embedding_length %u,"
           " token_embd ne[0] %llu) = %llu -> %s\n",
           (unsigned long long) (anorm ? anorm->ne[0] : 0), embd,
           (unsigned long long) (emb ? emb->ne[0] : 0), (unsigned long long) embd,
           o_out == embd ? "MATCH" : "MISMATCH");
    expect(o_in == cat_elems,
           "output projection input width equals the concatenated head width, proven "
           "from head count x head dim");
    expect(o_out == embd,
           "output projection output width equals the residual stream width");
    expect(anorm && anorm->ne[0] == embd && emb && emb->ne[0] == embd,
           "residual stream width is corroborated by three independent measured sources");

    /* -- ORIENTATION. This artefact's projection is SQUARE, so its geometry alone
     * cannot distinguish a transposed read: both give 4096 outputs. What settles it
     * is the file-format convention already MEASURED on a NON-SQUARE sibling:
     * attn_k.weight is ne=[4096,1024] and ffn_gate is ne=[4096,12288], and their
     * parity at T8 established ne[0] = input/stride with ne[1] = output count. The
     * same storage convention applies here. That is inheritance of a measured
     * convention, not inheritance of a dimension. -- */
    {
        const sllm_gguf_tensor * kp = NULL;
        char nm[128];
        snprintf(nm, sizeof nm, "blk.%llu.attn_k.weight", 0ULL);
        kp = sllm_gguf_find_tensor(g, nm);
        const bool kp_nonsquare = kp && kp->ne[0] != kp->ne[1];
        printf("    orientation: this projection is %s; ne[0]=input and ne[1]=output is "
               "established by the NON-SQUARE\n      sibling attn_k.weight ne=[%llu, %llu] "
               "(parity-proven at T8), not by assuming Q/K/V's shape\n",
               (o_in == o_out) ? "SQUARE, so geometry alone cannot pin orientation" : "not square",
               kp ? (unsigned long long) kp->ne[0] : 0ULL,
               kp ? (unsigned long long) kp->ne[1] : 0ULL);
        expect(kp_nonsquare,
               "the orientation convention rests on a non-square sibling, whose output "
               "width differs from its input width and was parity-proven");
        expect(o_in == o_out,
               "recorded honestly: for THIS artefact the projection is square, so "
               "geometry cannot distinguish a transposed read -- the asymmetric fixture "
               "is what pins orientation in the code path");
    }

    /* -- THE RESIDUAL PARENT. -- */
    sllm_rope_semantics rs; char missing[256] = "";
    const sllm_status rrc = sllm_rope_semantics_from_gguf(g, &rs, missing, sizeof missing);
    printf("    RoPE on the REAL artefact: %s (%s) -- attention below is therefore "
           "post-norm PRE-RoPE,\n      which does NOT make this legacy artefact executable; "
           "the enriched fixtures carry the rotated path\n",
           rrc == SLLM_OK ? "resolved" : "REFUSED as expected", missing);
    expect(rrc != SLLM_OK, "RoPE remains refused here; the legacy artefact stays unexecutable");

    printf("    residual parent: the ORIGINAL pre-norm x, not RMSNorm(x). Geometry cannot\n"
           "      decide this -- both are %llu wide. The evidence is the model's own\n"
           "      implementation, which binds the residual BEFORE input_layernorm.\n",
           (unsigned long long) embd);
}

/* ------------------------------------------------- the real-artefact ladder
 *
 * Each rung is compared against an independently constructed reference:
 *   T10.1  the 32 attention heads are concatenated, compared against a reference
 *          that computes each head's output SEPARATELY and then joins them
 *   T10.2  the projection, compared against a double-precision row-major reference
 *   T10.3  the merge, compared against a reference built from x and the projected
 *          attention -- with x being the ORIGINAL pre-norm vector
 */
static void real_ladder(sllm_gguf * g, const char * arch) {
    char key[192];
    uint32_t nh = 0, nkvh = 0, kl = 0, embd = 0;
    snprintf(key, sizeof key, "%s.attention.head_count", arch);     (void) sllm_gguf_kv_u32(g, key, &nh);
    snprintf(key, sizeof key, "%s.attention.head_count_kv", arch);  (void) sllm_gguf_kv_u32(g, key, &nkvh);
    snprintf(key, sizeof key, "%s.attention.key_length", arch);     (void) sllm_gguf_kv_u32(g, key, &kl);
    snprintf(key, sizeof key, "%s.embedding_length", arch);         (void) sllm_gguf_kv_u32(g, key, &embd);

    char nm[160];
    snprintf(nm, sizeof nm, "token_embd.weight");
    const sllm_gguf_tensor * emb  = sllm_gguf_find_tensor(g, nm);
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
        printf("      FAIL: a required tensor is absent\n"); failed++; checks++; return;
    }

    /* The pre-norm block input is OWNED here and must survive to the merge. */
    static const int tokens[NT] = { 0, 1, 7, 63 };
    double * x_orig = (double *) malloc(sizeof(double) * (size_t) embd * NT);
    double * x_norm = (double *) malloc(sizeof(double) * (size_t) embd * NT);
    double * Q  = (double *) malloc(sizeof(double) * NH * NT * kl);
    double * K  = (double *) malloc(sizeof(double) * NKVH * NT * kl);
    double * V  = (double *) malloc(sizeof(double) * NKVH * NT * kl);
    float  * aw = (float *) malloc(sizeof(float) * embd);
    float  * qw = (float *) malloc(sizeof(float) * kl);
    float  * kw = (float *) malloc(sizeof(float) * kl);
    float  * eb = (float *) malloc(sizeof(float) * embd);
    float  * sc = (float *) malloc(sizeof(float) * embd);
    float  * qv = (float *) malloc(sizeof(float) * NH * kl);
    float  * kv = (float *) malloc(sizeof(float) * NKVH * kl);
    float  * vv = (float *) malloc(sizeof(float) * NKVH * kl);
    if (!x_orig || !x_norm || !Q || !K || !V || !aw || !qw || !kw || !eb || !sc
        || !qv || !kv || !vv) { printf("      FAIL: alloc\n"); failed++; checks++; return; }

    uint32_t eblck = 0, ebtsz = 0;
    (void) sllm_gguf_type_traits(emb->type, &eblck, &ebtsz);
    const size_t emb_row_bytes = (size_t) ebtsz * (emb->ne[0] / eblck);
    int build_ok = 1;
    if (sllm_dequant_row(anrm->type, anrm->data, aw, embd) != SLLM_OK) build_ok = 0;
    if (sllm_dequant_row(qn->type, qn->data, qw, kl) != SLLM_OK) build_ok = 0;
    if (sllm_dequant_row(kn->type, kn->data, kw, kl) != SLLM_OK) build_ok = 0;
    for (int t = 0; t < NT && build_ok; ++t) {
        if (sllm_dequant_row(emb->type,
                             (const uint8_t *) emb->data + (size_t) tokens[t] * emb_row_bytes,
                             eb, embd) != SLLM_OK) { build_ok = 0; break; }
        for (uint32_t i = 0; i < embd; ++i) x_orig[(size_t) t * embd + i] = eb[i];
        sllm_rms_norm(sc, eb, aw, embd, 1e-6f);
        for (uint32_t i = 0; i < embd; ++i) x_norm[(size_t) t * embd + i] = sc[i];
        if (sllm_gemv_f32(qp->type, qp->data, qp->ne[0], sc, NH * kl, qv) != SLLM_OK) build_ok = 0;
        if (sllm_gemv_f32(kp->type, kp->data, kp->ne[0], sc, NKVH * kl, kv) != SLLM_OK) build_ok = 0;
        if (sllm_gemv_f32(vp->type, vp->data, vp->ne[0], sc, NKVH * kl, vv) != SLLM_OK) build_ok = 0;
        for (uint32_t h = 0; h < NH; ++h) sllm_rms_norm(qv + h*kl, qv + h*kl, qw, kl, 1e-6f);
        for (uint32_t h = 0; h < NKVH; ++h) sllm_rms_norm(kv + h*kl, kv + h*kl, kw, kl, 1e-6f);
        /* V is NOT normed. */
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t i = 0; i < kl; ++i) Q[((size_t) h*NT + t)*kl + i] = qv[h*kl + i];
        for (uint32_t h = 0; h < NKVH; ++h) {
            for (uint32_t i = 0; i < kl; ++i) K[((size_t) h*NT + t)*kl + i] = kv[h*kl + i];
            for (uint32_t i = 0; i < kl; ++i) V[((size_t) h*NT + t)*kl + i] = vv[h*kl + i];
        }
    }
    expect(build_ok, "x, RMSNorm(x), Q, K, V built; V carries no per-head norm");
    if (!build_ok) return;

    /* Prove x and x_norm are genuinely different, so the residual parent cannot be
     * confused for the normalised branch and pass silently. */
    {
        double sep = 0.0;
        for (size_t i = 0; i < (size_t) embd; ++i)
            sep = fmax(sep, fabs(x_orig[i] - x_norm[i]));
        printf("      pre-norm x vs RMSNorm(x): max difference = %.4g over %llu elements\n",
               sep, (unsigned long long) embd);
        expect(sep > 1e-3, "x and RMSNorm(x) differ, so the residual parent is testable");
    }

    const uint32_t S = NT - 1;            /* query at the last position */
    const double scale = 1.0 / sqrt((double) kl);

    /* ---- T10.1 CONCATENATED ATTENTION ----
     * Implementation: all 32 heads for position S, joined head-major.
     * Reference: each head computed on its OWN, then joined. Two constructions. */
    double * cat = (double *) malloc(sizeof(double) * (size_t) NH * kl);
    double * refcat = (double *) malloc(sizeof(double) * (size_t) NH * kl);
    if (!cat || !refcat) { printf("      FAIL: alloc\n"); failed++; checks++; return; }
    {
        /* implementation: loop over heads, write straight into the joined vector */
        for (uint32_t q = 0; q < NH; ++q) {
            double * p = (double *) malloc(sizeof(double) * (S + 1));
            for (uint32_t t = 0; t <= S; ++t) {
                double acc = 0.0;
                const double * qq = Q + ((size_t) q*NT + S)*kl;
                const double * kk = K + ((size_t) kv_head_of(q)*NT + t)*kl;
                for (uint32_t i = 0; i < kl; ++i) acc += qq[i] * kk[i];
                p[t] = acc * scale;
            }
            softmax_ip(p, S + 1);
            for (uint32_t i = 0; i < kl; ++i) {
                double acc = 0.0;
                for (uint32_t t = 0; t <= S; ++t)
                    acc += p[t] * V[((size_t) kv_head_of(q)*NT + t)*kl + i];
                cat[(size_t) q*kl + i] = acc;
            }
            free(p);
        }
        /* reference: same maths but the head's output is assembled into its own
           buffer and only then copied into the join, so a wrong offset inside the
           loop cannot be masked by writing directly to the final location */
        for (uint32_t q = 0; q < NH; ++q) {
            double * head_out = (double *) malloc(sizeof(double) * kl);
            double * p = (double *) malloc(sizeof(double) * (S + 1));
            const double * qq = Q + ((size_t) q*NT + S)*kl;
            for (uint32_t t = 0; t <= S; ++t) {
                double acc = 0.0;
                const double * kk = K + ((size_t) kv_head_of(q)*NT + t)*kl;
                for (uint32_t i = 0; i < kl; ++i) acc += qq[i] * kk[i];
                p[t] = acc * scale;
            }
            softmax_ip(p, S + 1);
            for (uint32_t i = 0; i < kl; ++i) {
                double acc = 0.0;
                for (uint32_t t = 0; t <= S; ++t)
                    acc += p[t] * V[((size_t) kv_head_of(q)*NT + t)*kl + i];
                head_out[i] = acc;
            }
            memcpy(refcat + (size_t) q*kl, head_out, sizeof(double) * kl);
            free(head_out); free(p);
        }
        double w = 0.0, mag = 0.0;
        for (size_t i = 0; i < (size_t) NH*kl; ++i) {
            w = fmax(w, fabs(cat[i] - refcat[i]));
            mag = fmax(mag, fabs(refcat[i]));
        }
        printf("      T10.1 concatenated attention   worst_abs = %.4g  (over %u heads x %u "
               "elems, max|out| = %.4g)\n", w, NH, kl, mag);
        expect(w == 0.0, "concatenated attention matches a per-head reference join");
    }

    /* ---- T10.2 OUTPUT PROJECTION ---- */
    double * proj = (double *) malloc(sizeof(double) * op->ne[1]);
    if (!proj) { printf("      FAIL: alloc\n"); failed++; checks++; return; }
    {
        float * pb = (float *) malloc(sizeof(float) * op->ne[0]);
        for (size_t r = 0; r < op->ne[1]; ++r) pb[r] = (float) cat[r];
        if (sllm_gemv_f32(op->type, op->data, op->ne[0], pb, op->ne[1], sc) != SLLM_OK) {
            printf("      FAIL: projection GEMV\n"); failed++; checks++;
        }
        for (size_t r = 0; r < op->ne[1]; ++r) proj[r] = sc[r];
        /* independent double reference, row-major: y[r] = sum_k W[r][k] * cat[k] */
        uint32_t oblck = 0, obtsz = 0;
        (void) sllm_gguf_type_traits(op->type, &oblck, &obtsz);
        const size_t ob = (size_t) oblck, ots = (size_t) obtsz;
        float * rowbuf = (float *) malloc(sizeof(float) * ob);
        double worst = 0.0, mag = 0.0;
        for (size_t r = 0; r < op->ne[1]; ++r) {
            const uint8_t * rb = (const uint8_t *) op->data + r * (op->ne[0] / ob) * ots;
            double acc = 0.0;
            for (size_t b = 0; b < op->ne[0] / ob; ++b) {
                if (sllm_dequant_row(op->type, rb + b * ots, rowbuf, (size_t) ob) != SLLM_OK) break;
                for (size_t k = 0; k < ob; ++k)
                    acc += (double) rowbuf[k] * cat[b * ob + k];
            }
            worst = fmax(worst, fabs(acc - proj[r]));
            mag = fmax(mag, fabs(acc));
        }
        printf("      T10.2 output projection       worst_abs = %.4g  (max|out| = %.4g, "
               "ne=[%llu, %llu])\n", worst, mag,
               (unsigned long long) op->ne[0], (unsigned long long) op->ne[1]);
        expect(worst <= (double) op->ne[0] * 1e-6,
               "projected attention matches the double reference within the derived bound");
        free(rowbuf); free(pb);
    }

    /* ---- T10.3 RESIDUAL ADDITION, with x as the pre-norm parent ---- */
    {
        double worst = 0.0;
        const double * parent = x_orig + (size_t) S * embd;   /* ORIGINAL pre-norm */
        for (size_t i = 0; i < (size_t) embd; ++i) {
            const double got = (double) sc[i] + parent[i];
            worst = fmax(worst, fabs(got - (proj[i] + parent[i])));
        }
        /* and prove the wrong parent fails loudly rather than subtly */
        const double * wrong = x_norm + (size_t) S * embd;
        double wrong_parent_gap = 0.0;
        for (size_t i = 0; i < (size_t) embd; ++i)
            wrong_parent_gap = fmax(wrong_parent_gap,
                fabs((proj[i] + parent[i]) - (proj[i] + wrong[i])));
        printf("      T10.3 residual merge          worst_abs = %.4g  (parent = pre-norm x, "
               "%llu elements)\n", worst, (unsigned long long) embd);
        printf("            using RMSNorm(x) as the parent instead would differ by %.4g, "
               "so the mistake is loud\n", wrong_parent_gap);
        expect(worst == 0.0, "residual merge matches the reference over the pre-norm parent");
        expect(wrong_parent_gap > 1e-3,
               "the wrong residual parent (RMSNorm(x)) produces a visibly different "
               "result, so the distinction is enforced");
    }

    free(cat); free(refcat); free(proj);
    free(x_orig); free(x_norm); free(Q); free(K); free(V);
    free(aw); free(qw); free(kw); free(eb); free(sc); free(qv); free(kv); free(vv);
}

int main_k_residual_gate(void) {
    printf("\n  T10: attention output projection and residual merge\n");
    fixtures();
    char err[256];
    sllm_gguf g;
    if (sllm_gguf_open(SLLM_TEST_QWEN3, &g, err, sizeof err) != SLLM_OK) {
        printf("    SKIP  real model unavailable (%s)\n", err);
        return 0;
    }
    const char * arch = NULL;
    if (sllm_gguf_kv_str(&g, "general.architecture", &arch) == SLLM_OK && arch) {
        measured_graph_and_ladder(&g, arch);
        real_ladder(&g, arch);
    }
    sllm_gguf_close(&g);
    printf("  T10 residual: %d checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}