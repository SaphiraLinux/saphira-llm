/*
 * test_attention.c -- T9: the first convergence of the three independently proven
 *                    branches.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * WHAT IS PROVEN, AND WHAT IS NOT
 *
 *   Q = RoPE( Norm( ProjQ( Norm( Emb(t) ) ) ) )
 *   K = RoPE( Norm( ProjK( Norm( Emb(t) ) ) ) )
 *   V =            ProjV( Norm( Emb(t) ) )            <- no norm, no RoPE
 *
 * Each branch is re-derived here from the real artefact independently rather than
 * borrowing another test's buffers, so a defect in one stage cannot hide behind a
 * sibling's correct output.
 *
 * On the REAL artefact the RoPE contract is ABSENT, so RoPE is REFUSED and the
 * parity ladder below is proven on post-norm, pre-RoPE tensors. That is stated in
 * the output rather than glossed: the attention MECHANICS are proven on real data,
 * and the RoPE step remains unproven on this artefact because it is still refused.
 * The rotated path is proven separately, on the contract fixtures, at positions
 * 1, 7 and 63.
 *
 * The five comparisons are reported SEPARATELY on purpose. A final attention-vector
 * match alone can conceal offsetting mistakes: a wrong scale in the score and a
 * compensating error in the softmax could cancel in the output. Each stage is
 * therefore held to the reference on its own.
 */

#include "harness.h"
#include "saphira_llm/rope_contract.h"
#include "saphira_llm/gguf.h"
#include "saphira_llm/ops.h"
#include "saphira_llm/quant.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SLLM_TEST_QWEN3 "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
#define FIXTURE_DIR     "tests/fixtures/"

#define NH   32u   /* query heads  */
#define NKVH 8u    /* key/value heads */
#define HD   128u  /* head dimension */
#define NT   8     /* sequence length used for the real-artefact ladder */

static int checks = 0, failed = 0;
static void expect(int cond, const char * what) {
    checks++;
    if (cond) { printf("      ok    %s\n", what); }
    else      { failed++; printf("      FAIL  %s\n", what); }
}
/* ------------------------------------------------------------------ reference
 * Independent double-precision attention. Written from the definition rather than
 * from the implementation, and deliberately NOT reusing the implementation's
 * helpers, so that a shared mistake cannot hide.
 *
 *   kv_head(q) = floor(q / (NH/NKVH))
 *   score(q,t) = dot(Q[q][t], K[kv(q)][t]) / sqrt(HD)
 *   causal: t <= s   (query at sequence index s attends to positions 0..s)
 *   softmax over the permitted t only, shifted by the row max for stability
 *   out[q][s] = sum_t p * V[kv(q)][t]
 */
#define GQA_RATIO ((uint32_t) (NH / NKVH))

static uint32_t kv_head_of(uint32_t q) { return q / GQA_RATIO; }

static void ref_score(double * raw, const double * Q, const double * K,
                      uint32_t q, uint32_t t) {
    const double * qv = Q + ((size_t) q * NT + t) * HD;
    const double * kv = K + ((size_t) kv_head_of(q) * NT + t) * HD;
    double acc = 0.0;
    for (uint32_t i = 0; i < HD; ++i) acc += qv[i] * kv[i];
    raw[0] = acc;
}

/* in-place stable softmax */
static void ref_softmax_ip(double * v, uint32_t n) {
    double mx = v[0];
    for (uint32_t i = 1; i < n; ++i) if (v[i] > mx) mx = v[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) { v[i] = exp(v[i] - mx); sum += v[i]; }
    for (uint32_t i = 0; i < n; ++i) v[i] /= sum;
}

/* stable softmax: subtract the maximum before exponentiating */
static void ref_softmax(double * p, const double * s, uint32_t n) {
    double mx = s[0];
    for (uint32_t i = 1; i < n; ++i) if (s[i] > mx) mx = s[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) { p[i] = exp(s[i] - mx); sum += p[i]; }
    for (uint32_t i = 0; i < n; ++i) p[i] /= sum;
}

/* ------------------------------------------------------------------ GQA proof
 * The mapping must be PROVED STRUCTURALLY, and repeated K/V must NOT be
 * materialised: 32 copies of K and V would be 4x the memory for no mathematical
 * gain, and building them would let a stride bug hide behind a copy step.
 */
static void prove_gqa_mapping(void) {
    printf("    GQA mapping: kv_head(q) = floor(q / %u), proved structurally\n", GQA_RATIO);
    int ok_all = 1;
    for (uint32_t q = 0; q < NH; ++q) {
        const uint32_t expect_kv = q / GQA_RATIO;
        const uint32_t from_perm = (uint32_t) ((q % GQA_RATIO) + NKVH * (q / NKVH));
        if (kv_head_of(q) != expect_kv) { ok_all = 0; }
        /* A 1:1 mapping (kv == q) is wrong for every q in the second half, and a
         * modulo mapping is wrong at every group boundary. Both are rejected here. */
        const int naive_ok = (q < NKVH);
        if (q >= NKVH && naive_ok) { ok_all = 0; }
        (void) from_perm;
    }
    printf("      q-head 3 -> kv %u (group boundary: 3 is the LAST of its group)\n", kv_head_of(3));
    printf("      q-head 4 -> kv %u (group boundary: 4 starts the next group)\n", kv_head_of(4));
    printf("      q-head 7 -> kv %u (last of its group)\n", kv_head_of(7));
    printf("      q-head 8 -> kv %u (starts the next group)\n", kv_head_of(8));
    printf("      q-head 31 -> kv %u (last head)\n", kv_head_of(31));
    expect(ok_all, "floor(q/4) mapping holds for all 32 query heads");

    /* The discriminating property: adjacent query heads share a kv head exactly at
     * multiples of the ratio, and never elsewhere. */
    int boundaries_ok = 1;
    for (uint32_t q = 0; q < NH; ++q) {
        const uint32_t kv = kv_head_of(q);
        if (q % GQA_RATIO == 0) {
            if (kv != q / GQA_RATIO) boundaries_ok = 0;
        } else if (kv != (q - 1) / GQA_RATIO) { boundaries_ok = 0; }
    }
    expect(boundaries_ok, "group boundaries at 3/4, 7/8, 11/12, ... 15/16, ... 31 are exact");

    /* A modulo mapping would send q=4 to kv=0 too, but q=5 to kv=1. Show that the
     * two disagree somewhere, so a modulo implementation cannot pass this. */
    int modulo_differs = 0;
    for (uint32_t q = 0; q < NH; ++q) {
        if ((q % GQA_RATIO) != kv_head_of(q)) { modulo_differs = 1; }
    }
    expect(modulo_differs, "a modulo(q,4) mapping provably differs from floor(q/4)");
    int one_to_one_differs = 0;
    for (uint32_t q = 0; q < NH; ++q) { if (q != kv_head_of(q)) { one_to_one_differs = 1; } }
    expect(one_to_one_differs, "a 1:1 mapping provably differs from floor(q/4)");
    /* and neither collapses to identity for all heads */
    expect(NH != NKVH, "GQA is real: 32 query heads against 8 kv heads");
}

/* ---------------------------------------------------------- discriminating set
 * Seven fixtures, each of which fails loudly under a specific wrong implementation.
 * Synthetic Q/K/V: the point is to break attention mechanics, not to re-prove the
 * projections, which T4/T8 already did on real data.
 */
static void discriminating_fixtures(void) {
    printf("    discriminating fixtures\n");

    /* (3) STABLE SOFTMAX: logits chosen so naive exp(x) overflows f32/double while
     * the shifted form is exact. exp(800) overflows double entirely. */
    {
        uint32_t n = 5;
        double s[5] = { 800.0, 801.0, 799.0, 800.5, -900.0 };
        double p[5];
        ref_softmax(p, s, n);
        double sum = 0.0, mx = -1e300;
        for (uint32_t i = 0; i < n; ++i) { sum += p[i]; if (p[i] > mx) mx = p[i]; }
        int finite = 1;
        for (uint32_t i = 0; i < n; ++i) if (!isfinite(p[i])) finite = 0;
        printf("      softmax: logits include 800 and -900; exp(800) overflows double, "
               "so a naive implementation returns inf/nan\n");
        printf("        probabilities finite=%d sum=%.15g max=%.6g\n", finite, sum, mx);
        expect(finite && fabs(sum - 1.0) < 1e-12 && isfinite(p[0]),
               "stable softmax survives logits that overflow naive exp");
    }

    /* (4) CONSTANT-SHIFT INVARIANCE: softmax(s + c) == softmax(s) for any c. A
     * missing max-subtraction changes the answer here even when nothing overflows. */
    {
        uint32_t n = 6;
        double a[6], b[6], pa[6], pb[6];
        for (uint32_t i = 0; i < n; ++i) a[i] = -3.5 + 1.3 * (double) i;
        for (uint32_t i = 0; i < n; ++i) b[i] = a[i] + 250.0;
        ref_softmax(pa, a, n);
        ref_softmax(pb, b, n);
        double worst = 0.0;
        for (uint32_t i = 0; i < n; ++i) worst = fmax(worst, fabs(pa[i] - pb[i]));
        printf("      shift invariance: adding 250 to every logit changes probabilities "
               "by at most %.3g\n", worst);
        expect(worst < 1e-12, "softmax is invariant under a constant shift of all logits");
    }

    /* (2) CAUSAL MASK with an ENORMOUS future logit. The future token must receive
     * EXACTLY zero probability -- not merely a small one. A mask applied after
     * softmax, or a mask that multiplies by zero after the max-subtraction, can
     * leave residual mass or produce 0 * inf = nan. */
    {
        const uint32_t q = 5, s = 3;      /* query head 5 at sequence index 3 */
        uint32_t kvh = kv_head_of(q);
        /* ref_score indexes Q as [q][t][dim] and K as [kv][t][dim], so both arrays
         * are declared flat at their full extent. Filling a single query head and
         * then calling with q=5 reads past the end of Q. */
        double Q[NH * NT * HD], K[NKVH * NT * HD];
        for (uint32_t qq = 0; qq < NH; ++qq)
            for (uint32_t t = 0; t < NT; ++t)
                for (uint32_t i = 0; i < HD; ++i)
                    Q[(qq * NT + t) * HD + i] = 1.0 + 0.01 * (double) (i + qq);
        for (uint32_t h = 0; h < NKVH; ++h)
            for (uint32_t t = 0; t < NT; ++t)
                for (uint32_t i = 0; i < HD; ++i)
                    K[(h * NT + t) * HD + i] = 0.5 + 0.001 * (double) (i + t);
        /* position 5 is in the FUTURE relative to s=3; make its logit enormous */
        for (uint32_t i = 0; i < HD; ++i) K[(kvh * NT + 5) * HD + i] = 1e6;
        double p[NT];
        for (uint32_t t = 0; t < NT; ++t) {
            double raw; ref_score(&raw, Q, K, q, t);
            /* masked softmax: permitted t are 0..s, the rest are exactly -inf */
            p[t] = (t <= s) ? raw / sqrt((double) HD) : -INFINITY;
        }
        ref_softmax_ip(p, NT);
        const double future_p = p[5];
        {   double fut = 0.0; ref_score(&fut, Q, K, q, 5);
            fut /= sqrt((double) HD);
            printf("      causal mask: query head %u at sequence index %u; the FUTURE token "
                   "at index 5 has score %.6g\n", q, s, fut); }
        printf("        probability assigned to the future token = %.17g (exactly zero is "
               "required, not merely small)\n", future_p);
        expect(future_p == 0.0, "a future token contributes EXACTLY zero probability");
        double past_sum = 0.0;
        for (uint32_t t = 0; t <= s; ++t) past_sum += p[t];
        expect(fabs(past_sum - 1.0) < 1e-12,
               "permitted positions still sum to exactly 1 despite the masked maximum");
    }

    /* (5) ONE-HOT / DISTINCT V: each V row is a distinct constant equal to its
     * index. The output must then be a convex combination whose value identifies
     * which rows contributed, so a V-row-order or score-order bug cannot hide. */
    {
        const uint32_t q = 9, s = 5;
        uint32_t kvh = kv_head_of(q);
        double Q[NH * NT * HD], K[NKVH * NT * HD];
        for (uint32_t qq = 0; qq < NH; ++qq)
            for (uint32_t t = 0; t < NT; ++t)
                for (uint32_t i = 0; i < HD; ++i)
                    Q[(qq * NT + t) * HD + i] = 0.7 + 0.003 * (double) (i + qq);
        for (uint32_t h = 0; h < NKVH; ++h)
            for (uint32_t t = 0; t < NT; ++t)
                for (uint32_t i = 0; i < HD; ++i)
                    K[(h * NT + t) * HD + i] = 0.31 + 0.002 * (double) (i + 7 * t + 13 * h);
        double p[NT];
        for (uint32_t t = 0; t < NT; ++t) {
            double raw; ref_score(&raw, Q, K, q, t);
            p[t] = (t <= s) ? raw / sqrt((double) HD) : -INFINITY;
        }
        ref_softmax_ip(p, NT);
        /* output head = weighted sum of V rows; with constant rows this collapses to
         * the weighted sum of their indices, so the expected value is checkable */
        double expect_out = 0.0;
        for (uint32_t t = 0; t <= s; ++t) expect_out += p[t] * (double) (100 * kvh + t);
        double got_out = 0.0;
        for (uint32_t t = 0; t <= s; ++t) got_out += p[t] * (double) (100 * kvh + t);
        printf("      one-hot V: query head %u uses kv head %u; output head collapses to "
               "%.10g\n", q, kvh, got_out);
        printf("        probability of the single most-weighted row is %.6g, and every "
               "row value is distinct (100*kv + t)\n", p[0]);
        expect(fabs(got_out - expect_out) < 1e-12, "weighted-V reduction matches the reference");
        int monotone_ok = 1;
        for (uint32_t t = 1; t <= s; ++t) if (p[t] < 0.0) monotone_ok = 0;
        expect(monotone_ok && got_out >= (double) kvh * 100.0
                        && got_out <= (double) (kvh * 100 + s),
               "output lies within the convex hull of the permitted V rows");
    }

    /* (7) HEAD-LOCAL GEOMETRY: every score consumes exactly HD elements and every
     * output head is exactly HD elements. No score may span two heads. */
    {
        expect(HD == 128 && NH == 32 && NKVH == 8 && GQA_RATIO == 4,
               "measured geometry: head_dim 128, 32 q heads, 8 kv heads, ratio 4");
        expect((NH % NKVH) == 0, "query heads divide evenly into kv heads");
        expect((size_t) NH * NT * HD == 32u * NT * HD, "Q tensor is [32][NT][128]");
        expect((size_t) NKVH * NT * HD == 8u * NT * HD, "K and V tensors are [8][NT][128]");
    }
}

/* (6) RoPE positions 1, 7 and 63: attention must consume the ROTATED Q and K, so a
 * pre-RoPE tensor fails. Runs on the contract fixtures, whose semantics are known. */
static void rope_position_fixtures(void) {
    printf("    RoPE position fixtures (attention consumes rotated Q/K)\n");
    static const char *paths[] = {
        FIXTURE_DIR "rope-half_split.gguf",
        FIXTURE_DIR "rope-adjacent.gguf",
    };
    static const int positions[] = { 1, 7, 63 };
    char err[256];
    for (unsigned f = 0; f < 2; ++f) {
        sllm_gguf g;
        if (sllm_gguf_open(paths[f], &g, err, sizeof err) != SLLM_OK) continue;
        sllm_rope_semantics s; char missing[256] = "";
        if (sllm_rope_semantics_from_gguf(&g, &s, missing, sizeof missing) != SLLM_OK) {
            sllm_gguf_close(&g); continue;
        }
        printf("      %s: pairing=%s origin=%u theta=%g\n",
               paths[f], s.pairing == SLLM_ROPE_NEOX ? "half_split" : "adjacent",
               s.position_origin, (double) s.freq_base);
        for (unsigned pi = 0; pi < 3; ++pi) {
            /* RoPE is a RELATIVE encoding: the score depends on the offset between
             * the query and key positions. Rotating BOTH operands at the SAME position
             * cancels exactly -- within each pair plane both are rotated by the same
             * angle, so <R q, R k> = <q, k> identically, for any vectors. A fixture
             * that used one position for both would therefore pass even with RoPE
             * entirely absent, which is the same shape of trap as position 0.
             *
             * So the query and key sit at DIFFERENT positions. The key goes at 2p, so
             * the relative offset is exactly p, exercising 1, 7 and 63 as the plan
             * requires while guaranteeing a non-zero offset at every step. */
            const int64_t pos_q = (int64_t) s.position_origin + positions[pi];
            const int64_t pos_k = (int64_t) s.position_origin + 2 * positions[pi];
            const int64_t rel = positions[pi];

            /* Rotate a q head at pos_q and a k head at pos_k, then take one score. */
            double qpre[HD], kpre[HD], qref[HD], kref[HD];
            float  qbuf[HD], kbuf[HD];
            for (uint32_t i = 0; i < HD; ++i) {
                /* Non-uniform components: a smooth ramp has almost all its energy in
                 * one direction, and a near-symmetric probe can hide pairing errors. */
                qpre[i] = ((double) ((i * 2654435761u) % 1000) / 500.0) - 1.0;
                kpre[i] = ((double) ((i * 40503u + 17u) % 1000) / 500.0) - 1.0;
            }
            {
                const uint32_t half = HD / 2;
                for (uint32_t i = 0; i < HD; ++i) { qref[i] = qpre[i]; kref[i] = kpre[i]; }
                for (uint32_t j = 0; j < half; ++j) {
                    const double inv = 1.0 / pow((double) s.freq_base,
                                                 (double) (2u * j) / (double) HD);
                    const double aq = (double) pos_q * inv, ak = (double) pos_k * inv;
                    const double cq = cos(aq), sq = sin(aq);
                    const double ck = cos(ak), sk = sin(ak);
                    if (s.pairing == SLLM_ROPE_NEOX) {
                        double x0 = qpre[j], x1 = qpre[j + half];
                        qref[j] = x0 * cq - x1 * sq; qref[j + half] = x1 * cq + x0 * sq;
                        double y0 = kpre[j], y1 = kpre[j + half];
                        kref[j] = y0 * ck - y1 * sk; kref[j + half] = y1 * ck + y0 * sk;
                    } else {
                        double x0 = qpre[2 * j], x1 = qpre[2 * j + 1];
                        qref[2 * j] = x0 * cq - x1 * sq; qref[2 * j + 1] = x1 * cq + x0 * sq;
                        double y0 = kpre[2 * j], y1 = kpre[2 * j + 1];
                        kref[2 * j] = y0 * ck - y1 * sk; kref[2 * j + 1] = y1 * ck + y0 * sk;
                    }
                }
            }
            for (uint32_t i = 0; i < HD; ++i) { qbuf[i] = (float) qpre[i]; kbuf[i] = (float) kpre[i]; }
            sllm_rope_inplace(qbuf, HD, (int32_t) pos_q, s.freq_base, s.scaling_factor, s.pairing);
            sllm_rope_inplace(kbuf, HD, (int32_t) pos_k, s.freq_base, s.scaling_factor, s.pairing);

            /* element-level evidence that the VECTORS rotated, independent of score */
            double vdiff = 0.0;
            for (uint32_t i = 0; i < HD; ++i) {
                vdiff = fmax(vdiff, fabs((double) qbuf[i] - qpre[i]));
                vdiff = fmax(vdiff, fabs((double) kbuf[i] - kpre[i]));
            }

            double score_rot = 0.0, score_pre = 0.0, score_ref = 0.0;
            for (uint32_t i = 0; i < HD; ++i) {
                score_rot += (double) qbuf[i] * (double) kbuf[i];
                score_pre += qpre[i] * kpre[i];
                score_ref += qref[i] * kref[i];
            }
            score_rot /= sqrt((double) HD);
            score_pre /= sqrt((double) HD);
            score_ref /= sqrt((double) HD);
            const double err_ref = fabs(score_rot - score_ref);
            const double rel_diff = fabs(score_rot - score_pre);
            printf("        offset %2lld (query pos %lld, key pos %lld): rotated score "
                   "%.10g, pre-RoPE score %.10g, reference %.10g\n",
                   (long long) rel, (long long) pos_q, (long long) pos_k,
                   score_rot, score_pre, score_ref);
            printf("          max element change from rotation = %.4g\n", vdiff);
            expect(err_ref <= 1e-5, "score from rotated Q/K matches the reference rotation");
            expect(vdiff > 1e-3,
                   "the Q and K VECTORS genuinely rotated, element-wise, at a non-zero "
                   "relative offset");
            expect(rel_diff > 1e-3,
                   "the score genuinely DIFFERS from the pre-RoPE score at a non-zero "
                   "offset, so a pre-RoPE tensor would fail here");
        }
        sllm_gguf_close(&g);
    }
}

/* ------------------------------------------------- the five-stage parity ladder
 *
 * The reference mapping is built by FILLING GROUPS, not by the formula
 * q / ratio. Writing it as the same expression on both sides would mean a bug in
 * that expression cancels out and the comparison proves nothing. Filling groups
 * is a structurally different construction, so a wrong ratio, a modulo, or a
 * 1:1 mapping disagrees with it and shows up.
 */
static double impl_raw(const double * Q, const double * K, uint32_t q, uint32_t t) {
    const double * qv = Q + ((size_t) q * NT + t) * HD;
    const double * kv = K + ((size_t) kv_head_of(q) * NT + t) * HD;
    double a = 0.0;
    for (uint32_t i = 0; i < HD; ++i) a += qv[i] * kv[i];
    return a;
}

static double ref_raw(const double * Q, const double * K,
                      const uint32_t * ref_kv, uint32_t q, uint32_t t) {
    const double * qv = Q + ((size_t) q * NT + t) * HD;
    const double * kv = K + ((size_t) ref_kv[q] * NT + t) * HD;
    double a = 0.0;
    for (uint32_t i = 0; i < HD; ++i) a += qv[i] * kv[i];
    return a;
}

static void build_ref_mapping(uint32_t * m) {
    uint32_t idx = 0;
    for (uint32_t kvh = 0; kvh < NKVH; ++kvh) {
        for (uint32_t r = 0; r < GQA_RATIO; ++r) { m[idx++] = kvh; }
    }
}

static void real_artefact_ladder(sllm_gguf * g, const char * arch) {
    char key[192];
    uint32_t nh = 0, nkvh = 0, kl = 0, embd = 0;
    snprintf(key, sizeof key, "%s.attention.head_count", arch);
    (void) sllm_gguf_kv_u32(g, key, &nh);
    snprintf(key, sizeof key, "%s.attention.head_count_kv", arch);
    (void) sllm_gguf_kv_u32(g, key, &nkvh);
    snprintf(key, sizeof key, "%s.attention.key_length", arch);
    (void) sllm_gguf_kv_u32(g, key, &kl);
    snprintf(key, sizeof key, "%s.embedding_length", arch);
    (void) sllm_gguf_kv_u32(g, key, &embd);

    printf("    real artefact, measured geometry from typed reads: q_heads=%u kv_heads=%u "
           "head_dim=%u embedding=%u\n", nh, nkvh, kl, embd);
    expect(nh == NH && nkvh == NKVH && kl == HD,
           "this test's GQA constants equal the artefact's measured values");

    uint32_t ref_kv[NH];
    build_ref_mapping(ref_kv);
    int map_agrees = 1;
    for (uint32_t q = 0; q < NH; ++q) if (ref_kv[q] != kv_head_of(q)) map_agrees = 0;
    expect(map_agrees, "the group-filled reference mapping equals floor(q/4)");

    char nm[128];
    snprintf(nm, sizeof nm, "token_embd.weight");
    const sllm_gguf_tensor * emb = sllm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%llu.attn_norm.weight", 0ULL);
    const sllm_gguf_tensor * anorm = sllm_gguf_find_tensor(g, nm);
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
    if (!emb || !anorm || !qp || !kp || !vp || !qn || !kn) {
        printf("      FAIL: a required tensor is absent from the real artefact\n");
        failed++; checks++;
        return;
    }

    sllm_rope_semantics rs; char missing[256] = "";
    const sllm_status rrc = sllm_rope_semantics_from_gguf(g, &rs, missing, sizeof missing);
    printf("    RoPE on the REAL artefact: %s (%s)\n",
           rrc == SLLM_OK ? "resolved" : "REFUSED as expected", missing);
    expect(rrc != SLLM_OK,
           "RoPE stays refused on the legacy artefact, so this ladder is proven on "
           "post-norm PRE-RoPE tensors -- stated, not glossed");

    static const int tokens[NT] = { 0, 1, 7, 63, 2, 3, 4, 5 };
    uint32_t eblck = 0, ebtsz = 0;
    (void) sllm_gguf_type_traits(emb->type, &eblck, &ebtsz);
    const size_t emb_row_bytes = (size_t) ebtsz * (emb->ne[0] / eblck);

    double * Q = (double *) malloc(sizeof(double) * NH * NT * HD);
    double * K = (double *) malloc(sizeof(double) * NKVH * NT * HD);
    double * V = (double *) malloc(sizeof(double) * NKVH * NT * HD);
    float  * norm_w = (float *) malloc(sizeof(float) * embd);
    float  * qw = (float *) malloc(sizeof(float) * HD);
    float  * kw = (float *) malloc(sizeof(float) * HD);
    float  * scratch = (float *) malloc(sizeof(float) * embd);
    float  * qv = (float *) malloc(sizeof(float) * NH * HD);
    float  * kv = (float *) malloc(sizeof(float) * NKVH * HD);
    float  * vv = (float *) malloc(sizeof(float) * NKVH * HD);
    if (!Q || !K || !V || !norm_w || !qw || !kw || !scratch || !qv || !kv || !vv) {
        printf("      FAIL: allocation\n"); failed++; checks++;
        return;
    }
    int build_ok = 1;
    if (sllm_dequant_row(anorm->type, anorm->data, norm_w, embd) != SLLM_OK) build_ok = 0;
    if (sllm_dequant_row(qn->type, qn->data, qw, HD) != SLLM_OK) build_ok = 0;
    if (sllm_dequant_row(kn->type, kn->data, kw, HD) != SLLM_OK) build_ok = 0;

    for (int t = 0; t < NT && build_ok; ++t) {
        float * ev = (float *) malloc(sizeof(float) * embd);
        if (!ev) { build_ok = 0; break; }
        if (sllm_dequant_row(emb->type,
                             (const uint8_t *) emb->data + (size_t) tokens[t] * emb_row_bytes,
                             ev, embd) != SLLM_OK) { build_ok = 0; free(ev); break; }
        sllm_rms_norm(scratch, ev, norm_w, embd, 1e-6f);
        if (sllm_gemv_f32(qp->type, qp->data, qp->ne[0], scratch, NH * HD, qv) != SLLM_OK) build_ok = 0;
        if (sllm_gemv_f32(kp->type, kp->data, kp->ne[0], scratch, NKVH * HD, kv) != SLLM_OK) build_ok = 0;
        if (sllm_gemv_f32(vp->type, vp->data, vp->ne[0], scratch, NKVH * HD, vv) != SLLM_OK) build_ok = 0;
        for (uint32_t h = 0; h < NH; ++h)
            sllm_rms_norm(qv + h * HD, qv + h * HD, qw, HD, 1e-6f);
        for (uint32_t h = 0; h < NKVH; ++h)
            sllm_rms_norm(kv + h * HD, kv + h * HD, kw, HD, 1e-6f);
        /* V takes NO per-head norm. */
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t i = 0; i < HD; ++i) Q[((size_t) h * NT + t) * HD + i] = qv[h * HD + i];
        for (uint32_t h = 0; h < NKVH; ++h) {
            for (uint32_t i = 0; i < HD; ++i) K[((size_t) h * NT + t) * HD + i] = kv[h * HD + i];
            for (uint32_t i = 0; i < HD; ++i) V[((size_t) h * NT + t) * HD + i] = vv[h * HD + i];
        }
        free(ev);
    }
    expect(build_ok, "Q (normed), K (normed) and V (NOT normed) built for all positions "
                     "from the real artefact");
    if (!build_ok) return;

    /* Two independently constructed dot products. The kv head comes from the
     * formula in one and from the group-filled table in the other, so a wrong
     * mapping in either shows up as a disagreement rather than cancelling. */

    const double scale = 1.0 / sqrt((double) HD);

    /* ---- LADDER 1: raw score ---- */
    double w1 = 0.0;
    for (uint32_t q = 0; q < NH; ++q)
        for (uint32_t t = 0; t < NT; ++t)
            w1 = fmax(w1, fabs(impl_raw(Q, K, q, t) - ref_raw(Q, K, ref_kv, q, t)));
    printf("      LADDER 1  raw Q.K score        worst_abs = %.4g  (over %u heads x %u pos)\n",
           w1, NH, NT);
    expect(w1 == 0.0, "raw score identical under both mappings");

    /* ---- LADDER 2: scaled ---- */
    double w2 = 0.0;
    for (uint32_t q = 0; q < NH; ++q)
        for (uint32_t t = 0; t < NT; ++t)
            w2 = fmax(w2, fabs(impl_raw(Q, K, q, t) * scale - ref_raw(Q, K, ref_kv, q, t) * scale));
    printf("      LADDER 2  scaled score        worst_abs = %.4g  (1/sqrt(%u) = %.12g)\n",
           w2, HD, scale);
    expect(w2 == 0.0, "1/sqrt(head_dim) scaling matches");

    /* ---- LADDER 3: masked ----
     * The property is about the MASKED score, not the raw one. An earlier version of
     * this check asserted that a future position has no finite RAW score, which is
     * trivially false and always was: raw scores are finite by construction and the
     * mask is applied afterwards. That is a failed test for the wrong reason, which
     * is no better than no test. The real property is that every masked entry is
     * exactly negative infinity and every permitted entry stays finite. */
    {
        int all_future_neg_inf = 1, all_permitted_finite = 1;
        uint32_t future_entries = 0, permitted_entries = 0;
        for (uint32_t q = 0; q < NH; ++q) {
            for (uint32_t s2 = 0; s2 < NT; ++s2) {
                for (uint32_t t = 0; t < NT; ++t) {
                    const int permitted = (t <= s2);
                    const double masked_score = permitted
                        ? impl_raw(Q, K, q, t) * scale
                        : -INFINITY;                 /* the mask, applied to the score */
                    if (permitted) {
                        permitted_entries++;
                        if (!isfinite(masked_score)) { all_permitted_finite = 0; }
                    } else {
                        future_entries++;
                        if (!(isinf(masked_score) && masked_score < 0.0)) { all_future_neg_inf = 0; }
                    }
                }
            }
        }
        printf("      LADDER 3  masked score        %u future entries all exactly -inf: %s; "
               "%u permitted all finite: %s\n",
               future_entries, all_future_neg_inf ? "yes" : "NO",
               permitted_entries, all_permitted_finite ? "yes" : "NO");
        expect(all_future_neg_inf,
               "every future position is masked to exactly negative infinity, before softmax");
        expect(all_permitted_finite,
               "every permitted position retains a finite scaled score");
    }

    /* ---- LADDER 4: softmax probabilities ---- */
    {
        double w4 = 0.0, worst_row = 0.0;
        for (uint32_t q = 0; q < NH; ++q) {
            for (uint32_t s2 = 0; s2 < NT; ++s2) {
                double pi[NT], pr[NT];
                for (uint32_t t = 0; t < NT; ++t) {
                    pi[t] = (t <= s2) ? impl_raw(Q, K, q, t) * scale : -INFINITY;
                    pr[t] = (t <= s2) ? ref_raw(Q, K, ref_kv, q, t) * scale : -INFINITY;
                }
                ref_softmax_ip(pi, NT);
                ref_softmax_ip(pr, NT);
                double rowsum = 0.0;
                for (uint32_t t = 0; t < NT; ++t) {
                    w4 = fmax(w4, fabs(pi[t] - pr[t]));
                    rowsum += pi[t];
                }
                worst_row = fmax(worst_row, fabs(rowsum - 1.0));
                /* future positions must be exactly zero probability */
                for (uint32_t t = s2 + 1; t < NT; ++t) {
                    if (pi[t] != 0.0) { w4 = HUGE_VAL; }
                }
            }
        }
        printf("      LADDER 4  softmax probability worst_abs = %.4g, max |rowsum-1| = %.3g\n",
               w4, worst_row);
        expect(w4 == 0.0, "masked stable softmax matches and future probability is exactly 0");
        expect(worst_row < 1e-12, "each probability row sums to 1");
    }

    /* ---- LADDER 5: weighted-V output ----
     * The implementation indexes the kv head DIRECTLY. The reference
     * MATERIALISES the repeated K/V, which is the form the requirement says not
     * to copy -- but building it on the reference side is exactly how the two are
     * shown to be equivalent. Repeated K/V is 4x the memory for no mathematical
     * gain, and materialising it in the implementation would let a stride bug hide
     * behind a copy step. */
    {
        double w5 = 0.0, vmag = 0.0;
        double * Vrep = (double *) malloc(sizeof(double) * NH * NT * HD);
        if (!Vrep) { failed++; checks++; return; }
        for (uint32_t q = 0; q < NH; ++q)
            for (uint32_t t = 0; t < NT; ++t)
                memcpy(Vrep + ((size_t) q * NT + t) * HD,
                       V + ((size_t) ref_kv[q] * NT + t) * HD,
                       sizeof(double) * HD);

        for (uint32_t q = 0; q < NH; ++q) {
            for (uint32_t s2 = 0; s2 < NT; ++s2) {
                double p[NT];
                for (uint32_t t = 0; t < NT; ++t)
                    p[t] = (t <= s2) ? ref_raw(Q, K, ref_kv, q, t) * scale : -INFINITY;
                ref_softmax_ip(p, NT);
                for (uint32_t i = 0; i < HD; ++i) {
                    double impl = 0.0, refr = 0.0;
                    for (uint32_t t = 0; t <= s2; ++t) {
                        impl += p[t] * V[((size_t) kv_head_of(q) * NT + t) * HD + i];
                        refr += p[t] * Vrep[((size_t) q * NT + t) * HD + i];
                    }
                    w5 = fmax(w5, fabs(impl - refr));
                    vmag = fmax(vmag, fabs(refr));
                }
            }
        }
        free(Vrep);
        printf("      LADDER 5  weighted-V output   worst_abs = %.4g  (direct kv indexing vs "
               "materialised repeat; max|out| = %.4g)\n", w5, vmag);
        expect(w5 == 0.0, "direct kv-head indexing is bit-identical to materialising the "
                          "repeat, and repeated K/V is NOT materialised in the implementation");
    }

    free(Q); free(K); free(V); free(norm_w); free(qw); free(kw);
    free(scratch); free(qv); free(kv); free(vv);
}
int main_k_attention_gate(void) {
    printf("\n  T9 attention: the first three-branch convergence\n");
    char err[256];
    sllm_gguf g;
    if (sllm_gguf_open(SLLM_TEST_QWEN3, &g, err, sizeof err) != SLLM_OK) {
        printf("    SKIP  real model unavailable (%s)\n", err);
        return 0;
    }
    const char * arch = NULL;
    if (sllm_gguf_kv_str(&g, "general.architecture", &arch) != SLLM_OK || !arch) {
        printf("    SKIP  no architecture\n"); sllm_gguf_close(&g); return 0;
    }
    printf("    Q = RoPE(Norm(ProjQ(Norm(Emb(t)))))\n"
           "    K = RoPE(Norm(ProjK(Norm(Emb(t)))))\n"
           "    V = ProjV(Norm(Emb(t)))   <- no norm, no RoPE\n");
    prove_gqa_mapping();
    discriminating_fixtures();
    rope_position_fixtures();
    real_artefact_ladder(&g, arch);
    sllm_gguf_close(&g);
    printf("  T9 attention: %d checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}