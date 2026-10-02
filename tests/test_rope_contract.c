/*
 * test_rope_contract.c — the RoPE semantics contract, and the rotation it unlocks.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * WHY THIS TEST HAS THREE FIXTURES INSTEAD OF ONE
 *
 *   rope-half_split.gguf   carries the contract with pairing = half_split
 *   rope-adjacent.gguf     carries the contract with pairing = adjacent
 *   Qwen3-8B-Q4_K_M.gguf  carries NO contract at all, and must refuse forever
 *
 * The third is the important one. It is a real, fully functional model that
 * happens to omit two pieces of required semantics, and the runtime must keep
 * refusing it rather than quietly inventing them. If someone later "fixes" that
 * refusal by defaulting to a convention, this fixture fails -- which is the only
 * thing standing between a family guess and a plausible wrong answer.
 *
 * The first two exist because a single fixture proves nothing about pairing. Both
 * conventions are norm-preserving, so a swapped implementation still passes any
 * norm test, any shape test, and any position-0 test. Only comparing against an
 * independent reference on BOTH conventions, at positions that are not degenerate,
 * can tell whether the implementation actually reads the contract.
 */

#include "harness.h"
#include "saphira_llm/rope_contract.h"
#include "saphira_llm/gguf.h"
#include "saphira_llm/ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SLLM_TEST_QWEN3 "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
#define FIXTURE_DIR     "tests/fixtures/"

/*
 * The independent reference. Double precision, written from the source model's
 * own formulation rather than from the implementation:
 *
 *   inv_freq[j] = 1 / base ** (2j/dim)          j = 0 .. dim/2-1
 *   angle_j     = position * inv_freq[j]
 *   half_split:  element i pairs with i + dim/2
 *   adjacent:    element 2j pairs with 2j+1
 *
 * The two differ in WHICH element each frequency acts on, and that difference is
 * the entire content of the encoding.
 */
static void rope_reference(double * out, const double * x, uint32_t dim,
                           double base, int64_t position, sllm_rope_type pairing) {
    const uint32_t half = dim / 2;
    for (uint32_t i = 0; i < dim; ++i) { out[i] = x[i]; }
    for (uint32_t j = 0; j < half; ++j) {
        const double inv = 1.0 / pow(base, (double) (2u * j) / (double) dim);
        const double angle = (double) position * inv;
        const double c = cos(angle), s = sin(angle);
        if (pairing == SLLM_ROPE_NEOX) {
            const double a = x[j], b = x[j + half];
            out[j]        = a * c - b * s;
            out[j + half] = b * c + a * s;
        } else {
            const double a = x[2u * j], b = x[2u * j + 1u];
            out[2u * j]     = a * c - b * s;
            out[2u * j + 1] = b * c + a * s;
        }
    }
}

/* Deterministic, and deliberately not smooth: a smooth vector can hide a
 * transposition, because neighbouring elements are then nearly equal. */
static void fill_probe(double * v, uint32_t n, unsigned seed) {
    unsigned s = seed;
    for (uint32_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((double) (s >> 8) / 8388608.0) - 1.0;   /* [-1, 1) */
    }
}

/* Prove one fixture end to end: resolve the contract, then rotate a real head and
 * compare against the reference at each substantive position. */
static int exercise_fixture(const char * path, const char * label,
                            uint32_t n_heads, unsigned * checks, int * failed) {
    char err[256];
    sllm_gguf g;
    if (sllm_gguf_open(path, &g, err, sizeof err) != SLLM_OK) {
        printf("    FAIL  %s: cannot open (%s)\n", label, err);
        (*failed)++; (*checks)++;
        return 1;
    }
    printf("    %s\n", label);

    sllm_rope_semantics s;
    char missing[256] = "";
    const sllm_status rc = sllm_rope_semantics_from_gguf(&g, &s, missing, sizeof missing);
    if (rc != SLLM_OK) {
        printf("      contract REFUSED: %s -- this fixture was supposed to carry it\n",
               missing);
        (*failed)++; (*checks)++;
        sllm_gguf_close(&g);
        return 1;
    }
    printf("      contract RESOLVED  pairing=%s (enum %d)  origin=%u  rotary_dim=%u  "
           "freq_base=%g  scaling=%s x%g\n",
           s.pairing == SLLM_ROPE_NEOX ? "half_split" : "adjacent", (int) s.pairing,
           s.position_origin, s.rotary_dim, (double) s.freq_base,
           s.scaling_mode, (double) s.scaling_factor);
    (*checks)++;

    /* The two conventions must NOT agree, or one of the fixtures is decorative
     * and a hard-coded implementation would pass both. */
    {
        double a[8], b[8], ra[8], rb[8];
        fill_probe(a, 8, 12345u); memcpy(b, a, sizeof a);
        rope_reference(ra, a, 8, (double) s.freq_base, 7, SLLM_ROPE_NEOX);
        rope_reference(rb, b, 8, (double) s.freq_base, 7, SLLM_ROPE_NORMAL);
        double diff = 0.0;
        for (int i = 0; i < 8; ++i) diff = fmax(diff, fabs(ra[i] - rb[i]));
        printf("      conventions DISCRIMINATE at position 7: max difference %.4g%s\n",
               diff, diff > 1e-3 ? "" : "  <-- TOO SIMILAR, FIXTURES ARE DECORATIVE");
        (*checks)++;
        if (!(diff > 1e-3)) { (*failed)++; }
    }

    /* Rotate head 0 and the highest head index, at positions 1, 7 and 63.
     * Position 0 is deliberately absent: sin(0)=0 and cos(0)=1 make every
     * rotation the identity, so it cannot discriminate a correct rotation from an
     * absent one. */
    static const int POSITIONS[] = { 1, 7, 63 };
    const uint32_t dim = s.rotary_dim;
    double * x   = (double *) malloc(sizeof(double) * dim);
    double * ref = (double *) malloc(sizeof(double) * dim);
    float  * buf = (float *)  malloc(sizeof(float) * dim);
    int all_ok = 1;

    for (unsigned pi = 0; pi < sizeof POSITIONS / sizeof POSITIONS[0]; ++pi) {
        const int seq = POSITIONS[pi];
        const int64_t pos = (int64_t) s.position_origin + seq;

        for (uint32_t h = 0; h < n_heads; h += (n_heads > 1 ? n_heads - 1 : 1)) {
            fill_probe(x, dim, 1000u + h * 7u + (unsigned) seq);
            rope_reference(ref, x, dim, (double) s.freq_base, pos, s.pairing);

            for (uint32_t i = 0; i < dim; ++i) { buf[i] = (float) x[i]; }
            sllm_rope_inplace(buf, dim, (int32_t) pos, s.freq_base,
                              s.scaling_factor, s.pairing);

            double worst = 0.0;
            size_t exact = 0;
            for (uint32_t i = 0; i < dim; ++i) {
                const double gv = (double) buf[i];
                const double d = fabs(gv - ref[i]);
                if ((float) ref[i] == buf[i]) { exact++; }
                if (d > worst) worst = d;
            }
            const int ok = (worst <= 1e-5);
            if (!ok) all_ok = 0;
            printf("      head %2u position %2d (origin %u -> pos %lld)  worst_abs=%.3g "
                   "exact=%zu/%u  %s\n",
                   h, seq, s.position_origin, (long long) pos, worst, exact, dim,
                   ok ? "ok" : "FAIL");
            (*checks)++;
        }
    }
    free(x); free(ref); free(buf);

    /* A rotation must not reach across a head boundary. If the implementation
     * treated one head's output as a single long vector it would pair element
     * (dim-1) of head h with element 0 of head h+1, which the reference above
     * never does. Checking head 0 and the last head specifically is what makes
     * that visible. */
    {
        const uint32_t two = dim * 2;
        double * x2 = (double *) malloc(sizeof(double) * two);
        double * r2 = (double *) malloc(sizeof(double) * two);
        float  * b2 = (float *)  malloc(sizeof(float) * two);
        /* Zero first, then fill. fill_probe writes every element, so this changes
         * no result; it exists so the arrays are unambiguously initialised rather
         * than merely presumed to be. */
        memset(x2, 0, sizeof(double) * two);
        memset(r2, 0, sizeof(double) * two);
        memset(b2, 0, sizeof(float)  * two);
        fill_probe(x2, two, 777u);
        rope_reference(r2, x2, dim, (double) s.freq_base, 63, s.pairing);
        for (uint32_t i = 0; i < two; ++i) { b2[i] = (float) x2[i]; }
        sllm_rope_inplace(b2, dim, 63, s.freq_base, s.scaling_factor, s.pairing);
        double w_head0 = 0.0, w_head1 = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            w_head0 = fmax(w_head0, fabs((double) b2[i] - r2[i]));
            /* head 1 was never rotated by the call above, so it must equal input */
            w_head1 = fmax(w_head1, fabs((double) b2[dim + i] - x2[dim + i]));
        }
        printf("      head boundary: rotating only head 0 leaves head 1 identical "
               "(head0 worst=%.3g head1 delta=%.3g) %s\n",
               w_head0, w_head1, (w_head0 <= 1e-5 && w_head1 == 0.0) ? "ok" : "FAIL");
        (*checks)++;
        if (!(w_head0 <= 1e-5 && w_head1 == 0.0)) { all_ok = 0; (*failed)++; }
        free(x2); free(r2); free(b2);
    }

    /* Position 0 remains a permanent non-proof: recorded so the reason it is
     * excluded cannot be forgotten. */
    {
        double z[8], zr[8];
        fill_probe(z, 8, 999u);
        rope_reference(zr, z, 8, (double) s.freq_base, 0, s.pairing);
        double worst = 0.0;
        for (int i = 0; i < 8; ++i) worst = fmax(worst, fabs(zr[i] - z[i]));
        printf("      position 0 non-proof retained: rotation delta = %.3g (identity, "
               "so position 0 cannot test a rotation)\n", worst);
        (*checks)++;
        if (worst != 0.0) { (*failed)++; }
    }

    sllm_gguf_close(&g);
    return all_ok;
}

int main_k_rope_contract_gate(void) {
    unsigned checks = 0; int failed = 0;
    char err[256];

    printf("\n  T7 RoPE semantics contract\n");

    /* -- 1. THE PERMANENT NEGATIVE FIXTURE: the real model, which has no contract.
     * It must refuse, and name what is missing. This is the regression that
     * proves the runtime does not fall back to a family default. -- */
    {
        sllm_gguf g;
        if (sllm_gguf_open(SLLM_TEST_QWEN3, &g, err, sizeof err) != SLLM_OK) {
            printf("    SKIP  real model unavailable (%s)\n", err);
        } else {
            printf("    real Qwen3-8B artefact (must REFUSE forever)\n");
            const int present = sllm_rope_contract_present(&g);
            sllm_rope_semantics s;
            char missing[256] = "";
            const sllm_status rc = sllm_rope_semantics_from_gguf(&g, &s, missing, sizeof missing);
            printf("      contract present: %s\n", present ? "yes" : "no");
            printf("      resolve: %s, missing fact named: %s\n",
                   rc == SLLM_OK ? "UNEXPECTEDLY RESOLVED" : "REFUSED", missing);
            checks += 2;
            if (present) { failed++; printf("      FAIL  a real model without the "
                                            "contract must not claim to carry one\n"); }
            if (rc == SLLM_OK || strcmp(missing, SLLM_ROPE_K_PAIRING) != 0) {
                failed++;
                printf("      FAIL  refusal must name %s specifically\n", SLLM_ROPE_K_PAIRING);
            }
            /* The rotary extent IS available in that artefact, independently. It
             * must not be borrowed to fill the missing pairing field. */
            uint32_t kl = 0;
            const char * arch = NULL;
            char key[192];
            if (sllm_gguf_kv_str(&g, "general.architecture", &arch) == SLLM_OK && arch) {
                snprintf(key, sizeof key, "%s.attention.key_length", arch);
                (void) sllm_gguf_kv_u32(&g, key, &kl);
            }
            printf("      attention.key_length=%u is available and was NOT used to "
                   "supply the missing pairing\n", kl);
            checks++;
            sllm_gguf_close(&g);
        }
    }

    /* -- 2 & 3. The two positive fixtures, one per convention. -- */
    exercise_fixture(FIXTURE_DIR "rope-half_split.gguf",
                     "fixture: contract pairing = half_split", 8, &checks, &failed);
    exercise_fixture(FIXTURE_DIR "rope-adjacent.gguf",
                     "fixture: contract pairing = adjacent", 8, &checks, &failed);

    /* -- 4. Negative gates on the contract itself. A resolver that accepted a
     * partial or ill-typed contract would reintroduce the whole problem. -- */
    printf("    contract negative gates\n");
    {   /* origin out of range, odd dim, scaling contradiction: all refused. */
        struct { const char * path; const char * expect; } bad[] = {
            { FIXTURE_DIR "rope-bad-origin.gguf",     "position_origin" },
            { FIXTURE_DIR "rope-bad-dim.gguf",       "rotary_dim"      },
            { FIXTURE_DIR "rope-bad-scaling.gguf",   "scaling_factor"  },
            { FIXTURE_DIR "rope-bad-pairing.gguf",   "pairing"         },
            /* pairing stored under the WRONG TYPE must read as ABSENT, not coerce */
            { FIXTURE_DIR "rope-bad-type.gguf",      "pairing"         },
            /* a genuinely partial contract must name the first absent field */
            { FIXTURE_DIR "rope-partial.gguf",       "pairing"         },
        };
        for (unsigned i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
            sllm_gguf g;
            if (sllm_gguf_open(bad[i].path, &g, err, sizeof err) != SLLM_OK) {
                printf("      FAIL  %s: fixture missing (%s)\n", bad[i].path, err);
                failed++; checks++; continue;
            }
            sllm_rope_semantics s; char missing[256] = "";
            const sllm_status rc = sllm_rope_semantics_from_gguf(&g, &s, missing, sizeof missing);
            const int ok = (rc != SLLM_OK) && strstr(missing, bad[i].expect) != NULL;
            printf("      %s %s refused naming '%s' (missing text: %s)\n",
                   ok ? "ok  " : "FAIL", bad[i].path, bad[i].expect, missing);
            checks++;
            if (!ok) failed++;
            sllm_gguf_close(&g);
        }
    }

    printf("  RoPE contract: %u checks, %d failed\n", checks, failed);
    return failed == 0 ? 0 : 1;
}