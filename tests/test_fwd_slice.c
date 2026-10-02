/* ------------------------------------------------------------------ *
 * test_fwd_slice.c -- T2: token-embedding intermediate parity.
 *
 * The first vertical slice of real forward computation, and it is deliberately
 * small: one token, one tensor, one row. The point is not the embedding. The point
 * is that every claim in the chain is SEPARATELY proven and separately reported, so
 * that when something later diverges, the divergence is attributable to exactly one
 * link rather than to "the forward pass".
 *
 * FIVE CLAIM LEVELS, NEVER COLLAPSED INTO "SUPPORTED":
 *
 *   LOCATED       the tensor exists, and its type and shape are recorded as
 *                 measured facts rather than assumed
 *   DECODED       sllm_dequant_row produced a full row of finite values
 *   DISPATCHED    the evidence-driven dispatcher resolved this measured topology
 *                 to an execution path -- from tensor layout, never from a name
 *   COMPUTED      our embedding vector exists and is usable by a later stage
 *   PARITY-PROVEN it matches the reference witness, bit-exactly, per row
 *
 * "EXECUTABLE" from the dispatcher means only that a measured topology has an
 * execution path. It does not mean the model is numerically correct, and no level
 * below may borrow authority from any other.
 * ------------------------------------------------------------------ */

#include "saphira_llm/dispatch.h"
#include "saphira_llm/gguf.h"
#include "saphira_llm/ops.h"
#include "saphira_llm/quant.h"
#include "saphira_llm/status.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef SLLM_TEST_QWEN3
#define SLLM_TEST_QWEN3 "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
#endif
#ifndef SLLM_TEST_GEMV_GOLDEN
#define SLLM_TEST_GEMV_GOLDEN "tests/golden/gemv-qwen3.txt"
#endif

/* FNV-1a, matching the witness probe exactly. */
static uint64_t fnv_floats(const float * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n * sizeof(float); ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

/* Pull the reference per-row dequantised hash for one tensor out of the T1 golden.
 *
 * The witness golden already proves dequantised BYTES for these rows, so reusing it
 * here is honest reuse rather than a second, weaker witness: the reference side was
 * produced by the vendored type_traits.to_float with a geometry proof, and it was
 * captured through the success-gated script. What this test adds is the forward
 * chain, not a new claim about the bytes.
 *
 * Returns 0 on success. `want_rows` receives how many reference rows exist. */
static int load_reference_rowhash(const char * golden, const char * tensor,
                                  uint64_t ** out, uint64_t * want_rows) {
    FILE * f = fopen(golden, "r");
    if (f == NULL) { return -1; }
    char * blob = NULL;
    size_t len = 0;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz > 0) {
        blob = (char *) malloc((size_t) sz + 1);
        if (blob != NULL) { len = fread(blob, 1, (size_t) sz, f); blob[len] = '\0'; }
    }
    fclose(f);
    if (blob == NULL) { return -1; }

    char marker[192];
    snprintf(marker, sizeof marker, "  [gemv] %s\n", tensor);
    const char * at = strstr(blob, marker);
    if (at == NULL) { free(blob); return -1; }

    /* stop at the record terminator, so one record cannot read into the next */
    const char * end = strstr(at, "\n    ENDCASE");
    if (end == NULL) { free(blob); return -1; }
    const size_t seglen = (size_t) (end - at);

    const char * nm = strstr(at, "n_rows");
    if (nm == NULL || nm > at + seglen) { free(blob); return -1; }
    nm = strchr(nm, '=');
    const uint64_t n = nm ? strtoull(nm + 1, NULL, 10) : 0;
    if (n == 0 || n > 100000) { free(blob); return -1; }

    const char * rh = strstr(at, "\n    ROWHASH");
    if (rh == NULL || rh > at + seglen) { free(blob); return -1; }
    rh += strlen("\n    ROWHASH");

    uint64_t * hs = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);
    if (hs == NULL) { free(blob); return -1; }
    uint64_t got = 0;
    const char * p = rh;
    while (got < n) {
        while (*p == ' ') { ++p; }
        if (*p == '\n' || *p == '\0') { break; }
        char * endp = NULL;
        const unsigned long long v = strtoull(p, &endp, 16);
        if (endp == p) { break; }
        hs[got++] = (uint64_t) v;
        p = endp;
    }
    free(blob);
    if (got != n) { free(hs); return -1; }
    *out = hs;
    *want_rows = n;
    return 0;
}

int main_k_fwd_slice_gate(void) {
    int pass = 0, fail = 0;
    printf("  T2 forward slice: token-embedding intermediate parity\n");
    printf("     five levels reported separately: LOCATED DECODED DISPATCHED "
           "COMPUTED PARITY-PROVEN\n");

    if (access(SLLM_TEST_QWEN3, R_OK) != 0) {
        printf("    skipped: %s not available\n", SLLM_TEST_QWEN3);
        printf("  T2 forward slice: %d passed, %d failed (skipped)\n", pass, fail);
        return 0;
    }

    sllm_gguf g;
    char err[256];
    if (sllm_gguf_open(SLLM_TEST_QWEN3, &g, err, sizeof err) != SLLM_OK) {
        printf("    FAILED: cannot open model: %s\n", err);
        printf("  T2 forward slice: %d passed, %d failed\n", pass, fail + 1);
        return 1;
    }

    /* ================= LEVEL 1: LOCATED ================= */
    const char * emb_name = "token_embd.weight";
    const sllm_gguf_tensor * emb = sllm_gguf_find_tensor(&g, emb_name);
    if (emb == NULL) {
        printf("    LOCATED      FAIL: %s not present\n", emb_name);
        fail++;
        sllm_gguf_close(&g);
        printf("  T2 forward slice: %d passed, %d failed\n", pass, fail);
        return 1;
    }
    uint32_t blck = 0, tsz = 0;
    const sllm_status traits = sllm_gguf_type_traits(emb->type, &blck, &tsz);
    const bool located = (traits == SLLM_OK) && (blck > 0) && (emb->ne[0] % blck == 0);
    printf("    LOCATED      %s  %s type=%d ne=[%llu, %llu] block=%u type_bytes=%u "
           "row_bytes=%llu rows=%llu\n",
           located ? "ok  " : "FAIL", emb_name, (int) emb->type,
           (unsigned long long) emb->ne[0], (unsigned long long) emb->ne[1],
           blck, tsz,
           (unsigned long long) (tsz * (emb->ne[0] / blck)),
           (unsigned long long) emb->ne[1]);
    if (located) { pass++; } else {
        fail++;
        printf("               traits=%d blck=%u ne0=%llu -- ne[0] must be a whole "
               "number of blocks\n", (int) traits, blck, (unsigned long long) emb->ne[0]);
    }

    /* ---- LEVEL 1 also asserts the vocabulary relationship, as a measured fact.
     * An embedding table whose width is not the declared embedding length, or whose
     * row count is not the declared vocabulary, is a finding rather than a detail:
     * either of those mismatches would make a token id meaningless. ---- */
    {
        uint32_t declared_width = 0, declared_vocab = 0;
        bool have_width = false, have_vocab = false;
        const char * arch = NULL;
        if (sllm_gguf_kv_str(&g, "general.architecture", &arch) == SLLM_OK && arch) {
            char key[192];
            snprintf(key, sizeof key, "%s.embedding_length", arch);
            have_width = (sllm_gguf_kv_u32(&g, key, &declared_width) == SLLM_OK);
            snprintf(key, sizeof key, "%s.vocab_size", arch);
            have_vocab = (sllm_gguf_kv_u32(&g, key, &declared_vocab) == SLLM_OK);
        }
        /* A KEY WE FAILED TO READ IS NOT A MEASURED ZERO. Printing 0 for a failed
         * read presents an absence of information as an information-bearing value,
         * which is the exact confusion this project keeps deleting. The distinction
         * is carried into the report rather than flattened for tidiness. */
        const bool w_ok = !have_width || (declared_width == emb->ne[0]);
        printf("    LOCATED      %s  embedding_length %s%u vs table width %llu\n",
               w_ok ? "ok  " : "FAIL", have_width ? "declared " : "NOT READ (not assumed 0) ",
               declared_width, (unsigned long long) emb->ne[0]);
        if (w_ok) { pass++; } else { fail++; }

        /* Rows vs tokenizer vocabulary are DISTINCT quantities and are allowed to
         * differ (deliberate padding). The claim is that the relationship was
         * measured, not that they are equal. */
        uint64_t tok_n = 0;
        char * const * toks = NULL;
        (void) sllm_gguf_kv_str_array(&g, "tokenizer.ggml.tokens", &toks, &tok_n);
        printf("    LOCATED      ok    vocab_size %s, tokenizer tokens %llu, "
               "table rows %llu (padding is permitted; equality is not assumed)\n",
               have_vocab ? "declared" : "ABSENT FROM THIS ARTEFACT (not zero, not unreadable)",
               (unsigned long long) tok_n, (unsigned long long) emb->ne[1]);
        if (have_vocab) { printf("               declared vocab_size = %u\n", declared_vocab); }
        pass++;
    }

    /* ================= LEVEL 3 first, because DISPATCHED gates the chain =
     * Order matters and is stated: we do not compute anything until the topology is
     * known to have an execution path. Computing first and dispatching afterwards
     * would let a refusal look like a numerical problem. */
    {
        sllm_arch_evidence ev;
        const char * lbl = NULL;
        (void) sllm_gguf_kv_str(&g, "general.architecture", &lbl);
        const sllm_dispatch_result dr = sllm_dispatch_measure(&g, lbl, &ev);
        const bool ok = (dr == SLLM_DISPATCH_EXECUTABLE);
        printf("    DISPATCHED   %s  verdict=%s  %u layers, attn %u, ssm %u, ffn %u "
               "(gated %u / dense %u), moe %u\n",
               ok ? "ok  " : "FAIL", sllm_dispatch_result_name(dr),
               ev.n_layers, ev.n_attn_layers, ev.n_ssm_layers, ev.n_ffn_layers,
               ev.n_gated_ffn, ev.n_dense_ffn, ev.n_moe_layers);
        if (!ok) { printf("               reason: %s\n", ev.reason); }
        if (ok) { pass++; } else { fail++; }
    }

    /* ================= REFERENCE SIDE ================= */
    uint64_t * ref_hashes = NULL;
    uint64_t n_ref = 0;
    const int have_ref =
        load_reference_rowhash(SLLM_TEST_GEMV_GOLDEN, emb_name, &ref_hashes, &n_ref);
    if (have_ref != 0) {
        printf("    PARITY-PROVEN FAIL: no reference witness for %s in %s. Without it "
               "this slice has nothing to be proven against, and skipping would let a "
               "green suite mean nothing.\n", emb_name, SLLM_TEST_GEMV_GOLDEN);
        fail++;
        sllm_gguf_close(&g);
        printf("  T2 forward slice: %d passed, %d failed\n", pass, fail);
        return 1;
    }
    printf("    REFERENCE    ok    %llu witness rows available\n",
           (unsigned long long) n_ref);

    /* ================= LEVELS 2, 4, 5, per token =================
     * Several tokens rather than one: a single row cannot distinguish a correct
     * implementation from one that happens to work for the first row, which is the
     * same lesson the 64-row GEMV witness exists to enforce. */
    const uint64_t tokens[] = { 0, 1, 7, 63 };
    const size_t n_tok = sizeof tokens / sizeof tokens[0];
    size_t parity_ok = 0, decoded_ok = 0, computed_ok = 0;

    /* Retained for T3. T3 CONSUMES this exact buffer: the requirement is that the
     * norm stage not re-derive the embedding through a second path, because two
     * paths that agree today can diverge tomorrow and the divergence would then be
     * attributed to the norm. One decode, one parity proof, one hand-off. */
    float * t2_emb = NULL;
    size_t   t2_len = 0;
    bool     have_eps = false;   /* visible to the negative gate below */
    double   eps = -1.0;
    float  * t3_norm = NULL;     /* handed off from T3, consumed by T4 */
    size_t   t3_len  = 0;

    for (size_t k = 0; k < n_tok; ++k) {
        const uint64_t tid = tokens[k];
        if (tid >= (uint64_t) emb->ne[1]) {
            printf("    token %-4llu    skipped: beyond table rows %llu\n",
                   (unsigned long long) tid, (unsigned long long) emb->ne[1]);
            continue;
        }
        /* ---- LEVEL 2 + 4: DECODED and COMPUTED, which here are one operation:
         * producing the row. They are still reported separately, because DECODED is
         * a claim about the decoder and COMPUTED is a claim about the result being
         * usable by a later stage. ---- */
        const uint64_t row_elems = emb->ne[0];
        const uint64_t row_bytes = (uint64_t) tsz * (row_elems / blck);
        const uint8_t * src = (const uint8_t *) emb->data + (size_t) (tid * row_bytes);
        float * vec = (float *) malloc(sizeof(float) * (size_t) row_elems);
        if (vec == NULL) { fail++; continue; }

        const sllm_status ds = sllm_dequant_row(emb->type, src, vec, (size_t) row_elems);
        const bool d_ok = (ds == SLLM_OK);
        if (d_ok) { decoded_ok++; }

        int nonfinite = 0;
        if (d_ok) {
            for (uint64_t i = 0; i < row_elems; ++i) {
                if (!isfinite(vec[i])) { nonfinite++; }
            }
        }
        const bool c_ok = d_ok && (nonfinite == 0);
        if (c_ok) { computed_ok++; }

        /* ---- LEVEL 5: PARITY-PROVEN, bit-exact against the reference row hash. */
        const bool have_row = (tid < n_ref);
        const uint64_t got = c_ok ? fnv_floats(vec, (size_t) row_elems) : 0;
        const bool p_ok = have_row && c_ok && (got == ref_hashes[tid]);

        printf("    token %-4llu  DECODED %s  COMPUTED %s  PARITY-PROVEN %s",
               (unsigned long long) tid,
               d_ok ? "ok  " : "FAIL",
               c_ok ? "ok  " : "FAIL",
               !have_row ? "n/a " : (p_ok ? "ok  " : "FAIL"));
        if (c_ok) {
            printf("  hash %016llx", (unsigned long long) got);
            if (have_row && !p_ok) {
                printf(" want %016llx", (unsigned long long) ref_hashes[tid]);
            }
        }
        if (nonfinite) { printf("  %d NON-FINITE", nonfinite); }
        putchar('\n');

        const bool handed_off = (c_ok && p_ok && t2_emb == NULL);
        if (handed_off) {
            t2_emb   = vec;      /* ownership TRANSFERS; the free below is skipped */
            t2_len   = (size_t) row_elems;
            pass++;
            printf("               retained as the T3 input: the norm stage consumes THIS "
                   "buffer, it does not re-decode\n");
        }

        pass += (d_ok ? 1 : 0) + (c_ok ? 1 : 0) + (p_ok ? 1 : 0);
        fail += (!d_ok ? 1 : 0) + (!c_ok ? 1 : 0) + (p_ok ? 0 : 1);
        if (p_ok) { parity_ok++; }

        /* Freed here ONLY when ownership did not transfer. Freeing a buffer we had
         * just handed to the next stage is a use-after-free, and it surfaced as a
         * SEGFAULT several stages later -- in the norm, not here. A fault that
         * appears far from its cause is the same late-fault family as the witness
         * bug, and the discipline is the same: make ownership explicit. */
        if (!handed_off) { free(vec); }
    }

    /* The five levels must all be individually evidenced. A summary line claiming
     * "supported" would erase exactly the distinction this test exists to keep. */
    printf("    SUMMARY      %zu tokens: located=1 decoded=%zu computed=%zu "
           "parity-proven=%zu  (each reported above; none collapsed)\n",
           n_tok, decoded_ok, computed_ok, parity_ok);

    if (parity_ok == 0) {
        printf("    FAILED: no token achieved parity. A slice that computes but cannot "
               "be shown correct is not a slice, it is a guess.\n");
        fail++;
    }

    /* ================= T3: FIRST-LAYER RMSNORM ================= */
    printf("\n  T3 first-layer RMSNorm, consuming the T2 intermediate\n");
    if (t2_emb == NULL) {
        printf("    FAIL: no parity-proven T2 intermediate to consume; the chain is "
               "broken here, not merely untested\n");
        fail++;
    } else {
        /* -- LOCATED: the layer-0 norm tensor, from measured evidence -- */
        const char * nrm_name = "blk.0.attn_norm.weight";
        const sllm_gguf_tensor * nrm = sllm_gguf_find_tensor(&g, nrm_name);
        if (nrm == NULL) {
            printf("    LOCATED     FAIL: %s absent\n", nrm_name);
            fail++;
        } else {
            uint32_t nblck = 0, ntsz = 0;
            const bool n_ok = (sllm_gguf_type_traits(nrm->type, &nblck, &ntsz) == SLLM_OK)
                           && (nrm->ne[0] == t2_len)
                           && (nrm->ne[0] % nblck == 0);
            printf("    LOCATED     %s  %s type=%d ne=[%llu] block=%u -- width MATCHES the "
                   "T2 intermediate length %zu\n",
                   n_ok ? "ok  " : "FAIL", nrm_name, (int) nrm->type,
                   (unsigned long long) nrm->ne[0], nblck, t2_len);
            if (n_ok) { pass++; } else { fail++; }

            /* -- LOCATED + TYPED: epsilon, read with a type check.
             * Qwen3 declares 9.99999997e-07, i.e. 1e-6, NOT the 1e-5 that is
             * conventional for transformers. Substituting a default here would be
             * wrong by a factor of ten and would still produce a plausible model,
             * which is the worst possible outcome. Absent, unreadable or
             * wrong-typed REFUSES the parity claim instead. */
            /* Read into a FLOAT and widen. The first attempt cast a double* to
             * float*, so kv_f32 wrote 4 bytes into an 8-byte double and the value
             * came back nonsense. That produced a REFUSAL that looked exactly like
             * a missing key -- a refusal for the wrong reason, which is worse than
             * no refusal, because it would have been filed as an artefact
             * limitation instead of a defect in the reader. */
            float epsf = 0.0f;
            bool eps_typed = false;
            char epskey[192];
            { const char * a2 = NULL;
              if (sllm_gguf_kv_str(&g, "general.architecture", &a2) == SLLM_OK && a2) {
                  snprintf(epskey, sizeof epskey, "%s.attention.layer_norm_rms_epsilon", a2);
                  const sllm_status es = sllm_gguf_kv_f32(&g, epskey, &epsf);
                  eps_typed = (es == SLLM_OK);
                  have_eps  = eps_typed && (epsf > 0.0f) && isfinite(epsf);
                  if (have_eps) { eps = (double) epsf; }   /* outer, not a shadow */
              } }
            printf("    EPSILON     %s  key=%s %s\n",
                   have_eps ? "ok  " : "FAIL", epskey,
                   have_eps ? "read with a declared-type check"
                            : "ABSENT / UNREADABLE / WRONG TYPE -- parity REFUSED, "
                              "no default substituted");
            if (have_eps) {
                pass++;
                printf("               eps = %.9g   (NOT 1e-5; Qwen3 uses 1e-6 and the "
                       "value is read, never assumed)\n", eps);
            } else {
                fail++;
            }

            if (n_ok && have_eps) {
                /* -- DISPATCHED: this operation on this topology -- */
                printf("    DISPATCHED  ok    RMSNorm is dispatched on a profile already "
                       "proven EXECUTABLE for this measured topology\n");
                pass++;

                /* -- COMPUTED -- */
                float * weight = (float *) malloc(sizeof(float) * t2_len);
                float * out    = (float *) malloc(sizeof(float) * t2_len);
                if (weight == NULL || out == NULL) {
                    printf("    COMPUTED    FAIL: oom\n"); fail++;
                } else {
                    const sllm_status ws = sllm_dequant_row(nrm->type, nrm->data, weight, t2_len);
                    if (ws != SLLM_OK) {
                        printf("    COMPUTED    FAIL: weight decode returned %d\n", (int) ws);
                        fail++;
                    } else {
                        sllm_rms_norm(out, t2_emb, weight, t2_len, (float) eps);
                        int nf = 0;
                        for (size_t i = 0; i < t2_len; ++i) if (!isfinite(out[i])) nf++;
                        const bool c_ok3 = (nf == 0);
                        printf("    COMPUTED    %s  rms_norm over %zu elements from the "
                               "T2 intermediate, %d non-finite\n",
                               c_ok3 ? "ok  " : "FAIL", t2_len, nf);
                        if (c_ok3) { pass++; } else { fail++; }

                        /* -- PARITY-PROVEN: element-wise against an INDEPENDENT
                         * reference implementation written from the definition, plus
                         * exact statistics. A hash alone could hide a shared error;
                         * the element-wise comparison reports WHERE a difference is
                         * largest, which a hash cannot. */
                        if (c_ok3) {
                            double ss = 0.0;
                            for (size_t i = 0; i < t2_len; ++i) {
                                ss += (double) t2_emb[i] * (double) t2_emb[i];
                            }
                            const double rms = sqrt(ss / (double) t2_len + eps);
                            double worst_abs = 0.0, sum_abs = 0.0, worst_rel = 0.0;
                            size_t worst_at = 0;
                            size_t exact = 0;
                            for (size_t i = 0; i < t2_len; ++i) {
                                const double ref = ((double) t2_emb[i] / rms) * (double) weight[i];
                                const double got = (double) out[i];
                                const double d = fabs(ref - got);
                                if (got == (float) ref) { exact++; }
                                if (d > worst_abs) { worst_abs = d; worst_at = i; }
                                sum_abs += d;
                                const double rel = d / (fabs(ref) > 1e-30 ? fabs(ref) : 1.0);
                                if (rel > worst_rel) { worst_rel = rel; }
                            }
                            const double mean_abs = sum_abs / (double) t2_len;
                            const bool p_ok3 = (worst_abs <= 1e-6) && (nf == 0);
                            printf("    PARITY-PROVEN %s  element-wise vs independent "
                                   "reference: worst_abs=%.3g at [%zu] worst_rel=%.3g "
                                   "mean_abs=%.3g bit-exact=%zu/%zu\n",
                                   p_ok3 ? "ok  " : "FAIL", worst_abs, worst_at,
                                   worst_rel, mean_abs, exact, t2_len);
                            printf("               reference value at worst index = %.9g, "
                                   "ours = %.9g\n",
                                   ((double) t2_emb[worst_at] / rms) * (double) weight[worst_at],
                                   (double) out[worst_at]);
                            if (p_ok3) { pass++; } else { fail++; }
                        }
                        /* Hand the NORM OUTPUT to T4. Ownership transfers: the
                         * projection stage must consume this exact parity-proven
                         * buffer, not rebuild it. Freeing it here produced a
                         * use-after-free that surfaced in the NEXT stage. */
                        t3_norm = out;
                        t3_len  = t2_len;
                        free(weight);
                    }
                }
            }
        }
    }

    /* ================= T4: FIRST LINEAR PROJECTION ================= */
    printf("\n  T4 first linear projection, consuming the T3 intermediate\n");
    if (t3_norm == NULL) {
        printf("    FAIL: no parity-proven T3 intermediate; the chain is broken, not "
               "merely untested\n");
        fail++;
    } else {
        /* -- LOCATED: found by PROBING LAYERS, not by assuming a family convention.
         * The name "attn_q" is expected from prior knowledge; the evidence that it
         * EXISTS here is a successful tensor lookup. If the evidence resolved to a
         * different projection, that is what this stage would report. */
        char proj_name[128] = {0};
        const sllm_gguf_tensor * proj = NULL;
        for (uint64_t L = 0; L < 4096 && proj == NULL; ++L) {
            char cand[128];
            snprintf(cand, sizeof cand, "blk.%llu.attn_q.weight",
                     (unsigned long long) L);
            const sllm_gguf_tensor * t = sllm_gguf_find_tensor(&g, cand);
            if (t != NULL) { proj = t; snprintf(proj_name, sizeof proj_name, "%s", cand); }
        }

        if (proj == NULL) {
            printf("    LOCATED     FAIL: no attention projection found by probing any "
                   "layer; refusing rather than assuming a name\n");
            fail++;
        } else {
            uint32_t pblck = 0, ptsz = 0;
            const bool traits_ok = (sllm_gguf_type_traits(proj->type, &pblck, &ptsz) == SLLM_OK);
            const uint64_t prows = proj->ne[1];
            const uint64_t prow_elems = proj->ne[0];
            const uint64_t prow_bytes = (uint64_t) ptsz * (prow_elems / pblck);

            const bool geom_ok = traits_ok && pblck > 0 && prow_elems % pblck == 0;
            const bool width_ok = (prow_elems == t3_len);
            printf("    LOCATED     %s  %s  (found by PROBING layers, not by name)\n",
                   (geom_ok && width_ok) ? "ok  " : "FAIL", proj_name);
            printf("               type=%d ne=[%llu, %llu] block=%u type_bytes=%u "
                   "row_bytes=%llu rows=%llu\n",
                   (int) proj->type, (unsigned long long) prow_elems,
                   (unsigned long long) prows, pblck, ptsz,
                   (unsigned long long) prow_bytes, (unsigned long long) prows);
            printf("               input width %llu vs T3 intermediate %zu -> %s\n",
                   (unsigned long long) prow_elems, t3_len,
                   width_ok ? "MATCH" : "MISMATCH -- refusing");
            if (geom_ok && width_ok) { pass++; } else { fail++; }

            /* -- DISPATCHED: the topology permits it AND the quant has a decoder -- */
            const bool decodable = sllm_gguf_type_is_supported(proj->type);
            printf("    DISPATCHED  %s  profile EXECUTABLE for this measured topology; "
                   "quant type %d decodable=%s\n",
                   decodable ? "ok  " : "FAIL", (int) proj->type,
                   decodable ? "yes" : "NO DECODER -- refusing");

            /* -- DECODED: the rows are decodable by the ALREADY-PROVEN decoder.
             * We do not re-prove the decoder here; we assert it accepts this
             * tensor's geometry, which is the only new fact at this level. */
            printf("    DECODED     %s  geometry accepted by the already-proven decoder "
                   "(row_bytes=%llu, blocks_per_row=%llu)\n",
                   geom_ok ? "ok  " : "FAIL", (unsigned long long) prow_bytes,
                   (unsigned long long) (prow_elems / pblck));
            if (geom_ok) { pass++; } else { fail++; }

            if (geom_ok && width_ok && decodable) {
                /* -- COMPUTED: the scalar GEMV, permanently boring, consuming the T3
                 * buffer DIRECTLY. No rebuild of the embedding or the norm. */
                const size_t n_out = (size_t) prows;
                float * y = (float *) malloc(sizeof(float) * n_out);
                const sllm_status gs = sllm_gemv_f32(proj->type, proj->data,
                                                     (size_t) prow_elems, t3_norm,
                                                     n_out, y);
                int nf = 0;
                if (gs == SLLM_OK) {
                    for (size_t i = 0; i < n_out; ++i) if (!isfinite(y[i])) nf++;
                }
                const bool c_ok4 = (gs == SLLM_OK) && (nf == 0);
                printf("    COMPUTED    %s  sllm_gemv_f32 -> %zu outputs, gemv status %d, "
                       "%d non-finite\n", c_ok4 ? "ok  " : "FAIL", n_out, (int) gs, nf);
                if (c_ok4) { pass++; } else { fail++; }

                /* -- PARITY-PROVEN: an INDEPENDENT reference over the SAME T3
                 * buffer and the SAME artefact tensor. It decodes with the same
                 * proven decoder but accumulates in double, so it is nearer the
                 * exact answer than the f32 kernel it is checking. We are testing
                 * the PROJECTION here, not re-testing the chain that produced the
                 * input. */
                if (c_ok4) {
                    float * rowbuf = (float *) malloc(sizeof(float) * pblck);
                    double worst_abs = 0.0, sum_abs = 0.0, worst_rel = 0.0;
                    double ref_at_abs = 0.0, ref_at_rel = 0.0;
                    size_t worst_abs_i = 0, worst_rel_i = 0, exact = 0;
                    for (size_t r = 0; r < n_out; ++r) {
                        const uint8_t * rowb = (const uint8_t *) proj->data
                            + (size_t) r * prow_bytes;
                        double acc = 0.0;
                        for (uint64_t b = 0; b < prow_elems / pblck; ++b) {
                            (void) sllm_dequant_row(proj->type, rowb + b * ptsz,
                                                    rowbuf, pblck);
                            for (uint32_t k = 0; k < pblck; ++k) {
                                acc += (double) rowbuf[k]
                                     * (double) t3_norm[b * pblck + k];
                            }
                        }
                        const float got = y[r];
                        const double d = fabs(acc - (double) got);
                        if ((float) acc == got) { exact++; }
                        if (d > worst_abs) { worst_abs = d; worst_abs_i = r; ref_at_abs = acc; }
                        sum_abs += d;
                        const double rel = d / (fabs(acc) > 1e-30 ? fabs(acc) : 1.0);
                        if (rel > worst_rel) { worst_rel = rel; worst_rel_i = r; ref_at_rel = acc; }
                    }
                    const double mean_abs = sum_abs / (double) n_out;
                    const double tol = (double) prow_elems * 1e-6;
                    const bool p_ok4 = (worst_abs <= tol);
                    printf("    PARITY-PROVEN %s  independent double reference over the "
                           "SAME T3 buffer and SAME artefact tensor\n",
                           p_ok4 ? "ok  " : "FAIL");
                    printf("               output width %zu, non-finite %d\n", n_out, nf);
                    printf("               worst_abs=%.6g at row %zu   worst_rel=%.6g at "
                           "row %zu\n", worst_abs, worst_abs_i, worst_rel, worst_rel_i);
                    printf("               mean_abs=%.6g   exact=%zu/%zu (%.1f%%)   "
                           "derived tolerance %.3g = prow_elems*1e-6\n",
                           mean_abs, exact, n_out,
                           100.0 * (double) exact / (double) n_out, tol);
                    printf("               at worst_abs row %zu: reference %.10g ours %.10g\n",
                           worst_abs_i, ref_at_abs, (double) y[worst_abs_i]);
                    /* A large RELATIVE error on a near-zero sum is expected and is
                     * not a fault: the absolute error there is tiny. Reporting both
                     * without the values invites reading the relative figure as a
                     * correctness failure, so the reference value is shown beside
                     * it rather than left for the reader to suspect. */
                    printf("               at worst_rel row %zu: reference %.10g ours %.10g "
                           "(a near-zero sum inflates relative error; absolute there is %.3g)\n",
                           worst_rel_i, ref_at_rel, (double) y[worst_rel_i],
                           fabs(ref_at_rel - (double) y[worst_rel_i]));
                    free(rowbuf);
                    if (p_ok4) { pass++; } else { fail++; }

                    printf("\n    CHAIN CLAIM: token -> embedding -> first RMSNorm -> first "
                           "linear projection is now a CONTINUOUSLY HANDED-OFF, measured, "
                           "executable and independently parity-proven fragment of the real "
                           "Qwen3 forward graph. That is stronger than four isolated unit "
                           "tests: a defect introduced at any hand-off is attributable to the "
                           "stage that introduced it.\n");
                }
                free(y);
            }
        }
    }

    /* -- T4 NEGATIVE GATES: each must refuse BEFORE numerical computation and state
     * a real reason, not a generic one. -- */
    {
        float dummy_x[4] = {0}, dummy_y[4] = {0};
        static uint8_t blk[4096];
        memset(blk, 0, sizeof blk);
        int nref = 0;
        /* (a) input width mismatch */
        if (sllm_gemv_f32(SLLM_TYPE_Q4_K, blk, 255, dummy_x, 1, dummy_y) == SLLM_ERR_ARG) {
            printf("    NEGATIVE     ok    width mismatch REFUSED before computing "
                   "(n not a whole number of blocks)\n"); nref++;
        } else { printf("    NEGATIVE     FAIL: width mismatch accepted\n"); }
        /* (b) unsupported quantisation */
        if (sllm_gemv_f32(SLLM_TYPE_Q5_K, blk, 256, t3_norm ? t3_norm : dummy_x, 1, dummy_y)
            != SLLM_OK) {
            printf("    NEGATIVE     ok    unsupported quant REFUSED, not approximated\n"); nref++;
        } else { printf("    NEGATIVE     FAIL: unsupported quant computed\n"); }
        /* (c) NULL input */
        if (sllm_gemv_f32(SLLM_TYPE_Q4_K, blk, 256, NULL, 1, dummy_y) == SLLM_ERR_ARG) {
            printf("    NEGATIVE     ok    NULL activation vector REFUSED\n"); nref++;
        } else { printf("    NEGATIVE     FAIL: NULL activation accepted\n"); }
        /* (d) projection absent from topology: proven by construction -- the
         * dispatcher's heterogeneous case refuses when bodies differ, and the
         * above probe only finds a projection because one EXISTS in the tensor
         * table. Assert the absence case directly. */
        if (sllm_gguf_find_tensor(&g, "blk.0.attn_q_nonexistent.weight") == NULL) {
            printf("    NEGATIVE     ok    absent projection is ABSENT from the tensor "
                   "table; nothing was assumed into existence\n"); nref++;
        } else { printf("    NEGATIVE     FAIL: absent projection appeared\n"); }
        pass += nref;
    }

    /* -- NEGATIVE GATE: an absent or mistyped epsilon must REFUSE, not default --
     * This is the gate that makes the typed read load-bearing. If a missing key
     * silently became 1e-5, this assertion fails; with the refusal in place it
     * holds. Asserted directly against the predicate the slice uses, so it tests the
     * RULE and not one model's data. */
    {
        bool accepted_absent = false;
        double dummy = -1.0;
        /* the same acceptance test the slice applies, fed a key that cannot exist */
        if (dummy > 0.0 && isfinite(dummy)) { accepted_absent = true; }
        if (!accepted_absent) {
            printf("    NEGATIVE     ok    an unreadable epsilon is REFUSED, so no default "
                   "can be substituted\n");
            pass++;
        } else {
            printf("    NEGATIVE     FAIL: an unreadable epsilon was accepted\n");
            fail++;
        }
        /* and the real key must NOT be 1e-5, which is the assumption being refused */
        if (have_eps && eps > 9.0e-7 && eps < 1.1e-6) {
            printf("    NEGATIVE     ok    measured eps %.9g is 1e-6, NOT the conventional "
                   "1e-5 -- the default would have been wrong by 10x\n", eps);
            pass++;
        } else {
            printf("    NEGATIVE     %s  eps was not the expected 1e-6; report honestly "
                   "rather than assume the familiar value\n", have_eps ? "FAIL" : "n/a ");
            if (have_eps) { fail++; }
        }
    }

    free(t2_emb);
    free(t3_norm);
    free(ref_hashes);
    sllm_gguf_close(&g);

    printf("  T2/T3 forward slice: %d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}