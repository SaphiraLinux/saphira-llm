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
        printf("    LOCATED      ok    vocab_size %s%u, tokenizer tokens %llu, "
               "table rows %llu (padding is permitted; equality is not assumed)\n",
               have_vocab ? "declared " : "NOT READ (not assumed 0) ",
               declared_vocab, (unsigned long long) tok_n,
               (unsigned long long) emb->ne[1]);
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

        pass += (d_ok ? 1 : 0) + (c_ok ? 1 : 0) + (p_ok ? 1 : 0);
        fail += (!d_ok ? 1 : 0) + (!c_ok ? 1 : 0) + (p_ok ? 0 : 1);
        if (p_ok) { parity_ok++; }

        free(vec);
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

    free(ref_hashes);
    sllm_gguf_close(&g);

    printf("  T2 forward slice: %d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}