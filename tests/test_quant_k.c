/* ------------------------------------------------------------------ *
 * test_quant_k.c -- the K-quant dequantiser gate.
 *
 * This is the first product test whose expected values come from a file
 * Saphira did not produce. tests/golden/mainstream-qwen3-dequant.txt was
 * captured by driving the VENDORED REFERENCE build's own type traits over a real
 * Q4_K_M model, so it is an independent witness rather than a recording of our
 * own output. Comparing our decode against it is the only thing that makes the
 * Q4_K/Q6_K `supported` flags honest.
 *
 * The discipline this file exists to enforce is the one the whole project turned
 * on: a capability flag must not be a promise without a kernel, and a kernel must
 * not be a promise without a reference. Both halves are checked here.
 *
 * For each golden case the test reads the real encoded bytes straight out of the
 * mapped GGUF, decodes with sllm_dequant_row, and requires:
 *
 *   - the decode SUCCEEDS (no UNSUPPORTED, no silent wrong answer)
 *   - the FNV-1a hash of the decoded floats equals the reference DECODED_HASH
 *   - the first/last samples match to f32 printing precision
 *   - the decoded row contains no non-finite value
 *
 * A hash comparison rather than a tolerance is deliberate. These are exact
 * transcriptions, not approximations, so "close enough" is not a pass here: a
 * scale unpacked one bit off produces a large error, and a scale applied in the
 * wrong order produces a small one, and only an exact comparison distinguishes
 * the mistake we care about from the mistake that is merely loud.
 * ------------------------------------------------------------------ */

#include "saphira_llm/gguf.h"
#include "saphira_llm/quant.h"
#include "saphira_llm/status.h"

#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SLLM_TEST_QWEN3
#define SLLM_TEST_QWEN3 "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
#endif
#ifndef SLLM_TEST_DEQ_GOLDEN
#define SLLM_TEST_DEQ_GOLDEN "tests/golden/mainstream-qwen3-dequant.txt"
#endif

/* FNV-1a over the raw bytes of `n` floats. The reference probe uses the same
 * 64-bit FNV-1a, so the two hashes are directly comparable; the exact byte order
 * of the float representation is identical because both sides run on the same
 * little-endian target and neither applies a transformation. */
static uint64_t hash_floats(const float * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n * sizeof(float); ++i) {
        h ^= (uint64_t) b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Parse one golden case. Returns 0 on success. Deliberately tolerant about
 * whitespace and ordering: the golden is a human-readable record, and a gate that
 * breaks when someone reformats a comment is a gate people will disable. */
struct deq_case {
    char name[128];
    char type_name[16];
    uint64_t row;
    uint64_t decoded_elems;
    uint64_t decoded_hash;
    int      have_hash;
    float    first[8];
    int      n_first;
    float    last[4];
    int      n_last;
};

static int parse_case(const char * block, struct deq_case * c) {
    memset(c, 0, sizeof *c);
    c->row = 0;
    const char * p = strstr(block, "[case]");
    if (p == NULL) { return -1; }
    p += 6;
    while (*p == ' ') { ++p; }
    size_t i = 0;
    while (*p && *p != '\n' && i + 1 < sizeof c->name) { c->name[i++] = *p++; }
    c->name[i] = '\0';

    const char * t = strstr(block, "type            = ");
    if (t != NULL) {
        t += 18;
        size_t k = 0;
        while (*t && *t != ' ' && *t != '\n' && k + 1 < sizeof c->type_name) {
            c->type_name[k++] = *t++;
        }
        c->type_name[k] = '\0';
    }
    const char * r = strstr(block, "row             = ");
    if (r != NULL) { c->row = strtoull(r + 17, NULL, 10); }
    const char * de = strstr(block, "decoded_elems   = ");
    if (de != NULL) { c->decoded_elems = strtoull(de + 18, NULL, 10); }
    const char * dh = strstr(block, "DECODED_HASH    = ");
    if (dh != NULL) {
        c->decoded_hash = strtoull(dh + 18, NULL, 16);
        c->have_hash = 1;
    }
    const char * f8 = strstr(block, "first8          = ");
    if (f8 != NULL) {
        f8 += 18;
        char * end = NULL;
        for (int k = 0; k < 8; ++k) {
            float v = strtof(f8, &end);
            if (end == f8) { break; }
            c->first[k] = v; c->n_first++;
            f8 = end;
        }
    }
    const char * l4 = strstr(block, "last4           = ");
    if (l4 != NULL) {
        l4 += 18;
        char * end = NULL;
        for (int k = 0; k < 4; ++k) {
            float v = strtof(l4, &end);
            if (end == l4) { break; }
            c->last[k] = v; c->n_last++;
            l4 = end;
        }
    }
    return 0;
}

/* Compare at the precision the golden prints, i.e. 9 significant digits. This is
 * not a numerical tolerance in the usual sense: both sides are computed the same
 * way, so anything beyond printing precision means the arithmetic diverged. */
static int close_enough(float a, float b) {
    if (a == b) { return 1; }
    if (!isfinite(a) || !isfinite(b)) { return 0; }
    const float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return fabsf(a - b) <= 1e-6f * scale;
}

static int check_case(const sllm_gguf * g, const struct deq_case * c,
                      char * why, size_t why_sz, int * n_checked) {
    const sllm_gguf_tensor * t = sllm_gguf_find_tensor(g, c->name);
    if (t == NULL) {
        snprintf(why, why_sz, "tensor %s not found in model", c->name);
        return 0;
    }

    uint32_t blck = 0, tsz = 0;
    if (sllm_gguf_type_traits(t->type, &blck, &tsz) != SLLM_OK) {
        snprintf(why, why_sz, "no type traits for type %d", (int) t->type);
        return 0;
    }
    if (blck == 0) { snprintf(why, why_sz, "block size zero"); return 0; }

    /* A ROW is ne[0] ELEMENTS, i.e. ne[0]/blck stored blocks. Decoding one block
     * and calling it a row is the bug this gate caught on its first run: for F32
     * blck is 1, so it silently decoded a single element and hashed 4 bytes
     * against a reference hash of 16384, and it reported a mismatch with no
     * explanation. The row stride is computed, not assumed. */
    const uint64_t row_elems = t->ne[0];
    if (row_elems % blck != 0) {
        snprintf(why, why_sz, "row %llu not a multiple of block %u",
                 (unsigned long long) row_elems, blck);
        return 0;
    }
    const uint64_t n_rows = row_elems / blck;
    if (c->row >= n_rows) {
        snprintf(why, why_sz, "row %llu beyond %llu rows",
                 (unsigned long long) c->row, (unsigned long long) n_rows);
        return 0;
    }

    /* Read the real encoded bytes straight out of the mapping. The byte offset is
     * computed from the row stride, not assumed. */
    const uint64_t row_bytes = tsz * (row_elems / blck);
    const uint8_t * src = (const uint8_t *) t->data + (size_t) c->row * row_bytes;
    float * dst = (float *) malloc(sizeof(float) * row_elems);
    if (dst == NULL) { snprintf(why, why_sz, "oom"); return 0; }

    const sllm_status st = sllm_dequant_row(t->type, src, dst, (size_t) row_elems);
    if (st != SLLM_OK) {
        snprintf(why, why_sz, "dequant_row returned %d (type %s)", (int) st,
                 c->type_name[0] ? c->type_name : "?");
        free(dst);
        return 0;
    }

    int ok = 1;
    for (uint64_t i = 0; i < row_elems && ok; ++i) {
        if (!isfinite(dst[i])) {
            snprintf(why, why_sz, "non-finite value at index %llu", (unsigned long long) i);
            ok = 0;
        }
    }

    if (ok && c->have_hash) {
        const uint64_t got = hash_floats(dst, (size_t) row_elems);
        if (got != c->decoded_hash) {
            snprintf(why, why_sz,
                     "hash mismatch: got %016llx want %016llx  (type %s row %llu)",
                     (unsigned long long) got,
                     (unsigned long long) c->decoded_hash,
                     c->type_name[0] ? c->type_name : "?", (unsigned long long) c->row);
            ok = 0;
        }
    }

    if (ok) {
        for (int k = 0; k < c->n_first && ok; ++k) {
            if (!close_enough(dst[k], c->first[k])) {
                snprintf(why, why_sz, "first[%d]: got %.9g want %.9g",
                         k, (double) dst[k], (double) c->first[k]);
                ok = 0;
            }
        }
        for (int k = 0; k < c->n_last && ok; ++k) {
            const float got = dst[row_elems - (uint64_t) c->n_last + (uint64_t) k];
            if (!close_enough(got, c->last[k])) {
                snprintf(why, why_sz, "last[%d]: got %.9g want %.9g",
                         k, (double) got, (double) c->last[k]);
                ok = 0;
            }
        }
    }

    free(dst);
    if (ok) { (*n_checked)++; }
    return ok;
}

int main_k_quant_gate(void) {
    int fail = 0, pass = 0;

    printf("  K-quant dequantiser gate (against the REFERENCE golden)\n");

    if (access(SLLM_TEST_QWEN3, R_OK) != 0) {
        printf("    skipped: %s not available\n", SLLM_TEST_QWEN3);
        printf("  K-quant dequantiser gate: %d passed, %d failed (skipped)\n", pass, fail);
        return 0;
    }
    if (access(SLLM_TEST_DEQ_GOLDEN, R_OK) != 0) {
        printf("    FAILED: golden %s missing. This gate has no expected values\n"
               "    without it, and skipping it would silently void the whole point.\n",
               SLLM_TEST_DEQ_GOLDEN);
        printf("  K-quant dequantiser gate: %d passed, %d failed\n", pass, fail + 1);
        return 1;
    }

    sllm_gguf g;
    char err[256];
    if (sllm_gguf_open(SLLM_TEST_QWEN3, &g, err, sizeof err) != SLLM_OK) {
        printf("    FAILED: cannot open model: %s\n", err);
        printf("  K-quant dequantiser gate: %d passed, %d failed\n", pass, fail + 1);
        return 1;
    }

    FILE * f = fopen(SLLM_TEST_DEQ_GOLDEN, "r");
    if (f == NULL) {
        printf("    FAILED: cannot open golden\n");
        sllm_gguf_close(&g);
        printf("  K-quant dequantiser gate: %d passed, %d failed\n", pass, fail + 1);
        return 1;
    }

    /* Read the golden in case-sized chunks. Cases begin at "[case]" and run until
     * the next one or EOF. */
    char * blob = NULL;
    size_t cap = 0, len = 0;
    {
        FILE * probe = fopen(SLLM_TEST_DEQ_GOLDEN, "r");
        if (probe != NULL) {
            fseek(probe, 0, SEEK_END);
            const long sz = ftell(probe);
            fseek(probe, 0, SEEK_SET);
            if (sz > 0) {
                cap = (size_t) sz;
                blob = (char *) malloc(cap + 1);
                if (blob != NULL) {
                    len = fread(blob, 1, cap, probe);
                    blob[len] = '\0';
                }
            }
            fclose(probe);
        }
    }
    fclose(f);

    if (blob == NULL) {
        printf("    FAILED: cannot read golden into memory\n");
        sllm_gguf_close(&g);
        printf("  K-quant dequantiser gate: %d passed, %d failed\n", pass, fail + 1);
        return 1;
    }

    int n_cases = 0, n_q4k = 0, n_q6k = 0, n_other = 0;
    const char * cur = blob;
    while (cur != NULL && *cur != '\0') {
        const char * next = strstr(cur + 1, "[case]");
        size_t seglen = (next != NULL) ? (size_t) (next - cur) : strlen(cur);
        char * seg = (char *) malloc(seglen + 1);
        if (seg == NULL) { break; }
        memcpy(seg, cur, seglen);
        seg[seglen] = '\0';

        if (strstr(seg, "[case]") != NULL) {
            struct deq_case c;
            if (parse_case(seg, &c) == 0) {
                n_cases++;
                const int is_k = (strstr(c.type_name, "q4_K") != NULL ||
                                  strstr(c.type_name, "q6_K") != NULL);
                if (strstr(c.type_name, "q4_K") != NULL) { n_q4k++; }
                else if (strstr(c.type_name, "q6_K") != NULL) { n_q6k++; }
                if (!is_k) { n_other++; }

                char why[512] = "";
                int checked = 0;
                if (check_case(&g, &c, why, sizeof why, &checked)) {
                    printf("    ok   %-34s %-6s hash %016llx\n", c.name, c.type_name,
                           (unsigned long long) c.decoded_hash);
                    pass++;
                } else {
                    printf("    FAIL %-34s %-6s %s\n", c.name, c.type_name, why);
                    fail++;
                }
            }
        }
        free(seg);
        cur = next;
    }

    free(blob);
    sllm_gguf_close(&g);

    printf("    cases=%d  q4_K=%d  q6_K=%d  non-K=%d\n",
           n_cases, n_q4k, n_q6k, n_other);

    /* The gate is only meaningful if it actually exercised the K-quants. A gate
     * that passes because it found no K-quant cases has proved nothing, and saying
     * so is more useful than a green line. */
    if (n_q4k == 0 && n_q6k == 0) {
        printf("    FAILED: no Q4_K or Q6_K cases in the golden, so the kernels were\n"
               "    never exercised. A gate with nothing to check is not a pass.\n");
        fail++;
    } else {
        printf("    K-quant kernels exercised against reference values: q4_K=%d q6_K=%d\n",
               n_q4k, n_q6k);
    }

    printf("  K-quant dequantiser gate: %d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}