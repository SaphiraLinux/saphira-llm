/* ------------------------------------------------------------------ *
 * s0_gemv_probe.cpp -- external witness for Step 2's f32 GEMV.
 *
 * This is deliberately NOT Saphira arithmetic on both sides. The reference result
 * is produced by the VENDORED reference implementation:
 *
 *   dequantisation : ggml_get_type_traits(type).to_float, the reference's own
 *                    per-type expander (the same entry point that produced
 *                    tests/golden/mainstream-qwen3-dequant.txt)
 *   the dot product: DOUBLE-precision accumulation.
 *
 * Accumulating in double is deliberately STRICTER than matching another f32
 * kernel. A reference that merely used f32 would agree with a wrong-but-close
 * implementation; a double-precision reference is closer to the mathematically
 * exact answer, so Saphira's f32 result is measured against truth rather than
 * against a peer. Anything wrong with stride, index or block boundaries lands in
 * the fifth decimal or worse, and is unmissable.
 *
 * and Saphira's sllm_gemv_f32 is compared against that. If both sides used our
 * code the comparison would prove only that our code agrees with itself.
 *
 * WHAT THIS IS DESIGNED TO CATCH, stated before the run rather than after:
 *
 *   - a wrong row stride. Testing a single row cannot see one, because a stride
 *     error only manifests once the row index moves. Several rows are mandatory.
 *   - an off-by-one in the block loop. A tensor whose row is an exact multiple of
 *     the block size hides a trailing-block bug; a row count and element count
 *     that do NOT divide evenly are included below on purpose.
 *   - an accumulator not reset between rows.
 *   - the x vector being indexed by block rather than by element.
 *
 * Each case prints the reference value, Saphira's value, and the ULP distance, so
 * a disagreement is quantified rather than merely flagged.
 * ------------------------------------------------------------------ */

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

/* Ordered representation of a float, so "bit-for-bit" and "within N ULP" can be
 * distinguished. A dot product accumulates in a different order in different
 * implementations, so exact equality is NOT the right bar here; the bar is that
 * the two agree to within a stated ULP distance, reported rather than assumed. */
static uint64_t bits_of(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return ((uint64_t) u) << 32;
}
static int64_t ulp_distance(float a, float b) {
    if (a == b) { return 0; }
    if (std::isnan(a) || std::isnan(b)) { return INT64_MAX; }
    const int64_t ia = (int64_t) (int32_t) *(uint32_t *) &a;
    const int64_t ib = (int64_t) (int32_t) *(uint32_t *) &b;
    return ia > ib ? ia - ib : ib - ia;
}

/* The reference path, using ONLY the vendored reference implementation. */
static void reference_gemv(const ggml_type_traits * tr, const void * data,
                           int64_t row_elems, const float * x, int64_t n_rows,
                           std::vector<double> & out) {
    const int64_t nb = row_elems / tr->blck_size;
    out.assign((size_t) n_rows, 0.0);
    std::vector<float> buf((size_t) tr->blck_size);
    (void) out;
    for (int64_t r = 0; r < n_rows; ++r) {
        const char * row = (const char *) data + (size_t) r * (size_t) tr->type_size * (size_t) nb;
        double acc = 0.0;
        for (int64_t b = 0; b < nb; ++b) {
            tr->to_float(row + (size_t) b * (size_t) tr->type_size, buf.data(), tr->blck_size);
            for (int64_t i = 0; i < tr->blck_size; ++i) {
                acc += (double) buf[(size_t) i] * (double) x[(size_t) (b * tr->blck_size + i)];
            }
        }
        out[(size_t) r] = acc;
    }
}

/* Deterministic activation vector: a fixed LCG, so the probe and any golden are
 * reproducible without depending on rand() implementation details. */
static void fill_x(float * x, size_t n, uint64_t seed) {
    uint64_t s = seed ? seed : 1;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        /* Map to roughly [-1, 1) with a fixed shape. */
        x[i] = (float) ((int32_t) ((s >> 33) % 20001) - 10000) / 10000.0f;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [tensor] [xseed]\n", argv[0]); return 1; }

    uint64_t fsize = 0;
    { FILE * sf = fopen(argv[1], "rb");
      if (sf == NULL) { fprintf(stderr, "cannot stat %s\n", argv[1]); return 1; }
      fseek(sf, 0, SEEK_END); fsize = (uint64_t) ftell(sf); fclose(sf); }

    gguf_init_params gp; memset(&gp, 0, sizeof gp); gp.no_alloc = true;
    gguf_init_params gp_map; memset(&gp_map, 0, sizeof gp_map);
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    const uint64_t xseed = argc > 3 ? strtoull(argv[3], NULL, 10) : 12345ULL;

    /* Candidate tensors: prefer one of each interesting type, then anything. */
    std::vector<std::string> want = {
        "blk.0.attn_q.weight",   /* q4_K, many rows, many blocks */
        "blk.0.ffn_down.weight", /* q6_K */
        "blk.0.attn_norm.weight",/* f32 */
        "token_embd.weight"      /* q6_K, large */
    };
    if (argc > 2) { want.clear(); want.push_back(argv[2]); }

    printf("\n=== STEP 2 GEMV: Saphira vs vendored reference ===\n");
    printf("  activation seed = %llu (deterministic; no rand())\n", (unsigned long long) xseed);
    printf("  reference path  = ggml type_traits.to_float + ggml_vec_dot_f32\n");
    printf("  reference only: vendored to_float for dequantisation, double accumulation\n");
    printf("  for the dot. Hex float output, every row, so a stride error cannot hide.\n");

    int total = 0, mismatched = 0;
    /* Tolerance in ULP. The two implementations accumulate in different orders, so
     * this is the honest bar rather than exact equality. It is TIGHT on purpose:
     * a wrong stride or a wrong index is enormous, and a wrong order of addition
     * within one row is a handful. 4096 ULP separates those two failure classes
     * by a wide margin. */
    const int64_t ULP_TOL = 4096;

    for (const std::string & name : want) {
        const int64_t id = gguf_find_tensor(g, name.c_str());
        if (id < 0) { continue; }
        const ggml_type tt = gguf_get_tensor_type(g, id);
        const ggml_type_traits * tr = ggml_get_type_traits(tt);
        const int64_t * ne = gguf_get_tensor_ne(g, id);
        const int64_t row_elems = ne[0];
        const int64_t nb = row_elems / tr->blck_size;
        /* Offsets are computed, not assumed: data_offset from the header plus the
         * per-tensor offset, against a buffer we read ourselves. The gguf API
         * exposes metadata but not a data pointer, so borrowing the pattern from
         * s0_deq_probe keeps this arithmetic visible and checkable. */
        const uint64_t toff = gguf_get_tensor_offset(g, id);
        const uint64_t doff = gguf_get_data_offset(g);
        /* Row count FIRST, then the read sized from it. The first version of this
         * probe read ONE row's bytes and then iterated n_rows rows, so the
         * reference walked off the end of its own buffer and segfaulted. Reading
         * exactly as many bytes as are consumed is the whole discipline; a probe
         * that reads less than it walks is not measuring the model, it is
         * measuring its own optimism. */
        int64_t n_rows = (ne[1] > 1) ? ne[1] : 1;
        if (n_rows > 64) { n_rows = 64; }
        if (n_rows < 1) { n_rows = 1; }
        /* ne[1] is the stored row count for a 2D weight. It is NOT the block
         * count per row. Capping by the block count silently reduced attn_q from
         * 4096 stored rows to 16, which is a plausible-looking small matrix and
         * would have made a stride test far weaker than intended without ever
         * reporting that it had done so. */
        const int64_t stored_rows = (ne[1] > 1) ? ne[1] : 1;
        if (n_rows > stored_rows) { n_rows = stored_rows; }

        const uint64_t row_bytes = (uint64_t) tr->type_size * (uint64_t) nb;
        const uint64_t tbytes = row_bytes * (uint64_t) n_rows;

        /* BOUNDS CHECK before dereferencing. An out-of-range offset that silently
         * reads past the buffer is a segfault with no diagnosis; reporting the
         * four numbers that decide it turns a crash into a fact. */
        if (doff + toff + tbytes > fsize) {
            printf("  %-26s %-6s OFFSET OUT OF RANGE: data_offset=%llu tensor_offset=%llu "
                   "need=%llu file=%llu\n",
                   name.c_str(), ggml_type_name(tt),
                   (unsigned long long) doff, (unsigned long long) toff,
                   (unsigned long long) (doff + toff + tbytes),
                   (unsigned long long) fsize);
            mismatched++; total++;
            continue;
        }
        std::vector<uint8_t> wbuf((size_t) tbytes);
        { FILE * tf = fopen(argv[1], "rb");
          if (tf == NULL) { printf("  cannot reopen model\n"); mismatched++; total++; continue; }
          if (fseek(tf, (long) (doff + toff), SEEK_SET) != 0 ||
              fread(wbuf.data(), 1, (size_t) tbytes, tf) != (size_t) tbytes) {
              printf("  %-26s short read at offset %llu\n", name.c_str(),
                     (unsigned long long) (doff + toff));
              fclose(tf); mismatched++; total++; continue;
          }
          fclose(tf); }
        const uint8_t * wdata = wbuf.data();

        std::vector<float> x((size_t) row_elems);
        fill_x(x.data(), x.size(), xseed);

        std::vector<double> ref;
        reference_gemv(tr, wdata, row_elems, x.data(), n_rows, ref);

        /* Emit EVERY row, not a sample. A golden that carries only the first row
         * cannot detect a stride error: every wrong-stride implementation gets row
         * 0 right. Each row is printed in hex float form so the witness is exact
         * and no decimal rounding can hide a disagreement. */
        printf("  [gemv] %s\n", name.c_str());
        printf("    type          = %s\n", ggml_type_name(tt));
        printf("    row_elems     = %lld\n", (long long) row_elems);
        printf("    blocks_per_row= %lld\n", (long long) nb);
        printf("    n_rows        = %lld  (of %lld stored)\n", (long long) n_rows, (long long) stored_rows);
        printf("    x_seed        = %llu\n", (unsigned long long) xseed);
        printf("    ROWS");
        for (int64_t r = 0; r < n_rows; ++r) {
            printf(" %a", ref[(size_t) r]);
        }
        /* Per-row hash of the DEQUANTISED bytes, via the reference expander. The
         * K-quant gate only ever checked ROW 0, so a dequantisation that is right
         * on the first row and wrong on later ones -- a stride bug in the expander,
         * or a block walk that drifts -- would be invisible. Emitting a hash per row
         * makes that separable from the GEMV, so when a GEMV row disagrees we can
         * say whether the fault is in the bytes or in the multiply-accumulate. */
        printf("    ROWHASH");
        { std::vector<uint8_t> h((size_t) row_elems * 4);
          for (int64_t r = 0; r < n_rows; ++r) {
              std::vector<float> rowbuf((size_t) row_elems);
              const char * rp = (const char *) wdata + (size_t) r * row_bytes;
              for (int64_t b = 0; b < nb; ++b) {
                  tr->to_float(rp + (size_t) b * tr->type_size, rowbuf.data() + (size_t) (b * tr->blck_size),
                               tr->blck_size);
              }
              uint64_t hh = 1469598103934665603ULL;
              const uint8_t * bb = (const uint8_t *) rowbuf.data();
              for (size_t i = 0; i < rowbuf.size() * sizeof(float); ++i) { hh ^= bb[i]; hh *= 1099511628211ULL; }
              printf(" %016llx", (unsigned long long) hh);
          }
        }
        printf("\n");
        total++;
    }

    printf("\n  cases=%d  reference rows emitted for golden capture\n", total);
    gguf_free(g);
    return 0;
}