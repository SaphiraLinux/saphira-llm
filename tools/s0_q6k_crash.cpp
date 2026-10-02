/* ------------------------------------------------------------------ *
 * s0_q6k_crash.cpp -- T1 instrumentation. WHERE does the q6_K probe die?
 *
 * Reproduces the reference-side walk over blk.0.ffn_down.weight with progress
 * reporting, so the failing ROW and BLOCK are captured rather than inferred from
 * how many values happened to reach stdout. A truncated output is a symptom; this
 * reports the cause.
 *
 * Nothing here is a golden and nothing here writes one. This is a diagnostic run
 * only, and its output is explicitly NOT a witness.
 * ------------------------------------------------------------------ */

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [tensor]\n", argv[0]); return 1; }
    const char * name = (argc > 2) ? argv[2] : "blk.0.ffn_down.weight";

    uint64_t fsize = 0;
    { FILE * sf = fopen(argv[1], "rb");
      if (!sf) { fprintf(stderr, "cannot stat\n"); return 1; }
      fseek(sf, 0, SEEK_END); fsize = (uint64_t) ftell(sf); fclose(sf); }

    gguf_init_params gp; memset(&gp, 0, sizeof gp); gp.no_alloc = true;
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) { fprintf(stderr, "gguf open failed\n"); return 1; }

    const int64_t id = gguf_find_tensor(g, name);
    if (id < 0) { fprintf(stderr, "tensor %s absent\n", name); return 1; }

    const ggml_type tt = gguf_get_tensor_type(g, id);
    const ggml_type_traits * tr = ggml_get_type_traits(tt);
    const int64_t * ne = gguf_get_tensor_ne(g, id);
    const uint64_t row_elems = (uint64_t) ne[0];
    const uint64_t nb = row_elems / (uint64_t) tr->blck_size;
    const uint64_t stored_rows = (uint64_t) (ne[1] > 1 ? ne[1] : 1);
    const uint64_t row_bytes = (uint64_t) tr->type_size * nb;
    const uint64_t doff = gguf_get_data_offset(g);
    const uint64_t toff = gguf_get_tensor_offset(g, id);

    printf("tensor=%s type=%s blck=%d type_size=%d\n", name, ggml_type_name(tt),
           tr->blck_size, tr->type_size);
    printf("ne[0]=%lld ne[1]=%lld row_elems=%llu nb=%llu stored_rows=%llu row_bytes=%llu\n",
           (long long) ne[0], (long long) ne[1], (unsigned long long) row_elems,
           (unsigned long long) nb, (unsigned long long) stored_rows,
           (unsigned long long) row_bytes);
    printf("data_offset=%llu tensor_offset=%llu abs=%llu file=%llu\n",
           (unsigned long long) doff, (unsigned long long) toff,
           (unsigned long long) (doff + toff), (unsigned long long) fsize);
    printf("to_float ptr = %p\n", (void *) (uintptr_t) tr->to_float);
    fflush(stdout);

    if (tr->to_float == NULL) { printf("FATAL: to_float is NULL for this type\n"); return 3; }

    const uint64_t need = row_bytes * stored_rows;
    printf("full tensor bytes = %llu ; available from abs = %llu -> %s\n",
           (unsigned long long) need,
           (unsigned long long) (fsize - (doff + toff)),
           (doff + toff + need <= fsize) ? "IN BOUNDS" : "OUT OF BOUNDS");
    fflush(stdout);

    /* Walk every row and block, reporting progress. To_float for q6_K takes the
     * pointer to ONE BLOCK, and the third argument is the element count. If that
     * signature is wrong for this type, the walk will run off the buffer and the
     * row/block printed just before death is the answer. */
    std::vector<unsigned char> buf((size_t) row_bytes);
    { FILE * tf = fopen(argv[1], "rb");
      if (!tf) { fprintf(stderr, "reopen failed\n"); return 1; }
      if (fseek(tf, (long) (doff + toff), SEEK_SET) != 0 ||
          fread(buf.data(), 1, (size_t) row_bytes, tf) != (size_t) row_bytes) {
          printf("FATAL: could not read one row at %llu\n",
                 (unsigned long long) (doff + toff));
          fclose(tf); return 1;
      }
      fclose(tf); }
    printf("read ONE row: %llu bytes OK\n", (unsigned long long) row_bytes);
    fflush(stdout);

    std::vector<float> out((size_t) tr->blck_size);
    for (uint64_t b = 0; b < nb; ++b) {
        printf("  block %llu/%llu  src=%llu\n", (unsigned long long) b,
               (unsigned long long) nb, (unsigned long long) (b * tr->type_size));
        fflush(stdout);
        tr->to_float((const char *) buf.data() + b * tr->type_size,
                     out.data(), tr->blck_size);
        /* Touch the output so the write is not optimised away and any overrun
         * faults HERE rather than silently later. */
        volatile float sink = out[0] + out[tr->blck_size - 1];
        (void) sink;
    }
    printf("ROW 0 COMPLETE: %llu blocks walked with no fault\n", (unsigned long long) nb);
    fflush(stdout);

    /* Now the multi-row walk, which is where the failure was observed. Read two
     * rows so a stride error is visible, and report each row before walking it. */
    /* Walk exactly as many rows as the gemv witness did, so the fault is
     * reproduced rather than approximated. */
    const uint64_t WALK = 64;
    std::vector<unsigned char> two((size_t) (row_bytes * WALK));
    { FILE * tf = fopen(argv[1], "rb");
      if (fseek(tf, (long) (doff + toff), SEEK_SET) == 0) {
          size_t got = fread(two.data(), 1, two.size(), tf);
          printf("read %llu rows: %zu of %zu bytes -> %s\n",
                 (unsigned long long) WALK, got, two.size(),
                 (got == two.size()) ? "COMPLETE" : "PARTIAL READ");
      }
      if (tf) { fclose(tf); } }
    printf("buffer holds %llu rows exactly: %llu bytes, buffer %zu bytes -> %s\n",
           (unsigned long long) WALK, (unsigned long long) (row_bytes * WALK), two.size(),
           ((uint64_t) two.size() == row_bytes * WALK) ? "EXACT FIT" : "MISMATCH");
    fflush(stdout);

    for (uint64_t r = 0; r < WALK; ++r) {
        printf("  ROW %llu walk\n", (unsigned long long) r);
        fflush(stdout);
        const char * rowp = (const char *) two.data() + r * row_bytes;
        if (r + 1 == WALK) { printf("  last row offset = %llu (buffer %zu)\n",
            (unsigned long long)(r * row_bytes), two.size()); fflush(stdout); }
        for (uint64_t b = 0; b < nb; ++b) {
            if (r + 1 == WALK) { printf("    block %llu/%llu\n",
                (unsigned long long) b, (unsigned long long) nb); fflush(stdout); }
            tr->to_float(rowp + b * tr->type_size, out.data(), tr->blck_size);
            volatile float sink = out[0];
            (void) sink;
        }
        printf("  ROW %llu COMPLETE\n", (unsigned long long) r);
        fflush(stdout);
    }

    /* NOW THE ROWHASH BLOCK, transcribed verbatim from the gemv probe. This is the
     * only structural difference between a clean walk and the witness, so it is
     * the prime suspect and it is tested here rather than guessed at. */
    printf("\n--- ROWHASH block, verbatim from the witness ---\n");
    fflush(stdout);
    { std::vector<uint8_t> h((size_t) row_elems * 4);
      for (uint64_t r = 0; r < WALK; ++r) {
          printf("  rowhash row %llu\n", (unsigned long long) r);
          fflush(stdout);
          std::vector<float> rowbuf((size_t) row_elems);
          const char * rp = (const char *) two.data() + (size_t) r * row_bytes;  /* CORRECTLY SIZED */
          for (uint64_t b = 0; b < nb; ++b) {
              tr->to_float(rp + (size_t) b * tr->type_size,
                           rowbuf.data() + (size_t) (b * tr->blck_size),
                           tr->blck_size);
          }
          uint64_t hh = 1469598103934665603ULL;
          const unsigned char * bb = (const unsigned char *) rowbuf.data();
          for (size_t i = 0; i < rowbuf.size() * sizeof(float); ++i) {
              hh ^= bb[i]; hh *= 1099511628211ULL;
          }
      }
    }
    printf("ROWHASH BLOCK COMPLETED without fault\n");
    return 0;
}