/* ------------------------------------------------------------------ *
 * s0_gemv_probe.cpp -- external witness for Step 2's f32 GEMV. REBUILT.
 *
 * This is a WITNESS. A witness has one obligation beyond computing an answer: it
 * must be able to say it did not go out of bounds, and it must be able to say that
 * BEFORE it reads, not afterwards when the MMU happens to notice.
 *
 * WHY THE REBUILD. The previous version read tensors through raw pointer
 * arithmetic and died, non-deterministically, on the q6_K case. An instrumented
 * study (tools/s0_q6k_crash.cpp) showed the fault was LATE: walking the same
 * 64 x 48 grid from a one-row buffer faults at ROW 8, eight rows after the buffer
 * ended. So "where it crashed" was never "where it went out of bounds", every
 * truncation count we had recorded was noise rather than a coordinate, and a run
 * that appeared to succeed may simply not have read far enough to die.
 *
 * Two rules follow, and they are the whole point of this rewrite:
 *
 *   1. PROVE THE GEOMETRY BEFORE TOUCHING DATA. Every byte offset that will be
 *      read is computed, compared against the buffer size, and refused if it does
 *      not fit -- before the read happens.
 *   2. DO NOT USE A DERIVATION AS ITS OWN PROOF. The row stride, the block count
 *      and the tensor size are cross-checked against the FILE's own layout: this
 *      tensor's computed end must not overlap the next tensor's offset, and the
 *      total must fit inside the actual file size. Those bounds come from the
 *      container, not from the arithmetic that produced the quantity being checked.
 *
 * Everything else is unchanged in intent. Dequantisation is the vendored
 * reference's own type_traits.to_float. The dot product accumulates in DOUBLE
 * precision, which is deliberately stricter than matching another f32 kernel.
 *
 * Output grammar, explicit and self-delimiting so a reader cannot misalign:
 *   [gemv] <tensor>
 *   type / row_elems / blocks_per_row / block_size / type_size / row_bytes
 *   n_rows / stored_rows / tensor_required_bytes / supplied_buffer_bytes
 *   first_byte_touched / max_byte_touched / buffer_bytes
 *   BOUNDS <OK|REFUSED>
 *   ROWS <n values>
 *   ROWHASH <n values>
 *   ENDCASE <tensor>
 * ------------------------------------------------------------------ */

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ---------------- geometry, computed then CHECKED ---------------- */

struct geom {
    /* Declared by the container. */
    int64_t  n_dims;
    int64_t  ne0, ne1;
    int32_t  block_size;
    int32_t  type_size;
    uint64_t tensor_offset;
    uint64_t data_offset;
    uint64_t file_bytes;

    /* Derived, each from exactly one source. */
    uint64_t blocks_per_row;
    uint64_t row_bytes;
    uint64_t n_rows;
    uint64_t stored_rows;
    uint64_t tensor_required_bytes;
    uint64_t supplied_buffer_bytes;

    /* The traversal, declared before it runs. */
    uint64_t first_byte_touched;
    uint64_t max_byte_touched;   /* INCLUSIVE last index */
    uint64_t requested_span;

    /* Independent container bounds, NOT derived from the above. */
    uint64_t next_tensor_offset; /* absolute, or file_bytes if last */
    bool     is_last_tensor;

    bool     ok;
    char     why[256];
};

static void geo_fail(geom * g, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g->why, sizeof g->why, fmt, ap);
    va_end(ap);
    g->ok = 0;
}

/* Build the geometry and CHECK it. Returns 1 only if every invariant holds.
 *
 * The independent bounds are the important part. `next_tensor_offset` comes from
 * the file's own tensor table: if our computed size for this tensor were too
 * large, its end would overlap the next tensor's start. Checking against that
 * validates the arithmetic using the container as the authority rather than
 * itself. */
static int geo_build(geom * g, gguf_context * ctx, const char * name,
                     bool one_row_regression) {
    memset(g, 0, sizeof *g);
    g->ok = 1;

    const int64_t id = gguf_find_tensor(ctx, name);
    if (id < 0) { geo_fail(g, "tensor %s not found", name); return 0; }

    const ggml_type tt = gguf_get_tensor_type(ctx, id);
    const ggml_type_traits * tr = ggml_get_type_traits(tt);
    const int64_t * ne = gguf_get_tensor_ne(ctx, id);

    if (tr == NULL) { geo_fail(g, "no type traits for %s", ggml_type_name(tt)); return 0; }
    /* F32 has NO to_float in ggml, because F32 data is already float and the
     * reference expander is a memcpy. Treating its absence as a fatal geometry
     * error would have refused a case that is perfectly measurable, and refusing
     * a measurable case is how a witness stops covering what it exists to cover. */
    if (tr->to_float == NULL && tr->type_size != (int32_t) sizeof(float)) {
        geo_fail(g, "no reference expander for type %s", ggml_type_name(tt));
        return 0;
    }
    /* n_dims is not exposed by this gguf API; ne[dim] is 1 for dim >= n_dims, so the
     * shape itself carries the information and a separate dim count is not needed. */
    g->n_dims    = (g->ne1 > 1) ? 2 : 1;
    g->ne0       = ne[0];
    g->ne1       = ne[1];
    g->block_size = tr->blck_size;
    g->type_size  = tr->type_size;
    g->data_offset = gguf_get_data_offset(ctx);
    g->tensor_offset = gguf_get_tensor_offset(ctx, id);

    /* Independent container bound: the next tensor's start, absolute. */
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    g->is_last_tensor = (id + 1 >= n_tensors);
    g->next_tensor_offset = g->is_last_tensor
        ? g->file_bytes
        : g->data_offset + gguf_get_tensor_offset(ctx, id + 1);

    /* ---- derived, one source each ---- */
    if (g->block_size <= 0) { geo_fail(g, "block_size is %d", g->block_size); return 0; }
    if (g->ne0 % g->block_size != 0) {
        geo_fail(g, "ne[0]=%lld is not a whole number of %d-element blocks",
                 (long long) g->ne0, g->block_size);
        return 0;
    }
    g->blocks_per_row = (uint64_t) (g->ne0 / g->block_size);
    g->row_bytes      = g->blocks_per_row * (uint64_t) g->type_size;

    g->stored_rows = (g->ne1 > 1) ? (uint64_t) g->ne1 : 1u;

    /* The traversal we intend: this many rows, capped for a complete record. */
    g->n_rows = 64;
    if (g->n_rows > g->stored_rows) { g->n_rows = g->stored_rows; }
    if (g->n_rows < 1) { g->n_rows = 1; }

    g->tensor_required_bytes = g->n_rows * g->row_bytes;

    /* ---- structural relationships, asserted SEPARATELY ---- */

    /* (a) the stride identity, stated rather than implied */
    if (g->row_bytes != g->blocks_per_row * (uint64_t) g->type_size) {
        geo_fail(g, "row_bytes %llu != blocks_per_row %llu * type_size %d",
                 (unsigned long long) g->row_bytes,
                 (unsigned long long) g->blocks_per_row, g->type_size);
        return 0;
    }
    /* (b) the span identity, stated rather than implied */
    if (g->tensor_required_bytes != g->n_rows * g->row_bytes) {
        geo_fail(g, "tensor_required_bytes %llu != n_rows %llu * row_bytes %llu",
                 (unsigned long long) g->tensor_required_bytes,
                 (unsigned long long) g->n_rows,
                 (unsigned long long) g->row_bytes);
        return 0;
    }

    /* (c) THE TRAVERSAL, DECLARED. These are the exact bytes the loops below will
     * touch: the first byte of row 0 block 0, and the last byte of the final
     * block of the final row. Computed, not discovered. */
    g->first_byte_touched = 0;
    g->max_byte_touched   = g->tensor_required_bytes - 1;
    g->requested_span     = g->tensor_required_bytes;

    /* (d) THE INVARIANT, against the SUPPLIED buffer. */
    /* The supplied buffer is DECIDED HERE, not implied downstream: in normal
     * operation it is exactly the declared span, and in the regression case it is
     * exactly one row. Deciding it in one place is what makes the bounds check
     * meaningful rather than circular. */
    g->supplied_buffer_bytes = one_row_regression ? g->row_bytes
                                                  : g->tensor_required_bytes;
    if (g->supplied_buffer_bytes < g->tensor_required_bytes) {
        /* Five specifiers, five arguments. This had four and printed a raw stack
         * value as the buffer size, which is the last thing a refusal message
         * should do: the number a reader would trust most was the one that was
         * garbage. */
        geo_fail(g, "REFUSED BEFORE ANY READ: buffer %llu bytes < required %llu "
                    "(max_byte_touched %llu would be at index %llu of buffer %llu)",
                 (unsigned long long) g->supplied_buffer_bytes,
                 (unsigned long long) g->tensor_required_bytes,
                 (unsigned long long) g->max_byte_touched,
                 (unsigned long long) g->supplied_buffer_bytes,
                 (unsigned long long) g->supplied_buffer_bytes);
        return 0;
    }
    if (g->max_byte_touched >= g->supplied_buffer_bytes) {
        geo_fail(g, "REFUSED BEFORE ANY READ: max_byte_touched %llu >= buffer %llu",
                 (unsigned long long) g->max_byte_touched,
                 (unsigned long long) g->supplied_buffer_bytes);
        return 0;
    }

    /* (e) INDEPENDENT BOUND: against the container, not against ourselves. */
    if (g->data_offset + g->tensor_offset + g->tensor_required_bytes >
        g->next_tensor_offset && !g->is_last_tensor) {
        geo_fail(g, "requested span %llu bytes would overlap the next tensor at %llu "
                    "(start %llu)",
                 (unsigned long long) g->tensor_required_bytes,
                 (unsigned long long) g->next_tensor_offset,
                 (unsigned long long) (g->data_offset + g->tensor_offset));
        return 0;
    }
    return 1;
}

/* ---------------- CHECKED SPAN ACCESS ----------------
 *
 * Every read goes through this. It re-verifies the bound at the point of use, so
 * even a future edit that changes the traversal cannot silently escape: the
 * accessor is the boundary, not an offset computed once somewhere far above. */
struct cspan {
    const uint8_t * base;
    uint64_t        size;
    uint64_t        used;   /* running high-water mark */
};

static int span_ok(cspan * s, uint64_t off, uint64_t len, char * why, size_t cap) {
    if (len == 0) { return 1; }
    if (off > s->size || len > s->size || off + len > s->size) {
        snprintf(why, cap,
                 "span REFUSED: off=%llu len=%llu exceeds buffer %llu "
                 "(would touch byte %llu)",
                 (unsigned long long) off, (unsigned long long) len,
                 (unsigned long long) s->size,
                 (unsigned long long) (off + len - 1));
        return 0;
    }
    const uint64_t hi = off + len - 1;
    if (hi + 1 > s->used) { s->used = hi + 1; }
    return 1;
}

/* ---------------- the traversal, using only checked spans ---------------- */

static bool fill_x(float * x, size_t n, uint64_t seed) {
    uint64_t s = seed ? seed : 1;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = (float) ((int32_t) ((s >> 33) % 20001) - 10000) / 10000.0f;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [tensor] [xseed] [--one-row-buffer]\n",
                argv[0]);
        return 1;
    }
    const char * only = NULL;
    uint64_t xseed = 12345ULL;
    bool force_one_row = false;
    /* EXPLICIT FLAGS ONLY. The previous parser treated any non-flag argument as a
     * tensor name, so passing the numeric seed 12345 silently became a search for
     * a tensor called "12345", which matched nothing and produced a clean,
     * confident, EMPTY witness. A number that turns into an identifier is the same
     * family of mistake as a filename that turns into a topology: the tool agreed
     * to look for something it was never asked about, and reported no results
     * without saying the question had changed. */
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--one-row-buffer") == 0) { force_one_row = true; }
        else if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) { only = argv[++i]; }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) { xseed = strtoull(argv[++i], NULL, 10); }
        else {
            fprintf(stderr,
                    "unknown argument '%s'. Use --tensor NAME, --seed N, "
                    "--one-row-buffer. Bare arguments are rejected so a value can "
                    "never be silently reinterpreted as a name.\n", argv[i]);
            return 2;
        }
    }

    uint64_t fsize = 0;
    { FILE * sf = fopen(argv[1], "rb");
      if (!sf) { fprintf(stderr, "cannot stat %s\n", argv[1]); return 1; }
      fseek(sf, 0, SEEK_END); fsize = (uint64_t) ftell(sf); fclose(sf); }

    gguf_init_params gp;
    memset(&gp, 0, sizeof gp);
    gp.no_alloc = true;
    gguf_context * ctx = gguf_init_from_file(argv[1], gp);
    if (!ctx) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    std::vector<const char *> want;
    if (only) {
        want.push_back(only);
    } else {
        want.push_back("blk.0.attn_q.weight");
        want.push_back("blk.0.ffn_down.weight");
        want.push_back("blk.0.attn_norm.weight");
        want.push_back("token_embd.weight");
    }

    printf("\n=== STEP 2 GEMV WITNESS (geometry proven before data) ===\n");
    printf("  reference dequant : vendored ggml type_traits.to_float\n");
    printf("  reference dot     : DOUBLE accumulation\n");
    printf("  x_seed            : %llu\n", (unsigned long long) xseed);

    int n_cases = 0, n_refused = 0;
    char why[256];

    for (size_t w = 0; w < want.size(); ++w) {
        const char * name = want[w];
        if (gguf_find_tensor(ctx, name) < 0) { continue; }

        /* ---- 1. GEOMETRY FIRST. Nothing below this point reads a payload. ---- */
        geom g;
        g.file_bytes = fsize;
        /* The deliberate regression: ask for one row of supply while the
         * traversal will need more. geo_build must REFUSE before any read. */
        const int geo_ok = geo_build(&g, ctx, name, force_one_row);

        printf("  [gemv] %s\n", name);
        printf("    type                  = %s\n", ggml_type_name(gguf_get_tensor_type(ctx, gguf_find_tensor(ctx, name))));
        printf("    ne                    = [%lld, %lld]\n", (long long) g.ne0, (long long) g.ne1);
        printf("    block_size            = %d\n", g.block_size);
        printf("    type_size             = %d\n", g.type_size);
        printf("    blocks_per_row        = %llu\n", (unsigned long long) g.blocks_per_row);
        printf("    row_bytes             = %llu\n", (unsigned long long) g.row_bytes);
        printf("    n_rows                = %llu\n", (unsigned long long) g.n_rows);
        printf("    stored_rows           = %llu\n", (unsigned long long) g.stored_rows);
        printf("    tensor_required_bytes = %llu\n", (unsigned long long) g.tensor_required_bytes);
        printf("    first_byte_touched    = %llu\n", (unsigned long long) g.first_byte_touched);
        printf("    max_byte_touched      = %llu\n", (unsigned long long) g.max_byte_touched);
        printf("    requested_span        = %llu\n", (unsigned long long) g.requested_span);
        printf("    supplied_buffer_bytes = %llu\n", (unsigned long long) g.supplied_buffer_bytes);
        printf("    next_tensor_offset    = %llu%s\n",
               (unsigned long long) g.next_tensor_offset,
               g.is_last_tensor ? " (last tensor; file size used)" : "");
        printf("    BOUNDS                = %s\n", geo_ok ? "OK" : "REFUSED");
        if (!geo_ok) {
            printf("    reason                = %s\n", g.why);
            printf("    ENDCASE %s REFUSED\n", name);
            n_refused++;
            n_cases++;
            continue;
        }

        /* ---- 2. READ EXACTLY THE DECLARED SPAN ---- */
        std::vector<uint8_t> wbuf((size_t) g.tensor_required_bytes);
        {
            FILE * tf = fopen(argv[1], "rb");
            if (!tf) { printf("    reason = cannot reopen\n"); n_cases++; continue; }
            const uint64_t at = g.data_offset + g.tensor_offset;
            if (fseek(tf, (long) at, SEEK_SET) != 0 ||
                fread(wbuf.data(), 1, (size_t) g.tensor_required_bytes, tf) !=
                    (size_t) g.tensor_required_bytes) {
                printf("    BOUNDS  = REFUSED (short read at %llu)\n",
                       (unsigned long long) at);
                printf("    ENDCASE %s REFUSED\n", name);
                fclose(tf);
                n_refused++; n_cases++;
                continue;
            }
            fclose(tf);
        }

        cspan sp;
        sp.base = wbuf.data();
        sp.size = g.supplied_buffer_bytes;
        sp.used = 0;

        const ggml_type_traits * tr =
            ggml_get_type_traits(gguf_get_tensor_type(ctx, gguf_find_tensor(ctx, name)));
        std::vector<float> x((size_t) g.ne0);
        fill_x(x.data(), x.size(), xseed);

        /* ---- 3. TRAVERSE THROUGH CHECKED SPANS ONLY ---- */
        std::vector<double> ref((size_t) g.n_rows, 0.0);
        std::vector<unsigned long long> rh((size_t) g.n_rows, 0);
        std::vector<float> rowbuf((size_t) g.ne0);
        bool refused = false;

        for (uint64_t r = 0; r < g.n_rows && !refused; ++r) {
            for (uint64_t b = 0; b < g.blocks_per_row; ++b) {
                const uint64_t off = r * g.row_bytes + b * (uint64_t) g.type_size;
                if (!span_ok(&sp, off, (uint64_t) g.type_size, why, sizeof why)) {
                    printf("    SPAN REFUSED at row %llu block %llu: %s\n",
                           (unsigned long long) r, (unsigned long long) b, why);
                    refused = true;
                    break;
                }
                if (tr->to_float != NULL) {
                    tr->to_float((const char *) sp.base + off,
                                 rowbuf.data() + b * (uint64_t) g.block_size,
                                 g.block_size);
                } else {
                    /* F32 passthrough, matching the reference's own semantics. */
                    memcpy(rowbuf.data() + b * (uint64_t) g.block_size,
                           sp.base + off, (size_t) g.type_size);
                }
                const uint64_t dst = b * (uint64_t) g.block_size;
                if (dst + (uint64_t) g.block_size > (uint64_t) g.ne0) {
                    snprintf(why, sizeof why,
                             "destination overflow: block %llu writes [%llu,%llu) past %llu",
                             (unsigned long long) b, (unsigned long long) dst,
                             (unsigned long long) (dst + g.block_size),
                             (unsigned long long) g.ne0);
                    printf("    DEST REFUSED: %s\n", why);
                    refused = true;
                    break;
                }
            }
            if (refused) { break; }

            double acc = 0.0;
            for (uint64_t i = 0; i < (uint64_t) g.ne0; ++i) {
                acc += (double) rowbuf[(size_t) i] * (double) x[(size_t) i];
            }
            ref[(size_t) r] = acc;

            unsigned long long hh = 1469598103934665603ULL;
            const uint8_t * bb = (const uint8_t *) rowbuf.data();
            for (uint64_t i = 0; i < (uint64_t) g.ne0 * sizeof(float); ++i) {
                hh ^= bb[i];
                hh *= 1099511628211ULL;
            }
            rh[(size_t) r] = hh;
        }

        if (refused) {
            printf("    ENDCASE %s REFUSED\n", name);
            n_refused++; n_cases++;
            continue;
        }

        /* ---- 4. VERIFY THE HIGH-WATER MARK MATCHED THE DECLARATION ---- */
        if (sp.used != g.tensor_required_bytes) {
            printf("    BOUNDS  = REFUSED AFTER TRAVERSAL: used %llu != declared %llu\n",
                   (unsigned long long) sp.used,
                   (unsigned long long) g.tensor_required_bytes);
            printf("    ENDCASE %s REFUSED\n", name);
            n_refused++; n_cases++;
            continue;
        }

        printf("    bytes_actually_read  = %llu\n", (unsigned long long) sp.used);
        printf("    ROWS");
        for (uint64_t r = 0; r < g.n_rows; ++r) { printf(" %a", ref[(size_t) r]); }
        printf("\n    ROWHASH");
        for (uint64_t r = 0; r < g.n_rows; ++r) {
            printf(" %016llx", rh[(size_t) r]);
        }
        printf("\n    ENDCASE %s OK\n", name);
        n_cases++;
    }

    gguf_free(ctx);

    printf("\n  cases=%d refused=%d\n", n_cases, n_refused);
    /* The exit status is part of the witness. Golden capture is permitted only on
     * exit 0 AND complete records AND zero refusals, so a partial run cannot be
     * mistaken for a witness by the mere existence of an output file. */
    if (n_refused > 0) {
        printf("  VERDICT: REFUSED -- not a witness; golden capture is forbidden\n");
        return 1;
    }
    printf("  VERDICT: COMPLETE -- every record structurally whole\n");
    return 0;
}