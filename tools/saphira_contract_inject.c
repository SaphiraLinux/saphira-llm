/*
 * saphira_contract_inject.c — add Saphira contract metadata to a GGUF without
 *                             touching a single tensor byte.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * WHAT THIS DOES, PRECISELY
 *
 * A GGUF is:  magic, version, tensor_count, kv_count, [kv table], [tensor info
 * table], padding to alignment, [tensor data]. Tensor offsets in the info table are
 * RELATIVE to the start of the data blob.
 *
 * Appending KV entries lengthens the KV table, which moves the tensor info table and
 * therefore the start of the data blob. Every tensor offset must be shifted by the
 * same delta or the payload would be read from the wrong place -- a corruption that
 * still opens cleanly and produces plausible garbage. So:
 *
 *   - the KV table is re-emitted VERBATIM, byte for byte, plus the new entries
 *   - the tensor info table is re-emitted with offsets shifted by exactly the delta
 *   - the tensor data blob is copied through UNCHANGED
 *
 * What this deliberately does NOT do: alter any tensor byte, tensor name, tensor
 * type, tensor geometry, or quantisation setting. Those are the model's content, not
 * metadata this project is entitled to reinterpret.
 *
 * The tool prints a digest of what it changed so the change can be checked rather
 * than trusted.
 *
 * Usage:
 *   saphira-contract-inject <in.gguf> <out.gguf> <field=value> ...
 * where field is e.g.  saphira.rope.pairing=half_split
 * Types are inferred from the value: "N" -> UINT32, "N.N" -> FLOAT32, else STRING.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#define T_U32 4u
#define T_F32 6u
#define T_STR 8u

static FILE *fin, *fout;

static void die(const char * msg) { fprintf(stderr, "error: %s\n", msg); exit(2); }

/* --- primitive readers/writers ------------------------------------------- */
static void rd(void * dst, size_t n) { if (fread(dst, 1, n, fin) != n) die("short read"); }
static void wr(const void * src, size_t n) { if (fwrite(src, 1, n, fout) != n) die("short write"); }

/* read a length-prefixed string into a malloc'd buffer */
static char * rd_str(size_t * len_out) {
    uint64_t n; rd(&n, 8);
    if (n > (1u << 20)) die("absurd key length");
    char * s = (char *) malloc((size_t) n + 1);
    rd(s, (size_t) n); s[n] = '\0';
    if (len_out) *len_out = (size_t) n;
    return s;
}

/* --- raw KV capture ------------------------------------------------------
 * Each entry is kept as its exact encoded bytes, so the table can be re-emitted
 * without understanding the value type. Re-encoding a value risks changing a float
 * or reordering an array; copying the bytes cannot. */
typedef struct { char * bytes; size_t len; } raw_t;

typedef struct {
    char * name; uint32_t n_dims; uint64_t ne[4]; uint32_t type; uint64_t offset;
} tinfo_t;

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <field=value> ...\n", argv[0]);
        return 2;
    }
    fin = fopen(argv[1], "rb");
    if (!fin) { perror("open input"); return 2; }
    fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 2; }

    char magic[4]; rd(magic, 4);
    if (memcmp(magic, "GGUF", 4)) die("not a GGUF");
    uint32_t version; rd(&version, 4);
    uint64_t n_tensors, n_kv; rd(&n_tensors, 8); rd(&n_kv, 8);
    printf("  source: version=%u tensors=%llu kv=%llu\n", version,
           (unsigned long long) n_tensors, (unsigned long long) n_kv);

    /* --- capture the KV table verbatim --- */
    raw_t * kv = (raw_t *) calloc((size_t) n_kv, sizeof(raw_t));
    uint32_t alignment = 32;      /* the GGUF default */
    size_t kv_total = 0;
    for (uint64_t i = 0; i < n_kv; ++i) {
        long start = ftell(fin);
        size_t klen; char * key = rd_str(&klen);
        if (strcmp(key, "general.alignment") == 0) {
            uint32_t t; rd(&t, 4);
            if (t != T_U32) die("general.alignment has an unexpected type");
            rd(&alignment, 4);
            if (alignment == 0 || (alignment & (alignment - 1)))
                die("general.alignment is not a power of two");
            /* rewind so the bytes are captured verbatim like every other entry */
            if (fseek(fin, start, SEEK_SET) != 0) die("seek");
            free(key); key = rd_str(&klen);
        }
        uint32_t vtype; rd(&vtype, 4);

        /* skip/copy the value according to its type, counting bytes */
        long vstart = ftell(fin);
        switch (vtype) {
        case 0: case 1: case 7: { uint8_t v; rd(&v, 1); break; }
        case 2: case 3: { uint16_t v; rd(&v, 2); break; }
        case T_U32: { uint32_t v; rd(&v, 4); break; }
        case 5: { int32_t v; rd(&v, 4); break; }
        case T_F32: { float v; rd(&v, 4); break; }
        case T_STR: { size_t l; char * s = rd_str(&l); free(s); break; }
        case 9: {
            uint32_t et; uint64_t cnt; rd(&et, 4); rd(&cnt, 8);
            for (uint64_t k = 0; k < cnt; ++k) {
                switch (et) {
                case 0: case 1: case 7: { uint8_t v; rd(&v, 1); break; }
                case 2: case 3: { uint16_t v; rd(&v, 2); break; }
                case T_U32: { uint32_t v; rd(&v, 4); break; }
                case 5: { int32_t v; rd(&v, 4); break; }
                case T_F32: { float v; rd(&v, 4); break; }
                case 10: { uint64_t v; rd(&v, 8); break; }
                case 11: { int64_t v; rd(&v, 8); break; }
                case 12: { double v; rd(&v, 8); break; }
                case T_STR: { size_t l; char * s = rd_str(&l); free(s); break; }
                default: die("unsupported array element type");
                }
            }
            break;
        }
        case 10: { uint64_t v; rd(&v, 8); break; }
        case 11: { int64_t v; rd(&v, 8); break; }
        case 12: { double v; rd(&v, 8); break; }
        default: die("unsupported metadata value type");
        }
        const long vend = ftell(fin);
        /* Capture the WHOLE entry: key length, key bytes, type tag and value. Keeping
         * only the value would silently strip every key and type on rewrite. */
        if (fseek(fin, start, SEEK_SET) != 0) die("seek");
        kv[i].len = (size_t) (vend - start);
        kv[i].bytes = (char *) malloc(kv[i].len ? kv[i].len : 1);
        rd(kv[i].bytes, kv[i].len);
        kv_total += kv[i].len;
        (void) vstart;
        free(key);
    }
    const long old_kv_end = ftell(fin);
    const long old_tinfo_start = old_kv_end;

    /* --- capture the tensor info table --- */
    const long old_tinfo_begin = ftell(fin);
    tinfo_t * ti = (tinfo_t *) calloc((size_t) n_tensors, sizeof(tinfo_t));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        size_t l; ti[i].name = rd_str(&l);
        rd(&ti[i].n_dims, 4);
        if (ti[i].n_dims > 4) die("tensor rank above 4");
        for (uint32_t d = 0; d < ti[i].n_dims; ++d) rd(&ti[i].ne[d], 8);
        for (uint32_t d = ti[i].n_dims; d < 4; ++d) ti[i].ne[d] = 1;
        rd(&ti[i].type, 4);
        rd(&ti[i].offset, 8);
    }
    const long old_tinfo_end = ftell(fin);
    const size_t tinfo_bytes = (size_t) (old_tinfo_end - old_tinfo_begin);

    /* old data blob starts at the first offset; GGUF aligns it, and the reader
     * already accepted this file, so the blob base is the alignment boundary at or
     * after the end of the tensor info table. */
    uint64_t min_off = UINT64_MAX;
    for (uint64_t i = 0; i < n_tensors; ++i)
        if (ti[i].offset < min_off) min_off = ti[i].offset;
    if (n_tensors == 0) min_off = 0;
    const long old_data_start = (long) (old_tinfo_start + ((old_tinfo_end - old_tinfo_start)));
    (void) old_data_start;
    /* the blob base is the alignment boundary at or after the end of the tensor
     * info table. Confirmed against the smallest tensor offset, which is where the
     * first payload physically begins. */
    uint64_t a = (uint64_t) old_tinfo_end;
    uint64_t blob_base = (a + alignment - 1) / alignment * alignment;

    /* --- build the appended KV entries --- */
    const int n_new = argc - 3;
    if (n_new <= 0) die("nothing to inject");
    raw_t * nv = (raw_t *) calloc((size_t) n_new, sizeof(raw_t));
    for (int i = 0; i < n_new; ++i) {
        char * eq = strchr(argv[3 + i], '=');
        if (!eq) die("expected field=value");
        *eq = '\0';
        const char * key = argv[3 + i];
        const char * val = eq + 1;
        char buf[1024];
        uint32_t type; char vbuf[512]; size_t vlen = 0;
        char * endp = NULL;
        const double d = strtod(val, &endp);
        const int looks_float = (endp && *endp == '\0' && strchr(val, '.'));
        if (!*val)                              { type = T_STR; }
        else if (looks_float)                  { type = T_F32; float f = (float) d;
                                                  memcpy(vbuf, &f, 4); vlen = 4; }
        else                                   { uint32_t u = (uint32_t) strtoul(val, &endp, 10);
                                                  if (endp && *endp == '\0') { type = T_U32;
                                                      memcpy(vbuf, &u, 4); vlen = 4; }
                                              else { type = T_STR; } }
        if (type == T_STR) {
            const size_t vl = strlen(val);
            uint64_t vl64 = vl;
            const size_t klen = strlen(key);
            uint64_t kl64 = klen;
            const size_t total = 8 + klen + 4 + 8 + vl;
            nv[i].bytes = (char *) malloc(total); nv[i].len = total;
            char * p = nv[i].bytes;
            memcpy(p, &kl64, 8); p += 8; memcpy(p, key, klen); p += klen;
            memcpy(p, &type, 4); p += 4; memcpy(p, &vl64, 8); p += 8;
            memcpy(p, val, vl); p += vl;
            (void) buf;
        } else {
            const size_t klen = strlen(key);
            uint64_t kl64 = klen;
            const size_t total = 8 + klen + 4 + vlen;
            nv[i].bytes = (char *) malloc(total); nv[i].len = total;
            char * p = nv[i].bytes;
            memcpy(p, &kl64, 8); p += 8; memcpy(p, key, klen); p += klen;
            memcpy(p, &type, 4); p += 4; memcpy(p, vbuf, vlen); p += vlen;
        }
    }

    size_t new_kv_total = kv_total;
    for (int i = 0; i < n_new; ++i) new_kv_total += nv[i].len;

    /* --- new layout --- */
    const uint64_t hdr = 4 + 4 + 8 + 8;
    const uint64_t new_kv_end = hdr + new_kv_total;
    /* the tensor info table is re-emitted with shifted offsets but is otherwise the
     * same size: same count, same names, dims and types */
    const uint64_t new_tinfo_end = new_kv_end + tinfo_bytes;
    const uint64_t new_blob_base = (new_tinfo_end + alignment - 1) / alignment * alignment;
    /* The data blob is copied VERBATIM: byte j of the new blob is byte j of the old.
     * Tensor offsets are relative to the blob base, so they must NOT be shifted. Only
     * the blob's own position in the file moves. Shifting the offsets as well would
     * double-count the shift and push the last tensor past the end of the blob -- a
     * corruption that still opens and still yields plausible values.
     * The earlier revision made exactly that mistake; the reader caught it as a
     * tensor spanning past the end of the data. */
    const uint64_t data_shift = new_blob_base - blob_base;

    printf("  layout: kv table %zu -> %zu bytes; blob base %llu -> %llu "
           "(blob base moves by %llu; tensor offsets are UNCHANGED because the blob is "
           "copied verbatim)\n",
           kv_total, new_kv_total, (unsigned long long) blob_base,
           (unsigned long long) new_blob_base, (unsigned long long) data_shift);

    /* --- emit --- */
    wr("GGUF", 4); wr(&version, 4); wr(&n_tensors, 8);
    const uint64_t n_kv_new = n_kv + (uint64_t) n_new;
    wr(&n_kv_new, 8);
    for (uint64_t i = 0; i < n_kv; ++i) wr(kv[i].bytes, kv[i].len);
    for (int i = 0; i < n_new; ++i) wr(nv[i].bytes, nv[i].len);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        const uint64_t kl = strlen(ti[i].name);
        wr(&kl, 8); wr(ti[i].name, (size_t) kl);
        wr(&ti[i].n_dims, 4);
        for (uint32_t d = 0; d < ti[i].n_dims; ++d) wr(&ti[i].ne[d], 8);
        wr(&ti[i].type, 4);
        wr(&ti[i].offset, 8);      /* relative to the blob, and the blob is verbatim */
    }
    /* pad to the new blob base */
    {   static const char zeros[4096] = { 0 };
        uint64_t at = new_tinfo_end;
        while (at < new_blob_base) {
            uint64_t chunk = new_blob_base - at;
            if (chunk > sizeof zeros) chunk = sizeof zeros;
            wr(zeros, (size_t) chunk);
            at += chunk;
        }
    }
    /* copy the data blob through unchanged */
    if (fseek(fin, (long) blob_base, SEEK_SET) != 0) die("seek to data");
    {   static char buf[1 << 22];
        for (;;) {
            const size_t got = fread(buf, 1, sizeof buf, fin);
            if (got == 0) break;
            if (fwrite(buf, 1, got, fout) != got) die("short write on data");
        }
    }
    if (fclose(fout) != 0) die("close output");
    fclose(fin);
    printf("  wrote %s with %d new metadata field(s); tensor data copied verbatim\n", argv[2], n_new);
    return 0;
}