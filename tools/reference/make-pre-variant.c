/*
 * make-pre-variant — produce a model whose pre-tokeniser declaration differs.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * A fixture-preparation tool. It is not part of the runtime and never ships.
 *
 * It rewrites a GGUF's metadata, adding or replacing `tokenizer.ggml.pre`, and
 * copies the tensor blob through unchanged. Nothing else is touched: the
 * vocabulary, the merges, the token types and every weight are exactly the
 * source model's.
 *
 * It exists to settle one question with evidence rather than assertion:
 *
 *     is the pre-tokeniser a property of the model file, or a constant in the
 *     tokenizer?
 *
 * If the same vocabulary tokenises differently when one metadata key changes,
 * then it is model metadata, and a runtime that hardcodes either behaviour is
 * wrong for some model. That matters before Phase 7, where ordinary GGUF models
 * each carry their own `pre` value, and it cannot be demonstrated with the
 * models on hand, because the only one available declares no `pre` at all.
 *
 * Usage:
 *   make-pre-variant <in.gguf> <pre-value|-> <out.gguf>
 *
 * A `pre-value` of "-" removes the key, reproducing the DEFAULT fallback.
 */

#include "saphira_llm/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>

#define KEY_PRE "tokenizer.ggml.pre"

static void die(const char * msg) {
    fprintf(stderr, "make-pre-variant: %s\n", msg);
    exit(1);
}

static void die2(const char * msg, const char * detail) {
    fprintf(stderr, "make-pre-variant: %s: %s\n", msg, detail);
    exit(1);
}

static void w32(FILE * f, uint32_t v) { if (fwrite(&v, 4, 1, f) != 1) { die("write failed"); } }
static void w64(FILE * f, uint64_t v) { if (fwrite(&v, 8, 1, f) != 1) { die("write failed"); } }

static void wstr(FILE * f, const char * s) {
    const size_t n = strlen(s);
    w64(f, (uint64_t) n);
    if (n && fwrite(s, 1, n, f) != n) { die("write failed"); }
}

static void wblob(FILE * f, const void * p, size_t n) {
    if (n && fwrite(p, 1, n, f) != n) { die("write failed"); }
}

static void wkv(FILE * f, const char * key, const sllm_gguf_kv * kv) {
    wstr(f, key);
    w32(f, (uint32_t) kv->type);

    if (kv->type == SLLM_VT_STRING) {
        /*
         * A scalar string carries no meaningful element size -- the parser
         * records 1 -- so it must be written with its own length prefix. Taking
         * elem_size at face value writes a single byte and desynchronises the
         * whole key-value stream, which the next reader reports as an absurd
         * string length several keys later.
         */
        wstr(f, (const char *) kv->data);
        return;
    }

    if (kv->type != SLLM_VT_ARRAY) {
        wblob(f, kv->data, kv->elem_size);
        return;
    }

    w32(f, (uint32_t) kv->arr_type);
    w64(f, kv->n);

    if (kv->arr_type == SLLM_VT_STRING) {
        const char * const * strs = (const char * const *) kv->data;
        for (uint64_t i = 0; i < kv->n; ++i) {
            wstr(f, strs[i]);
        }
        return;
    }
    wblob(f, kv->data, (size_t) kv->n * kv->elem_size);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        die("usage: make-pre-variant <in.gguf> <pre-value|-> <out.gguf>");
    }
    const char * in_path  = argv[1];
    const char * pre      = argv[2];
    const char * out_path = argv[3];
    const bool  remove    = (strcmp(pre, "-") == 0);

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(in_path, &g, err, sizeof(err)) != SLLM_OK) {
        die2("cannot read source model", err);
    }

    /* Does the source already declare one? If so it is replaced, so the tool
     * is idempotent and a variant can be regenerated from either side. */
    bool had_pre = false;
    for (uint64_t i = 0; i < g.n_kv; ++i) {
        if (strcmp(g.kv[i].key, KEY_PRE) == 0) { had_pre = true; break; }
    }

    uint64_t alignment = 32;
    {
        const sllm_gguf_kv * a = sllm_gguf_find_kv(&g, "general.alignment");
        if (a != NULL && a->type == SLLM_VT_UINT32) {
            alignment = *(const uint32_t *) a->data;
        }
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        die2("general.alignment is not a power of two", "refusing to write a file the reader would reject");
    }

    /* An empty string is how the key is written when we add one; an explicit
     * empty value is not the same as an absent key, so "-" removes instead. */
    sllm_gguf_kv pre_kv;
    memset(&pre_kv, 0, sizeof(pre_kv));
    pre_kv.type = SLLM_VT_STRING;
    pre_kv.n    = 1;
    pre_kv.elem_size = strlen(pre) + 1;
    pre_kv.data = (void *) pre;

    const uint64_t n_kv_out = g.n_kv + (remove ? 0 : (had_pre ? 0 : 1));

    FILE * f = fopen(out_path, "wb");
    if (f == NULL) { die2("cannot create output", out_path); }

    w32(f, 0x46554747u);   /* 'GGUF' */
    w32(f, g.version);
    w64(f, g.n_tensors);
    w64(f, n_kv_out);

    for (uint64_t i = 0; i < g.n_kv; ++i) {
        if (strcmp(g.kv[i].key, KEY_PRE) == 0) { continue; }   /* drop, if present */
        wkv(f, g.kv[i].key, &g.kv[i]);
    }
    if (!remove && !had_pre) {
        wkv(f, KEY_PRE, &pre_kv);
    }

    for (uint64_t i = 0; i < g.n_tensors; ++i) {
        const sllm_gguf_tensor * t = &g.tensors[i];
        wstr(f, t->name);
        w32(f, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; ++d) {
            w64(f, t->ne[d]);
        }
        w32(f, (uint32_t) t->type);
        w64(f, t->offset);
    }

    /* Pad to the alignment the reader will insist on, then copy the blob
     * verbatim. The blob's internal offsets are relative to its own start, so
     * they survive the move unchanged. */
    const long pos = ftell(f);
    if (pos < 0) { die("ftell failed"); }
    const long pad = (long) ((alignment - ((uint64_t) pos % alignment)) % alignment);
    for (long i = 0; i < pad; ++i) { fputc(0, f); }

    /* A separate handle for the source. Seeking the output file back to the
     * source's data offset would read the bytes just written into it, which
     * fails as a short read and looks like a truncated input file. */
    FILE * src = fopen(in_path, "rb");
    if (src == NULL) { die2("cannot reopen the source model", in_path); }
    if (fseek(src, (long) g.data_offset, SEEK_SET) != 0) { die("cannot seek to the data blob"); }
    {
        size_t left = (size_t) g.data_size;
        char buf[1 << 16];
        while (left > 0) {
            const size_t want = left < sizeof(buf) ? left : sizeof(buf);
            const size_t got = fread(buf, 1, want, src);
            if (got != want) { die2("short read on the data blob", in_path); }
            if (fwrite(buf, 1, got, f) != got) { die("write failed"); }
            left -= got;
        }
    }
    fclose(src);

    if (fclose(f) != 0) { die("close failed"); }
    const uint64_t n_tensors_out = g.n_tensors;
    const uint64_t data_size_out  = g.data_size;
    (void) data_size_out;
    sllm_gguf_close(&g);

    printf("wrote %s: %" PRIu64 " tensors, %" PRIu64 " kv, %s\n",
           out_path, n_tensors_out, n_kv_out,
           remove ? "pre-tokeniser key removed (DEFAULT)"
                  : (had_pre ? "pre-tokeniser key replaced" : "pre-tokeniser key added"));
    return 0;
}
