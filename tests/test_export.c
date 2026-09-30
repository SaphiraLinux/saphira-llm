/*
 * test_export.c — the I2_S export contract, pinned.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA Limited.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Why this file exists, in one sentence: a 64-wide layer exported cleanly,
 * loaded cleanly, passed every shape check, and then computed nothing, and
 * nothing in the project could see it because every shipped model is
 * 2048-wide.
 *
 * The I2_S block is 128 elements in 32 bytes (SLLM_I2S_QK), and the runtime
 * imposes two hard requirements on a row: sllm_i2s_gemv strides rows by n/4,
 * and sllm_i2s_dot returns 0 outright when n < 128. A row is therefore only
 * consumable when its input width is a multiple of 128. For those widths this
 * exporter's flattened pack and the gemv's row-wise read are the same bytes,
 * which is why every shipped tensor is unaffected. For narrower rows there is
 * no packing that works, so the exporter refuses rather than emitting a model
 * that loads and lies.
 *
 * The three properties pinned here are deliberately separate, because they fail
 * independently:
 *
 *   1. byte stability -- a change to the writer must not move a single byte of
 *      an aligned export. Checked against fingerprints captured from the writer
 *      as it stood, not regenerated from the code under test.
 *   2. refusal -- narrow widths are rejected, with no file left behind.
 *   3. equality -- the bytes in the file, decoded the way the runtime decodes
 *      them, are the trainer's quantised weights, code for code and bit for
 *      bit on the scale.
 */
#include <saphira_llm/forward.h>
#include <saphira_llm/gguf.h>
#include <saphira_llm/i2s_convert.h>
#include <saphira_llm/i2s_gemm.h>
#include <saphira_llm/qat.h>
#include <saphira_llm/sllm.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "harness.h"

#define EX_GGUF "test_export.gguf"

static const int32_t EX_TOKENS[24] = {
      3,  17,  42,   7,  99,  13, 200,  55,
     71,   2,  88,  31, 150,  64,  19, 233,
     40,  77, 128,   9, 201,  26, 110,  61
};

/* A deterministic model at a given width. 4 heads / 2 kv, so n_embd_gqa is
 * n_embd/2, and n_ff is set equal to n_embd so every ternary row is the same
 * width and the case is unambiguous. */
static sllm_qat * model_at(int n_embd, int n_ff) {
    sllm_qat_config c;
    sllm_qat_default_config(&c);
    c.n_embd = n_embd; c.n_ff = n_ff; c.n_layer = 1;
    c.n_head = 4; c.n_head_kv = 2;
    c.n_embd_head = n_embd / 4; c.n_embd_gqa = n_embd / 2;
    sllm_qat * q = sllm_qat_new(&c, 777);
    if (q == NULL) { return NULL; }
    sllm_qat_init(q, 777);
    sllm_qat_batch b;
    b.tokens = (int32_t *) EX_TOKENS; b.n_tokens = 24;
    for (int i = 0; i < 10; ++i) {
        (void) sllm_qat_loss(q, &b);
        sllm_qat_clip_grad(q, 1.0f);
        sllm_qat_step(q, 0.05f);
    }
    return q;
}

/* FNV-1a over the I2_S payload of every I2_S tensor, combined. Compact, and
 * enough to move on any single-byte change. */
static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static uint64_t combined_i2s_fingerprint(const sllm_gguf * g) {
    uint64_t all = 1469598103934665603ULL;
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        const sllm_gguf_tensor * t = &g->tensors[i];
        if (t->type != SLLM_TYPE_I2_S) { continue; }
        all ^= fnv1a((const uint8_t *) t->data, (size_t) t->nbytes);
        all *= 1099511628211ULL;
    }
    return all;
}

/* ------------------------------------------------------------------ */
/* 1. narrow rows are refused                                          */
/* ------------------------------------------------------------------ */

TEST(export_refuses_row_widths_the_runtime_cannot_consume) {
    /* 64 is the width that started this: below SLLM_I2S_QK, so sllm_i2s_dot
     * returns 0 for every projection. It used to export successfully. */
    sllm_qat * q = model_at(64, 128);
    CHECK(q != NULL);
    if (q == NULL) { return; }
    const int rc = sllm_qat_export_i2s_gguf(q, EX_GGUF, SLLM_I2S_RULE_GGUF_PY);
    CHECK(rc != 0);
    /* And it must leave nothing behind: a half-written file that a later step
     * picks up is worse than no file. */
    struct stat st;
    CHECK(stat(EX_GGUF, &st) != 0);
    sllm_qat_free(q);

    /* 192 is a subtler case and worth pinning separately: sllm_i2s_dot does
     * NOT return 0 for it (192 >= 128), but gemv strides rows by 192/4 = 48
     * while the block layout needs 2 x 32 = 64 bytes, so rows would be read
     * from the wrong offsets. A silent wrong answer, not a zero. */
    sllm_qat * r = model_at(128, 192);
    CHECK(r != NULL);
    if (r == NULL) { return; }
    CHECK(sllm_qat_export_i2s_gguf(r, EX_GGUF, SLLM_I2S_RULE_GGUF_PY) != 0);
    sllm_qat_free(r);
    remove(EX_GGUF);
}

/* ------------------------------------------------------------------ */
/* 2. aligned exports are byte-stable                                  */
/* ------------------------------------------------------------------ */

/*
 * Captured from the writer as it stood, at 9682664, BEFORE the row-width
 * precondition was added. These are not regenerated from the code under test,
 * which is the whole point: a fingerprint recomputed on every run would pass
 * whatever the writer did.
 *
 * 2048 is the shipped n_embd, so this is the byte-stability claim for models
 * that actually exist. 128 and 256 cover the block boundary and a multi-block
 * row.
 */
TEST(export_is_byte_stable_for_aligned_widths) {
    struct { int e, f; uint64_t fnv; } cases[] = {
        {  128,  128, 0x3d8afbc91665b037ULL },
        {  256,  256, 0xcceae8e057a0186dULL },
        { 2048, 2048, 0xde1dfed496c24ff3ULL },
    };
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; ++i) {
        sllm_qat * q = model_at(cases[i].e, cases[i].f);
        CHECK(q != NULL);
        if (q == NULL) { continue; }
        CHECK(sllm_qat_export_i2s_gguf(q, EX_GGUF, SLLM_I2S_RULE_GGUF_PY) == 0);
        sllm_gguf g;
        if (sllm_gguf_open(EX_GGUF, &g, NULL, 0) == SLLM_OK) {
            const uint64_t got = combined_i2s_fingerprint(&g);
            CHECK(got == cases[i].fnv);
            sllm_gguf_close(&g);
        }
        sllm_qat_free(q);
    }
    remove(EX_GGUF);
}

/* ------------------------------------------------------------------ */
/* 3. the file decodes back to the trainer's quantised weights           */
/* ------------------------------------------------------------------ */

/*
 * Decoded the way sllm_i2s_gemv does -- row base r*(cols/4), and within a row
 * the interleaved 128-block mapping j = 32*field + lane, byte `lane`, shift
 * 6 - 2*field -- and compared against the trainer's own quantisation of its
 * master, code for code, with the scale compared as raw bits.
 */
TEST(exported_bytes_equal_the_trainers_quantised_weights) {
    sllm_qat * q = model_at(128, 128);
    CHECK(q != NULL);
    if (q == NULL) { return; }
    CHECK(sllm_qat_export_i2s_gguf(q, EX_GGUF, SLLM_I2S_RULE_GGUF_PY) == 0);

    sllm_gguf g;
    if (sllm_gguf_open(EX_GGUF, &g, NULL, 0) != SLLM_OK) {
        sllm_qat_free(q); remove(EX_GGUF); return;
    }

    static const char * names[7] = {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ffn_up.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"
    };
    static const char * params[7] = {
        "blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v",
        "blk.0.attn_output", "blk.0.ffn_up", "blk.0.ffn_gate", "blk.0.ffn_down"
    };
    int rows_of[7] = { 128, 64, 64, 128, 128, 128, 128 };
    int cols_of[7] = { 128, 128, 128, 128, 128, 128, 128 };

    for (int i = 0; i < 7; ++i) {
        const sllm_gguf_tensor * t = sllm_gguf_find_tensor(&g, names[i]);
        const sllm_qat_tensor * m = sllm_qat_find(q, params[i]);
        CHECK(t != NULL);
        CHECK(m != NULL);
        if (t == NULL || m == NULL) { continue; }
        const int rows = rows_of[i], cols = cols_of[i];
        const int n = rows * cols;
        CHECK(t->nbytes == (uint64_t) sllm_i2s_packed_size((size_t) n));
        CHECK(t->ne[0] == (uint64_t) cols);
        CHECK(t->ne[1] == (uint64_t) rows);

        /* the scale, as bits, because "close" is not the claim */
        const float fscale = sllm_i2s_block_scale(t->data, (size_t) n);
        const float tscale = sllm_qat_weight_scale(m);
        CHECK(fscale == tscale);

        int mismatch = 0;
        const uint8_t * p = (const uint8_t *) t->data;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const long off = (long) r * (cols / 4) + (long) (c / 128) * 32 + (c % 32);
                const int field = (int) ((c % 128) / 32);
                const int code = (p[off] >> (6 - 2 * field)) & 3;
                /* the trainer's own rule, from ternarise(): sign of w/scale,
                 * thresholded at 0.5, encoded as 0/1/2 for -1/0/+1 */
                const double w = (double) m->master[(size_t) r * cols + c];
                const double t = w / (double) tscale;
                const int want = (t > 0.5) ? 2 : (t < -0.5) ? 0 : 1;
                if (code != want) { ++mismatch; }
            }
        }
        CHECK(mismatch == 0);
    }

    sllm_gguf_close(&g);
    sllm_qat_free(q);
    remove(EX_GGUF);
}


/* ------------------------------------------------------------------ */
/* 4. the ordinary loader refuses an unsupported foreign model          */
/* ------------------------------------------------------------------ */

/*
 * The exporter knows the rule, but a GGUF from anywhere must get the same
 * answer. This builds a file the exporter would never produce -- a valid GGUF
 * with a 64-wide I2_S matrix -- and asserts that the ordinary reader refuses
 * it, so the contract lives at the boundary that enforces it rather than in
 * every producer that has to remember it.
 *
 * The width is corrupted in place on a real export: the header and the tensor
 * table are left alone, only the declared ne[0] of one matrix is rewritten.
 * That is exactly the shape of the failure -- a file that opens, parses,
 * passes every shape check, and then computes nothing.
 */
TEST(ordinary_loader_refuses_an_unsupported_i2s_width) {
    /* width 64 is the case that made a finite logit out of nothing */
    sllm_qat * q = model_at(128, 128);
    CHECK(q != NULL);
    if (q == NULL) { return; }
    CHECK(sllm_qat_export_i2s_gguf(q, EX_GGUF, SLLM_I2S_RULE_GGUF_PY) == 0);

    sllm_gguf g;
    if (sllm_gguf_open(EX_GGUF, &g, NULL, 0) != SLLM_OK) {
        sllm_qat_free(q); remove(EX_GGUF); return;
    }
    const sllm_gguf_tensor * t = sllm_gguf_find_tensor(&g, "blk.0.attn_q.weight");
    CHECK(t != NULL);
    /* sanity: the honest file is accepted */
    sllm_model * good = NULL;
    CHECK(sllm_model_load(&g, &good) == SLLM_OK);
    CHECK(good != NULL);
    sllm_model_free(good);
    sllm_gguf_close(&g);

    /* Rewrite ne[0] to 64 in place. Located by searching the file for the
     * tensor's name: the name appears exactly once, in the tensor table, and
     * a tensor entry is name(u64 len + bytes) then n_dims(u32) then the dims,
     * so ne[0] is a fixed distance past the end of the name. */
    {
        FILE * f = fopen(EX_GGUF, "rb");
        CHECK(f != NULL);
        if (f == NULL) { sllm_qat_free(q); remove(EX_GGUF); return; }
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t * buf = (uint8_t *) malloc((size_t) sz);
        CHECK(buf != NULL);
        if (buf == NULL) { fclose(f); sllm_qat_free(q); remove(EX_GGUF); return; }
        CHECK(fread(buf, 1, (size_t) sz, f) == (size_t) sz);
        fclose(f);

        static const char needle[] = "blk.0.attn_q.weight";
        const size_t nlen = sizeof needle - 1;
        long at = -1;
        for (long i = 0; i + (long) nlen < sz; ++i) {
            if (memcmp(buf + i, needle, nlen) == 0) { at = i; break; }
        }
        CHECK(at >= 0);
        if (at >= 0) {
            const long dims_at = at + (long) nlen + 4;   /* +4 skips n_dims */
            CHECK(dims_at + 8 <= sz);
            uint64_t ne0 = 64;
            memcpy(buf + dims_at, &ne0, 8);
            FILE * o = fopen(EX_GGUF, "wb");
            CHECK(o != NULL);
            if (o != NULL) {
                CHECK(fwrite(buf, 1, (size_t) sz, o) == (size_t) sz);
                fclose(o);
            }
        }
        free(buf);
    }

    /* the reader still opens it -- the file is structurally valid ... */
    sllm_gguf g3;
    if (sllm_gguf_open(EX_GGUF, &g3, NULL, 0) == SLLM_OK) {
        const sllm_gguf_tensor * t3 = sllm_gguf_find_tensor(&g3, "blk.0.attn_q.weight");
        CHECK(t3 != NULL);
        if (t3 != NULL) { CHECK((int64_t) t3->ne[0] == 64); }
        /* ... and the loader refuses it, which is the contract */
        sllm_model * bad = NULL;
        CHECK(sllm_model_load(&g3, &bad) != SLLM_OK);
        CHECK(bad == NULL);
        sllm_gguf_close(&g3);
    }
    sllm_qat_free(q);
    remove(EX_GGUF);
}

void sllm_test_export(void) {
    printf("export\n");
    RUN(export_refuses_row_widths_the_runtime_cannot_consume);
    RUN(export_is_byte_stable_for_aligned_widths);
    RUN(exported_bytes_equal_the_trainers_quantised_weights);
    RUN(ordinary_loader_refuses_an_unsupported_i2s_width);
}
