/*
 * test_gguf.c — GGUF container tests, including malformed and truncated input.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The rejection tests matter more than the acceptance ones. A loader that
 * accepts a corrupt file produces wrong answers silently; a loader that
 * rejects one with a specific reason is doing its job.
 */

#include "harness.h"

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* a tiny GGUF writer, so tests do not depend on a model on disk       */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t * buf;
    size_t    len;
    size_t    cap;
} buf;

static void bput(buf * b, const void * p, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 64;
        b->buf = realloc(b->buf, b->cap);
    }
    memcpy(b->buf + b->len, p, n);
    b->len += n;
}

static void bu8(buf * b, uint8_t v)   { bput(b, &v, 1); }
static void bu32(buf * b, uint32_t v) { bput(b, &v, 4); }
static void bu64(buf * b, uint64_t v) { bput(b, &v, 8); }

static void bstr(buf * b, const char * s) {
    const uint64_t n = strlen(s);
    bu64(b, n);
    bput(b, s, (size_t) n);
}

static void bfree(buf * b) { free(b->buf); b->buf = NULL; b->len = b->cap = 0; }

/*
 * A minimal but structurally complete GGUF v3 container:
 *   - metadata: general.architecture (string), test.count (uint32)
 *   - two tensors: 1024 elements of I2_S and 512 elements of F32
 *   - a correctly aligned data blob
 *
 * The header counts are derived from what is actually written rather than
 * hard-coded, because a wrong count makes the parser read tensor descriptors
 * out of the metadata and fail for the wrong reason. That mistake produced two
 * bogus test failures here before it was fixed, which is exactly why the
 * writer now counts as it goes.
 */
typedef struct {
    buf  b;
    int  n_kv;
    int  n_tensors;
} builder;

static void bkv_u32(builder * bd, const char * key, uint32_t v) {
    bstr(&bd->b, key);
    bu32(&bd->b, SLLM_VT_UINT32);
    bu32(&bd->b, v);
    bd->n_kv++;
}

static void bkv_str(builder * bd, const char * key, const char * v) {
    bstr(&bd->b, key);
    bu32(&bd->b, SLLM_VT_STRING);
    bstr(&bd->b, v);
    bd->n_kv++;
}

enum { BREAK_NONE = 0, BREAK_BAD_VTYPE, BREAK_DUP_KEY, BREAK_HUGE_STR };

static void build_valid(buf * out, uint32_t version, int break_what) {
    builder bd;
    memset(&bd, 0, sizeof(bd));

    /* Reserve the header; it is rewritten once the counts are known. */
    bput(&bd.b, "GGUF", 4);
    bu32(&bd.b, version);
    bu64(&bd.b, 0);   /* tensor_count placeholder */
    bu64(&bd.b, 0);   /* kv_count placeholder */

    bkv_str(&bd, "general.architecture", "bitnet-b1.58");

    /*
     * The second slot is either written normally or deliberately malformed.
     * It replaces the slot rather than appending, because appending would
     * leave n_kv unchanged and the parser would never reach the malformed
     * bytes at all -- it would read them as a tensor descriptor instead and
     * fail for entirely the wrong reason. That happened here once already.
     */
    if (break_what == BREAK_BAD_VTYPE) {
        bstr(&bd.b, "test.count");
        bu32(&bd.b, 999);
        bu32(&bd.b, 0);
        bu64(&bd.b, 0);
        bd.n_kv++;
    } else if (break_what == BREAK_DUP_KEY) {
        bkv_u32(&bd, "test.count", 1234);
        bkv_u32(&bd, "test.count", 4321);
    } else if (break_what == BREAK_HUGE_STR) {
        bstr(&bd.b, "test.count");
        bu32(&bd.b, SLLM_VT_STRING);
        bu64(&bd.b, UINT64_MAX);
        bd.n_kv++;
    } else {
        bkv_u32(&bd, "test.count", 1234);
    }

    /* tensor 0: 256x4 = 1024 elements of I2_S, at blob offset 0 */
    bstr(&bd.b, "blk.0.weight");
    bu32(&bd.b, 2);
    bu64(&bd.b, 256);
    bu64(&bd.b, 4);
    bu32(&bd.b, SLLM_TYPE_I2_S);
    bu64(&bd.b, 0);
    bd.n_tensors++;

    /* tensor 1: 512 elements of F32, right after the I2_S payload */
    bstr(&bd.b, "norm.weight");
    bu32(&bd.b, 1);
    bu64(&bd.b, 512);
    bu32(&bd.b, SLLM_TYPE_F32);
    bu64(&bd.b, 1024 / 4 + 32);
    bd.n_tensors++;

    /* Pad to the 32-byte alignment boundary, then the blob. */
    while (bd.b.len % 32 != 0) {
        bu8(&bd.b, 0);
    }
    for (int i = 0; i < (1024 / 4 + 32) / 4; ++i) {
        bu32(&bd.b, 0x3f800000u);
    }
    for (int i = 0; i < 512; ++i) {
        bu32(&bd.b, 0x40000000u);
    }

    /* Now that the counts are known, patch them into the header. */
    memcpy(bd.b.buf +  8, &version,  4);
    uint64_t nt = (uint64_t) bd.n_tensors;
    uint64_t nk = (uint64_t) bd.n_kv;
    memcpy(bd.b.buf +  8, &nt, 8);
    memcpy(bd.b.buf + 16, &nk, 8);

    *out = bd.b;
}

/* ------------------------------------------------------------------ */

TEST(gguf_accepts_a_valid_container) {
    buf b;
    build_valid(&b, SLLM_GGUF_VERSION, BREAK_NONE);

    sllm_gguf g;
    char err[256] = {0};
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err)), SLLM_OK);

    CHECK_EQ_U64(g.n_tensors, 2);
    CHECK_EQ_U64(g.n_kv, 2);
    CHECK_EQ_U64(g.alignment, 32);
    CHECK_EQ_INT(g.version, 3);

    const char * arch = NULL;
    CHECK_STATUS(sllm_gguf_kv_str(&g, "general.architecture", &arch), SLLM_OK);
    CHECK_STR(arch, "bitnet-b1.58");

    uint32_t n = 0;
    CHECK_STATUS(sllm_gguf_kv_u32(&g, "test.count", &n), SLLM_OK);
    CHECK_EQ_U64(n, 1234);

    sllm_gguf_close(&g);
    bfree(&b);
}

TEST(gguf_rejects_bad_magic) {
    buf b;
    build_valid(&b, SLLM_GGUF_VERSION, BREAK_NONE);
    b.buf[0] = 'X';

    sllm_gguf g;
    char err[256] = {0};
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err)),
                 SLLM_ERR_GGUF_MAGIC);
    CHECK(err[0] != '\0');
    bfree(&b);
}

TEST(gguf_rejects_unsupported_version) {
    buf b;
    build_valid(&b, 99, 0);

    sllm_gguf g;
    char err[256] = {0};
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err)),
                 SLLM_ERR_GGUF_VERSION);
    CHECK(err[0] != '\0');
    bfree(&b);
}

/* Truncation is the most important rejection: a short file must never be
 * read past, and must never be reported as merely "unsupported". */
TEST(gguf_rejects_every_truncation) {
    buf full;
    build_valid(&full, SLLM_GGUF_VERSION, 0);

    /* Cut at many points, including inside the metadata and inside a tensor
     * descriptor. Every one must be rejected, and none may crash. */
    const size_t cuts[] = { 0, 1, 3, 4, 7, 8, 12, 16, 20, 24, 40, 64, 100, 200 };
    for (size_t i = 0; i < sizeof(cuts) / sizeof(cuts[0]); ++i) {
        const size_t n = cuts[i];
        if (n >= full.len) {
            continue;
        }
        sllm_gguf g;
        char err[256] = {0};
        const sllm_status st = sllm_gguf_open_memory(full.buf, n, &g, err, sizeof(err));
        sllm_tests_run++;
        if (st == SLLM_OK) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL truncation at %zu bytes was accepted\n", n);
            sllm_gguf_close(&g);
        } else if (st != SLLM_ERR_GGUF_TRUNCATED && st != SLLM_ERR_GGUF_LAYOUT) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL truncation at %zu gave %s, want truncated or layout\n",
                    n, sllm_status_string(st));
        }
    }
    bfree(&full);
}

TEST(gguf_rejects_unknown_value_type) {
    buf b;
    build_valid(&b, SLLM_GGUF_VERSION, BREAK_BAD_VTYPE);

    sllm_gguf g;
    char err[256] = {0};
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err)),
                 SLLM_ERR_GGUF_TYPE);
    CHECK(err[0] != '\0');
    bfree(&b);
}

TEST(gguf_rejects_duplicate_keys) {
    buf b;
    build_valid(&b, SLLM_GGUF_VERSION, BREAK_DUP_KEY);

    sllm_gguf g;
    char err[256] = {0};
    /* Duplicates are ambiguous, so they are refused rather than resolved by
     * taking the first or the last occurrence. */
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err)),
                 SLLM_ERR_GGUF_KEYDUP);
    CHECK(err[0] != '\0');
    bfree(&b);
}

TEST(gguf_rejects_absurd_string_length) {
    buf b;
    build_valid(&b, SLLM_GGUF_VERSION, BREAK_HUGE_STR);

    sllm_gguf g;
    char err[256] = {0};
    const sllm_status st = sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err));
    sllm_tests_run++;
    if (st == SLLM_OK) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL an absurd string length was accepted\n");
        sllm_gguf_close(&g);
    }
    bfree(&b);
}

TEST(gguf_rejects_absurd_counts) {
    /* A header claiming more tensors than the limit must be refused before
     * anything is allocated. */
    buf b;
    memset(&b, 0, sizeof(b));
    bput(&b, "GGUF", 4);
    bu32(&b, SLLM_GGUF_VERSION);
    bu64(&b, UINT64_MAX);          /* tensor_count */
    bu64(&b, 0);                   /* kv_count     */

    sllm_gguf g;
    char err[256] = {0};
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err)),
                 SLLM_ERR_TOO_LARGE);
    bfree(&b);
}

TEST(gguf_rejects_tensor_offset_past_end) {
    buf b;
    build_valid(&b, SLLM_GGUF_VERSION, BREAK_NONE);

    /* Shrink the file so the declared tensor extents no longer fit inside the
     * blob. The descriptors are intact, so this must be a layout rejection and
     * not a truncation one. */
    sllm_gguf g;
    char err[256] = {0};
    /* Keep only the header plus descriptors, drop the data blob. */
    const size_t keep = b.len - 16;
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, keep, &g, err, sizeof(err)),
                 SLLM_ERR_GGUF_LAYOUT);
    CHECK(err[0] != '\0');
    bfree(&b);
}

TEST(gguf_missing_key_is_distinct_from_wrong_type) {
    buf b;
    build_valid(&b, SLLM_GGUF_VERSION, BREAK_NONE);

    sllm_gguf g;
    char err[256] = {0};
    CHECK_STATUS(sllm_gguf_open_memory(b.buf, b.len, &g, err, sizeof(err)), SLLM_OK);

    uint32_t u = 0;
    const char * s = NULL;
    float f = 0.0f;

    /* Absent, wrong-type, and present must all be distinguishable. */
    CHECK_STATUS(sllm_gguf_kv_u32(&g, "no.such.key", &u), SLLM_ERR_KV_MISSING);
    CHECK_STATUS(sllm_gguf_kv_str(&g, "no.such.key", &s),  SLLM_ERR_KV_MISSING);
    CHECK_STATUS(sllm_gguf_kv_f32(&g, "no.such.key", &f),  SLLM_ERR_KV_MISSING);

    CHECK_STATUS(sllm_gguf_kv_u32(&g, "general.architecture", &u), SLLM_ERR_KV_TYPE);
    CHECK_STATUS(sllm_gguf_kv_f32(&g, "test.count", &f),           SLLM_ERR_KV_TYPE);
    CHECK_STATUS(sllm_gguf_kv_u32(&g, "test.count", &u),           SLLM_OK);

    CHECK(sllm_gguf_find_tensor(&g, "no.such.tensor") == NULL);
    CHECK(sllm_gguf_find_tensor(&g, "norm.weight") != NULL);

    sllm_gguf_close(&g);
    bfree(&b);
}

/* ------------------------------------------------------------------ */
/* tensor type sizes                                                   */
/* ------------------------------------------------------------------ */

TEST(gguf_i2_s_size_is_four_per_byte_plus_a_tail) {
    uint64_t n = 0;
    /* This is the single most important size in the BitNet path. Getting it
     * wrong shifts the scale and produces plausible but wrong numbers
     * instead of a crash. */
    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_I2_S, 1024, &n), SLLM_OK);
    CHECK_EQ_U64(n, 1024 / 4 + 32);

    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_I2_S, 4, &n), SLLM_OK);
    CHECK_EQ_U64(n, 1 + 32);

    /* Not a whole number of bytes: must be refused, not truncated. */
    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_I2_S, 1023, &n), SLLM_ERR_GGUF_TENSOR);
}

TEST(gguf_fixed_block_sizes) {
    uint64_t n = 0;
    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_F32, 512, &n), SLLM_OK);
    CHECK_EQ_U64(n, 512 * 4);

    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_F16, 512, &n), SLLM_OK);
    CHECK_EQ_U64(n, 512 * 2);

    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_BF16, 512, &n), SLLM_OK);
    CHECK_EQ_U64(n, 512 * 2);

    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_Q8_0, 512, &n), SLLM_OK);
    CHECK_EQ_U64(n, (512 / 32) * 34);

    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_Q4_K, 512, &n), SLLM_OK);
    CHECK_EQ_U64(n, (512 / 256) * 144);

    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_Q6_K, 512, &n), SLLM_OK);
    CHECK_EQ_U64(n, (512 / 256) * 210);

    /* A partial block is a malformed tensor, not a rounded-down size. */
    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_Q4_K, 100, &n), SLLM_ERR_GGUF_TENSOR);
    CHECK_STATUS(sllm_gguf_type_nbytes(SLLM_TYPE_Q8_0, 17,  &n), SLLM_ERR_GGUF_TENSOR);
}

TEST(gguf_known_versus_supported_are_different_questions) {
    /* A type the format defines but we have not implemented must be
     * reported as such, so the message can say "valid type, no kernel" rather
     * than "malformed file". */
    CHECK(sllm_gguf_type_is_known((uint32_t) SLLM_TYPE_Q5_K));
    CHECK(!sllm_gguf_type_is_supported(SLLM_TYPE_Q5_K));

    CHECK(sllm_gguf_type_is_known((uint32_t) SLLM_TYPE_I2_S));
    CHECK(sllm_gguf_type_is_supported(SLLM_TYPE_I2_S));

    CHECK(!sllm_gguf_type_is_known(200));
    CHECK(!sllm_gguf_type_is_known(43));

    uint64_t n = 0;
    CHECK_STATUS(sllm_gguf_type_nbytes((sllm_ggml_type) 200, 16, &n),
                 SLLM_ERR_TYPE_UNSUPPORTED);
    CHECK_STATUS(sllm_gguf_type_traits(SLLM_TYPE_Q5_K, NULL, NULL),
                 SLLM_ERR_TYPE_UNSUPPORTED);

    CHECK_STR(sllm_gguf_type_name(SLLM_TYPE_I2_S), "I2_S");
    CHECK_STR(sllm_gguf_type_name((sllm_ggml_type) 200), "UNKNOWN(200)");
}

void sllm_test_gguf(void) {
    printf("gguf\n");
    RUN(gguf_accepts_a_valid_container);
    RUN(gguf_rejects_bad_magic);
    RUN(gguf_rejects_unsupported_version);
    RUN(gguf_rejects_every_truncation);
    RUN(gguf_rejects_unknown_value_type);
    RUN(gguf_rejects_duplicate_keys);
    RUN(gguf_rejects_absurd_string_length);
    RUN(gguf_rejects_absurd_counts);
    RUN(gguf_rejects_tensor_offset_past_end);
    RUN(gguf_missing_key_is_distinct_from_wrong_type);
    RUN(gguf_i2_s_size_is_four_per_byte_plus_a_tail);
    RUN(gguf_fixed_block_sizes);
    RUN(gguf_known_versus_supported_are_different_questions);
}
