/*
 * harness.h — minimal test harness for saphira-llm.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SLLM_TEST_HARNESS_H
#define SLLM_TEST_HARNESS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <saphira_llm/sllm.h>

/*
 * The counters live in tests/main.c and are declared here. They must NOT be
 * `static` in this header: each translation unit would get its own copy and
 * the summary would report zero no matter how many checks ran. That is a
 * silent-test-failure bug, not a cosmetic one.
 */
extern int sllm_tests_run;
extern int sllm_tests_failed;
extern const char * sllm_current;

#define TEST(name) \
    static void name(void); \
    static void run_##name(void) { sllm_current = #name; name(); } \
    static void name(void)

#define RUN(name) do { run_##name(); } while (0)

#define CHECK(cond) do { \
    sllm_tests_run++; \
    if (!(cond)) { \
        sllm_tests_failed++; \
        fprintf(stderr, "  FAIL %s:%d in %s: %s\n", __FILE__, __LINE__, sllm_current, #cond); \
    } \
} while (0)

/* Status equality with a readable message on failure. */
#define CHECK_STATUS(expr, want) do { \
    sllm_status got_ = (expr); \
    sllm_status want_ = (want); \
    sllm_tests_run++; \
    if (got_ != want_) { \
        sllm_tests_failed++; \
        fprintf(stderr, "  FAIL %s:%d in %s: %s\n        got  %s (%d)\n        want %s (%d)\n", \
                __FILE__, __LINE__, sllm_current, #expr, \
                sllm_status_string(got_), (int) got_, \
                sllm_status_string(want_), (int) want_); \
    } \
} while (0)

#define CHECK_STR(got, want) do { \
    const char * g_ = (got); const char * w_ = (want); \
    sllm_tests_run++; \
    if (g_ == NULL || strcmp(g_, w_) != 0) { \
        sllm_tests_failed++; \
        fprintf(stderr, "  FAIL %s:%d in %s: %s\n        got  \"%s\"\n        want \"%s\"\n", \
                __FILE__, __LINE__, sllm_current, #got, g_ ? g_ : "(null)", w_); \
    } \
} while (0)

/* Bit-exact float comparison. Casting a float to an integer is undefined when
 * the value is negative or out of range, which UBSan flagged here: these
 * kernels are compared on routinely negative values. */
static inline int sllm_same_bits(float x, float y) {
    return memcmp(&x, &y, sizeof(float)) == 0;
}

#define CHECK_SAME_BITS(got, want) do { \
    sllm_tests_run++; \
    if (!sllm_same_bits((got), (want))) { \
        sllm_tests_failed++; \
        fprintf(stderr, "  FAIL %s:%d in %s: got %.9g want %.9g\n", \
                __FILE__, __LINE__, sllm_current, (double) (got), (double) (want)); \
    } \
} while (0)

/*
 * Build a packed I2_S block straight from the converter's definition:
 *   utils/convert-hf-to-gguf-bitnet.py reshapes each 128-element block to
 *   (4, 32) and packs (q[:,0,:] << 6) | (q[:,1,:] << 4) | (q[:,2,:] << 2) | q[:,3,:]
 * so element j = 32*a + b goes into byte b of the block at bit shift 6 - 2a.
 *
 * Written from the converter, not from sllm_i2s_code, so the test and the
 * reader can disagree. An earlier version of this helper derived the shift
 * from the reader, which made the test incapable of catching the reader
 * having the fields backwards -- and it did not catch it.
 */
static inline void sllm_pack_i2s_like_converter(uint8_t * dst,
                                                 const uint8_t * codes, size_t n) {
    memset(dst, 0, n / 4 + 32);
    for (size_t blk = 0; blk * 128 < n; ++blk) {
        for (size_t j = 0; j < 128; ++j) {
            const size_t a = j / 32;
            const size_t b = j % 32;
            dst[blk * 32 + b] |= (uint8_t) (codes[blk * 128 + j] << (6 - 2 * a));
        }
    }
}

#define CHECK_EQ_U64(got, want) do { \
    unsigned long long g_ = (unsigned long long)(got); \
    unsigned long long w_ = (unsigned long long)(want); \
    sllm_tests_run++; \
    if (g_ != w_) { \
        sllm_tests_failed++; \
        fprintf(stderr, "  FAIL %s:%d in %s: %s\n        got  %llu\n        want %llu\n", \
                __FILE__, __LINE__, sllm_current, #got, g_, w_); \
    } \
} while (0)

#define CHECK_EQ_INT(got, want) do { \
    long g_ = (long)(got); long w_ = (long)(want); \
    sllm_tests_run++; \
    if (g_ != w_) { \
        sllm_tests_failed++; \
        fprintf(stderr, "  FAIL %s:%d in %s: %s\n        got  %ld\n        want %ld\n", \
                __FILE__, __LINE__, sllm_current, #got, g_, w_); \
    } \
} while (0)

#endif /* SLLM_TEST_HARNESS_H */
