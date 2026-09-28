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
