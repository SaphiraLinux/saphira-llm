/*
 * main.c — saphira-llm test runner.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */

#include "harness.h"

void sllm_test_isa(void);
void sllm_test_gguf(void);

int sllm_tests_run    = 0;
int sllm_tests_failed = 0;
const char * sllm_current = "";

int main(void) {
    (void) sllm_current;
    printf("saphira-llm test suite (baseline %s)\n\n", SLLM_BASELINE_ISA);

    sllm_test_isa();
    sllm_test_gguf();

    printf("\n%d checks, %d failed\n", sllm_tests_run, sllm_tests_failed);
    return sllm_tests_failed == 0 ? 0 : 1;
}
