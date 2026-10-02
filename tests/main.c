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
void sllm_test_thread(void);
void sllm_test_ops(void);
void sllm_test_i2s(void);
void sllm_test_tokenizer(void);
void sllm_test_forward(void);
void sllm_test_phase5(void);
void sllm_test_dot_f16(void);
void sllm_test_eval(void);
void sllm_test_i2s_convert(void);
void sllm_test_i2s_hermetic(void);
void sllm_test_qat(void);
void sllm_test_lifecycle(void);
void sllm_test_export(void);
int  main_k_quant_gate(void);
int  main_k_gemv_gate(void);

int sllm_tests_run    = 0;
int sllm_tests_failed = 0;
const char * sllm_current = "";

int main(void) {
    (void) sllm_current;
    printf("saphira-llm test suite (baseline %s)\n\n", SLLM_BASELINE_ISA);

    sllm_test_isa();
    sllm_test_gguf();
    sllm_test_thread();
    sllm_test_ops();
    sllm_test_i2s();
    sllm_test_tokenizer();
    sllm_test_forward();
    sllm_test_dot_f16();
    sllm_test_phase5();
    sllm_test_eval();
    sllm_test_i2s_convert();
    sllm_test_i2s_hermetic();
    sllm_test_qat();
    sllm_test_lifecycle();
    sllm_test_export();
    /* Step 1 gate: the K-quant dequantisers checked against a REFERENCE golden. */
    main_k_quant_gate();
    /* Step 2 gate: the f32 GEMV against a reference double-accumulated golden. */
    main_k_gemv_gate();

    printf("\n%d checks, %d failed\n", sllm_tests_run, sllm_tests_failed);
    return sllm_tests_failed == 0 ? 0 : 1;
}
