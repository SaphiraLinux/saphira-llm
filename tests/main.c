/*
 * main.c — saphira-llm test runner.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */

#include "harness.h"
#include "gate_runner.h"

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
int  main_k_dispatch_gate(void);
int  main_k_fwd_slice_gate(void);
int  main_k_rope_contract_gate(void);
int  main_k_attention_gate(void);
int  main_k_residual_gate(void);
int  main_k_ffn_gate(void);

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
    /* The four product gates each RETURN a status. Calling them bare discarded that
     * status, so a failing forward slice still let the suite print "0 failed" and the
     * build go green. This is the single most dangerous defect class in a test rig:
     * a gate that cannot fail. It stayed latent only because T2-T4 all passed; the
     * first genuine failure inside a gate exposed it, and the failure is invisible
     * unless the gate's verdict is actually folded into the global counters. */
    sllm_run_product_gate(main_k_quant_gate,   "K-quant dequantisers vs reference golden");
    sllm_run_product_gate(main_k_gemv_gate,    "f32 GEMV vs reference double-accumulated golden");
    sllm_run_product_gate(main_k_dispatch_gate,"execution dispatched by measured evidence");
    sllm_run_product_gate(main_k_fwd_slice_gate,"forward slice: T2/T3/T4/T5/T6 claim levels");
    sllm_run_product_gate(main_k_rope_contract_gate,"RoPE semantics contract + rotation parity");
    sllm_run_product_gate(main_k_attention_gate,"T9 attention: three-branch convergence + GQA + causal + softmax");
    sllm_run_product_gate(main_k_residual_gate,"T10 attention output projection + residual merge");
    sllm_run_product_gate(main_k_ffn_gate,"T11 FFN block + second residual");

    printf("\n%d checks, %d failed\n", sllm_tests_run, sllm_tests_failed);
    return sllm_tests_failed == 0 ? 0 : 1;
}
