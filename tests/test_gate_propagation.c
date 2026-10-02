/*
 * test_gate_propagation.c -- the harness must be able to FAIL. This is the test.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * HARNESS-FAILURE PROPAGATION IS AN INVARIANT, NOT A TEST THAT MAY BE DELETED WHEN
 * THE DEFECT IT GUARDS AGAINST IS FORGOTTEN.
 *
 * The bug this guards against is already fixed, so nothing in the product now fails
 * and this suite reports green forever. That is precisely why it must exist: the
 * absence of a failure is not proof that a failure would be visible. Every gate in
 * the product passing is consistent both with "the gates work" and with "the gates
 * cannot fail", and only deliberately induced failures tell those apart.
 *
 * For EACH product gate we induce a KNOWN failure in a fixture and prove all three
 * of the properties that were previously false:
 *
 *     1. the gate returns failure
 *     2. the global failed count increments
 *     3. the process exits non-zero
 *
 * Property 3 is checked by re-executing this binary as a child process, because
 * asserting "the exit code would be non-zero" from inside a process that is itself
 * running green is a claim about code that was not exercised.
 *
 * The runner under test is the SAME sllm_run_product_gate() that main.c calls,
 * included from gate_runner.h. A test of a copy would prove only the copy.
 */

#include "gate_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int sllm_tests_run    = 0;
int sllm_tests_failed = 0;
const char * sllm_current = "";

static int checks_run = 0, checks_failed = 0;

static void expect(int cond, const char *what) {
    checks_run++;
    if (cond) {
        printf("    ok    %s\n", what);
    } else {
        checks_failed++;
        printf("    FAIL  %s\n", what);
    }
}

/* -- The fixtures: one deliberately failing gate per product gate slot. --
 *
 * Each returns a distinct non-zero value, so the test can prove the runner counted
 * the failure rather than merely noticing that something was nonzero. A fixture that
 * always returned 1 could not distinguish "counted the gate's verdict" from
 * "increment on any nonzero", which is a weaker and more bug-prone property. */
static int gate_quant_fails(void)        { return 3; }
static int gate_gemv_fails(void)         { return 1; }
static int gate_dispatch_fails(void)     { return 2; }
static int gate_fwd_slice_fails(void)    { return 7; }

static int gate_quant_passes(void)       { return 0; }
static int gate_gemv_passes(void)        { return 0; }
static int gate_dispatch_passes(void)    { return 0; }
static int gate_fwd_slice_passes(void)   { return 0; }

struct slot { sllm_product_gate fail_fixture; sllm_product_gate pass_fixture;
              const char *name; int expected_rc; };

static const struct slot SLOTS[] = {
    { gate_quant_fails,     gate_quant_passes,     "K-quant dequantisers vs reference golden", 3 },
    { gate_gemv_fails,      gate_gemv_passes,      "f32 GEMV vs reference double-accumulated golden", 1 },
    { gate_dispatch_fails,  gate_dispatch_passes,  "execution dispatched by measured evidence", 2 },
    { gate_fwd_slice_fails, gate_fwd_slice_passes, "forward slice: T2/T3/T4/T5/T6 claim levels", 7 },
};
#define NSLOTS ((int)(sizeof SLOTS / sizeof SLOTS[0]))

/* Prove properties 1 and 2 for every gate: the return value is non-zero AND the
 * global failed count moved. A gate that fails without incrementing the counter is
 * exactly the original defect, reintroduced. */
static void prove_each_gate_propagates(void) {
    printf("  each product gate: induced failure must reach the global counter\n");
    for (int i = 0; i < NSLOTS; ++i) {
        const int before_failed = sllm_tests_failed;
        const int before_run    = sllm_tests_run;

        /* property 1: the gate itself reports failure */
        const int rc = SLOTS[i].fail_fixture();
        expect(rc == SLOTS[i].expected_rc, "gate returns its own distinct failure code");

        /* property 2: running it through the real runner increments the counter */
        sllm_run_product_gate(SLOTS[i].fail_fixture, SLOTS[i].name);
        expect(sllm_tests_run == before_run + 1,
               "global run count incremented by exactly one");
        expect(sllm_tests_failed == before_failed + 1,
               "global FAILED count incremented by exactly one");

        /* and a passing gate must NOT increment failed, or the counter is useless */
        const int pf_before = sllm_tests_failed;
        sllm_run_product_gate(SLOTS[i].pass_fixture, SLOTS[i].name);
        expect(sllm_tests_failed == pf_before,
               "a passing gate does not increment the failed count");
    }
}

/* Prove the counter is not merely monotonic: many failures accumulate. */
static void prove_failures_accumulate(void) {
    printf("  failures accumulate rather than being overwritten\n");
    const int start = sllm_tests_failed;
    sllm_run_product_gate(gate_quant_fails,     "accumulate 1");
    sllm_run_product_gate(gate_gemv_fails,      "accumulate 2");
    sllm_run_product_gate(gate_dispatch_fails,  "accumulate 3");
    sllm_run_product_gate(gate_fwd_slice_fails, "accumulate 4");
    expect(sllm_tests_failed == start + 4,
           "four failing gates produce four counted failures, not one");
}

int main(int argc, char **argv) {
    /* Child mode: run exactly one failing gate and exit on the propagated verdict.
     * The parent re-executes us this way to observe a real process exit status. */
    if (argc > 1 && strcmp(argv[1], "--child-one-failing-gate") == 0) {
        sllm_run_product_gate(gate_fwd_slice_fails, "child induced failure");
        printf("  child: run=%d failed=%d\n", sllm_tests_run, sllm_tests_failed);
        return sllm_tests_failed == 0 ? 0 : 1;
    }
    if (argc > 1 && strcmp(argv[1], "--child-one-passing-gate") == 0) {
        sllm_run_product_gate(gate_fwd_slice_passes, "child passing gate");
        printf("  child: run=%d failed=%d\n", sllm_tests_run, sllm_tests_failed);
        return sllm_tests_failed == 0 ? 0 : 1;
    }

    printf("  HARNESS-FAILURE PROPAGATION: the gate verdict must be able to fail the build\n");

    prove_each_gate_propagates();
    prove_failures_accumulate();

    /* Property 3: the process exits non-zero when a gate fails. Observed by
     * re-executing this binary, because a green parent cannot testify about an exit
     * code it never produced. */
    printf("  process exit status reflects a failing gate\n");
    {   /* Locate ourselves so the child can be re-executed. */
        char self[4096];
        const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
        int ok_child_fail = 0, ok_child_pass = 0, ok_fail_nonzero = 0, ok_pass_zero = 0;
        if (n > 0) {
            self[n] = '\0';
            pid_t pid = fork();
            if (pid == 0) {
                execl(self, self, "--child-one-failing-gate", (char *)NULL);
                _exit(127);
            }
            int st = 0; waitpid(pid, &st, 0);
            if (WIFEXITED(st)) {
                ok_child_fail  = 1;
                ok_fail_nonzero = (WEXITSTATUS(st) != 0);
                printf("    child with a failing gate exited %d\n", WEXITSTATUS(st));
            }
            pid_t pid2 = fork();
            if (pid2 == 0) {
                execl(self, self, "--child-one-passing-gate", (char *)NULL);
                _exit(127);
            }
            int st2 = 0; waitpid(pid2, &st2, 0);
            if (WIFEXITED(st2)) {
                ok_child_pass = 1;
                ok_pass_zero  = (WEXITSTATUS(st2) == 0);
                printf("    child with a passing gate exited %d\n", WEXITSTATUS(st2));
            }
        } else {
            printf("    could not resolve /proc/self/exe\n");
        }
        expect(ok_child_fail,  "child process with a failing gate was observed to exit");
        expect(ok_fail_nonzero,"child process with a FAILING gate exits NON-ZERO");
        expect(ok_child_pass,  "child process with a passing gate was observed to exit");
        expect(ok_pass_zero,   "child process with a PASSING gate exits zero");
    }

    /* The non-vacuity gate on this very test: if these fixtures stopped failing,
     * this suite would go green while proving nothing. Assert they are still live. */
    printf("  non-vacuity: the induced failures must still be failing\n");
    expect(gate_quant_fails()     != 0, "quant fixture still returns failure");
    expect(gate_gemv_fails()      != 0, "GEMV fixture still returns failure");
    expect(gate_dispatch_fails()  != 0, "dispatch fixture still returns failure");
    expect(gate_fwd_slice_fails() != 0, "forward-slice fixture still returns failure");
    expect(gate_quant_passes()    == 0, "passing fixture still returns success");

    printf("  gate propagation: %d checks, %d failed\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}