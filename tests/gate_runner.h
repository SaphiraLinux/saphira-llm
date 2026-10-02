/*
 * gate_runner.h — the one path by which a product gate's verdict reaches the build.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * WHY THIS IS A HEADER RATHER THAN A STATIC IN main.c
 *
 * The four product gates each RETURN a status. For an entire session main.c called
 * them bare, discarding that return value while the gates' own pass/fail counters
 * stayed local to each gate. The suite therefore printed "35993 checks, 0 failed"
 * and the build went green WHILE the forward slice was reporting failures on the
 * same run. It stayed latent only because every gate happened to pass; the first
 * genuine failure exposed it.
 *
 * The lesson generalises past that one bug: a green summary line is NOT evidence
 * that the thing it summarises can fail. A gate never observed failing is not known
 * to work.
 *
 * So the runner lives here, in a place the regression can reach, and the regression
 * in tests/test_gate_propagation.c exercises THIS EXACT FUNCTION rather than a copy
 * of it. A test of a reimplementation proves the reimplementation.
 */

#ifndef SAPHIRA_GATE_RUNNER_H
#define SAPHIRA_GATE_RUNNER_H

#include <stdio.h>

extern int sllm_tests_run;
extern int sllm_tests_failed;

/* A product gate returns 0 to pass and non-zero to fail. */
typedef int (*sllm_product_gate)(void);

/* Fold a gate's own verdict into the global counters, so the verdict is
 * AUTHORITATIVE and a dropped return value can no longer pass silently. */
static void sllm_run_product_gate(sllm_product_gate fn, const char *name) {
    const int rc = fn();
    sllm_tests_run += 1;
    if (rc != 0) {
        sllm_tests_failed += 1;
        printf("  GATE FAIL    %s -> returned %d, and this verdict is counted\n", name, rc);
    } else {
        printf("  GATE ok      %s\n", name);
    }
}

#endif /* SAPHIRA_GATE_RUNNER_H */