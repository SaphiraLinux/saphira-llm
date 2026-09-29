/*
 * prof.h — in-process sampling profiler.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * See prof.c for why the target machine has no usable `perf`.
 */

#ifndef SAPHIRA_PROF_H
#define SAPHIRA_PROF_H

#include <stddef.h>

/* ~4 ms worst case at 250 Hz over a 20 s run is ample; the cap is a guard
 * against a long run silently wrapping the buffer. */
#define SLLM_PROF_MAX (1u << 22)

/* Begin sampling. Returns 0 on success. */
int sllm_prof_start(unsigned hz);

/* Stop sampling. */
void sllm_prof_stop(void);

/* Write captured program counters as link-time addresses. Returns 0 on
 * success. Call after sllm_prof_stop(). */
int sllm_prof_dump(const char * path);

size_t sllm_prof_count(void);

#endif
