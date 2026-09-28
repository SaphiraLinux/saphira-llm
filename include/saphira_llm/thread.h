/*
 * thread.h — native pthread worker pool and parallel-for.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_THREAD_H
#define SAPHIRA_LLM_THREAD_H

#include <stdbool.h>
#include <stddef.h>

#include <saphira_llm/status.h>
#include <saphira_llm/topology.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The job function. `ctx` is opaque to the pool. `begin` and `end` bound the
 * chunk this call must cover. Chunks are disjoint and cover the whole range
 * exactly once, so a job may be written as if it were the only thread.
 */
typedef void (*sllm_job_fn)(void * ctx, size_t begin, size_t end);

/*
 * How a pool is set up.
 *
 * `n_threads`  total workers including the calling thread, so 1 means no
 *              worker threads are created at all.
 * `plan`       per-thread CPU placement from sllm_topology_plan(), or NULL to
 *              leave placement alone.
 * `n_plan`     length of `plan`; may be less than `n_threads`, in which case
 *              the remaining workers keep the inherited mask.
 */
typedef struct sllm_pool_config {
    int                     n_threads;
    const int             * plan;
    int                     n_plan;
    const sllm_topology   * topology;
} sllm_pool_config;

typedef struct sllm_pool sllm_pool;

/*
 * Create a pool. Never returns NULL: on any failure it degrades to a
 * single-threaded pool, because a model that runs slowly is better than one
 * that refuses to start, and the caller can read the actual thread count back
 * with sllm_pool_threads().
 */
sllm_pool * sllm_pool_create(const sllm_pool_config * cfg);
void        sllm_pool_destroy(sllm_pool * pool);

/* Actual worker count, which may be lower than requested. */
int         sllm_pool_threads(const sllm_pool * pool);

/* The pool the current thread is running under, or NULL on the main thread. */
sllm_pool * sllm_pool_current(void);

/*
 * Run `fn(ctx, begin, end)` over [0, n) in parallel.
 *
 * There is no barrier between workers. Work is handed out through a single
 * atomic cursor, so workers never block one another, and the calling thread
 * takes chunks too. Only the caller waits, and it waits on a condition
 * variable rather than spinning.
 *
 * That is a deliberate response to a measurement, not a style choice. The
 * reference implementation loses a factor of seventeen between 24 and 28
 * threads on the target because its barrier spins: once every logical CPU is
 * running a spinner, the thread that has to make progress cannot be scheduled.
 * A design with no inter-worker barrier cannot exhibit that failure, whatever
 * the thread count.
 *
 * `grain` is the smallest number of elements worth handing out. Too small and
 * the atomic cursor dominates; too large and the tail is left idle. When 0, a
 * value derived from the work size is used.
 */
void sllm_parallel_for(sllm_pool * pool, sllm_job_fn fn, void * ctx,
                       size_t n, size_t grain);

/* Convenience wrappers. */
void sllm_parallel_rows(sllm_pool * pool, sllm_job_fn fn, void * ctx,
                        size_t rows, size_t grain);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_THREAD_H */
