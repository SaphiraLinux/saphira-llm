/*
 * thread.c — native pthread worker pool and parallel-for.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Design note, because it is the whole point of this file.
 *
 * The reference implementation we are measured against loses a factor of
 * seventeen on the target between 24 and 28 threads, reproducibly, with low
 * variance. The cause was measured rather than guessed: it is not placement,
 * not the number of cores, and not memory bandwidth. It is barrier saturation.
 * 28 threads on 24 CPUs runs at 15.2 t/s; 28 threads on 28 CPUs runs at 1.7.
 * Every logical CPU ends up spinning on an atomic barrier, and the thread that
 * has to break the barrier cannot get a CPU to run on.
 *
 * The fix here is structural rather than a tuning knob: workers never wait for
 * one another. Chunks are claimed from a single atomic cursor, and only the
 * calling thread ever blocks, on a condition variable. There is no inter-worker
 * barrier for full occupancy to starve, so the failure mode cannot occur.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  /* must precede every system header */
#endif

#include <saphira_llm/thread.h>
#include <saphira_llm/log.h>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* The number of chunks handed out per element of work. Small enough to keep
 * the tail busy, large enough that the atomic cursor is not the bottleneck. */
#define SLLM_CHUNKS_PER_THREAD 16

/*
 * How long a worker looks for a new region before it sleeps.
 *
 * Sized from a measurement rather than picked: decode posts a region every few
 * tens of microseconds, so a budget in the tens of thousands of pause
 * instructions covers several regions and turns most wakeups into an atomic
 * load. It is bounded on purpose -- an unbounded spin is the reference's bug,
 * not a fix for it -- and a worker that exhausts it falls through to the
 * condition variable exactly as before.
 */
#define SLLM_WORKER_SPIN 40000
#define SLLM_MIN_CHUNK 1

typedef struct sllm_worker sllm_worker;

struct sllm_pool {
    pthread_t       * threads;
    sllm_worker     * workers;
    int               n_threads;      /* workers, excluding the caller */
    int               total;          /* workers + 1, the caller       */
    int              * plan;
    int               n_plan;

    /* The region currently being executed. */
    sllm_job_fn        job;
    void             * job_ctx;
    size_t             job_n;
    size_t             job_grain;
    size_t             job_chunk;
    int                job_generation;

    cpu_set_t          original_mask;   /* the caller's, restored on destroy */
    bool               have_original;

    pthread_mutex_t    mtx;
    pthread_cond_t     work_ready;     /* workers wait here              */
    pthread_cond_t     work_done;      /* the caller waits here           */
    size_t             cursor;         /* next unclaimed chunk            */
    int                outstanding;    /* chunks dispatched, not finished */
    int                arrived;        /* workers that have entered        */
    bool               shutdown;
    bool               dispatch;       /* a region is active              */

};

static __thread sllm_pool * t_pool = NULL;

/* ------------------------------------------------------------------ */

/*
 * `index` is 0 for the calling thread and 1..n-1 for the workers, because
 * plan[0] belongs to the caller. Sharing plan[0] with worker 0 would put two
 * threads on one CPU while leaving another CPU idle, which is a measurable
 * loss for exactly the case placement is meant to help.
 */
static void apply_affinity(const sllm_pool * pool, int index) {
    if (pool->plan == NULL || pool->n_plan <= 0) {
        return;
    }
    if (index < 0 || index >= pool->n_plan) {
        return;
    }
    const int cpu = pool->plan[index];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        /* Not fatal. A pool with no placement still parallelises correctly,
         * just less predictably, and saying so once beats a hard failure on a
         * system where affinity is restricted. */
        sllm_log(SLLM_LOG_DEBUG, "could not pin thread %d to cpu %d", index, cpu);
    }
}

/* Claim up to one chunk. Returns false when the region is exhausted. */
static bool claim_chunk(sllm_pool * pool, size_t * begin, size_t * end) {
    const size_t start = __atomic_fetch_add(&pool->cursor, pool->job_chunk,
                                            __ATOMIC_RELAXED);
    if (start >= pool->job_n) {
        return false;
    }
    size_t stop = start + pool->job_chunk;
    if (stop > pool->job_n) {
        stop = pool->job_n;
    }
    *begin = start;
    *end = stop;
    return true;
}

static void run_chunks(sllm_pool * pool) {
    size_t begin, end;
    while (claim_chunk(pool, &begin, &end)) {
        pool->job(pool->job_ctx, begin, end);
    }

    pthread_mutex_lock(&pool->mtx);
    if (--pool->outstanding == 0) {
        pthread_cond_broadcast(&pool->work_done);
    }
    pthread_mutex_unlock(&pool->mtx);
}

/*
 * Announce that a worker has entered the current region.
 *
 * This is the piece whose absence made the pool serial. The calling thread
 * used to claim chunks in a tight loop the moment it dispatched, and on a
 * multi-core machine that is a race it wins: it claims the whole region before
 * any freshly-signalled worker has been scheduled, so every worker wakes to
 * find nothing to do and the "parallel" region runs on one thread. Pinning
 * hid the bug, because a worker pinned to another core gets a chance to run
 * the instant it is signalled -- which is why the placed numbers looked fine
 * while the unplaced ones were pinned at exactly single-thread throughput.
 *
 * The caller waits until the workers have actually arrived before it starts,
 * which fixed the starvation above. The reasoning recorded here was that this
 * "costs nothing like what it replaces and it cannot starve at full
 * occupancy".
 *
 * The second half of that was wrong, and the decode path is what proved it.
 * cond_var wait does avoid burning a CPU, but it also means a woken worker has
 * to be *scheduled* before it can arrive. On a machine where every logical CPU
 * already has a runnable thread on it, nothing is scheduled until something
 * yields, so the arrival wait becomes a full-occupancy rendezvous -- the exact
 * pathology this file was written to claim was impossible by construction.
 *
 * The Phase 2 microbenchmarks never saw it because their regions are few and
 * large, so a rendezvous per region is amortised over a lot of work. Decode
 * issues 211 regions per generated token, one per projection per layer plus
 * the output projection, each only tens of microseconds long. At 28 threads
 * that is roughly 5900 wake-and-arrive round trips per token, and measured
 * end-to-end throughput fell from 21.6 t/s at 4 threads to 6.25 t/s at 28 --
 * monotonically worse with every extra thread, which is the opposite of what a
 * bandwidth limit does.
 *
 * So the rendezvous stays, because removing it reintroduces the starvation
 * this comment was written about, but the workers now spin briefly for a new
 * region before sleeping. Regions arrive back to back, so a worker that is
 * still spinning when the next one is posted claims a chunk immediately and
 * the rendezvous costs nothing. The spin is bounded and falls back to the
 * condition variable, so it cannot become the indefinite barrier that destroys
 * the reference implementation.
 */
static void worker_arrived(sllm_pool * pool) {
    pthread_mutex_lock(&pool->mtx);
    if (pool->arrived < pool->n_threads) {
        ++pool->arrived;
        pthread_cond_broadcast(&pool->work_ready);
    }
    pthread_mutex_unlock(&pool->mtx);
}

struct sllm_worker {
    sllm_pool * pool;
    int         index;   /* 0-based among the spawned workers */
};

static void * worker_main(void * arg) {
    sllm_worker * w = (sllm_worker *) arg;
    sllm_pool * pool = w->pool;
    t_pool = pool;
    /* Plan index 0 belongs to the calling thread, so workers start at 1. */
    apply_affinity(pool, w->index + 1);

    int seen_generation = 0;
    for (;;) {
        /*
         * Bounded spin for the next region before sleeping.
         *
         * Read with acquire so the job, context and cursor published by the
         * caller are visible, and without the mutex so waking costs an atomic
         * load rather than a futex round trip. The loop is bounded, so a
         * worker that finds no work parks in cond_wait as before.
         */
        for (int spin = 0; spin < SLLM_WORKER_SPIN; ++spin) {
            if (__atomic_load_n(&pool->shutdown, __ATOMIC_ACQUIRE)) { break; }
            if (__atomic_load_n(&pool->dispatch, __ATOMIC_ACQUIRE) &&
                __atomic_load_n(&pool->job_generation, __ATOMIC_ACQUIRE) != seen_generation) {
                break;
            }
            __builtin_ia32_pause();
        }

        pthread_mutex_lock(&pool->mtx);

        /*
         * One wait, not two.
         *
         * The obvious shape -- an outer wait for "a region is active" and an
         * inner wait for "the generation has advanced" -- deadlocks. A worker
         * that finishes its chunks while the region is still active falls past
         * the outer wait and parks in the inner one, which does not re-check
         * shutdown. Shutdown then broadcasts, the worker wakes, re-checks the
         * same generation it already saw, and waits again. join never returns.
         *
         * That was a real hang here at every thread count above one. Checking
         * shutdown on every wake, in a single loop, removes the whole class.
         */
        for (;;) {
            if (pool->shutdown) {
                pthread_mutex_unlock(&pool->mtx);
                t_pool = NULL;
                return NULL;
            }
            if (pool->dispatch && pool->job_generation != seen_generation) {
                break;
            }
            pthread_cond_wait(&pool->work_ready, &pool->mtx);
        }

        seen_generation = pool->job_generation;
        pthread_mutex_unlock(&pool->mtx);

        worker_arrived(pool);
        run_chunks(pool);
    }
}

/* ------------------------------------------------------------------ */

static size_t choose_grain(size_t n, int total, size_t requested) {
    if (requested > 0) {
        return requested;
    }
    if (n == 0) {
        return 1;
    }
    const size_t target = (size_t) total * SLLM_CHUNKS_PER_THREAD;
    size_t g = n / (target > 0 ? target : 1);
    if (g < SLLM_MIN_CHUNK) {
        g = SLLM_MIN_CHUNK;
    }
    return g;
}

/* Capture the calling thread's current affinity mask, before we change it. */
static int get_caller_affinity(sllm_pool * pool) {
    if (sched_getaffinity(0, sizeof(pool->original_mask), &pool->original_mask) == 0) {
        pool->have_original = true;
        return 1;
    }
    pool->have_original = false;
    return 0;
}

sllm_pool * sllm_pool_create(const sllm_pool_config * cfg) {
    sllm_pool * pool = calloc(1, sizeof(*pool));
    if (pool == NULL) {
        return NULL;
    }

    int want = (cfg != NULL && cfg->n_threads > 0) ? cfg->n_threads : 1;
    if (want > SLLM_MAX_CPUS) {
        want = SLLM_MAX_CPUS;
    }

    pool->total = want;
    pool->n_threads = want - 1;   /* the caller is a worker too */

    if (cfg != NULL && cfg->plan != NULL && cfg->n_plan > 0) {
        pool->n_plan = cfg->n_plan < want ? cfg->n_plan : want;
        pool->plan = calloc((size_t) pool->n_plan, sizeof(int));
        if (pool->plan != NULL) {
            memcpy(pool->plan, cfg->plan, (size_t) pool->n_plan * sizeof(int));
        } else {
            pool->n_plan = 0;
        }
    }

    pthread_mutex_init(&pool->mtx, NULL);
    pthread_cond_init(&pool->work_ready, NULL);
    pthread_cond_init(&pool->work_done, NULL);
    if (pool->n_threads > 0) {
        pool->threads = calloc((size_t) pool->n_threads, sizeof(pthread_t));
        pool->workers = calloc((size_t) pool->n_threads, sizeof(sllm_worker));
        if (pool->threads == NULL || pool->workers == NULL) {
            pool->n_threads = 0;
            pool->total = 1;
        }
    }

    for (int i = 0; i < pool->n_threads; ++i) {
        pool->workers[i].pool  = pool;
        pool->workers[i].index = i;
        if (pthread_create(&pool->threads[i], NULL, worker_main, &pool->workers[i]) != 0) {
            /* Fewer threads than asked for is a degradation, not a failure.
             * Report it rather than pretending we got what we asked for. */
            sllm_log(SLLM_LOG_WARN, "could not create worker %d of %d: %s",
                     i, pool->n_threads, strerror(errno));
            pool->n_threads = i;
            pool->total = i + 1;
            break;
        }
    }

    /*
     * Remember the caller's affinity and restore it on destroy. Without this,
     * a program that creates and destroys pools repeatedly -- a benchmark, or
     * a server handling a request at a time -- leaves itself pinned to one CPU
     * for the rest of its life, and every later measurement is contaminated.
     */
    if (get_caller_affinity(pool) != 0) {
        apply_affinity(pool, 0);
    }

    sllm_log(SLLM_LOG_DEBUG, "pool created with %d threads (%d workers + caller)%s",
             pool->total, pool->n_threads, pool->n_plan ? ", placed" : "");
    return pool;
}

void sllm_pool_destroy(sllm_pool * pool) {
    if (pool == NULL) {
        return;
    }
    pthread_mutex_lock(&pool->mtx);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mtx);

    for (int i = 0; i < pool->n_threads; ++i) {
        (void) pthread_join(pool->threads[i], NULL);
    }

    if (pool->have_original) {
        (void) sched_setaffinity(0, sizeof(pool->original_mask), &pool->original_mask);
    }

    pthread_mutex_destroy(&pool->mtx);
    pthread_cond_destroy(&pool->work_ready);
    pthread_cond_destroy(&pool->work_done);
    free(pool->threads);
    free(pool->workers);
    free(pool->plan);
    free(pool);
}

int sllm_pool_threads(const sllm_pool * pool) {
    return pool != NULL ? pool->total : 1;
}

sllm_pool * sllm_pool_current(void) {
    return t_pool;
}

void sllm_parallel_for(sllm_pool * pool, sllm_job_fn fn, void * ctx,
                       size_t n, size_t grain) {
    if (fn == NULL || n == 0) {
        return;
    }
    if (pool == NULL || pool->total <= 1) {
        fn(ctx, 0, n);
        return;
    }

    pool->job_grain = choose_grain(n, pool->total, grain);
    pool->job_chunk = n / ((size_t) pool->total * SLLM_CHUNKS_PER_THREAD);
    if (pool->job_chunk < pool->job_grain) {
        pool->job_chunk = pool->job_grain;
    }
    if (pool->job_chunk > n) {
        pool->job_chunk = n;
    }

    pthread_mutex_lock(&pool->mtx);
    pool->job          = fn;
    pool->job_ctx      = ctx;
    pool->job_n        = n;
    pool->cursor       = 0;
    pool->arrived      = 0;
    pool->outstanding  = pool->n_threads + 1;   /* workers plus this thread */
    __atomic_store_n(&pool->dispatch, true, __ATOMIC_RELEASE);
    __atomic_store_n(&pool->job_generation, pool->job_generation + 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&pool->work_ready);

    /*
     * Wait for the workers to be running before starting to claim. Without
     * this the caller starves them and the region is serial no matter how many
     * threads were created. The escape condition matters for regions too small
     * to be worth distributing: if the workers somehow retire without arriving
     * we must not wait forever.
     */
    while (pool->arrived < pool->n_threads && pool->shutdown == false) {
        pthread_cond_wait(&pool->work_ready, &pool->mtx);
    }
    pthread_mutex_unlock(&pool->mtx);

    /*
     * Hand the CPU over once before starting.
     *
     * The arrival wait above guarantees every worker is *about to* claim, but
     * the caller is the one that woke first and is still on-CPU. Without this
     * it wins the race outright, and at two threads it won every chunk: the
     * region ran on one thread while a thread sat idle. One yield per region
     * is a few hundred nanoseconds and it is the difference between
     * "parallel" and "serial with extra threads".
     */
    sched_yield();

    /* The caller works too. A pool that leaves the initiating thread idle
     * discards a whole core, which is worse than any synchronisation cost. */
    run_chunks(pool);

    /*
     * Only this thread ever waits, and it waits on a condition variable. No
     * worker is ever blocked by another worker, so there is no barrier for a
     * fully-occupied machine to starve on.
     */
    pthread_mutex_lock(&pool->mtx);
    while (pool->outstanding > 0) {
        pthread_cond_wait(&pool->work_done, &pool->mtx);
    }
    pool->dispatch = false;
    pool->job = NULL;
    pool->job_ctx = NULL;
    pthread_mutex_unlock(&pool->mtx);
}

void sllm_parallel_rows(sllm_pool * pool, sllm_job_fn fn, void * ctx,
                        size_t rows, size_t grain) {
    sllm_parallel_for(pool, fn, ctx, rows, grain);
}
