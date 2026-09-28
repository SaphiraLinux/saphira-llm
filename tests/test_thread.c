/*
 * test_thread.c — topology detection and the worker pool.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Correctness here is not "it ran". A work-stealing pool can produce
 * perfectly correct output while running on one thread, which is exactly the
 * failure this file exists to catch: an earlier version of the pool counted
 * every element exactly once and still used a single CPU, and the only way
 * that was caught was by counting which thread did which chunk.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* must precede every system header: CPU_* and
                       * sched_getaffinity are GNU extensions */
#endif

#include "harness.h"

#include <saphira_llm/thread.h>
#include <saphira_llm/topology.h>

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* topology                                                            */
/* ------------------------------------------------------------------ */

TEST(topology_detection_is_self_consistent) {
    sllm_topology topo;
    sllm_topology_detect(&topo);

    CHECK(topo.n_cpus >= 1);
    CHECK(topo.n_cores >= 1);
    CHECK(topo.n_cores <= topo.n_cpus);
    CHECK(topo.n_cpus <= SLLM_MAX_CPUS);

    /* Every online CPU is recorded, ids are unique, and each logical CPU has a
     * package and a core. A duplicated id would silently shrink the machine. */
    int primaries = 0;
    for (int i = 0; i < topo.n_cpus; ++i) {
        CHECK(topo.cpus[i].online);
        CHECK(topo.cpus[i].id >= 0);
        if (topo.cpus[i].primary) {
            ++primaries;
        }
        for (int j = i + 1; j < topo.n_cpus; ++j) {
            if (topo.cpus[i].id == topo.cpus[j].id) {
                fprintf(stderr, "  FAIL duplicate cpu id %d\n", topo.cpus[i].id);
                sllm_tests_failed++;
            }
        }
    }
    /* Exactly one primary hardware thread per physical core. */
    CHECK_EQ_INT(primaries, topo.n_cores);

    char buf[256];
    sllm_topology_describe(&topo, buf, sizeof(buf));
    CHECK(strstr(buf, "logical cpus") != NULL);
    sllm_topology_describe(NULL, buf, sizeof(buf));
    sllm_topology_describe(&topo, NULL, 0);
    CHECK(1);

    CHECK_STR(sllm_core_class_name(SLLM_CORE_PERF), "perf");
    CHECK_STR(sllm_core_class_name(SLLM_CORE_EFFICIENCY), "efficiency");
    CHECK_STR(sllm_core_class_name(SLLM_CORE_UNKNOWN), "unknown");
}

TEST(topology_plan_is_a_permutation_of_the_online_cpus) {
    sllm_topology topo;
    sllm_topology_detect(&topo);

    int plan[SLLM_MAX_CPUS];
    const int n = sllm_topology_plan(&topo, topo.n_cpus, plan, SLLM_MAX_CPUS);
    CHECK_EQ_INT(n, topo.n_cpus);

    /* No duplicates, and every id is one we detected. */
    for (int i = 0; i < n; ++i) {
        bool found = false;
        for (int k = 0; k < topo.n_cpus; ++k) {
            if (topo.cpus[k].id == plan[i]) {
                found = true;
                break;
            }
        }
        CHECK(found);
        for (int j = i + 1; j < n; ++j) {
            CHECK(plan[i] != plan[j]);
        }
    }

    /* Asking for fewer threads must yield a prefix of the plan, so that thread
     * 0 keeps its CPU as the count grows. */
    if (topo.n_cpus >= 2) {
        int small[SLLM_MAX_CPUS];
        const int m = sllm_topology_plan(&topo, 2, small, SLLM_MAX_CPUS);
        CHECK_EQ_INT(m, 2);
        CHECK_EQ_INT(small[0], plan[0]);
        CHECK_EQ_INT(small[1], plan[1]);
    }

    /*
     * A nonsensical request means "one thread", not "all of them". Clamping a
     * negative count to the whole machine would be a surprising reading of a
     * typo, and a caller that wants every CPU has to say so.
     */
    CHECK_EQ_INT(sllm_topology_plan(&topo, 0, plan, SLLM_MAX_CPUS), 1);
    CHECK_EQ_INT(sllm_topology_plan(&topo, -5, plan, SLLM_MAX_CPUS), 1);
    CHECK_EQ_INT(sllm_topology_plan(&topo, topo.n_cpus, plan, 1), 1);
    CHECK_EQ_INT(sllm_topology_plan(NULL, 4, plan, SLLM_MAX_CPUS), 0);
    CHECK_EQ_INT(sllm_topology_plan(&topo, 4, NULL, 4), 0);
}

TEST(topology_never_recommends_full_occupancy) {
    sllm_topology topo;
    sllm_topology_detect(&topo);

    const int rec = sllm_topology_recommended_threads(&topo, 0);
    CHECK(rec >= 1);
    /*
     * The reference implementation loses about seventeen times at
     * threads == logical CPUs because its barrier spins. Leaving one hardware
     * thread free is the cheapest insurance against that class of failure, so
     * the recommendation must not fill the machine on a multi-CPU host.
     */
    if (topo.n_cpus > 1) {
        CHECK(rec < topo.n_cpus);
    }
    /* A ceiling is respected. */
    if (topo.n_cpus >= 4) {
        CHECK(sllm_topology_recommended_threads(&topo, 2) <= 2);
    }
    CHECK_EQ_INT(sllm_topology_recommended_threads(NULL, 0), 1);
}

/* ------------------------------------------------------------------ */
/* the pool                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    long  per_thread[64];
    int   distinct_threads;
    int   distinct_cpus;
} tally;

static pthread_mutex_t g_tally_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_t       g_seen[64];
static int             g_n_seen = 0;
static tally           g_tally;
static int             g_tally_n = 0;

static int register_self(void) {
    const pthread_t self = pthread_self();
    pthread_mutex_lock(&g_tally_mtx);
    for (int i = 0; i < g_n_seen; ++i) {
        if (pthread_equal(g_seen[i], self)) {
            pthread_mutex_unlock(&g_tally_mtx);
            return i;
        }
    }
    int slot = 0;
    if (g_n_seen < 64) {
        g_seen[g_n_seen] = self;
        g_tally.per_thread[g_n_seen] = 0;
        slot = g_n_seen++;
    }
    pthread_mutex_unlock(&g_tally_mtx);
    return slot;
}

static void tally_reset(int nthreads) {
    pthread_mutex_lock(&g_tally_mtx);
    memset(&g_tally, 0, sizeof(g_tally));
    g_n_seen = 0;
    g_tally_n = nthreads;
    pthread_mutex_unlock(&g_tally_mtx);
}

static void tally_bump(int cpu) {
    const int slot = register_self();
    pthread_mutex_lock(&g_tally_mtx);
    g_tally.per_thread[slot]++;
    (void) cpu;
    pthread_mutex_unlock(&g_tally_mtx);
}

static void tally_finish(tally * out) {
    pthread_mutex_lock(&g_tally_mtx);
    *out = g_tally;
    out->distinct_cpus = -1;   /* not asserted: sched_getcpu is not reliable here */
    out->distinct_threads = 0;
    for (int i = 0; i < g_n_seen; ++i) {
        if (g_tally.per_thread[i] > 0) {
            out->distinct_threads++;
        }
    }
    pthread_mutex_unlock(&g_tally_mtx);
}

/* Counts every element, so a chunk claimed twice or not at all is visible. */
static long g_count = 0;
static void count_job(void * ctx, size_t begin, size_t end) {
    (void) ctx;
    for (size_t i = begin; i < end; ++i) {
        __atomic_fetch_add(&g_count, 1, __ATOMIC_RELAXED);
    }
    tally_bump(sched_getcpu());
}

TEST(pool_counts_every_element_exactly_once) {
    const size_t N = 100003;   /* deliberately not a round number */

    for (int t = 1; t <= 8; ++t) {
        sllm_pool_config cfg = { .n_threads = t, .plan = NULL, .n_plan = 0, .topology = NULL };
        sllm_pool * pool = sllm_pool_create(&cfg);
        CHECK_EQ_INT(sllm_pool_threads(pool), t);

        for (int rep = 0; rep < 4; ++rep) {
            g_count = 0;
            sllm_parallel_for(pool, count_job, NULL, N, 0);
            sllm_tests_run++;
            if (g_count != (long) N) {
                sllm_tests_failed++;
                fprintf(stderr, "  FAIL %d threads rep %d: counted %ld of %zu\n",
                        t, rep, g_count, N);
            }
        }
        sllm_pool_destroy(pool);
    }
}

TEST(pool_really_uses_more_than_one_thread) {
    /*
     * The failure this catches: a pool that reports 8 threads, produces
     * perfectly correct output, and runs entirely on one of them. Counting
     * elements cannot see it. Counting which thread claimed which chunk can.
     */
    const size_t N = 262144;

    for (int t = 2; t <= 8; t *= 2) {
        sllm_pool_config cfg = { .n_threads = t, .plan = NULL, .n_plan = 0, .topology = NULL };
        sllm_pool * pool = sllm_pool_create(&cfg);
        CHECK_EQ_INT(sllm_pool_threads(pool), t);

        tally_reset(t);
        /* Several regions, so a thread that only ever missed the work by bad
         * luck gets another chance. */
        for (int rep = 0; rep < 16; ++rep) {
            sllm_parallel_for(pool, count_job, NULL, N, 0);
        }
        tally got;
        tally_finish(&got);

        sllm_tests_run++;
        if (got.distinct_threads < 2) {
            sllm_tests_failed++;
            fprintf(stderr,
                "  FAIL %d threads: only %d thread(s) ever claimed a chunk -- "
                "the region is running serially\n", t, got.distinct_threads);
        }
        /* Report the split, because a heavily skewed one is worth knowing about
         * even when it is not a failure. */
        {
            long total = 0;
            for (int i = 0; i < 64; ++i) {
                total += got.per_thread[i];
            }
            printf("    %d threads: %d active, busiest %ld of %ld chunks\n",
                   t, got.distinct_threads, got.per_thread[0], total);
        }
        sllm_pool_destroy(pool);
    }
}

TEST(pool_handles_edge_cases) {
    sllm_pool_config cfg = { .n_threads = 4, .plan = NULL, .n_plan = 0, .topology = NULL };
    sllm_pool * pool = sllm_pool_create(&cfg);

    /* Zero work must be a no-op, not a hang. */
    g_count = 0;
    sllm_parallel_for(pool, count_job, NULL, 0, 0);
    CHECK_EQ_INT(g_count, 0);

    /* One element. */
    g_count = 0;
    sllm_parallel_for(pool, count_job, NULL, 1, 0);
    CHECK_EQ_INT(g_count, 1);

    /* A NULL job must be ignored rather than called. */
    sllm_parallel_for(pool, NULL, NULL, 1000, 0);

    /* A grain larger than the work must still be correct. */
    g_count = 0;
    sllm_parallel_for(pool, count_job, NULL, 100, 1000);
    CHECK_EQ_INT(g_count, 100);

    /* A NULL pool runs inline. */
    g_count = 0;
    sllm_parallel_for(NULL, count_job, NULL, 777, 0);
    CHECK_EQ_INT(g_count, 777);

    CHECK(sllm_pool_threads(NULL) == 1);
    sllm_pool_destroy(pool);
    sllm_pool_destroy(NULL);
    CHECK(1);
}

TEST(pool_survives_create_destroy_cycles) {
    /*
     * The deadlock this catches: a worker parked in a wait that does not
     * re-check shutdown, so join never returns. It hung at every thread count
     * above one and would have shown up as a hung test run, which is a poor
     * way to find out.
     */
    for (int cycle = 0; cycle < 4; ++cycle) {
        for (int t = 1; t <= 6; ++t) {
            sllm_pool_config cfg = { .n_threads = t, .plan = NULL, .n_plan = 0, .topology = NULL };
            sllm_pool * pool = sllm_pool_create(&cfg);
            for (int rep = 0; rep < 3; ++rep) {
                sllm_parallel_for(pool, count_job, NULL, 10000, 0);
            }
            sllm_pool_destroy(pool);
        }
    }
    CHECK(1);
}

TEST(pool_restores_the_callers_affinity) {
    /*
     * If the caller stays pinned after destroy, every later measurement in
     * the process is contaminated. Comparing the mask before and after is the
     * only way to notice.
     */
    cpu_set_t before, after;
    CPU_ZERO(&before);
    CPU_ZERO(&after);
    sched_getaffinity(0, sizeof(before), &before);
    const int n_before = CPU_COUNT(&before);

    sllm_topology topo;
    sllm_topology_detect(&topo);
    int plan[SLLM_MAX_CPUS];
    const int pn = sllm_topology_plan(&topo, topo.n_cpus, plan, SLLM_MAX_CPUS);

    sllm_pool_config cfg = { .n_threads = 4, .plan = plan, .n_plan = pn, .topology = &topo };
    sllm_pool * pool = sllm_pool_create(&cfg);
    sllm_parallel_for(pool, count_job, NULL, 10000, 0);
    sllm_pool_destroy(pool);

    sched_getaffinity(0, sizeof(after), &after);
    sllm_tests_run++;
    if (CPU_COUNT(&after) != n_before) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL caller affinity not restored: %d cpus before, %d after\n",
                n_before, CPU_COUNT(&after));
    }
}

void sllm_test_thread(void) {
    printf("topology\n");
    RUN(topology_detection_is_self_consistent);
    RUN(topology_plan_is_a_permutation_of_the_online_cpus);
    RUN(topology_never_recommends_full_occupancy);

    printf("thread\n");
    RUN(pool_counts_every_element_exactly_once);
    RUN(pool_really_uses_more_than_one_thread);
    RUN(pool_handles_edge_cases);
    RUN(pool_survives_create_destroy_cycles);
    RUN(pool_restores_the_callers_affinity);
}
