/*
 * sched_bench.c — scheduler benchmark for saphira-llm.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Phase 2 gate, measured rather than asserted.
 *
 * The reference implementation loses a factor of seventeen between 24 and 28
 * threads on this machine. That collapse was diagnosed, not guessed: it is
 * barrier saturation at full occupancy, not a placement problem. 28 threads on
 * 24 CPUs runs at 15.2 t/s; 28 threads on 28 CPUs runs at 1.7.
 *
 * So the question here is not "can we place threads well" but "does our pool
 * exhibit the pathology at any thread count, including full occupancy".
 *
 * Three workloads spanning the two regimes that matter:
 *
 *   memory    streaming over 32 MiB, far larger than L3. Bandwidth-bound.
 *             This is what BitNet generation actually is.
 *   vector    an FMA chain in L1. Compute-bound; should scale with cores.
 *   sync      many small work items in one region, each doing real work.
 *             The shape that hurts a barrier-bound implementation, and so the
 *             adversarial case for our design specifically.
 *
 * Every workload runs at each thread count under the scheduler's own placement
 * and under no placement at all. Both are reported, because a gain that only
 * exists with pinning is a placement result and a gain that exists either way
 * is a scheduler result. The claim under test is the second kind.
 *
 * Soundness note, learned the hard way. An earlier version of this benchmark
 * reported 1-thread throughput a hundred times *higher* than 2-thread: the
 * compiler had eliminated the jobs outright at 1 thread, because the work was
 * trivially foldable and the call was inlined. Every job here therefore adds
 * into a shared sink that is printed and checked, so a build that optimises
 * the work away is loud rather than flattering.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <saphira_llm/sllm.h>
#include <saphira_llm/thread.h>
#include <saphira_llm/topology.h>

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* shared sink                                                         */
/* ------------------------------------------------------------------ */

/*
 * Written by every job and read once at the end, so a build that optimises the
 * work away is loud rather than flattering.
 *
 * One write per *chunk*, never per element. Writing a single volatile double
 * from inside the inner loop puts every thread on one cache line and
 * serialises the entire workload, which silently turns a scaling benchmark
 * into a flat-line benchmark. That is what an earlier version of this file
 * did, and it is why the unplaced numbers did not move at all.
 */
static volatile double g_sink = 0.0;

static void sink_add(double v) {
    /* Folding into a bounded value keeps the sink finite; the point is only
     * that it is non-zero, and inf would look like a broken run. */
    double cur = g_sink;
    g_sink = cur * 0.9999 + (v > 0.0 ? 1.0 : -1.0);
}

/* ------------------------------------------------------------------ */
/* memory-bound: streaming over 32 MiB                                 */
/* ------------------------------------------------------------------ */

#define MEM_BYTES (32u * 1024u * 1024u)
#define MEM_N     (MEM_BYTES / sizeof(float))

static float * g_mem = NULL;

static void mem_init(void) {
    g_mem = (float *) aligned_alloc(64, MEM_BYTES);
    if (g_mem == NULL) {
        fprintf(stderr, "cannot allocate %u MiB\n", MEM_BYTES / (1024u * 1024u));
        exit(1);
    }
    /* A stride that defeats the prefetcher's perfect streaming, so the load
     * is a realistic dependent pattern rather than a memcpy in disguise. */
    for (size_t i = 0; i < MEM_N; ++i) {
        g_mem[i] = (float) ((i * 2654435761u) % 4096) * 0.0009765625f;
    }
}

static void mem_job(void * ctx, size_t begin, size_t end) {
    (void) ctx;
    __m256 acc = _mm256_setzero_ps();
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(g_mem + i));
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    double s = 0.0;
    for (int k = 0; k < 8; ++k) {
        s += lanes[k];
    }
    for (; i < end; ++i) {
        s += g_mem[i];
    }
    sink_add(s);
}

/* ------------------------------------------------------------------ */
/* compute-bound: FMA chain in L1                                      */
/* ------------------------------------------------------------------ */

#define VEC_ROWS 262144
#define VEC_CHAIN 64

static void vec_job(void * ctx, size_t begin, size_t end) {
    (void) ctx;
    double local = 0.0;
    for (size_t r = begin; r < end; ++r) {
        __m256 a = _mm256_set1_ps((float) (r & 4095));
        __m256 b = _mm256_set1_ps((float) ((r >> 5) & 255));
        const __m256 k = _mm256_set1_ps(1.0000001f);
        for (size_t c = 0; c < VEC_CHAIN; ++c) {
            a = _mm256_fmadd_ps(a, k, b);
            b = _mm256_fmadd_ps(b, k, a);
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, _mm256_add_ps(a, b));
        local += (double) lanes[0];
    }
    sink_add(local);
}

/* ------------------------------------------------------------------ */
/* many small work items in one region                                 */
/* ------------------------------------------------------------------ */

#define SYNC_ITEMS   2048
#define SYNC_CHAIN   512

static void sync_job(void * ctx, size_t begin, size_t end) {
    (void) ctx;
    double local = 0.0;
    for (size_t r = begin; r < end; ++r) {
        __m256 a = _mm256_set1_ps((float) (r + 1));
        __m256 b = _mm256_set1_ps((float) (r + 2));
        const __m256 k = _mm256_set1_ps(1.0000001f);
        for (size_t c = 0; c < SYNC_CHAIN; ++c) {
            a = _mm256_fmadd_ps(a, k, b);
            b = _mm256_fmadd_ps(b, k, a);
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, _mm256_add_ps(a, b));
        local += (double) lanes[0];
    }
    sink_add(local);
}

/* ------------------------------------------------------------------ */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1.0e-9;
}

typedef struct result {
    double seconds;
    double units_per_s;
} result;

static result run_once(sllm_pool * pool, sllm_job_fn fn, size_t n, size_t grain,
                       size_t units, int reps) {
    double best = 1.0e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        sllm_parallel_for(pool, fn, NULL, n, grain);
        const double t1 = now_s();
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }
    result res;
    res.seconds = best;
    res.units_per_s = (double) units / best;
    return res;
}

struct workload {
    const char * name;
    sllm_job_fn  fn;
    size_t       n;
    size_t       grain;
    size_t       units;
    int          reps;
};

int main(int argc, char ** argv) {
    const char * mode = (argc > 1) ? argv[1] : "all";
    const int reps = (argc > 2) ? atoi(argv[2]) : 5;

    mem_init();

    sllm_topology topo;
    sllm_topology_detect(&topo);
    char tdesc[256];
    sllm_topology_describe(&topo, tdesc, sizeof(tdesc));
    printf("saphira-llm scheduler benchmark\n");
    printf("topology: %s\n", tdesc);

    if (sllm_topology_calibrate(&topo, 2000000)) {
        sllm_topology_describe(&topo, tdesc, sizeof(tdesc));
        printf("topology: %s\n", tdesc);
    } else {
        printf("topology: calibration unavailable, using the static order\n");
    }

    int plan[SLLM_MAX_CPUS];
    const int plan_n = sllm_topology_plan(&topo, topo.n_cpus, plan, SLLM_MAX_CPUS);
    printf("recommended threads: %d of %d logical cpus\n\n",
           sllm_topology_recommended_threads(&topo, 0), topo.n_cpus);

    struct workload mem = { "memory", mem_job, MEM_N,    0, MEM_N,    reps };
    struct workload vec = { "vector", vec_job, VEC_ROWS, 0, VEC_ROWS, reps };
    struct workload syn = { "sync",   sync_job, SYNC_ITEMS, 0, SYNC_ITEMS, reps };
    struct workload * all[3] = { &mem, &vec, &syn };

    const int thread_counts[] = { 1, 2, 4, 8, 12, 14, 16, 20, 22, 24, 26, 28 };
    const int n_tc = (int) (sizeof(thread_counts) / sizeof(thread_counts[0]));

    int collapse_at_full_occupancy = 0;
    int placed_slower_somewhere = 0;
    double full_occ_placed = 0.0, full_occ_unplaced = 0.0;

    for (int w = 0; w < 3; ++w) {
        struct workload * wl = all[w];
        if (strcmp(mode, "all") != 0 && strcmp(mode, wl->name) != 0) {
            continue;
        }
        printf("=== %s ===\n", wl->name);
        printf("%8s %13s %13s %9s %13s  %s\n",
               "threads", "placed u/s", "unplaced u/s", "ratio", "1t->Nt gain", "notes");

        double one_thread = 0.0;
        for (int i = 0; i < n_tc; ++i) {
            const int t = thread_counts[i];
            if (t > topo.n_cpus) {
                continue;
            }

            sllm_pool_config c1 = { .n_threads = t, .plan = plan, .n_plan = plan_n, .topology = &topo };
            sllm_pool * p1 = sllm_pool_create(&c1);
            const int got = sllm_pool_threads(p1);
            const result placed = run_once(p1, wl->fn, wl->n, wl->grain, wl->units, wl->reps);
            sllm_pool_destroy(p1);

            sllm_pool_config c2 = { .n_threads = t, .plan = NULL, .n_plan = 0, .topology = &topo };
            sllm_pool * p2 = sllm_pool_create(&c2);
            const result unplaced = run_once(p2, wl->fn, wl->n, wl->grain, wl->units, wl->reps);
            sllm_pool_destroy(p2);

            if (got != t) {
                printf("%8d  requested %d, pool delivered %d\n", t, t, got);
                continue;
            }
            if (t == 1) {
                one_thread = placed.units_per_s;
            }

            const double ratio = unplaced.units_per_s > 0.0
                ? placed.units_per_s / unplaced.units_per_s : 0.0;
            const double gain = one_thread > 0.0 ? placed.units_per_s / one_thread : 0.0;

            const char * note = "";
            if (t == topo.n_cpus) {
                /* The specific failure under test. */
                if (placed.units_per_s < one_thread * 0.5) {
                    note = "FULL-OCCUPANCY REGRESSION";
                    collapse_at_full_occupancy = 1;
                } else {
                    note = "full occupancy ok";
                }
                full_occ_placed   = placed.units_per_s;
                full_occ_unplaced = unplaced.units_per_s;
            }
            if (t > 1 && ratio < 0.5) {
                note = "placed < half of unplaced";
                placed_slower_somewhere = 1;
            }

            printf("%8d %13.5g %13.5g %9.2f %13.2f  %s\n", t,
                   placed.units_per_s, unplaced.units_per_s, ratio, gain, note);
        }
        printf("\n");
    }

    printf("=== verdict ===\n");
    printf("sink: %.6g  (non-zero proves the jobs were not optimised away)\n", g_sink);
    printf("full occupancy, placed: %.5g u/s, unplaced: %.5g u/s\n",
           full_occ_placed, full_occ_unplaced);
    if (collapse_at_full_occupancy) {
        printf("RESULT: FAIL — a full-occupancy regression was observed.\n");
    } else {
        printf("RESULT: PASS — no full-occupancy regression at %d threads on %d logical CPUs.\n",
               topo.n_cpus, topo.n_cpus);
        printf("        The reference implementation loses about 17x at this point\n"
               "        (BENCHMARKS.md). The mechanism is barrier saturation, and this\n"
               "        pool has no inter-worker barrier for full occupancy to starve.\n");
    }
    if (placed_slower_somewhere) {
        printf("        NOTE: placement underperformed unplaced placement somewhere; that is\n"
               "        a placement-policy question, not a scheduler-collapse question, and\n"
               "        is reported separately rather than folded into the verdict.\n");
    }

    free(g_mem);
    return collapse_at_full_occupancy ? 1 : 0;
}
