/*
 * topo_probe.c -- diagnostics for the Phase 6 topology classification.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Why this exists.
 *
 * This probe exists because the obvious approach was wrong and the failure
 * was worth keeping a record of.
 *
 * sllm_topology_calibrate used to classify this host's cores by timing each
 * one and thresholding the result. It does not work here, and the reason is
 * specific rather than general noise: guest sched_setaffinity pins a vCPU to a
 * guest logical CPU, but the host is still free to run that vCPU's thread on
 * any host core. A single logical CPU's samples therefore swing by a factor of
 * two between rounds, at random, uncorrelated with round or CPU. Observed
 * across six rounds: ~5.08e8 and ~1.05e9, best-of-three landing high almost
 * every time, 27 of 28 logical CPUs classified full-rate at every threshold
 * from 0.60 to 0.95. Four separate runs answered 14/0, 13/1, 12/2 and 11/3
 * perf/efficiency cores on a 13900K that has 8 P-cores and 6 E-cores.
 *
 * Classification is now taken from CPUID leaf 0x1A, EAX[31:24], read pinned
 * to each logical CPU, which is why the tree still builds this: the numbers
 * below are what the timing approach was resting on, kept so the replacement
 * is checkable rather than merely asserted. On this host the hypervisor zeroes
 * that leaf, so the honest result is that core type is unavailable.
*/

#include <saphira_llm/isa.h>
#include <saphira_llm/sllm.h>
#include <saphira_llm/topology.h>

#define _GNU_SOURCE
#include <immintrin.h>
#include <sched.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

static int pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

/* Identical workload to sllm_topology_calibrate's probe, so the numbers here
 * are directly comparable with what the classifier sees. */
static double probe_once(size_t units) {
    enum { LANES = 16 };
    static float buf[LANES * 8];
    for (int i = 0; i < LANES * 8; ++i) { buf[i] = (float) (i % 13) - 6.0f; }

    __m256 a0 = _mm256_loadu_ps(buf + 0),  a1 = _mm256_loadu_ps(buf + 8);
    __m256 a2 = _mm256_loadu_ps(buf + 16), a3 = _mm256_loadu_ps(buf + 24);
    __m256 a4 = _mm256_loadu_ps(buf + 32), a5 = _mm256_loadu_ps(buf + 40);
    __m256 a6 = _mm256_loadu_ps(buf + 48), a7 = _mm256_loadu_ps(buf + 56);
    const __m256 k1 = _mm256_set1_ps(1.0000001f);
    const __m256 k2 = _mm256_set1_ps(0.9999999f);

    for (size_t i = 0; i < units / 8 + 1; ++i) {
        a0 = _mm256_fmadd_ps(a0, k1, a1); a1 = _mm256_fmadd_ps(a1, k2, a2);
        a2 = _mm256_fmadd_ps(a2, k1, a3); a3 = _mm256_fmadd_ps(a3, k2, a4);
        a4 = _mm256_fmadd_ps(a4, k1, a5); a5 = _mm256_fmadd_ps(a5, k2, a6);
        a6 = _mm256_fmadd_ps(a6, k1, a7); a7 = _mm256_fmadd_ps(a7, k2, a0);
    }
    const double t0 = now_s();
    for (size_t i = 0; i < units; ++i) {
        a0 = _mm256_fmadd_ps(a0, k1, a1); a1 = _mm256_fmadd_ps(a1, k2, a2);
        a2 = _mm256_fmadd_ps(a2, k1, a3); a3 = _mm256_fmadd_ps(a3, k2, a4);
        a4 = _mm256_fmadd_ps(a4, k1, a5); a5 = _mm256_fmadd_ps(a5, k2, a6);
        a6 = _mm256_fmadd_ps(a6, k1, a7); a7 = _mm256_fmadd_ps(a7, k2, a0);
    }
    const double t1 = now_s();

    float sink[8];
    _mm256_storeu_ps(sink, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    if (sink[0] == 12345.678f) { return 0.0; }
    return (double) units / (t1 - t0);
}

static int cmp_d(const void * a, const void * b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

int main(int argc, char ** argv) {
    size_t units = 2000000;
    int rounds = 5;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-u") && i + 1 < argc) { units = strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) { rounds = atoi(argv[++i]); }
        else { fprintf(stderr, "usage: [-u UNITS] [-r ROUNDS]\n"); return 2; }
    }

    sllm_topology topo;
    sllm_topology_detect(&topo);

    const int n = (int) topo.n_cpus;
    double (*all)[32] = calloc((size_t) n, sizeof *all);
    if (all == NULL) { return 1; }

    printf("# topology probe: %d logical cpus, %d cores, units=%zu rounds=%d\n",
           n, topo.n_cores, units, rounds);
    printf("baseline_isa %s\n\n", SLLM_BASELINE_ISA);

    /* Round-major, so any drift in machine state is shared across CPUs rather
     * than penalising whichever CPUs happen to be measured first. */
    for (int r = 0; r < rounds; ++r) {
        for (int c = 0; c < n; ++c) {
            if (pin_to_cpu(c) != 0) { all[c][r] = 0.0; continue; }
            all[c][r] = probe_once(units);
        }
    }

    double best = 0.0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < rounds; ++r)
            if (all[c][r] > best) { best = all[c][r]; }

    printf("%-5s %-6s %-11s %-11s %-11s %-8s %-7s %s\n",
           "cpu", "core", "siblings", "min", "max", "spread%",
           "rel_max", "rel_med");
    for (int c = 0; c < n; ++c) {
        double v[32];
        for (int r = 0; r < rounds; ++r) { v[r] = all[c][r]; }
        qsort(v, (size_t) rounds, sizeof(double), cmp_d);
        const double lo = v[0], hi = v[rounds - 1], med = v[rounds / 2];
        const double spread = (hi > 0.0) ? (hi - lo) / hi * 100.0 : 0.0;
        char sib[24] = "-";
        char path[128];
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE * f = fopen(path, "r");
        if (f) {
            if (fgets(sib, sizeof sib, f)) { char * nl = strchr(sib, '\n'); if (nl) { *nl = 0; } }
            fclose(f);
        }
        printf("%-5d %-6d %-11s %-11.4g %-11.4g %-8.2f %-7.3f %.3f\n",
               c, topo.cpus[c].core_id, sib,
               lo, hi, spread, best > 0 ? hi / best : 0.0, best > 0 ? med / best : 0.0);
    }

    /* How well would each candidate rule separate the two populations? */
    /* Per-round values, unsorted, for a few CPUs: is the fast/slow choice
     * random per round, or structured by round? That decides whether a robust
     * central statistic can rescue the signal or whether there is no signal. */
    printf("\n# per-round raw values (round order preserved)\n");
    for (int c = 0; c < n; c += 1) {
        if (c % 4 != 0) { continue; }
        printf("  cpu%-3d", c);
        for (int r = 0; r < rounds; ++r) { printf(" %8.4g", all[c][r] / 1e8); }
        printf("\n");
    }

    printf("\n# threshold sweep: how many logical CPUs classify as PERF per rule\n");
    for (double th = 0.60; th <= 0.95001; th += 0.05) {
        int perf = 0;
        for (int c = 0; c < n; ++c) {
            double v[32];
            for (int r = 0; r < rounds; ++r) { v[r] = all[c][r]; }
            qsort(v, (size_t) rounds, sizeof(double), cmp_d);
            if (best > 0 && v[rounds - 1] / best >= th) { ++perf; }
        }
        printf("  rel_max >= %.2f -> %2d perf logical cpus\n", th, perf);
    }
    printf("\n# a 13900K has 8 P-cores (16 logical) and 6 E-cores (12 logical)\n");
    free(all);
    return 0;
}
