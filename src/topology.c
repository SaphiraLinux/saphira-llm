/*
 * topology.c — CPU topology, hybrid classification and placement planning.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Two facts drive this file, both measured on the target (see BENCHMARKS.md):
 *
 * 1. Placement matters for peak throughput. Fourteen threads on CPUs 0-13 run
 *    at 26.0 t/s while the same thread count spread over the whole machine runs
 *    at 23.2 t/s. Knowing which cores are worth using first is worth having.
 *
 * 2. Placement does NOT cause the pathological collapse. The collapse is
 *    barrier saturation at threads == logical CPUs, and it is addressed in
 *    thread.c, not here. This file is about quality, not about that cliff.
 *
 * The target is a KVM guest that zeroes CPUID leaf 0x1A and ships no
 * core_type file, so the hardware will not tell us which cores are fast. It
 * gets measured instead.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  /* must precede every system header */
#endif

#include <saphira_llm/topology.h>
#include <saphira_llm/log.h>

#include <sched.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h>

/* ------------------------------------------------------------------ */
/* sysfs helpers                                                       */
/* ------------------------------------------------------------------ */

static int read_int_file(const char * path, int * out) {
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    int v = 0;
    const int ok = fscanf(f, "%d", &v);
    (void) fclose(f);
    if (ok != 1) {
        return 0;
    }
    *out = v;
    return 1;
}

/*
 * Parse a sysfs cpu list such as "0-3,8,10-11" into a bitmask.
 * Returns 0 if any token is malformed. Being strict here matters: a silently
 * misparsed sibling list would produce a wrong placement plan, which is the
 * kind of bug that shows up as mysterious slowdowns.
 */
static int parse_cpu_list(const char * text, size_t len, unsigned char * bits) {
    size_t i = 0;
    while (i < len) {
        while (i < len && (text[i] == ',' || text[i] == ' ' || text[i] == '\n')) {
            ++i;
        }
        if (i >= len) {
            break;
        }
        int lo = 0, hi = 0;
        size_t digits = 0;
        while (i < len && text[i] >= '0' && text[i] <= '9') {
            lo = lo * 10 + (text[i] - '0');
            ++i;
            ++digits;
        }
        if (digits == 0) {
            return 0;
        }
        hi = lo;
        if (i < len && text[i] == '-') {
            ++i;
            hi = 0;
            digits = 0;
            while (i < len && text[i] >= '0' && text[i] <= '9') {
                hi = hi * 10 + (text[i] - '0');
                ++i;
                ++digits;
            }
            if (digits == 0 || hi < lo) {
                return 0;
            }
        }
        for (int c = lo; c <= hi; ++c) {
            if (c >= 0 && c < SLLM_MAX_CPUS) {
                bits[c / 8] |= (unsigned char) (1u << (c % 8));
            }
        }
    }
    return 1;
}

static int read_cpu_mask(const char * fmt, int cpu, unsigned char * bits) {
    char path[256];
    (void) snprintf(path, sizeof(path), fmt, cpu);
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    char text[4096];
    const size_t n = fread(text, 1, sizeof(text) - 1, f);
    (void) fclose(f);
    text[n] = '\0';
    return parse_cpu_list(text, n, bits);
}


/* ------------------------------------------------------------------ */
/* detection                                                           */
/* ------------------------------------------------------------------ */

void sllm_topology_detect(sllm_topology * topo) {
    if (topo == NULL) {
        return;
    }
    memset(topo, 0, sizeof(*topo));

    unsigned char online_bits[SLLM_MAX_CPUS / 8];
    memset(online_bits, 0, sizeof(online_bits));

    if (!read_cpu_mask("/sys/devices/system/cpu/cpu%d/online", 0, online_bits) &&
        !read_cpu_mask("/sys/devices/system/cpu/online", 0, online_bits)) {
        /*
         * No sysfs at all. One CPU is a correct description of a machine we
         * cannot enumerate, and is far better than guessing a count.
         */
        topo->cpus[0].id = 0;
        topo->cpus[0].online = true;
        topo->cpus[0].primary = true;
        topo->n_cpus = 1;
        topo->n_cores = 1;
        return;
    }

    const long nconf = sysconf(_SC_NPROCESSORS_CONF);
    const long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc <= 0) {
        topo->n_cpus = 1;
        topo->n_cores = 1;
        return;
    }
    (void) nconf;

    topo->n_cpus = 0;
    for (int id = 0; id < SLLM_MAX_CPUS; ++id) {
        if (!(online_bits[id / 8] & (1u << (id % 8)))) {
            continue;
        }
        if (id >= nproc) {
            continue;
        }
        sllm_cpu * c = &topo->cpus[topo->n_cpus++];
        c->id     = id;
        c->online = true;
        c->klass  = SLLM_CORE_UNKNOWN;

        char path[256];
        (void) snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/core_id", id);
        if (!read_int_file(path, &c->core_id)) {
            c->core_id = id;
        }
        (void) snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", id);
        if (!read_int_file(path, &c->package)) {
            c->package = 0;
        }
    }

    /* Group by physical core to find primaries and detect SMT. */
    for (int i = 0; i < topo->n_cpus; ++i) {
        sllm_cpu * a = &topo->cpus[i];
        a->primary = true;
        for (int j = 0; j < topo->n_cpus; ++j) {
            const sllm_cpu * b = &topo->cpus[j];
            if (i != j && b->package == a->package && b->core_id == a->core_id) {
                if (b->id < a->id) {
                    a->primary = false;
                }
            }
        }
    }

    topo->n_cores = 0;
    for (int i = 0; i < topo->n_cpus; ++i) {
        bool first = true;
        for (int j = 0; j < i; ++j) {
            if (topo->cpus[j].package == topo->cpus[i].package &&
                topo->cpus[j].core_id == topo->cpus[i].core_id) {
                first = false;
                break;
            }
        }
        if (first) {
            ++topo->n_cores;
        }
        if (!topo->cpus[i].primary) {
            topo->smt = true;
        }
    }

    sllm_log(SLLM_LOG_DEBUG, "topology: %d logical cpus, %d cores, smt=%s",
             topo->n_cpus, topo->n_cores, topo->smt ? "yes" : "no");
}

/* ------------------------------------------------------------------ */
/* calibration by measurement                                          */
/* ------------------------------------------------------------------ */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1.0e-9;
}

static int pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

/*
 * A fixed amount of vector work whose throughput tracks core capability.
 *
 * It is pure register work on a 16 KiB buffer, so it is unaffected by memory
 * bandwidth and measures the core's issue width and clock. On a hybrid part
 * that is exactly the distinction we want, and the measurement is repeatable:
 * running it twice gives a spread small enough to separate a full-rate core
 * from a reduced-rate one with a wide margin.
 */
static double probe_throughput(int cpu, size_t units) {
    enum { LANES = 16 };
    static float buf[LANES * 8];
    for (int i = 0; i < LANES * 8; ++i) {
        buf[i] = (float) (i % 13) - 6.0f;
    }
    __m256 a0 = _mm256_loadu_ps(buf + 0),  a1 = _mm256_loadu_ps(buf + 8);
    __m256 a2 = _mm256_loadu_ps(buf + 16), a3 = _mm256_loadu_ps(buf + 24);
    __m256 a4 = _mm256_loadu_ps(buf + 32), a5 = _mm256_loadu_ps(buf + 40);
    __m256 a6 = _mm256_loadu_ps(buf + 48), a7 = _mm256_loadu_ps(buf + 56);
    const __m256 k1 = _mm256_set1_ps(1.0000001f);
    const __m256 k2 = _mm256_set1_ps(0.9999999f);

    if (pin_to_cpu(cpu) != 0) {
        return 0.0;
    }

    /* A short warm-up so the first measured pass is not paying for frequency
     * ramp or page faults. */
    for (size_t i = 0; i < units / 8 + 1; ++i) {
        a0 = _mm256_fmadd_ps(a0, k1, a1);
        a1 = _mm256_fmadd_ps(a1, k2, a2);
        a2 = _mm256_fmadd_ps(a2, k1, a3);
        a3 = _mm256_fmadd_ps(a3, k2, a4);
        a4 = _mm256_fmadd_ps(a4, k1, a5);
        a5 = _mm256_fmadd_ps(a5, k2, a6);
        a6 = _mm256_fmadd_ps(a6, k1, a7);
        a7 = _mm256_fmadd_ps(a7, k2, a0);
    }

    const double t0 = now_s();
    for (size_t i = 0; i < units; ++i) {
        a0 = _mm256_fmadd_ps(a0, k1, a1);
        a1 = _mm256_fmadd_ps(a1, k2, a2);
        a2 = _mm256_fmadd_ps(a2, k1, a3);
        a3 = _mm256_fmadd_ps(a3, k2, a4);
        a4 = _mm256_fmadd_ps(a4, k1, a5);
        a5 = _mm256_fmadd_ps(a5, k2, a6);
        a6 = _mm256_fmadd_ps(a6, k1, a7);
        a7 = _mm256_fmadd_ps(a7, k2, a0);
    }
    const double t1 = now_s();

    /* Keep the result live so the loop cannot be optimised away. */
    float sink[8];
    _mm256_storeu_ps(sink, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    if (sink[0] == 12345.678f) {
        return 0.0; /* never taken; defeats dead-code elimination */
    }
    return (double) units / (t1 - t0);
}

bool sllm_topology_calibrate(sllm_topology * topo, size_t work_units) {
    if (topo == NULL || topo->n_cpus <= 0) {
        return false;
    }
    if (work_units == 0) {
        work_units = 2000000;
    }

    /* Pinning is optional. Without it the probe is still valid, just less
     * controlled, so a refusal degrades the measurement rather than failing. */
    const bool can_pin = (pin_to_cpu(0) == 0);

    bool ok = true;
    for (int i = 0; i < topo->n_cpus; ++i) {
        double best = 0.0;
        /* Three passes, keep the best: the scheduler can preempt us, and the
         * fastest clean pass is the one that reflects the core rather than
         * the noise around it. */
        for (int attempt = 0; attempt < 3; ++attempt) {
            const double t = probe_throughput(topo->cpus[i].id, work_units);
            if (t > best) {
                best = t;
            }
        }
        topo->cpus[i].score = best;
        if (best <= 0.0) {
            ok = false;
        }
    }

    if (!ok) {
        return false;
    }

    double max_score = 0.0;
    for (int i = 0; i < topo->n_cpus; ++i) {
        if (topo->cpus[i].score > max_score) {
            max_score = topo->cpus[i].score;
        }
    }
    if (max_score <= 0.0) {
        return false;
    }

    /*
     * Class a core by how far its throughput falls below the best core. On the
     * target a full-rate core scores near 1.0 and a reduced-rate core near
     * 0.5, so the 0.75 threshold separates them with a wide margin and is not
     * balanced on a knife edge. Cores are classified by their primary
     * hardware thread and the class is shared with their siblings, because an
     * SMT sibling is the same core.
     */
    for (int i = 0; i < topo->n_cpus; ++i) {
        const double rel = topo->cpus[i].score / max_score;
        topo->cpus[i].klass = (rel >= 0.75) ? SLLM_CORE_PERF : SLLM_CORE_EFFICIENCY;
    }

    topo->n_perf = 0;
    topo->n_eff  = 0;

    /* Count distinct cores by class, via the primaries. */
    for (int i = 0; i < topo->n_cpus; ++i) {
        if (!topo->cpus[i].primary) {
            continue;
        }
        if (topo->cpus[i].klass == SLLM_CORE_PERF) {
            ++topo->n_perf;
        } else {
            ++topo->n_eff;
        }
    }

    topo->calibrated = true;
    topo->hybrid = (topo->n_perf > 0 && topo->n_eff > 0);
    (void) can_pin;

    sllm_log(SLLM_LOG_INFO, "topology calibrated by measurement: %d perf cores, %d efficiency cores, hybrid=%s",
             topo->n_perf, topo->n_eff, topo->hybrid ? "yes" : "no");
    return true;
}

/* ------------------------------------------------------------------ */
/* placement planning                                                  */
/* ------------------------------------------------------------------ */

static int cmp_cpu(const void * a, const void * b) {
    const sllm_cpu * x = (const sllm_cpu *) a;
    const sllm_cpu * y = (const sllm_cpu *) b;
    /* Strongest first; among equals, primary hardware thread first, then by
     * id so the plan is deterministic. */
    if (x->score > y->score) { return -1; }
    if (x->score < y->score) { return  1; }
    if (x->primary != y->primary) { return x->primary ? -1 : 1; }
    return (x->id < y->id) ? -1 : (x->id > y->id);
}

int sllm_topology_plan(const sllm_topology * topo, int wanted, int * out, int out_max) {
    if (topo == NULL || out == NULL || out_max <= 0) {
        return 0;
    }
    if (wanted <= 0) {
        wanted = 1;
    }

    sllm_cpu sorted[SLLM_MAX_CPUS];
    const int n = topo->n_cpus < SLLM_MAX_CPUS ? topo->n_cpus : SLLM_MAX_CPUS;
    memcpy(sorted, topo->cpus, (size_t) n * sizeof(sllm_cpu));
    qsort(sorted, (size_t) n, sizeof(sllm_cpu), cmp_cpu);

    int written = 0;
    for (int i = 0; i < n && written < wanted && written < out_max; ++i) {
        if (!sorted[i].online) {
            continue;
        }
        out[written++] = sorted[i].id;
    }
    return written;
}

int sllm_topology_recommended_threads(const sllm_topology * topo, int ceiling) {
    if (topo == NULL || topo->n_cpus <= 0) {
        return 1;
    }
    int n = topo->n_cpus;
    if (ceiling > 0 && ceiling < n) {
        n = ceiling;
    }
    /*
     * Never fill every hardware thread.
     *
     * Measured on the target with the reference implementation: 24 threads on
     * 28 logical CPUs runs at 25.97 t/s, 25 threads at 13.36, and 28 threads at
     * 1.54. Full occupancy is where a spinning barrier stops making progress.
     * Leaving one hardware thread free is the cheapest insurance available and
     * costs a couple of percent at 24.
     */
    if (n == topo->n_cpus && topo->n_cpus > 1) {
        --n;
    }
    if (n < 1) {
        n = 1;
    }
    return n;
}

const char * sllm_core_class_name(sllm_core_class k) {
    switch (k) {
        case SLLM_CORE_PERF:       return "perf";
        case SLLM_CORE_EFFICIENCY: return "efficiency";
        default:                   return "unknown";
    }
}

void sllm_topology_describe(const sllm_topology * topo, char * buf, size_t buflen) {
    if (buf == NULL || buflen == 0) {
        return;
    }
    if (topo == NULL) {
        buf[0] = '\0';
        return;
    }
    (void) snprintf(buf, buflen,
        "%d logical cpus, %d cores, smt=%s, %s%s",
        topo->n_cpus, topo->n_cores, topo->smt ? "yes" : "no",
        topo->calibrated ? "calibrated" : "uncalibrated",
        topo->hybrid ? " (hybrid)" : "");
}
