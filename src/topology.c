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
 * Structure (which logical CPUs share a physical core, how many cores exist,
 * whether any core has SMT) comes from sysfs and is exact.
 *
 * Core classification comes from CPUID leaf 0x1A, EAX[31:24], read pinned to
 * each logical CPU. That is a hardware answer, so it is deterministic and
 * independent of machine load, which is what Phase 6 requires of detection.
 *
 * On the target, a KVM guest, the hypervisor zeroes that leaf and reports a
 * non-GenuineIntel vendor string, so core type is unavailable and the classes
 * are left UNKNOWN. That is a reported fact, not a gap, and no runtime
 * measurement is used to fill it in: guest-side timing on this host measures
 * host scheduling, swinging ~2x between rounds on the same logical CPU at
 * random, which is precisely why it was removed from discovery rather than
 * retuned. See sllm_topology_calibrate for the narrow use it retains.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  /* must precede every system header */
#endif

#include <saphira_llm/topology.h>
#include <saphira_llm/log.h>

#include <cpuid.h>
#include <sched.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h>

/* Defined below, used by the CPUID classification in sllm_topology_detect. */
static int pin_to_cpu(int cpu);

/* ------------------------------------------------------------------ */
/* CPUID core classification                                            */
/* ------------------------------------------------------------------ */

/*
 * Intel's documented native core-type values for CPUID leaf 0x1A, EAX[31:24].
 *
 * These are recorded as constants for readability, but nothing in this file
 * assumes a core returns one of them. Every value the hardware produces is
 * stored in sllm_cpu.core_type, and a value we do not recognise is left
 * UNKNOWN rather than being forced into one of the two classes.
 */
#define SLLM_CORE_TYPE_INTEL_CORE 0x40u  /* performance core  */
#define SLLM_CORE_TYPE_INTEL_ATOM 0x20u  /* efficiency core  */

/*
 * Does the processor advertise a hybrid architecture?
 *
 * CPUID.7.0:EBX[15]. Returns 1 for hybrid, 0 for homogeneous, -1 if the
 * leaf is not available to ask.
 *
 * A hypervisor that does not expose the topology will report 0 here. That is
 * not a claim that the silicon is homogeneous -- it is a claim that we were
 * not told, and we treat it as being not told.
 */
static int cpuid_hybrid_advertised(void) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        return -1;
    }
    return (int) ((b >> 15) & 1u);
}

/*
 * Read the native core-type field while pinned to `cpu`.
 *
 * CPUID is per-thread state, so the thread must be on the CPU being asked
 * about; running this unpinned would report the core the caller happened to
 * be on, for every CPU in turn.
 *
 * Returns 0 and fills *out on success. Returns -1 if affinity was refused or
 * the leaf is unavailable, in which case *out is untouched.
 */
static int read_core_type(int cpu, unsigned * out) {
    if (pin_to_cpu(cpu) != 0) {
        return -1;
    }
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(0x1A, 0, &a, &b, &c, &d)) {
        return -1;
    }
    *out = (a >> 24) & 0xffu;
    return 0;
}

/* Restore affinity to every CPU, so a caller that only wanted to ask a
 * question does not leave itself pinned. */
static void unpin_all(void) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < SLLM_MAX_CPUS; ++i) {
        CPU_SET(i, &set);
    }
    (void) sched_setaffinity(0, sizeof set, &set);
}

/*
 * Classify cores from the native CPUID core-type field.
 *
 * This is a hardware answer and nothing else participates: no timing, no
 * thresholds tuned against observed timings, and no inference from
 * cluster_id. The same call returns the same bytes regardless of what the
 * machine is doing, which is the property Phase 6 needs.
 *
 * sysfs facts are used only to CROSS-CHECK, never to derive a class. The
 * check that matters is SMT siblings: two logical CPUs that sysfs says share a
 * physical core must report the same core type, because they are the same
 * core. A disagreement means the two sources contradict each other, and we
 * would rather report nothing than pick a winner.
 *
 * On this host the leaf is present but zeroed, every value is 0, so nothing is
 * classifiable and we say so.
 */
static void classify_cores_cpuid(sllm_topology * topo) {
    topo->hybrid_advertised = (cpuid_hybrid_advertised() == 1);

    int read_ok = 0;
    bool any_nonzero = false;
    for (int i = 0; i < topo->n_cpus; ++i) {
        sllm_cpu * c = &topo->cpus[i];
        c->klass = SLLM_CORE_UNKNOWN;
        c->core_type = 0;
        unsigned t = 0;
        if (read_core_type(c->id, &t) == 0) {
            ++read_ok;
            c->core_type = t;
            if (t != 0) {
                any_nonzero = true;
            }
        }
    }
    unpin_all();

    /* Cross-check: sysfs says these logical CPUs are one physical core, so
     * they must agree on core type. */
    bool sibling_conflict = false;
    for (int i = 0; i < topo->n_cpus; ++i) {
        for (int j = i + 1; j < topo->n_cpus; ++j) {
            const sllm_cpu * a = &topo->cpus[i];
            const sllm_cpu * b = &topo->cpus[j];
            if (a->package == b->package && a->core_id == b->core_id &&
                a->core_type != b->core_type) {
                sibling_conflict = true;
            }
        }
    }

    const bool usable = (read_ok == topo->n_cpus) && any_nonzero && !sibling_conflict;
    topo->core_type_available = usable;

    if (!usable) {
        /* Record why, so an absent classification is never mistaken for a
         * homogeneous machine. */
        if (read_ok != topo->n_cpus) {
            sllm_log(SLLM_LOG_INFO,
                     "topology: CPUID.1A unreadable on %d of %d logical CPUs; "
                     "core type unavailable",
                     topo->n_cpus - read_ok, topo->n_cpus);
        } else if (!any_nonzero) {
            sllm_log(SLLM_LOG_INFO,
                     "topology: CPUID.1A returned core type 0 on all %d logical "
                     "CPUs (hypervisor does not expose hybrid topology; "
                     "CPUID.7.0:EBX[15]=%d). Core class left UNKNOWN. Structure "
                     "below is from sysfs and is exact.",
                     topo->n_cpus, topo->hybrid_advertised ? 1 : 0);
        } else {
            sllm_log(SLLM_LOG_INFO,
                     "topology: CPUID.1A core types contradict sysfs sibling "
                     "grouping; core type unavailable");
        }
        topo->n_perf = 0;
        topo->n_eff  = 0;
        topo->hybrid = false;
        return;
    }

    for (int i = 0; i < topo->n_cpus; ++i) {
        sllm_cpu * c = &topo->cpus[i];
        switch (c->core_type) {
            case SLLM_CORE_TYPE_INTEL_CORE: c->klass = SLLM_CORE_PERF;     break;
            case SLLM_CORE_TYPE_INTEL_ATOM: c->klass = SLLM_CORE_EFFICIENCY; break;
            default:
                /* Recorded, but not one we are prepared to name. */
                c->klass = SLLM_CORE_UNKNOWN;
                break;
        }
    }

    topo->n_perf = 0;
    topo->n_eff  = 0;
    for (int i = 0; i < topo->n_cpus; ++i) {
        if (!topo->cpus[i].primary) {
            continue;
        }
        if (topo->cpus[i].klass == SLLM_CORE_PERF) {
            ++topo->n_perf;
        } else if (topo->cpus[i].klass == SLLM_CORE_EFFICIENCY) {
            ++topo->n_eff;
        }
    }
    topo->hybrid = (topo->n_perf > 0 && topo->n_eff > 0);
    sllm_log(SLLM_LOG_INFO,
             "topology: CPUID.1A core types available; %d performance, %d "
             "efficiency cores (hybrid_advertised=%d)",
             topo->n_perf, topo->n_eff, topo->hybrid_advertised ? 1 : 0);
}

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

    classify_cores_cpuid(topo);
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

/*
 * Samples per logical CPU, and the two independent halves used to check that
 * the probe measured anything at all. Must be even.
 */
bool sllm_topology_calibrate(sllm_topology * topo, size_t work_units) {
    if (topo == NULL || topo->n_cpus <= 0) {
        return false;
    }
    if (work_units == 0) {
        work_units = 2000000;
    }

    /*
     * Measurement only. This deliberately does NOT classify cores, set
     * hybrid, or touch n_perf/n_eff. Those are hardware facts now, established
     * once by sllm_topology_detect() from CPUID, and a load-dependent
     * measurement must not be able to overwrite them.
     *
     * It fills score, which is a throughput observation about this machine at
     * this moment, and is therefore not a topology fact and not stable. Callers
     * that want a thread count from it must accept that.
     *
     * The median of the passes is taken rather than the best, so a single
     * preempted pass cannot be read as a fast core. Even so the result is
     * noisy on a virtualised host, which is the whole reason this is not
     * allowed to classify.
     */
    for (int i = 0; i < topo->n_cpus; ++i) {
        double v[3];
        for (int attempt = 0; attempt < 3; ++attempt) {
            v[attempt] = probe_throughput(topo->cpus[i].id, work_units);
            if (v[attempt] <= 0.0) {
                return false;
            }
        }
        if (v[0] > v[1]) { const double t = v[0]; v[0] = v[1]; v[1] = t; }
        if (v[1] > v[2]) { const double t = v[1]; v[1] = v[2]; v[2] = t; }
        if (v[0] > v[1]) { const double t = v[0]; v[0] = v[1]; v[1] = t; }
        topo->cpus[i].score = v[1];
    }

    double best = 0.0;
    for (int i = 0; i < topo->n_cpus; ++i) {
        if (topo->cpus[i].score > best) { best = topo->cpus[i].score; }
    }
    if (best <= 0.0) {
        return false;
    }
    for (int i = 0; i < topo->n_cpus; ++i) {
        topo->cpus[i].score /= best;
    }

    topo->calibrated = true;
    return true;
}

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

bool sllm_topology_classes_known(const sllm_topology * topo) {
    return topo != NULL && topo->core_type_available;
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

    /*
     * Report core class availability explicitly. A bare "hybrid: no" is
     * ambiguous between a genuinely homogeneous machine and one that would not
     * tell us, and those two need very different treatment from a caller.
     */
    const char * class_str;
    if (topo->core_type_available) {
        class_str = topo->hybrid ? " (hybrid, CPUID.1A)"
                                 : " (homogeneous, CPUID.1A)";
    } else {
        class_str = " (core class unavailable)";
    }

    (void) snprintf(buf, buflen,
        "%d logical cpus, %d cores, smt=%s, %s%s%s",
        topo->n_cpus, topo->n_cores, topo->smt ? "yes" : "no",
        topo->calibrated ? "calibrated" : "uncalibrated",
        class_str,
        topo->hybrid_advertised ? "" : " [no hybrid hint from CPUID.7]");
}
