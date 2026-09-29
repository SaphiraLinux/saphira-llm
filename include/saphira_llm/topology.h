/*
 * topology.h — CPU topology, hybrid classification and placement planning.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_TOPOLOGY_H
#define SAPHIRA_LLM_TOPOLOGY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SLLM_MAX_CPUS 1024

typedef enum sllm_core_class {
    SLLM_CORE_UNKNOWN  = 0,  /* not classified                              */
    SLLM_CORE_PERF     = 1,  /* measured as a full-rate core               */
    SLLM_CORE_EFFICIENCY = 2,/* measured as a reduced-rate core            */
    SLLM_CORE_COUNT    = 3
} sllm_core_class;

typedef struct sllm_cpu {
    int             id;         /* logical CPU id                     */
    int             core_id;    /* physical core within the package   */
    int             package;
    bool            online;
    bool            primary;    /* first hardware thread of its core  */
    sllm_core_class klass;       /* from CPUID, or UNKNOWN            */
    double          score;      /* measured throughput, 1.0 == best   */
    unsigned        core_type;  /* raw CPUID.1A EAX[31:24], 0 if none */
} sllm_cpu;

typedef struct sllm_topology {
    sllm_cpu cpus[SLLM_MAX_CPUS];
    int      n_cpus;            /* online logical CPUs                    */
    int      n_cores;           /* distinct physical cores                */
    int      n_perf;            /* cores classified as full rate         */
    int      n_eff;             /* cores classified as reduced rate       */
    bool     calibrated;        /* true once measured classification ran  */
    bool     hybrid;            /* more than one core class was found     */
    bool     smt;               /* any core has more than one hw thread   */

    /*
     * Whether the core classes above are facts or absent.
     *
     * hybrid_advertised is CPUID.7.0:EBX[15]: the processor says it has a
     * hybrid architecture. core_type_available is whether CPUID leaf 0x1A
     * actually returned a usable EAX[31:24] on the CPUs we sampled.
     *
     * When core_type_available is false, n_perf and n_eff are zero, hybrid is
     * false, and every klass is SLLM_CORE_UNKNOWN. That is a complete
     * description, not a missing one: the machine may well be hybrid, we
     * simply cannot tell from inside this environment, and a caller that needs
     * the number should find that out rather than receive a guess.
     */
    bool     hybrid_advertised;
    bool     core_type_available;
} sllm_topology;

/*
 * Read the topology from sysfs and classify cores from CPUID. Never benchmarks.
 *
 * Structure -- which logical CPUs share a physical core, how many cores there
 * are, whether any core has SMT -- comes from sysfs and is exact.
 *
 * Classification of performance versus efficiency cores comes from CPUID leaf
 * 0x1A, EAX[31:24], read while pinned to each logical CPU. That is a hardware
 * answer, so it is deterministic and independent of machine load, which is the
 * property this function exists to guarantee.
 *
 * If leaf 0x1A is unavailable or zeroed -- as it is under a hypervisor that
 * does not expose it -- the core classes are left UNKNOWN. Runtime timing is
 * deliberately NOT used as a fallback: measured on this class of host it
 * answers a different question. Guest sched_setaffinity pins a vCPU to a guest
 * logical CPU, but the host remains free to run that vCPU's thread on any host
 * core, so guest-side timing measures host scheduling rather than the guest
 * CPU's capability. On this host a single logical CPU's samples swing by 2x
 * between rounds at random, which is why timing-based classification is
 * removed from discovery rather than tuned.
 *
 * An unreadable or absent sysfs yields a single-CPU topology, which is correct
 * behaviour rather than an error.
 */
void sllm_topology_detect(sllm_topology * topo);

/*
 * Measure relative throughput per logical CPU, pinned, by running a fixed
 * amount of vector work and ranking the results.
 *
 * This is NOT topology discovery and no longer classifies cores. It was
 * removed from discovery because it does not answer the question it was asked:
 * on a virtualised host a guest's samples swing by about 2x between rounds at
 * random, so a measurement of this kind reflects host scheduling rather than
 * the guest CPU's capability. Repeated runs on this host classified 14/0,
 * 13/1, 12/2 and 11/3 performance/efficiency cores on a 13900K, and 27 of 28
 * logical CPUs came out full-rate at every threshold from 0.60 to 0.95.
 *
 * What remains useful is a throughput number that happens to be measured on
 * this machine, for choosing how many threads to run. It fills score and sets
 * calibrated, and nothing else: n_perf, n_eff, hybrid and every klass are left
 * exactly as sllm_topology_detect() established them.
 *
 * Because those numbers are load-dependent, a caller must not treat them as
 * topology facts. The distinction is why this is a separate entry point.
 *
 * `work_units` controls the probe length. Returns false if the topology was
 * too small, in which case score is left untouched.
 */
bool sllm_topology_calibrate(sllm_topology * topo, size_t work_units);

/*
 * Build the order in which threads should be placed.
 *
 * `wanted` is the thread count. The plan is filled with logical CPU ids in
 * placement order: the strongest core's primary hardware thread, then the next
 * strongest, and only then the SMT siblings, so that adding a thread beyond
 * the core count never costs more than it must.
 *
 * Returns the number of CPUs written. `out` must have room for `wanted` ids.
 * `wanted` is clamped to what the machine can actually supply.
 */
int sllm_topology_plan(const sllm_topology * topo, int wanted, int * out, int out_max);

/*
 * The thread count this machine should use for a memory-bound workload, given
 * a ceiling and the occupancy behaviour measured in BENCHMARKS.md.
 *
 * At full occupancy the reference implementation loses a factor of seventeen
 * on the target, because its barrier spins and every logical CPU ends up
 * spinning. Leaving at least one hardware thread free costs a few percent and
 * removes the cliff entirely, so the default is deliberately short of the
 * logical CPU count.
 */
int sllm_topology_recommended_threads(const sllm_topology * topo, int ceiling);

/* A one-line human-readable summary for the log. */
void sllm_topology_describe(const sllm_topology * topo, char * buf, size_t buflen);

const char * sllm_core_class_name(sllm_core_class k);

/*
 * Are the core classes in this topology actually known?
 *
 * Read this BEFORE consulting topo->hybrid, topo->n_perf, topo->n_eff or any
 * cpus[i].klass. Those fields are zeroed rather than left stale when core type
 * is unavailable, so hybrid == false on an unknown-classification topology is
 * indistinguishable from hybrid == false on a genuinely homogeneous one by
 * looking at hybrid alone.
 *
 * The distinction matters: "this machine has no efficiency cores" justifies
 * spreading work across every core, whereas "we were not told" justifies
 * nothing. Only the first is a fact about the silicon.
 *
 * On a host where CPUID leaf 0x1A is hidden or zeroed -- which includes a
 * hypervisor that does not expose hybrid topology -- this returns false.
 */
bool sllm_topology_classes_known(const sllm_topology * topo);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_TOPOLOGY_H */
