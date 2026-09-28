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
    sllm_core_class klass;       /* measurement-derived, not assumed  */
    double          score;      /* relative throughput, 1.0 == best   */
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
} sllm_topology;

/*
 * Read the topology from sysfs. Cheap, never benchmarks, never fails: an
 * unreadable or absent sysfs yields a single-CPU topology, which is correct
 * behaviour rather than an error.
 *
 * On this class of machine sysfs cannot tell performance cores from
 * efficiency cores: there is no core_type file, and cluster_id merely mirrors
 * core_id. The authoritative source, CPUID leaf 0x1A, is zeroed by the
 * hypervisor. So classification is left to measurement.
 */
void sllm_topology_detect(sllm_topology * topo);

/*
 * Classify cores by running a fixed amount of vector work pinned to each
 * logical CPU and ranking the results.
 *
 * This is the only method that is both correct and portable on the target: it
 * works on a hybrid core, on a homogeneous core, and inside a VM that hides
 * the hardware topology. It also means we never encode a guess about which
 * cores are fast; we measure which ones are.
 *
 * `work_units` controls the probe length. It is intentionally modest: this runs
 * once and the result is meant to be cached by the caller. Returns false if the
 * topology was too small or affinity was refused, in which case the topology is
 * left uncalibrated and a static ordering is used instead.
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

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_TOPOLOGY_H */
