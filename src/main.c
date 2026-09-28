/*
 * main.c — saphira-llm command line entry point.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Phase 1 scope: the container layer and ISA dispatch. Model loading,
 * tokenisation and generation arrive in later phases; the interface is fixed
 * now so the user-facing shape does not churn.
 */

#include <saphira_llm/sllm.h>
#include <saphira_llm/thread.h>
#include <saphira_llm/topology.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE * out) {
    fprintf(out,
"usage: saphira-llm -m MODEL -p PROMPT [options]\n"
"\n"
"  -m, --model FILE        GGUF model to load\n"
"  -p, --prompt TEXT       prompt text\n"
"  -c, --ctx N             context size\n"
"  -t, --threads N         worker threads (0 = recommended for this machine)\n"
"      --no-place         do not pin threads; for measuring placement\n"
"      --calibrate        measure cores before placing (default on)\n"
"      --temp F            sampling temperature (0 = greedy)\n"
"  -n, --n-predict N       maximum tokens to generate\n"
"      --seed N            sampling seed\n"
"      --isa LEVEL         force an ISA floor: v3, vnni, avx512, amx, auto\n"
"      --log LEVEL         quiet, error, warn, info, debug\n"
"  -h, --help              this text\n"
"  -V, --version           version and baseline\n"
"\n"
"The final interface is intended to stay this small:\n"
"  saphira-llm -m model.gguf -p \"Hello\"\n");
}

int main(int argc, char ** argv) {
    const char * model = NULL;
    const char * prompt = NULL;
    sllm_isa_level isa_floor = SLLM_ISA_LEVEL_AUTO;
    int  n_threads = 0;      /* 0 means "use the recommendation" */
    bool no_place   = false;
    bool calibrate  = true;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = (i + 1 < argc) ? argv[i + 1] : NULL;

        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        }
        if (!strcmp(a, "-V") || !strcmp(a, "--version")) {
            printf("saphira-llm %s (baseline %s)\n", SLLM_VERSION_STRING, SLLM_BASELINE_ISA);
            return 0;
        }
        if (!strcmp(a, "-m") || !strcmp(a, "--model")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            model = v; i++;
            continue;
        }
        if (!strcmp(a, "-p") || !strcmp(a, "--prompt")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            prompt = v; i++;
            continue;
        }
        if (!strcmp(a, "--isa")) {
            if (!v) { fprintf(stderr, "--isa needs a value\n"); return 2; }
            if (sllm_isa_level_parse(v, &isa_floor) != SLLM_OK) {
                fprintf(stderr,
                    "--isa '%s' is not one of: v3, vnni, avx512, amx, auto\n", v);
                return 2;
            }
            i++;
            continue;
        }
        if (!strcmp(a, "--log")) {
            sllm_log_level lvl = SLLM_LOG_INFO;
            if (!v) { fprintf(stderr, "--log needs a value\n"); return 2; }
            if (sllm_log_level_parse(v, &lvl) != SLLM_OK) {
                fprintf(stderr,
                    "--log '%s' is not one of: quiet, error, warn, info, debug\n", v);
                return 2;
            }
            sllm_log_set_level(lvl);
            i++;
            continue;
        }
        if (!strcmp(a, "-t") || !strcmp(a, "--threads")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            const int t = atoi(v);
            if (t < 1) {
                fprintf(stderr, "%s must be at least 1, got '%s'\n", a, v);
                return 2;
            }
            n_threads = t;
            i++;
            continue;
        }
        if (!strcmp(a, "--no-place")) {
            no_place = true;
            continue;
        }
        if (!strcmp(a, "--calibrate")) {
            calibrate = true;
            continue;
        }

        /* Options accepted for interface stability but not yet implemented.
         * Rejecting them loudly beats silently ignoring a flag the user
         * believes is having an effect. */
        if (!strcmp(a, "-c") || !strcmp(a, "--ctx") ||
            !strcmp(a, "--temp") ||
            !strcmp(a, "-n") || !strcmp(a, "--n-predict") ||
            !strcmp(a, "--seed")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            fprintf(stderr,
                "saphira-llm: %s is accepted but not implemented yet "
                "(arrives with model execution)\n", a);
            return 2;
        }

        fprintf(stderr, "saphira-llm: unknown argument '%s'\n", a);
        usage(stderr);
        return 2;
    }

    /* --- ISA dispatch, decided once, before anything else runs --- */
    const sllm_isa_dispatch isa = sllm_isa_build(isa_floor);

    sllm_isa_caps caps = isa.caps;
    /*
     * An impossible floor is a hard error, never a silent clamp: a deployment
     * that pinned --isa amx and quietly got v3 instead would be running
     * different kernels from the ones it asked for, with nothing to say so.
     *
     * The message names the request and the ceiling separately, because the
     * dispatch table's "selected" field already reflects the override and
     * reporting that here would just be confusing.
     */
    const sllm_status floor_ok = sllm_isa_check_floor(isa_floor, &caps);
    if (floor_ok != SLLM_OK) {
        (void) sllm_fail(floor_ok,
            "--isa %s was requested but this CPU tops out at %s%s%s",
            sllm_isa_level_name(isa_floor),
            sllm_isa_level_name(sllm_isa_ceiling(&caps)),
            caps.avx_vnni ? " (has avx_vnni)" : "",
            caps.avx512f  ? " (has avx512)" : "");
        return 1;
    }

    /*
     * Topology and threading. Placement is on by default because it was
     * measured to be load-bearing here: without it, threads that repeatedly
     * sleep and are woken per region do not get spread across the machine by
     * the scheduler, and throughput stays at the single-thread figure.
     */
    sllm_topology topo;
    sllm_topology_detect(&topo);
    if (calibrate) {
        (void) sllm_topology_calibrate(&topo, 2000000);
    }
    char topo_desc[256];
    sllm_topology_describe(&topo, topo_desc, sizeof(topo_desc));
    sllm_log(SLLM_LOG_INFO, "topology: %s", topo_desc);

    if (n_threads <= 0) {
        n_threads = sllm_topology_recommended_threads(&topo, 0);
    }

    int plan[SLLM_MAX_CPUS];
    int plan_n = 0;
    if (!no_place) {
        plan_n = sllm_topology_plan(&topo, topo.n_cpus, plan, SLLM_MAX_CPUS);
    }

    sllm_pool_config pcfg = {
        .n_threads = n_threads,
        .plan      = plan_n > 0 ? plan : NULL,
        .n_plan    = plan_n,
        .topology  = &topo,
    };
    sllm_pool * pool = sllm_pool_create(&pcfg);
    sllm_log(SLLM_LOG_INFO, "threading: %d threads, placement %s",
             sllm_pool_threads(pool), plan_n > 0 ? "on" : "off");

    char isa_desc[256];
    sllm_isa_describe(&isa, isa_desc, sizeof(isa_desc));
    sllm_log(SLLM_LOG_INFO, "saphira-llm %s baseline %s (%s)",
             SLLM_VERSION_STRING, SLLM_BASELINE_ISA, isa_desc);

    if (!sllm_isa_level_supported(SLLM_ISA_V3, &caps)) {
        sllm_log(SLLM_LOG_WARN,
            "this CPU does not report the x86-64-v3 baseline; the binary "
            "assumes it and may misexecute");
    }

    if (model == NULL) {
        usage(stderr);
        return 2;
    }

    /* --- container layer --- */
    sllm_gguf g;
    char err[512];
    const sllm_status st = sllm_gguf_open(model, &g, err, sizeof(err));
    if (st != SLLM_OK) {
        fprintf(stderr, "saphira-llm: %s: %s\n", sllm_status_string(st),
                err[0] ? err : "(no detail)");
        return 1;
    }

    char desc[512];
    sllm_gguf_describe(&g, desc, sizeof(desc));
    sllm_log(SLLM_LOG_INFO, "%s", desc);

    const char * arch = NULL;
    if (sllm_gguf_kv_str(&g, "general.architecture", &arch) == SLLM_OK) {
        sllm_log(SLLM_LOG_INFO, "architecture: %s", arch);
    }

    /*
     * Model execution is not built yet. Saying so plainly is better than
     * exiting zero, and better than pretending to generate text.
     */
    if (prompt == NULL) {
        sllm_log(SLLM_LOG_INFO,
            "container parsed; model execution arrives in a later phase "
            "(no prompt given, nothing to run)");
    } else {
        sllm_log(SLLM_LOG_INFO,
            "container parsed; model execution arrives in a later phase "
            "(prompt accepted but not yet run)");
    }

    sllm_gguf_close(&g);
    sllm_pool_destroy(pool);
    return 0;
}
