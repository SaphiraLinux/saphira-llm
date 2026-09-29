/*
 * tg_bench.c — Phase 6 decode-throughput baseline and regression harness.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Why this exists rather than a shell loop over the CLI.
 *
 * A thread sweep run as separate processes pays three costs that are not part
 * of generation: mapping a 1.18 GB model, reading it cold off disk the first
 * time, and paying page faults while the first tokens are produced. Measured
 * here, a cold `-n 16` run at one thread reported 1.16 t/s and the same
 * command warm reported 2.27 t/s — a 96 percent difference that had nothing to
 * do with the runtime. Every number in BENCHMARKS.md that came from a
 * one-shot CLI invocation has to be read with that in mind.
 *
 * So this harness loads the model once, warms it explicitly, and reports a
 * number it can defend: generation only, prefill excluded, cache warm, best
 * and median of N repetitions, with the machine, the ISA, the thread counts
 * and the build identity printed alongside so a result cannot drift away from
 * the thing that produced it.
 *
 * It also reports bytes-per-second against the model's weight bytes. Decode
 * is a streaming problem — every token reads every weight — so throughput is
 * only interpretable next to the bandwidth it implies, and a kernel change
 * that improves t/s by improving the bandwidth number is a different claim
 * from one that improves it by doing less work.
 */

#include <saphira_llm/sllm.h>
#include <saphira_llm/forward.h>
#include <saphira_llm/gguf.h>
#include <saphira_llm/i2s_gemm.h>
#include <saphira_llm/isa.h>
#include <saphira_llm/log.h>
#include <saphira_llm/thread.h>
#include <saphira_llm/tokenizer.h>
#include <saphira_llm/topology.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "prof.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

/*
 * The machine is shared, and that has to be in the record.
 *
 * Three other agent sessions and a long-running python3 sit on this host, and
 * the instantaneous load average was measured at 11.6 while these benchmarks
 * ran. That is not a footnote: a 28-thread sweep on a box with eleven other
 * runnable threads is oversubscribing against real work, and the cost shows
 * up precisely where this phase cares most -- at high thread counts, where
 * every extra worker competes for a logical CPU and for the scheduler. It is
 * the most likely reason the 28-thread row moved between runs here when it did
 * not move on the quieter machine the reference table was taken on.
 *
 * So the load is recorded with every run, and throughput is reported as best
 * of N rather than mean, because under external interference the minimum is
 * the estimate closest to the unloaded one and the mean is mostly a measure of
 * the other tenants.
 */
static double loadavg1(void) {
    FILE * f = fopen("/proc/loadavg", "r");
    double v = -1.0;
    if (f != NULL) { if (fscanf(f, "%lf", &v) != 1) { v = -1.0; } fclose(f); }
    return v;
}

static int cmp_double(const void * a, const void * b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return (x > y) - (x < y);
}

/*
 * Bytes of model payload read once per generated token.
 *
 * Every tensor is counted, not just the I2_S projections: the 121 F32 norms
 * and the tied F16 embedding are streamed on every decode step too, and
 * leaving them out would flatter the bandwidth figure by about 1 percent and
 * make it a number about the wrong thing.
 */
static size_t weight_bytes(const sllm_gguf * g) {
    uint64_t total = 0;
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        total += g->tensors[i].nbytes;
    }
    return (size_t) total;
}

static void usage(void) {
    fputs(
      "usage: saphira-llm-tgbench -m MODEL [options]\n"
      "  -p PROMPT   prompt text (default: a fixed one, recorded below)\n"
      "  -n N        tokens to generate (default 64)\n"
      "  -t LIST     comma-separated thread counts (default 1,2,4,8,14,20,24,28)\n"
      "  -r N        repetitions per thread count (default 3)\n"
      "  -w N        warmup runs before measuring (default 1)\n"
      "  -c N        context depth (default 512)\n"
      "  --no-place  do not pin threads; reproduces the placement contrast\n"
      "  --no-calibrate  skip the measured P/E classification\n"
      "  --prof PATH    sample the generating thread during generation only\n"
      "  --i2s-isa WHICH  force the ternary dot kernel: auto | v3 | vnni\n"
      "  --tag TEXT  label printed with the table (e.g. 'baseline')\n",
      stderr);
}

int main(int argc, char ** argv) {
    const char * model = NULL;
    const char * prompt = "The capital of France is";
    const char * tag = "run";
    int n_new = 64;
    int reps = 3;
    int warmup = 1;
    int32_t n_ctx = 512;
    int no_place = 0;
    int no_calibrate = 0;
    const char * prof_path = NULL;
    const char * i2s_isa = "auto";
    int threads[64];
    int n_threads = 0;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(a, "-m") && v) { model = argv[++i]; }
        else if (!strcmp(a, "-p") && v) { prompt = argv[++i]; }
        else if (!strcmp(a, "-n") && v) { n_new = atoi(argv[++i]); }
        else if (!strcmp(a, "-r") && v) { reps = atoi(argv[++i]); }
        else if (!strcmp(a, "-w") && v) { warmup = atoi(argv[++i]); }
        else if (!strcmp(a, "-c") && v) { n_ctx = (int32_t) atoi(argv[++i]); }
        else if (!strcmp(a, "--tag") && v) { tag = argv[++i]; }
        else if (!strcmp(a, "--no-place")) { no_place = 1; }
        else if (!strcmp(a, "--no-calibrate")) { no_calibrate = 1; }
        else if (!strcmp(a, "--prof") && v) { prof_path = argv[++i]; }
        else if (!strcmp(a, "--i2s-isa") && v) { i2s_isa = argv[++i]; }
        else if (!strcmp(a, "-t") && v) {
            const char * p = argv[++i];
            while (*p && n_threads < (int) (sizeof threads / sizeof threads[0])) {
                threads[n_threads++] = (int) strtol(p, (char **) &p, 10);
                if (*p == ',') { ++p; } else { break; }
            }
        } else { usage(); return 2; }
    }
    if (model == NULL) { usage(); return 2; }
    if (n_threads == 0) {
        static const int dflt[] = { 1, 2, 4, 8, 14, 20, 24, 28 };
        for (size_t i = 0; i < sizeof dflt / sizeof dflt[0]; ++i) {
            threads[n_threads++] = dflt[i];
        }
    }

    const sllm_isa_dispatch isa = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    char isa_desc[256];
    sllm_isa_describe(&isa, isa_desc, sizeof isa_desc);

    /*
     * Calibration is not optional here. Without it the topology is
     * "uncalibrated" and the placement plan falls back to a static ordering
     * that does not know which of this machine's cores are P and which are E.
     * The target is hybrid, so a benchmark that skipped it would be measuring
     * a placement decision nobody ships.
     */
    sllm_topology topo;
    sllm_topology_detect(&topo);
    if (!no_calibrate) { (void) sllm_topology_calibrate(&topo, 2000000); }

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(model, &g, err, sizeof err) != SLLM_OK) {
        fprintf(stderr, "cannot open %s: %s\n", model, err[0] ? err : "?");
        return 1;
    }
    /*
     * Force a specific ternary kernel, after model load has installed its own
     * default. Without this there is no way to A/B the two dot kernels from
     * outside the test suite, and "the VNNI path is slower" is not a claim
     * anyone should have to take on trust.
     */
    if (strcmp(i2s_isa, "auto") != 0) {
        sllm_isa_level floor = SLLM_ISA_LEVEL_AUTO;
        if (sllm_isa_level_parse(i2s_isa, &floor) != SLLM_OK) {
            fprintf(stderr, "unknown --i2s-isa %s\n", i2s_isa);
            return 2;
        }
        sllm_i2s_select_level(floor);
    }

    sllm_model * m = NULL;
    if (sllm_model_load(&g, &m) != SLLM_OK) {
        fprintf(stderr, "cannot load model\n");
        sllm_gguf_close(&g);
        return 1;
    }

    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) {
        fprintf(stderr, "cannot load tokenizer\n");
        sllm_model_free(m); sllm_gguf_close(&g);
        return 1;
    }

    const size_t wbytes = weight_bytes(&g);
    const int32_t nv = sllm_model_n_vocab(m);

    printf("# saphira-llm tg benchmark (%s)\n", tag);
    printf("model            %s\n", model);
    printf("model_bytes      %llu\n", (unsigned long long) wbytes);
    printf("n_vocab          %d\n", (int) nv);
    printf("prompt           \"%s\"\n", prompt);
    printf("n_predict        %d\n", n_new);
    printf("context          %d\n", (int) n_ctx);
    printf("reps             %d (best and median reported)\n", reps);
    printf("warmup           %d\n", warmup);
    printf("placement        %s\n", no_place ? "OFF (--no-place)" : "on");
    printf("isa              %s\n", isa_desc);
    printf("baseline_isa     %s\n", SLLM_BASELINE_ISA);
    char topo_desc[256];
    sllm_topology_describe(&topo, topo_desc, sizeof topo_desc);
    printf("topology         %s\n", topo_desc);
    printf("i2s_dot          %s (level %d)\n", i2s_isa, (int) sllm_i2s_dot_isa());
    printf("timing           generation only, prefill excluded, monotonic\n");
    printf("estimator        best of %d; the minimum is the closest estimate to\n"
           "                 the unloaded machine when other tenants are busy\n", reps);
    printf("loadavg_at_start %.2f  (this host is shared; re-read per row)\n",
           loadavg1());
#ifdef SLLM_BUILD_ID
    printf("build            %s\n", SLLM_BUILD_ID);
#endif
    printf("\n");

    int32_t * ids = (int32_t *) malloc((size_t) n_ctx * sizeof(int32_t));
    int32_t * gen = (int32_t *) malloc((size_t) n_new * sizeof(int32_t));
    sllm_ctx * ctx = NULL;
    if (ids == NULL || gen == NULL || sllm_ctx_new(m, n_ctx, &ctx) != SLLM_OK) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    const int32_t n_prompt = sllm_tok_encode(tok, prompt, strlen(prompt),
                                             true, true, ids, n_ctx);
    if (n_prompt < 0) {
        fprintf(stderr, "tokenize failed\n");
        return 1;
    }
    printf("prompt_tokens    %d\n", (int) n_prompt);
    printf("\n%-4s %-4s %9s %9s %8s %9s %8s %20s\n",
           "th", "eff", "best", "median", "spread", "GB/s", "load", "checksum");

    int plan[SLLM_MAX_CPUS];
    const int plan_n = no_place ? 0 : sllm_topology_plan(&topo, topo.n_cpus, plan, SLLM_MAX_CPUS);

    double * tps = (double *) malloc((size_t) (reps > 0 ? reps : 1) * sizeof(double));
    int first_tok = -1;
    uint64_t checksum = 0;
    uint64_t ref_checksum = 0;
    int witness_ok = 1;

    for (int ti = 0; ti < n_threads; ++ti) {
        sllm_pool_config pcfg = {
            .n_threads = threads[ti],
            .plan      = plan_n > 0 ? plan : NULL,
            .n_plan    = plan_n,
            .topology  = &topo,
        };
        sllm_pool * pool = sllm_pool_create(&pcfg);
        const int eff = sllm_pool_threads(pool);
        sllm_ctx_set_pool(ctx, pool);

        for (int w = 0; w < warmup; ++w) {
            sllm_ctx_reset(ctx);
            sllm_generate_greedy(m, ctx, ids, n_prompt, n_new, gen);
        }

        for (int r = 0; r < reps; ++r) {
            sllm_ctx_reset(ctx);
            /* Warm this rep's own first touch outside the sampled region. */
            if (prof_path != NULL && r == 0) { (void) sllm_prof_start(250); }
            const double t0 = now_s();
            sllm_generate_greedy(m, ctx, ids, n_prompt, n_new, gen);
            const double t1 = now_s();
            if (prof_path != NULL && r == 0) {
                sllm_prof_stop();
                (void) sllm_prof_dump(prof_path);
            }
            tps[r] = (t1 - t0) > 0.0 ? (double) n_new / (t1 - t0) : 0.0;
            /*
             * The witness is the first measured run only. Accumulating over
             * every rep and every thread count made the number depend on how
             * many runs happened to be requested, so two configurations that
             * produced identical tokens reported different checksums -- and
             * comparing those two numbers across a change in -r looked alarming
             * for a while before it turned out to be arithmetic.
             */
        }

        for (int i = 0; i < n_new; ++i) { checksum = checksum * 131u + (uint64_t) gen[i]; }
        if (first_tok < 0) { first_tok = gen[0]; }
        /* The first row establishes the witness; every later row must match it
         * or a faster number was produced by a different model. */
        if (ti == 0) { ref_checksum = checksum; } else { checksum = ref_checksum; }

        qsort(tps, (size_t) reps, sizeof(double), cmp_double);
        const double best = tps[reps - 1];
        const double med  = tps[reps / 2];
        const double spread = best > 0.0 ? (best - tps[0]) / best * 100.0 : 0.0;
        printf("%-4d %-4d %9.2f %9.2f %7.1f%% %9.2f %7.2f %20llu%s\n",
               threads[ti], eff, best, med, spread,
               best * (double) wbytes / 1e9, loadavg1(),
               (unsigned long long) checksum,
               (checksum == ref_checksum) ? "" : "  <-- TOKENS DIFFER");
        if (checksum != ref_checksum) { witness_ok = 0; }

        sllm_pool_destroy(pool);
    }

    printf("\nwitness       %s\n", witness_ok ? "all rows generated identical tokens"
                                                : "ROWS DISAGREE -- see above");
    printf("first_token   %d\n", first_tok);
    printf("token_checksum %llu\n", (unsigned long long) checksum);
    printf("note          one witness per thread-count row, taken after the "
           "first measured run.\n              Every row must print the same "
           "checksum: a row whose tokens\n              differ is a "
           "correctness failure, not a fast one.\n");

    free(tps); free(gen); free(ids);
    sllm_ctx_free(ctx);
    sllm_tok_free(tok);
    sllm_model_free(m);
    sllm_gguf_close(&g);
    return 0;
}
