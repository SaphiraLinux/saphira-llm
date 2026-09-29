/*
 * main.c — saphira-llm command line entry point.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The interface was fixed in Phase 1 so the user-facing shape does not churn;
 * model loading, tokenisation and greedy generation have since arrived.
 */

#include <saphira_llm/forward.h>
#include <saphira_llm/sllm.h>
#include <saphira_llm/tokenizer.h>
#include <saphira_llm/thread.h>
#include <saphira_llm/topology.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static struct timespec now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

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
    int32_t ctx_size = 0;     /* 0 = the model's trained context */
    int32_t n_predict = 0;   /* 0 = the default continuation length */

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

        if (!strcmp(a, "-c") || !strcmp(a, "--ctx")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            ctx_size = (int32_t) atoi(v);
            if (ctx_size < 1) {
                fprintf(stderr, "%s must be at least 1, got '%s'\n", a, v);
                return 2;
            }
            i++;
            continue;
        }
        if (!strcmp(a, "-n") || !strcmp(a, "--n-predict")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            n_predict = (int32_t) atoi(v);
            if (n_predict < 1) {
                fprintf(stderr, "%s must be at least 1, got '%s'\n", a, v);
                return 2;
            }
            i++;
            continue;
        }

        /* Still accepted for interface stability but not implemented. Rejecting
         * them loudly beats silently ignoring a flag the user believes is
         * having an effect. */
        if (!strcmp(a, "--temp") || !strcmp(a, "--seed")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            fprintf(stderr,
                "saphira-llm: %s is accepted but not implemented yet "
                "(sampling beyond greedy arrives with a later phase)\n", a);
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

    if (prompt == NULL) {
        sllm_log(SLLM_LOG_INFO, "container parsed; nothing to run (no -p)");
        sllm_gguf_close(&g);
        sllm_pool_destroy(pool);
        return 0;
    }

    /* --- model, tokenizer, context --- */
    sllm_model * model_h = NULL;
    sllm_status rc = sllm_model_load(&g, &model_h);
    if (rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm: cannot load this model: %s\n",
                sllm_status_string(rc));
        sllm_gguf_close(&g);
        sllm_pool_destroy(pool);
        return 1;
    }
    sllm_tok * tok = NULL;
    rc = sllm_tok_load(&g, &tok);
    if (rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm: cannot load the tokenizer: %s\n",
                sllm_status_string(rc));
        sllm_model_free(model_h);
        sllm_gguf_close(&g);
        sllm_pool_destroy(pool);
        return 1;
    }
    sllm_log(SLLM_LOG_INFO, "pre-tokeniser: %s (declared \"%s\")",
             sllm_tok_pre_type_name(sllm_tok_pre_type(tok)),
             sllm_tok_pre_declared(tok));

    const int32_t n_ctx = ctx_size > 0 ? ctx_size : 4096;
    sllm_ctx * c = NULL;
    rc = sllm_ctx_new(model_h, n_ctx, &c);
    if (rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm: cannot create a context: %s\n",
                sllm_status_string(rc));
        sllm_tok_free(tok);
        sllm_model_free(model_h);
        sllm_gguf_close(&g);
        sllm_pool_destroy(pool);
        return 1;
    }

    const size_t max_tok = (size_t) n_ctx;
    int32_t * ids = (int32_t *) malloc(max_tok * sizeof(int32_t));
    if (ids == NULL) {
        fprintf(stderr, "saphira-llm: out of memory\n");
        sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(model_h);
        sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }
    const int32_t n_prompt = sllm_tok_encode(tok, prompt, strlen(prompt),
                                             true, true, ids, (int32_t) max_tok);
    if (n_prompt < 0) {
        fprintf(stderr, "saphira-llm: tokenize failed: %s\n",
                sllm_status_string((sllm_status) n_prompt));
        free(ids);
        sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(model_h);
        sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    const int32_t n_new = n_predict > 0 ? n_predict : 64;
    if ((size_t) n_prompt + (size_t) n_new > max_tok) {
        fprintf(stderr, "saphira-llm: %d prompt tokens plus %d new exceeds "
                "the %d context\n", (int) n_prompt, (int) n_new, n_ctx);
        free(ids);
        sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(model_h);
        sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    int32_t * gen = (int32_t *) malloc((size_t) n_new * sizeof(int32_t));
    if (gen == NULL) {
        fprintf(stderr, "saphira-llm: out of memory\n");
        free(ids);
        sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(model_h);
        sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    /*
     * Generation is single-threaded in Phase 4. The pool exists and is measured,
     * but the forward pass does not use it yet, and reporting a thread count
     * that the forward pass ignores would be a lie told by a benchmark. Phase 6
     * is where the forward is parallelised and the thread count becomes real.
     */
    const struct timespec t0 = now();
    rc = sllm_generate_greedy(model_h, c, ids, n_prompt, n_new, gen);
    const struct timespec t1 = now();
    if (rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm: generation failed: %s\n",
                sllm_status_string(rc));
        free(gen); free(ids);
        sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(model_h);
        sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    const double secs = (double) (t1.tv_sec - t0.tv_sec) +
                        (double) (t1.tv_nsec - t0.tv_nsec) / 1e9;
    const double tps = secs > 0.0 ? (double) n_new / secs : 0.0;

    /*
     * The continuation is decoded, not printed as pieces. A piece is the
     * byte-encoded form, so a space is U+0120 and printing it raw produces
     * "The capital is a small town" with the separator glyphs still in it.
     * Decoding is also the check that decode works on real generated ids
     * rather than only on ids the encoder produced.
     */
    {
        char * text = (char *) malloc((size_t) (n_new + 1) * 64 + 1);
        if (text != NULL) {
            const int32_t len = sllm_tok_decode(tok, gen, n_new, true,
                                                text, (int32_t) ((size_t) (n_new + 1) * 64 + 1));
            if (len > 0) { fputs(text, stdout); }
            putchar('\n');
            free(text);
        }
    }

    sllm_log(SLLM_LOG_INFO, "prompt %d tokens, generated %d in %.3f s = %.2f t/s",
             (int) n_prompt, (int) n_new, secs, tps);
    printf("sllm_bench tg n_predict %d time %.4f s tps %.2f threads %d\n",
           (int) n_new, secs, tps, 1);

    free(gen); free(ids);
    sllm_ctx_free(c); sllm_tok_free(tok); sllm_model_free(model_h);
    sllm_gguf_close(&g);
    sllm_pool_destroy(pool);
    return 0;
}
