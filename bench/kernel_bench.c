/*
 * kernel_bench.c — isolated timing for the ternary dot kernels.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * Why a separate harness for one kernel.
 *
 * End-to-end tg throughput is the number that matters, but it is a poor
 * instrument for kernel work: it carries a 0.66 GB streaming read, thread
 * placement, page-cache state and enough run-to-run drift (measured at up to
 * 8 percent between processes on an otherwise idle machine) to hide a
 * difference the kernel actually has. Two back-to-back end-to-end runs
 * disagreed by more than the effect being looked for.
 *
 * So the dot kernel is timed here on its own, over real I2_S weights from the
 * model, with the two paths interleaved inside one process and many
 * repetitions. Interleaving matters: it means a thermal or frequency drift
 * hits both candidates equally instead of whichever one happened to run second.
 */

#include <saphira_llm/sllm.h>
#include <saphira_llm/forward.h>
#include <saphira_llm/gguf.h>
#include <saphira_llm/i2s_gemm.h>
#include <saphira_llm/isa.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec / 1e9;
}

static int64_t g_sink;

int main(int argc, char ** argv) {
    const char * model = NULL;
    int iters = 40;
    int rounds = 7;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) { model = argv[++i]; }
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) { iters = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) { rounds = atoi(argv[++i]); }
        else { fprintf(stderr, "usage: -m MODEL [-i ITERS] [-r ROUNDS]\n"); return 2; }
    }
    if (model == NULL) { fprintf(stderr, "usage: -m MODEL\n"); return 2; }

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(model, &g, err, sizeof err) != SLLM_OK) {
        fprintf(stderr, "open: %s\n", err); return 1;
    }

    /* Pick the real I2_S projections, which is what decode actually runs. */
    const sllm_gguf_tensor * t0 = NULL;
    const sllm_gguf_tensor * t1 = NULL;
    for (uint64_t i = 0; i < g.n_tensors; ++i) {
        if (g.tensors[i].type != SLLM_TYPE_I2_S) { continue; }
        if (g.tensors[i].ne[0] == 2560 && t0 == NULL) { t0 = &g.tensors[i]; }
        if (g.tensors[i].ne[0] == 6912 && t1 == NULL) { t1 = &g.tensors[i]; }
    }
    if (t0 == NULL || t1 == NULL) { fprintf(stderr, "no I2_S tensors found\n"); return 1; }


    printf("# ternary dot kernel, isolated, real I2_S weights\n");
    printf("model        %s\n", model);
    printf("iters/round  %d\n", iters);
    printf("rounds       %d (interleaved: v3, vnni, v3, vnni, ...)\n", rounds);

    const sllm_isa_dispatch probe = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    const sllm_isa_caps * caps = &probe.caps;
    const int have_vnni = sllm_isa_level_supported(SLLM_ISA_VNNI, caps);
    printf("vnni support %s\n", have_vnni ? "yes" : "no");

    const sllm_gguf_tensor * ts[2] = { t0, t1 };
    for (int which = 0; which < 2; ++which) {
        const sllm_gguf_tensor * t = ts[which];
        const size_t n     = (size_t) t->ne[0];
        const size_t rows  = (size_t) t->ne[1];
        const size_t rbytes = n / 32;
        const uint8_t * w = (const uint8_t *) t->data;

        /* A representative activation: the distribution only has to be
         * non-degenerate, since the kernel is integer and branch-free. */
        int8_t * a = (int8_t *) malloc(n);
        if (a == NULL) { return 1; }
        uint32_t rng = 0x1234u + (uint32_t) n;
        for (size_t d = 0; d < n; ++d) {
            rng = rng * 1664525u + 1013904223u;
            a[d] = (int8_t) ((rng >> 17) & 0xff);
        }

        double best[2] = { 1e30, 1e30 };

        for (int r = 0; r < rounds; ++r) {
            for (int k = 0; k < 2; ++k) {
                if (k == 1 && !have_vnni) { continue; }
                sllm_i2s_select_level(k == 0 ? SLLM_ISA_V3 : SLLM_ISA_VNNI);
                /* warm */
                for (size_t row = 0; row < 64 && row < rows; ++row) {
                    g_sink += sllm_i2s_dot(w + row * rbytes, a, n);
                }
                const double t0s = now_s();
                for (int it = 0; it < iters; ++it) {
                    for (size_t row = 0; row < rows; ++row) {
                        g_sink += sllm_i2s_dot(w + row * rbytes, a, n);
                    }
                }
                const double dt = now_s() - t0s;
                if (dt < best[k]) { best[k] = dt; }
            }
        }

        const double dots = (double) iters * (double) rows;
        printf("\ntensor n=%zu rows=%zu  (%.1f M values per pass)\n",
               n, rows, (double) n * rows / 1e6);
        const char * names[2] = { "v3   (maddubs)", "vnni (dpbusd)" };
        for (int k = 0; k < 2; ++k) {
            if (k == 1 && !have_vnni) { continue; }
            const double per    = best[k] / dots * 1e9;          /* ns per dot  */
            const double dots_s = (double) dots / best[k];         /* dots/s      */
            const double gval   = dots_s * (double) n / 1e9;       /* Gvalues/s   */
            const double gbyte  = dots_s * (double) rbytes / 1e9;  /* GB/s weights*/
            printf("  %-16s %8.3f ns/dot  %8.1f Mvalues/s  %6.2f GB/s weights\n",
                   names[k], per, gval * 1e3, gbyte);
        }
        if (have_vnni && best[1] > 0.0) {
            printf("  vnni vs v3: %.3fx  (negative means vnni is faster)\n",
                   best[0] / best[1]);
        }
        free(a);
    }
    sllm_gguf_close(&g);
    /* keep the compiler from deleting the work */
    if (g_sink == 0x7fffffff) { printf("%lld\n", (long long) g_sink); }
    return 0;
}
