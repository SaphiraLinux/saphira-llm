/*
 * capture_i2s_golden.c — reference-only: capture I2_S dequantisation fixtures.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * NOT a shipped tool. It links the upstream reference so that its output can
 * be recorded as a permanent, reference-free test fixture.
 *
 * Why this exists. The I2_S reader was wrong twice: it reversed the four
 * two-bit fields within every byte, and it returned the raw code where the
 * semantic value was wanted. Neither error was caught by the golden vector
 * written alongside it, because that vector built its packed bytes and read
 * them back with the same wrong assumption. A test that shares an assumption
 * with the code under test cannot detect that the code is wrong.
 *
 * The only thing that settles it is the reference's own dequantiser, run on
 * real tensors from the real model. This tool records what it produces:
 *
 *   - the f32 scale, exactly as the reference reports it
 *   - the first PREVIEW values, so a mismatch is diagnosable by eye
 *   - an FNV-1a 64 hash over EVERY element of the tensor, so the gate covers
 *     millions of values without storing them
 *
 * The resulting tests/golden/i2s-reference.txt is checked by the ordinary test
 * suite with no reference library present.
 *
 * Build and run:
 *   cc -O2 -o /tmp/capture tools/reference/capture_i2s_golden.c \
 *      -Iinclude -Ithird_party/BitNet/3rdparty/llama.cpp/include \
 *      -Ithird_party/BitNet/3rdparty/llama.cpp/ggml/include \
 *      src/gguf.c src/quant.c src/log.c src/status.c \
 *      -L build-base/bin -lggml -lggml-base -lggml-cpu -lm -lpthread -ldl
 *   LD_LIBRARY_PATH=build-base/bin /tmp/capture \
 *      /var/lib/spoon/models/ggml-model-i2_s.gguf
 */

#include <saphira_llm/sllm.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* The reference's own dequantiser. Declared here rather than included so this
 * file does not need the reference's private headers. */
void dequantize_row_i2_s(const uint8_t * x, float * y, int64_t n, const float i2_scale);

/* The reference's own matrix kernels, called with nr = number of activation
 * columns and nc = number of weight rows -- the order is not what the parameter
 * names suggest, and getting it wrong looks like a numerical mismatch rather
 * than a wrong call. */
void ggml_gemv_i2_i8_s(int n, float * s, size_t bs, const void * vx,
                       const void * vy, int nr, int nc);
void ggml_gemm_i2_i8_s(int n, float * s, size_t bs, const void * vx,
                       const void * vy, int nr, int nc);

#define PREVIEW 32

static uint64_t fnv1a64(const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* One fixture record. */
static void emit(FILE * f, const sllm_gguf_tensor * t, const void * data) {
    size_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; ++d) {
        n *= (size_t) t->ne[d];
    }

    const float scale = sllm_i2s_scale(data, n);
    float * ref = malloc(n * sizeof(float));
    if (ref == NULL) {
        fprintf(stderr, "out of memory for %s\n", t->name);
        exit(1);
    }
    dequantize_row_i2_s((const uint8_t *) data, ref, (int64_t) n, scale);

    fprintf(f, "tensor %s\n", t->name);
    fprintf(f, "  elements %zu\n", n);
    fprintf(f, "  scale %.9g\n", (double) scale);
    fprintf(f, "  fnv1a64 %016llx\n", (unsigned long long) fnv1a64(ref, n * sizeof(float)));
    fprintf(f, "  preview");
    const size_t m = n < PREVIEW ? n : PREVIEW;
    for (size_t i = 0; i < m; ++i) {
        fprintf(f, " %.6f", (double) ref[i]);
    }
    fprintf(f, "\n\n");
    free(ref);
}

/*
 * Record the reference's GEMV and GEMM results on real weights against a
 * deterministic synthetic activation.
 *
 * This is the kernel-level parity gate. The activation is synthetic but the
 * weights are the model's, so the packed layout, the tile structure and the
 * activation quantisation are all exercised on real data. The generator is
 * fixed, so the fixture is reproducible byte for byte.
 */
static void emit_gemv(FILE * f, const sllm_gguf_tensor * t) {
    const size_t n = t->ne[0];
    size_t total = 1;
    for (uint32_t d = 0; d < t->n_dims; ++d) { total *= (size_t) t->ne[d]; }
    const size_t rows = total / n;
    const size_t R = rows < 512 ? rows : 512;

    float * x = malloc(n * sizeof(float));
    int8_t * q = malloc(n);
    if (!x || !q) { exit(1); }
    uint32_t rng = 0xBEEF01u;
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        x[i] = (((float) ((rng >> 8) & 0xffff) / 32768.0f) - 1.0f) * 3.0f;
    }
    sllm_i2s_act act;
    sllm_i2s_quant_act(x, n, q, &act);

    float * gemv = calloc(R, sizeof(float));
    float * gemm = calloc(R, sizeof(float));
    if (!gemv || !gemm) { exit(1); }
    ggml_gemv_i2_i8_s((int) n, gemv, 1, t->data, q, 1, (int) R);
    ggml_gemm_i2_i8_s((int) n, gemm, 1, t->data, q, 1, (int) R);

    fprintf(f, "gemv %s\n", t->name);
    fprintf(f, "  n %zu\n", n);
    fprintf(f, "  rows %zu\n", R);
    fprintf(f, "  act_scale %.9g\n", (double) act.scale);
    fprintf(f, "  act_sum %d\n", act.sum);
    const float w_scale = sllm_i2s_scale(t->data, total);
    fprintf(f, "  w_scale %.9g\n", (double) w_scale);
    fprintf(f, "  fnv1a64 %016llx\n",
            (unsigned long long) fnv1a64(gemv, R * sizeof(float)));
    fprintf(f, "  gemv_gemm_agree %d\n",
            memcmp(gemv, gemm, R * sizeof(float)) == 0 ? 1 : 0);
    /*
     * The reference's kernels return the RAW dot products; the epilogue is
     * applied by the caller, in ggml-bitnet-compute.c. So the raw hash above
     * gates the kernel, and a second hash gates the epilogue, computed here
     * from the reference's OWN raw output and the documented formula:
     *
     *     post_scale = w_scale / act_scale
     *     dst       = (dot - act_sum) * post_scale
     */
    {
        float * epi = calloc(R, sizeof(float));
        if (!epi) { exit(1); }
        for (size_t r = 0; r < R; ++r) {
            /* The reference's forward path (ggml-cpu.c) does the division ONCE
             * per column, then (dot - sum) * post_scale. Matching that grouping
             * matters: the per-element form rounds differently. */
            const float post_scale = (float) w_scale / act.scale;
            epi[r] = (gemv[r] - (float) act.sum) * post_scale;
        }
        fprintf(f, "  epilogue_fnv1a64 %016llx\n",
                (unsigned long long) fnv1a64(epi, R * sizeof(float)));
        fprintf(f, "  epilogue_values");
        for (size_t r = 0; r < R && r < 8; ++r) { fprintf(f, " %.6f", (double) epi[r]); }
        fprintf(f, "\n");
        free(epi);
    }
    fprintf(f, "  values");
    for (size_t r = 0; r < R && r < 16; ++r) { fprintf(f, " %.1f", (double) gemv[r]); }
    fprintf(f, "\n\n");

    free(gemv); free(gemm); free(q); free(x);
}

int main(int argc, char ** argv) {
    const char * model = (argc > 1) ? argv[1] : "/var/lib/spoon/models/ggml-model-i2_s.gguf";
    const char * out   = (argc > 2) ? argv[2] : "tests/golden/i2s-reference.txt";

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(model, &g, err, sizeof(err)) != SLLM_OK) {
        fprintf(stderr, "open: %s\n", err);
        return 1;
    }

    FILE * f = fopen(out, "w");
    if (f == NULL) {
        fprintf(stderr, "cannot write %s\n", out);
        return 1;
    }
    fprintf(f, "# saphira-llm I2_S golden vectors\n");
    fprintf(f, "# Captured from the REFERENCE, never from us.\n");
    fprintf(f, "#\n");
    fprintf(f, "#   tensor records  : dequantize_row_i2_s over every element of a real\n");
    fprintf(f, "#                     tensor. Verified by FNV-1a 64 over all of it, so the\n");
    fprintf(f, "#                     gate covers millions of values without storing them.\n");
    fprintf(f, "#   gemv records    : ggml_gemv_i2_i8_s and ggml_gemm_i2_i8_s on real weights\n");
    fprintf(f, "#                     against a fixed synthetic activation, plus the act\n");
    fprintf(f, "#                     scale and sum our quantiser must produce to match.\n");
    fprintf(f, "#\n");
    fprintf(f, "# The activation quantiser is the only free variable here: the reference\n");
    fprintf(f, "# takes an already-quantised int8 row, so the fixture pins the exact input\n");
    fprintf(f, "# our quantiser has to reproduce before the kernels mean anything.\n");
    fprintf(f, "model %s\n\n", model);

    /* A representative spread: a square projection, the wide FFN matrix, and
     * the deepest layer, so the fixture is not one shape. */
    static const char * wanted[] = {
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "blk.15.ffn_gate.weight",
        "blk.29.attn_output.weight",
    };

    for (size_t w = 0; w < sizeof(wanted) / sizeof(wanted[0]); ++w) {
        const sllm_gguf_tensor * t = sllm_gguf_find_tensor(&g, wanted[w]);
        if (t == NULL) {
            fprintf(stderr, "no tensor %s, skipping\n", wanted[w]);
            continue;
        }
        if (t->type != SLLM_TYPE_I2_S) {
            fprintf(stderr, "%s is %s, not I2_S, skipping\n", t->name,
                    sllm_gguf_type_name(t->type));
            continue;
        }
        emit(f, t, t->data);
        emit_gemv(f, t);
        fprintf(stderr, "captured %s\n", t->name);
    }

    fclose(f);
    sllm_gguf_close(&g);
    return 0;
}
