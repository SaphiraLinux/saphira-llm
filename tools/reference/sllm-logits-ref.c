/*
 * sllm-logits-ref — reference golden-vector dumper for saphira-llm.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * THIS IS A REFERENCE TOOL. It is not part of the saphira-llm runtime, it is
 * not installed, and it is never shipped. It links against the *upstream*
 * BitNet-capable llama.cpp so that it can capture golden vectors from the
 * implementation we are being compared against. It exists only to produce
 * the fixtures in tests/golden/.
 *
 * It must not be used at inference time and it introduces no Python.
 *
 * Output per prompt position:
 *   - every logit as raw f32, little-endian, into a .f32 binary
 *   - a text manifest with token ids, top-k tokens, logit statistics and an
 *     FNV-1a hash of the raw logit bytes for exact-match checking
 *   - a greedy continuation, for the token-level parity gate
 */

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* The pinned reference header declares these as bare struct tags rather than
 * providing typedefs. Alias them here instead of editing upstream. */
typedef struct llama_model          llama_model;
typedef struct llama_context        llama_context;
typedef struct llama_vocab          llama_vocab;
typedef struct llama_model_params   llama_model_params;
typedef struct llama_context_params llama_context_params;

static void die(const char * msg) {
    fprintf(stderr, "sllm-logits-ref: %s\n", msg);
    exit(1);
}

/* FNV-1a 64, so a bit-exact match can be asserted with a single line. */
static uint64_t fnv1a64(const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

struct options {
    const char * model;
    const char * prompt;
    const char * out_prefix;
    int         n_ctx;
    int         n_threads;
    int         n_predict;
    int         top_k;
    bool        add_special;
};

static struct options opts = {
    .model       = NULL,
    .prompt      = NULL,
    .out_prefix  = NULL,
    .n_ctx       = 512,
    .n_threads   = 1,
    .n_predict   = 0,
    .top_k       = 10,
    .add_special = true,
};

static void usage(void) {
    fprintf(stderr,
        "usage: sllm-logits-ref -m MODEL -p PROMPT -o PREFIX [options]\n"
        "  -c N    context size (default %d)\n"
        "  -t N    threads (default %d; keep fixed, reduction order affects bits)\n"
        "  -n N    greedy tokens to generate after the prompt (default %d)\n"
        "  -k N    top-k recorded per position (default %d)\n"
        "  -s      do not add special tokens when tokenising (default: add them)\n",
        opts.n_ctx, opts.n_threads, opts.n_predict, opts.top_k);
    exit(2);
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(a, "-m") && v) { opts.model      = v; i++; }
        else if (!strcmp(a, "-p") && v) { opts.prompt     = v; i++; }
        else if (!strcmp(a, "-o") && v) { opts.out_prefix = v; i++; }
        else if (!strcmp(a, "-c") && v) { opts.n_ctx      = atoi(v); i++; }
        else if (!strcmp(a, "-t") && v) { opts.n_threads  = atoi(v); i++; }
        else if (!strcmp(a, "-n") && v) { opts.n_predict  = atoi(v); i++; }
        else if (!strcmp(a, "-k") && v) { opts.top_k      = atoi(v); i++; }
        else if (!strcmp(a, "-s"))       { opts.add_special = false; }
        else usage();
    }
    if (!opts.model || !opts.prompt || !opts.out_prefix) {
        usage();
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = true;

    const double t_load0 = now_ms();
    llama_model * model = llama_model_load_from_file(opts.model, mparams);
    if (!model) {
        die("failed to load model");
    }
    const double t_load1 = now_ms();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        die("model has no vocab");
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = (uint32_t) opts.n_ctx;
    cparams.n_batch       = (uint32_t) opts.n_ctx;
    cparams.n_ubatch      = (uint32_t) opts.n_ctx;
    cparams.n_seq_max     = 1;
    cparams.n_threads     = opts.n_threads;
    cparams.n_threads_batch = opts.n_threads;
    cparams.n_outputs_max = (uint32_t) opts.n_ctx;
    cparams.embeddings    = false;
    /* KV type left at the model default so the fixture records what upstream
     * actually chose, rather than what we asked for. */
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        die("failed to create context");
    }
    const double t_ctx1 = now_ms();

    /* ---- tokenise ---- */
    const int n_text = (int) strlen(opts.prompt);
    llama_token * toks = calloc((size_t) opts.n_ctx, sizeof(llama_token));
    if (!toks) {
        die("out of memory");
    }
    const int n_tok = llama_tokenize(vocab, opts.prompt, n_text, toks, opts.n_ctx,
                                     opts.add_special, /*parse_special*/ true);
    if (n_tok < 0) {
        die("tokenise failed");
    }
    if (n_tok >= opts.n_ctx) {
        die("prompt does not fit in the context");
    }

    /* ---- evaluate the prompt in one batch ----
     * llama_batch_get_one() only marks the final token for logits, and we want
     * every position, so the batch is built explicitly. */
    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    if (!batch.token) {
        die("llama_batch_init failed");
    }
    /* llama_batch_init leaves every member uninitialised, including n_tokens
     * and the per-token logits flags. */
    batch.n_tokens = n_tok;
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i]      = toks[i];
        batch.pos[i]        = i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = 1;
    }
    const double t_pp0 = now_ms();
    if (llama_decode(ctx, batch) != 0) {
        die("llama_decode failed on the prompt");
    }
    const double t_pp1 = now_ms();

    /* ---- capture logits for every prompt position ---- */
    float * all_logits = calloc((size_t) n_tok * (size_t) n_vocab, sizeof(float));
    if (!all_logits) {
        die("out of memory allocating the logit buffer");
    }
    for (int i = 0; i < n_tok; ++i) {
        const float * src = llama_get_logits_ith(ctx, i);
        if (!src) {
            die("llama_get_logits_ith returned NULL; is n_outputs_max set?");
        }
        memcpy(all_logits + (size_t) i * (size_t) n_vocab, src, (size_t) n_vocab * sizeof(float));
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s.f32", opts.out_prefix);
    FILE * f = fopen(path, "wb");
    if (!f) {
        die("cannot open the .f32 output");
    }
    fwrite(all_logits, sizeof(float), (size_t) n_tok * (size_t) n_vocab, f);
    fclose(f);

    /* ---- manifest ---- */
    snprintf(path, sizeof(path), "%s.txt", opts.out_prefix);
    f = fopen(path, "w");
    if (!f) {
        die("cannot open the manifest output");
    }
    fprintf(f, "# saphira-llm reference golden vector\n");
    fprintf(f, "model            %s\n", opts.model);
    fprintf(f, "prompt           %s\n", opts.prompt);
    fprintf(f, "n_tokens         %d\n", n_tok);
    fprintf(f, "n_vocab          %d\n", n_vocab);
    fprintf(f, "n_ctx            %d\n", opts.n_ctx);
    fprintf(f, "n_threads        %d\n", opts.n_threads);
    fprintf(f, "add_special      %d\n", opts.add_special ? 1 : 0);
    fprintf(f, "type_k           %d\n", (int) cparams.type_k);
    fprintf(f, "type_v           %d\n", (int) cparams.type_v);
    fprintf(f, "flash_attn_type  %d\n", (int) cparams.flash_attn_type);
    fprintf(f, "load_ms          %.3f\n", t_load1 - t_load0);
    fprintf(f, "ctx_ms           %.3f\n", t_ctx1 - t_load1);
    fprintf(f, "prompt_ms        %.3f\n", t_pp1 - t_pp0);
    fprintf(f, "prompt_tps       %.4f\n", n_tok / ((t_pp1 - t_pp0) / 1000.0));

    fprintf(f, "logits_fnv1a64   %016llx\n",
            (unsigned long long) fnv1a64(all_logits, (size_t) n_tok * (size_t) n_vocab * sizeof(float)));

    fprintf(f, "\ntokens\n");
    for (int i = 0; i < n_tok; ++i) {
        char piece[256];
                int np = llama_token_to_piece(vocab, toks[i], piece, sizeof(piece), 0, false);
        if (np < 0) {
            np = 0;
        }
        piece[np] = '\0';
        for (int k = 0; k < np; ++k) {
            if (piece[k] == '\n') { piece[k] = ' '; }
        }
        fprintf(f, "  %4d  %8d  %s\n", i, (int) toks[i], piece);
    }

    fprintf(f, "\nlogits\n");
    for (int i = 0; i < n_tok; ++i) {
        const float * lg = all_logits + (size_t) i * (size_t) n_vocab;
        double sum = 0.0, sumsq = 0.0, amax = -INFINITY, amin = INFINITY;
        int imax = -1;
        for (int v = 0; v < n_vocab; ++v) {
            const double x = lg[v];
            sum  += x;
            sumsq += x * x;
            if (x > amax) { amax = x; imax = v; }
            if (x < amin) { amin = x; }
        }
        const double mean = sum / (double) n_vocab;
        fprintf(f, "  pos %4d  argmax %8d  max %.6f  min %.6f  mean %.6f  rms %.6f  sum %.6f\n",
                i, imax, amax, amin, mean, sqrt(sumsq / (double) n_vocab), sum);

        /* Top-k by selection. k is small, so k linear passes is both obviously
         * correct and utterly negligible next to the decode. */
        const int k = opts.top_k < n_vocab ? opts.top_k : n_vocab;
        int * picked = calloc((size_t) k, sizeof(int));
        char * taken = calloc((size_t) n_vocab, sizeof(char));
        if (!picked || !taken) {
            die("out of memory");
        }
        for (int j = 0; j < k; ++j) {
            int best = -1;
            for (int v = 0; v < n_vocab; ++v) {
                if (taken[v]) {
                    continue;
                }
                if (best < 0 || lg[v] > lg[best]) {
                    best = v;
                }
            }
            if (best < 0) {
                break;
            }
            taken[best] = 1;
            picked[j] = best;
        }
        fprintf(f, "    top:");
        for (int j = 0; j < k; ++j) {
            if (picked[j] < 0) {
                break;
            }
            fprintf(f, " %d:%.6f", picked[j], lg[picked[j]]);
        }
        fprintf(f, "\n");
        free(picked);
        free(taken);
    }

    /* ---- greedy continuation ---- */
    llama_token next = 0;
    float next_logit = 0.0f;
    {
        const float * lg = llama_get_logits_ith(ctx, n_tok - 1);
        if (!lg) {
            die("no logits for the final prompt position");
        }
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (lg[v] > lg[best]) {
                best = v;
            }
        }
        next = best;
        next_logit = lg[best];
    }
    if (opts.n_predict > 0) {
        fprintf(f, "\ngreedy\n");
        const int total = n_tok + opts.n_predict;
        int pos = n_tok;
        const double t_gen0 = now_ms();

        /*
         * Emit the token, then consume it.
         *
         * The previous version decoded `next` and then printed the argmax of
         * the logits that decode produced, labelling it with the position of
         * the token just consumed. That printed the prediction for position
         * pos+1 under the label pos, and it never printed the first generated
         * token at all, because that one came from the prompt decode. The
         * .f32 logits and their FNV hash were unaffected -- those are written
         * above this block -- but the listing was off by one, and a listing
         * that mislabels its positions is worse than no listing.
         */
        while (pos < total) {
            char piece[256];
            int np = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, false);
            if (np < 0) {
                np = 0;
            }
            piece[np] = '\0';
            for (int k = 0; k < np; ++k) {
                if (piece[k] == '\n') { piece[k] = ' '; }
            }
            fprintf(f, "  %4d  %8d  %.6f  %s\n", pos, (int) next, next_logit, piece);

            /* Consume it at this position; that produces position pos+1. */
            llama_memory_t mem = llama_get_memory(ctx);
            llama_memory_seq_rm(mem, 0, pos, pos);
            llama_batch gb = llama_batch_init(1, 0, 1);
            if (!gb.token) {
                die("llama_batch_init failed during generation");
            }
            gb.n_tokens   = 1;
            gb.token[0]     = next;
            gb.pos[0]       = pos;
            gb.n_seq_id[0]  = 1;
            gb.seq_id[0][0] = 0;
            gb.logits[0]    = 1;
            if (llama_decode(ctx, gb) != 0) {
                die("llama_decode failed during generation");
            }
            llama_batch_free(gb);

            const float * lg = llama_get_logits_ith(ctx, 0);
            if (!lg) {
                die("no logits during generation");
            }
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) {
                if (lg[v] > lg[best]) {
                    best = v;
                }
            }
            next = best;
            next_logit = lg[best];
            ++pos;
        }
        const double t_gen1 = now_ms();
        fprintf(f, "gen_tps         %.4f\n", opts.n_predict / ((t_gen1 - t_gen0) / 1000.0));
    }

    llama_batch_free(batch);
    fclose(f);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(toks);
    free(all_logits);
    return 0;
}
