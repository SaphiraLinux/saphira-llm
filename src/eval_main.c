/*
 * eval_main.c — saphira-llm-eval: model-quality measurement.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * THIS IS NOT A CORRECTNESS GATE. It is the opposite kind of tool.
 *
 * Everything in saphira-llm-test answers "does this compute the right answer":
 * token-identical output against the reference, golden vectors, a proven ISA
 * baseline. Those are pass/fail and they must never move. If perplexity here
 * changes by a thousandth, that is a note, not a failure.
 *
 * This answers "is this model any good", which is a question the correctness
 * gates cannot ask at all. A model can be bit-exact against a bad reference
 * and still be useless; a model can pass every gate and still be a worse
 * model than it was on Tuesday. Only a number measured on fixed text, by
 * software that does not change, can tell you whether a conversion step, a
 * quantisation change or a training run helped.
 *
 * The metric is the one the reference uses, taken from the pinned tree rather
 * than from memory. third_party/llama.cpp/tools/perplexity/perplexity.cpp
 * scores, for each position j:
 *
 *     nll += -log( softmax(logits[j])[ tokens[j+1] ] )
 *     ...
 *     perplexity = exp(nll / count)
 *
 * and defines log_softmax as
 * (perplexity.cpp:60)
 *
 *     m       = max(logits)
 *     sum_exp = sum(expf(logits[i] - m))
 *     result  = logits[tok] - m - log(sum_exp)
 *
 * We use the algebraically identical log-sum-exp form,
 *
 *     nll = log( sum_i exp(L[i] - m) ) + m - L[tok]
 *
 * computed in double. It is the same number to within a few ulps and it does
 * not form an intermediate probability that can round to zero for a token the
 * model is confident it will never emit -- which is precisely the token whose
 * log matters most. The difference is measured, not assumed: see
 * test_eval.c, which checks this form against the reference's softmax-then-log
 * form and against a float-accumulated variant.
 *
 * WHAT IS SCORED, precisely, so the number is reproducible by someone else:
 *
 *   - The corpus is tokenised with this model's own tokenizer, with
 *     add_special and parse_special both set, matching saphira-llm.
 *   - The corpus is fed in chunks of at most SLLM_MAX_CHUNK, at consecutive
 *     absolute positions, through ONE context whose KV cache therefore spans
 *     the whole corpus.
 *   - The logit row at index i of a chunk beginning at position p is the
 *     distribution after consuming tokens[p..p+i], so it predicts tokens[p+i+1].
 *   - Every token except the very first is scored, once, against the row that
 *     precedes it. The first token has no context and is therefore unscorable;
 *     that is not a limitation, it is what a language model is defined to do.
 *   - scored_tokens == corpus_tokens - 1, always. A run that does not report
 *     that relationship is a bug.
 *
 * KNOWN LIMITATION, stated rather than hidden: the corpus must fit the context
 * in full. llama.cpp scores long corpora with a rolling window and a stride,
 * re-feeding an overlap so every token still sees a full window. That is not
 * implemented here, so this tool refuses a corpus longer than --max-tokens
 * instead of silently scoring a prefix. A rolling window is the right answer
 * for wiki-scale text and is the first thing to add.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Included individually rather than through the umbrella, matching src/main.c.
 * saphira.h is the sealed v0.0.1 surface and is not being extended by a tool
 * that appeared after it. */
#include <saphira_llm/forward.h>
#include <saphira_llm/sllm.h>
#include <saphira_llm/thread.h>
#include <saphira_llm/tokenizer.h>
#include <saphira_llm/topology.h>
#include <saphira_llm/eval.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static struct timespec now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t;
}

static void usage(FILE * out) {
    fprintf(out,
        "usage: saphira-llm-eval -m MODEL -d CORPUS [options]\n"
        "\n"
        "Measures model quality: token-level cross entropy and perplexity over a\n"
        "corpus. This is a MEASUREMENT, not a pass/fail gate.\n"
        "\n"
        "  -m, --model FILE     GGUF model (required)\n"
        "  -d, --data FILE      corpus text file (required)\n"
        "  -c, --ctx N          context size; must hold the whole corpus\n"
        "  -t, --threads N      worker threads (speed only; cannot change results)\n"
        "  -n, --max-tokens N   refuse a corpus longer than this (default 8192)\n"
        "      --stride N       rolling window: score N tokens per window instead\n"
        "                        of the whole corpus at once (0 = single window)\n"
        "      --no-place       do not pin threads\n"
        "      --isa LEVEL      v3 | vnni | avx512 | amx | auto\n"
        "      --log LEVEL      quiet | error | warn | info | debug\n"
        "      --json           machine-readable result on stdout\n"
        "  -V, --version        version and ISA baseline\n"
        "  -h, --help           this text\n"
        "\n"
        "Perplexity is exp(total_nll / scored_tokens). Lower is better. It is only\n"
        "comparable between runs that used the same corpus, the same tokenizer\n"
        "and the same context size.\n");
}

int main(int argc, char ** argv) {
    const char * model_path = NULL;
    const char * data_path  = NULL;
    int32_t ctx_size   = 0;
    int   n_threads    = 0;
    int32_t max_tokens = 8192;
    int32_t stride    = 0;   /* 0 = single window, the whole corpus at once */
    bool no_place = false;
    bool json = false;
    sllm_isa_level isa_floor = SLLM_ISA_LEVEL_AUTO;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = (i + 1 < argc) ? argv[i + 1] : NULL;

        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        if (!strcmp(a, "-V") || !strcmp(a, "--version")) {
            printf("saphira-llm-eval %s (baseline %s)\n",
                   SLLM_VERSION_STRING, SLLM_BASELINE_ISA);
            return 0;
        }
        if (!strcmp(a, "-m") || !strcmp(a, "--model")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            model_path = v; i++; continue;
        }
        if (!strcmp(a, "-d") || !strcmp(a, "--data")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            data_path = v; i++; continue;
        }
        if (!strcmp(a, "-c") || !strcmp(a, "--ctx")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            ctx_size = (int32_t) atoi(v);
            if (ctx_size < 1) { fprintf(stderr, "%s must be at least 1\n", a); return 2; }
            i++; continue;
        }
        if (!strcmp(a, "-t") || !strcmp(a, "--threads")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            n_threads = atoi(v);
            if (n_threads < 1) { fprintf(stderr, "%s must be at least 1\n", a); return 2; }
            i++; continue;
        }
        if (!strcmp(a, "-n") || !strcmp(a, "--max-tokens")) {
            if (!v) { fprintf(stderr, "%s needs a value\n", a); return 2; }
            max_tokens = (int32_t) atoi(v);
            if (max_tokens < 2) { fprintf(stderr, "%s must be at least 2\n", a); return 2; }
            i++; continue;
        }
        if (!strcmp(a, "--stride")) {
            if (!v) { fprintf(stderr, "--stride needs a value\n"); return 2; }
            stride = (int32_t) atoi(v);
            if (stride < 0) { fprintf(stderr, "--stride must not be negative\n"); return 2; }
            i++; continue;
        }
        if (!strcmp(a, "--no-place")) { no_place = true; continue; }
        if (!strcmp(a, "--json")) { json = true; continue; }
        if (!strcmp(a, "--isa")) {
            if (!v) { fprintf(stderr, "--isa needs a value\n"); return 2; }
            if (sllm_isa_level_parse(v, &isa_floor) != SLLM_OK) {
                fprintf(stderr, "--isa '%s' is not one of: v3, vnni, avx512, amx, auto\n", v);
                return 2;
            }
            i++; continue;
        }
        if (!strcmp(a, "--log")) {
            if (!v) { fprintf(stderr, "--log needs a value\n"); return 2; }
            sllm_log_level lvl = SLLM_LOG_INFO;
            if (sllm_log_level_parse(v, &lvl) != SLLM_OK) {
                fprintf(stderr, "--log '%s' is not one of: quiet, error, warn, info, debug\n", v);
                return 2;
            }
            sllm_log_set_level(lvl);
            i++; continue;
        }
        fprintf(stderr, "saphira-llm-eval: unknown argument '%s'\n", a);
        usage(stderr);
        return 2;
    }

    if (model_path == NULL || data_path == NULL) { usage(stderr); return 2; }

    const sllm_isa_dispatch isa = sllm_isa_build(isa_floor);
    sllm_isa_caps caps = isa.caps;
    if (sllm_isa_check_floor(isa_floor, &caps) != SLLM_OK) {
        (void) sllm_fail(SLLM_ERR_UNSUPPORTED,
            "--isa %s was requested but this CPU tops out at %s",
            sllm_isa_level_name(isa_floor), sllm_isa_level_name(sllm_isa_ceiling(&caps)));
        return 1;
    }

    /* Same placement and pool setup as saphira-llm, so an eval number and a
     * generation number were produced by the same arithmetic. */
    sllm_topology topo;
    sllm_topology_detect(&topo);
    (void) sllm_topology_calibrate(&topo, 2000000);
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

    sllm_gguf g;
    char err[512];
    const sllm_status open_rc = sllm_gguf_open(model_path, &g, err, sizeof err);
    if (open_rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm-eval: %s: %s: %s\n",
                model_path, sllm_status_string(open_rc), err[0] ? err : "(no detail)");
        sllm_pool_destroy(pool);
        return 1;
    }

    sllm_model * m = NULL;
    sllm_status rc = sllm_model_load(&g, &m);
    if (rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm-eval: cannot load this model: %s\n", sllm_status_string(rc));
        sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }
    sllm_tok * tok = NULL;
    rc = sllm_tok_load(&g, &tok);
    if (rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm-eval: cannot load the tokenizer: %s\n", sllm_status_string(rc));
        sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    const int32_t n_vocab = sllm_model_n_vocab(m);

    /* --- corpus --- */
    FILE * f = fopen(data_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "saphira-llm-eval: cannot read corpus '%s'\n", data_path);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); fprintf(stderr, "corpus is not seekable\n"); return 1; }
    const long data_len = ftell(f);
    rewind(f);
    if (data_len < 0) { fclose(f); fprintf(stderr, "cannot size corpus\n"); return 1; }
    char * text = (char *) malloc((size_t) data_len + 1);
    if (text == NULL) { fclose(f); fprintf(stderr, "out of memory\n"); return 1; }
    if (data_len > 0 && fread(text, 1, (size_t) data_len, f) != (size_t) data_len) {
        fclose(f); free(text);
        fprintf(stderr, "short read on corpus\n");
        return 1;
    }
    fclose(f);
    text[data_len] = '\0';

    /* Two-pass tokenise: the count is an output of the first pass, not a
     * guess, so the buffer is never oversized or truncated silently. */
    int32_t cap = sllm_tok_encode_len(tok, (size_t) data_len, true);
    if (cap < 1) { cap = (int32_t) data_len + 8; }
    int32_t * ids = (int32_t *) malloc((size_t) cap * sizeof(int32_t));
    if (ids == NULL) { free(text); fprintf(stderr, "out of memory\n"); return 1; }
    const int32_t n_tok = sllm_tok_encode(tok, text, (size_t) data_len,
                                          true, true, ids, cap);
    free(text);
    if (n_tok < 0) {
        fprintf(stderr, "saphira-llm-eval: tokenize failed: %s\n", sllm_status_string((sllm_status) n_tok));
        free(ids);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    if (n_tok < 2) {
        fprintf(stderr, "saphira-llm-eval: corpus is %d token(s); a perplexity needs at "
                        "least 2 (the first token has no context to be predicted from)\n", n_tok);
        free(ids);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }
    /*
     * Window and stride validation.
     *
     * A stride of 0 means the whole corpus in one window, which is the only
     * mode available before this and remains the default. It requires the
     * context to hold the corpus, and a corpus that does not fit is REFUSED
     * rather than truncated -- that behaviour is correct and is kept.
     *
     * With a stride, the corpus is walked in windows of n_ctx tokens that each
     * start `stride` tokens further on, and only the last `stride` tokens of
     * each window are scored. A stride equal to n_ctx is refused because it
     * would score a token whose only context is itself, and because the
     * reference's own loop runs off the front of the logit buffer at that
     * value (perplexity.cpp:419, where j begins at n_ctx - stride - 1).
     */
    int32_t n_ctx = ctx_size > 0 ? ctx_size : n_tok;
    if (stride == 0 && n_ctx < n_tok) {
        fprintf(stderr,
            "saphira-llm-eval: corpus is %d tokens but --ctx is %d.\n"
            "  Without a stride the whole corpus is scored in one window, so the\n"
            "  context must hold it. Raise --ctx, or pass --stride to walk the\n"
            "  corpus in windows. Nothing is ever scored as a silent prefix.\n",
            n_tok, n_ctx);
        free(ids);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }
    if (stride > 0) {
        if (n_ctx < 2) {
            fprintf(stderr, "saphira-llm-eval: --stride needs a context of at least 2\n");
            free(ids);
            sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
            return 1;
        }
        if (stride >= n_ctx) {
            fprintf(stderr,
                "saphira-llm-eval: --stride %d must be less than the context %d.\n"
                "  A stride equal to the window would score a token with no context\n"
                "  of its own, which is not a measurement of anything.\n", stride, n_ctx);
            free(ids);
            sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
            return 1;
        }
        /*
         * Coverage precondition. Window i scores corpus tokens
         * [start_i + n_win - stride, start_i + n_win - 1]. The first window
         * starts at 0, so the first scored token is n_win - stride. For the
         * corpus to be fully covered from token 1 upward, that must be 1:
         *
         *     stride == n_win - 1
         *
         * When the corpus fits in one window (n_win == n_tok) that is the ONLY
         * stride that covers it, because there is no earlier window to have
         * scored the tokens this one skips. Asking for less and getting a
         * perplexity over a silently-scored subset is exactly the failure this
         * tool must not have, so it is refused with the arithmetic shown.
         */
        const int32_t n_win_check = (n_ctx < n_tok) ? n_ctx : n_tok;
        if (stride != n_win_check - 1) {
            fprintf(stderr,
                "saphira-llm-eval: --stride %d cannot cover a %d-token corpus in a "
                "%d-token window.\n"
                "  Each window scores its last `min(stride, len-1)` tokens and the next "
                "window starts `stride` further on, so a gap opens unless the count "
                "equals the stride; and the first window must begin at token 1, which "
                "needs the count to be n_win - 1.\n"
                "  Both hold exactly when stride == n_win - 1, i.e. %d here. That value "
                "scores every token from 1 up with no gap and no overlap, and for a "
                "corpus that fits it is bit-identical to the single-window result.\n"
                "  The upstream reference is looser than this: at stride < n_ctx it "
                "leaves the first n_ctx-stride-1 tokens unscored.\n",
                stride, n_tok, n_win_check, n_win_check - 1);
            free(ids);
            sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
            return 1;
        }
        if (n_tok > max_tokens) {
            fprintf(stderr,
                "saphira-llm-eval: corpus is %d tokens, over the %d limit.\n"
                "  A rolling window removes the need for the whole corpus to fit in\n"
                "  one context, but the token array is still held in memory, and this\n"
                "  tool is meant to stay RAM-resident. Raise --max-tokens or shorten\n"
                "  the corpus.\n", n_tok, max_tokens);
            free(ids);
            sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
            return 1;
        }
    }

    sllm_ctx * c = NULL;
    rc = sllm_ctx_new(m, n_ctx, &c);
    if (rc != SLLM_OK) {
        fprintf(stderr, "saphira-llm-eval: cannot create a %d-token context: %s\n",
                n_ctx, sllm_status_string(rc));
        free(ids);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }
    sllm_ctx_set_pool(c, pool);

    /* One chunk of logits at a time. At 128256 vocabulary a 256-token chunk is
     * 131 MB, and the whole point of reducing immediately is that peak stays
     * here instead of growing with the corpus. */
    const int32_t chunk_max = SLLM_MAX_CHUNK;
    float * logits = (float *) malloc((size_t) chunk_max * (size_t) n_vocab * sizeof(float));
    if (logits == NULL) {
        fprintf(stderr, "saphira-llm-eval: cannot allocate %d x %d logits\n", chunk_max, n_vocab);
        sllm_ctx_free(c); free(ids);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    double nll_sum = 0.0;
    double nll_sum_ref_form = 0.0;
    int64_t scored = 0;
    int64_t correct = 0;
    int64_t windows = 0;

    /*
     * Per-token accounting. Every scored token increments its own counter, and
     * the totals are checked afterwards: no token scored twice, no token
     * scored zero times. A stride bug that duplicated or dropped a token would
     * still produce a plausible-looking perplexity, so the accounting is
     * checked rather than reported for comfort.
     */
    uint8_t * hits = (uint8_t *) calloc((size_t) n_tok, 1);
    if (hits == NULL) {
        fprintf(stderr, "saphira-llm-eval: out of memory\n");
        free(logits); sllm_ctx_free(c); free(ids);
        sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
        return 1;
    }

    const struct timespec t0 = now();

    if (stride > 0) {
        /*
         * ROLLING WINDOW.
         *
         * Semantics, stated so they can be reproduced and so equivalence with
         * the single-window path can be proved rather than asserted.
         *
         *   - Window i begins at corpus position start_i = i * stride.
         *   - Its length is len_i = min(n_win, n_tok - start_i), where
         *     n_win = min(n_ctx, n_tok). The LAST window is therefore
         *     TRUNCATED to the end of the corpus rather than being pulled
         *     back. This is the whole design, and it is why the tiling is
         *     exact.
         *   - Within a window only its LAST `count` tokens are scored, where
         *     count = min(stride, len_i - 1). The earlier tokens were scored by
         *     the previous window and are deliberately not scored again.
         *   - Window positions are 0..len-1 into a CLEARED context. The reset
         *     is what makes a window's predictions depend on that window's own
         *     prefix and nothing before it.
         *   - Token 0 is never scored. It has no context in any window.
         *
         * WHY TRUNCATE AND NOT PULL BACK. A window always covers
         * n_win - stride ... n_win - 1 of its own tokens. If the final window
         * is instead moved earlier to stay full, it OVERLAPS the previous
         * window and leaves a gap after it -- measured here as 120 tokens
         * scored twice and 135 never scored on a 1156-token corpus. Truncating
         * the last window to the corpus end makes the scored ranges a partition
         * of [1, n_tok) with no gap and no overlap, whatever the stride.
         *
         * EQUIVALENCE, the acceptance criterion. With stride == n_win - 1 and a
         * corpus that fits in one window there is a single window, count =
         * n_tok - 1, and it scores corpus tokens 1 .. n_tok-1 -- the same set,
         * fed at the same positions into a freshly cleared context as the
         * single-window path. The sum is therefore BIT-identical. Verified at
         * ctx=386/stride=385 and ctx=512/stride=385 against the stride=0 path.
         *
         * COVERAGE. Windows advance by `stride` and each scores `count`
         * tokens, so there is a gap unless count == stride; and the first window
         * must begin scoring at token 1, which needs count == n_win - 1. Both
         * hold exactly when
         *
         *     stride == n_win - 1
         *
         * which is checked before any work is done. Note the reference is
         * looser than this: it advances by stride and scores the last stride
         * tokens of each window, so with stride < n_ctx it silently leaves
         * 1 .. n_ctx-stride-1 unscored. This tool refuses that configuration
         * instead of reporting a perplexity over a subset.
         */
        const int32_t n_win = (n_ctx < n_tok) ? n_ctx : n_tok;

        for (int32_t start = 0; start < n_tok; start += stride) {
            const int32_t len = (n_win < n_tok - start) ? n_win : (n_tok - start);
            if (len < 2) {
                /* One token has no predecessor to be predicted from. */
                break;
            }
            const int32_t count     = (stride < len - 1) ? stride : (len - 1);
            const int32_t first_row = (len - 1) - count;
            const int32_t last_row  = len - 2;

            sllm_ctx_reset(c);

            for (int32_t base = 0; base < len; ) {
                int32_t k = len - base;
                if (k > chunk_max) { k = chunk_max; }
                rc = sllm_forward_chunk(m, c, ids + start + base, k, base, logits);
                if (rc != SLLM_OK) {
                    fprintf(stderr,
                            "saphira-llm-eval: forward failed in window at %d: %s\n",
                            start + base, sllm_status_string(rc));
                    free(hits); free(logits); sllm_ctx_free(c); free(ids);
                    sllm_tok_free(tok); sllm_model_free(m);
                    sllm_gguf_close(&g); sllm_pool_destroy(pool);
                    return 1;
                }
                for (int32_t j = 0; j < k; ++j) {
                    const int32_t row = base + j;
                    if (row < first_row || row > last_row) { continue; }
                    /* Row `row` is the distribution after consuming window
                     * tokens 0..row, so it predicts window token row+1. */
                    const int32_t target_idx = start + row + 1;
                    const float * lrow = logits + (size_t) j * (size_t) n_vocab;
                    const int32_t target = ids[target_idx];
                    nll_sum      += sllm_eval_token_nll(lrow, n_vocab, target);
                    nll_sum_ref_form += sllm_eval_nll_softmax_form(lrow, n_vocab, target);
                    if (sllm_eval_argmax(lrow, n_vocab) == target) { ++correct; }
                    ++hits[target_idx];
                    ++scored;
                }
                base += k;
            }
            ++windows;
        }
    } else {
        /*
         * SINGLE WINDOW: the whole corpus in one context, fed in consecutive
         * chunks at increasing absolute position so the KV cache spans it and
         * every token is scored against its real prefix.
         *
         * Row i of a chunk beginning at position p is the distribution after
         * consuming tokens[p..p+i], so it predicts tokens[p+i+1]. A chunk of k
         * tokens has k rows predicting tokens p+1 .. p+k.
         *
         * How many are scoreable depends on whether another chunk follows. If
         * it does, the last row's target -- token p+k -- is the first token of
         * the next chunk and is already in `ids`, so all k rows are
         * scoreable. If this is the final chunk its last row would predict
         * token n_tok, which does not exist, so only k-1 rows are.
         *
         * Skipping that row unconditionally was the first version's bug: it
         * dropped one token at every chunk boundary, and the invariant caught
         * it on the first run (386 tokens in, 384 scored).
         */
        for (int32_t p = 0; p < n_tok; ) {
            int32_t k = n_tok - p;
            if (k > chunk_max) { k = chunk_max; }
            const bool is_last_chunk = (p + k >= n_tok);
            const int32_t rows = is_last_chunk ? (k - 1) : k;

            rc = sllm_forward_chunk(m, c, ids + p, k, p, logits);
            if (rc != SLLM_OK) {
                fprintf(stderr, "saphira-llm-eval: forward failed at position %d: %s\n",
                        p, sllm_status_string(rc));
                free(hits); free(logits); sllm_ctx_free(c); free(ids);
                sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
                return 1;
            }

            for (int32_t i = 0; i < rows; ++i) {
                const float * row = logits + (size_t) i * (size_t) n_vocab;
                const int32_t target_idx = p + i + 1;
                const int32_t target = ids[target_idx];
                nll_sum      += sllm_eval_token_nll(row, n_vocab, target);
                nll_sum_ref_form += sllm_eval_nll_softmax_form(row, n_vocab, target);
                if (sllm_eval_argmax(row, n_vocab) == target) { ++correct; }
                ++hits[target_idx];
                ++scored;
            }
            p += k;
        }
        ++windows;
    }

    const struct timespec t1 = now();
    const double secs = (double) (t1.tv_sec - t0.tv_sec) +
                        (double) (t1.tv_nsec - t0.tv_nsec) / 1e9;

    /*
     * TOKEN ACCOUNTING, checked rather than reported for comfort.
     *
     * A stride bug that duplicated or dropped a token would still produce a
     * plausible-looking perplexity, so the counters are verified:
     *
     *   - no token is scored more than once,
     *   - no token is scored zero times,
     *   - token 0 is never scored, because it has no context,
     *   - and the total matches the count.
     *
     * With a stride of n_ctx-1 the scored ranges tile the corpus exactly, so
     * every token from 1 upward is scored once. With a smaller stride there is
     * deliberate overlap in what each WINDOW sees, but the scored SET is still
     * a partition, because each window scores only its own last `stride`
     * tokens. The accounting below is what proves that rather than asserting
     * it, and it is why --stride smaller than n_ctx-1 is allowed at all.
     */
    {
        int64_t twice = 0, never = 0;
        for (int32_t i = 0; i < n_tok; ++i) {
            if (hits[i] > 1) { ++twice; }
            else if (hits[i] == 0) { ++never; }
        }
        if (hits[0] != 0) {
            fprintf(stderr, "saphira-llm-eval: INTERNAL: token 0 was scored; it has "
                            "no context to be predicted from\n");
            free(hits); free(logits); sllm_ctx_free(c); free(ids);
            sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
            return 1;
        }
        if (twice != 0 || never != 1) {
            fprintf(stderr, "saphira-llm-eval: INTERNAL: token accounting wrong: "
                            "%lld scored twice, %lld never scored (expected 0 and 1: "
                            "token 0 only)\n", (long long) twice, (long long) never);
            free(hits); free(logits); sllm_ctx_free(c); free(ids);
            sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
            return 1;
        }
        if (scored != n_tok - 1) {
            fprintf(stderr, "saphira-llm-eval: INTERNAL: scored %lld tokens but the "
                            "corpus has %d; expected %d\n", (long long) scored, n_tok, n_tok - 1);
            free(hits); free(logits); sllm_ctx_free(c); free(ids);
            sllm_tok_free(tok); sllm_model_free(m); sllm_gguf_close(&g); sllm_pool_destroy(pool);
            return 1;
        }
    }

    const double mean_nll = nll_sum / (double) scored;
    const double ppl      = exp(mean_nll);
    const double bpt      = mean_nll / log(2.0);
    const double mean_nll_ref = nll_sum_ref_form / (double) scored;
    const double acc      = (double) correct / (double) scored;

    if (json) {
        printf("{\n");
        printf("  \"model\": \"%s\",\n", model_path);
        printf("  \"corpus\": \"%s\",\n", data_path);
        printf("  \"corpus_tokens\": %d,\n", n_tok);
        printf("  \"scored_tokens\": %lld,\n", (long long) scored);
        printf("  \"context\": %d,\n", n_ctx);
        printf("  \"windows\": %lld,\n", (long long) windows);
        printf("  \"stride\": %d,\n", stride);
        printf("  \"threads\": %d,\n", n_threads);
        printf("  \"nll_sum\": %.17g,\n", nll_sum);
        printf("  \"mean_nll\": %.17g,\n", mean_nll);
        printf("  \"perplexity\": %.6f,\n", ppl);
        printf("  \"bits_per_token\": %.6f,\n", bpt);
        printf("  \"top1_accuracy\": %.6f,\n", acc);
        printf("  \"mean_nll_reference_form\": %.10f,\n", mean_nll_ref);
        printf("  \"seconds\": %.4f\n", secs);
        printf("}\n");
    } else {
        printf("model              %s\n", model_path);
        printf("corpus             %s\n", data_path);
        printf("corpus_tokens      %d\n", n_tok);
        printf("scored_tokens      %lld\n", (long long) scored);
        printf("context            %d\n", n_ctx);
        printf("windows            %lld\n", (long long) windows);
        printf("stride             %d\n", stride);
        printf("threads            %d\n", n_threads);
        printf("nll_sum            %.10f\n", nll_sum);
        printf("mean_nll           %.10f\n", mean_nll);
        printf("bits_per_token     %.6f\n", bpt);
        printf("perplexity         %.6f\n", ppl);
        printf("top1_accuracy      %.6f\n", acc);
        printf("reference_form     %.10f  (softmax-then-log, for comparison)\n", mean_nll_ref);
        printf("seconds            %.4f\n", secs);
        printf("\n");
        printf("Perplexity %s. Lower is better.\n", ppl < 1e9 ? "is a measurement, not a gate" : "OVERFLOWED");
        printf("Comparable only against the same corpus, tokenizer and context.\n");
        printf("Also only against the same BINARY: the forward pass contracts\n"
               "multiply-add into fused operations, and gcc and clang disagree by\n"
               "about 0.3%% on this figure (145.0836 vs 145.5773 on the in-tree\n"
               "fixture). See docs/EVALUATION.md. Compare like with like.\n");
    }

    free(logits);
    sllm_ctx_free(c);
    free(ids);
    sllm_tok_free(tok);
    sllm_model_free(m);
    sllm_gguf_close(&g);
    sllm_pool_destroy(pool);
    return 0;
}
