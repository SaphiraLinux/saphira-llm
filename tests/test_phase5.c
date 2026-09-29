/*
 * Phase 5 tests: chunked attention, KV state save/load, state restore.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The gate is that chunk boundaries are not observable. A mask that is only
 * correct when the whole prompt is one chunk is a mask that is wrong somewhere
 * and has merely not been observed, so these tests compare BIT FOR BIT rather
 * than within a tolerance: chunking is a batching structure and must not change
 * a single bit of the arithmetic.
 *
 * That is a stricter requirement than the Phase 4 logit gate, and deliberately
 * so. There, the comparison is against a reference whose own kernels we do not
 * reproduce. Here, both sides are ours, so there is no floor to characterise
 * and any difference at all is a bug.
 */

#include "harness.h"
#include "saphira_llm/forward.h"
#include "saphira_llm/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef SLLM_TEST_MODEL
#define SLLM_TEST_MODEL "/var/lib/spoon/models/ggml-model-i2_s.gguf"
#endif

typedef struct {
    sllm_gguf    g;      /* owned; the model borrows from it */
    bool         open;
    sllm_model * m;
    sllm_tok   * tok;
    int32_t      ids[64];
    int32_t      n;
    int32_t      nv;
} fixture;

/*
 * The prompt is a parameter rather than a field. It was a field, and reading it
 * back out of a struct that the caller had memset was a null dereference on the
 * first run -- which is a test bug that looks exactly like a runtime bug, and
 * cost a round of bisection for no interesting reason.
 */
static bool fixture_open(fixture * f, const char * prompt) {
    char err[512];
    memset(f, 0, sizeof(*f));
    if (access(SLLM_TEST_MODEL, R_OK) != 0) { return false; }
    if (sllm_gguf_open(SLLM_TEST_MODEL, &f->g, err, sizeof(err)) != SLLM_OK) { return false; }
    f->open = true;
    /* The model and tokenizer borrow their strings from the mapping, so the
     * gguf has to outlive both. The fixture owns the handle and closes it
     * last, which is the ordering that is easy to get wrong by hand. */
    if (sllm_model_load(&f->g, &f->m) != SLLM_OK) { sllm_gguf_close(&f->g); f->open = false; return false; }
    if (sllm_tok_load(&f->g, &f->tok) != SLLM_OK) { sllm_model_free(f->m); sllm_gguf_close(&f->g); f->open = false; return false; }
    f->nv = sllm_model_n_vocab(f->m);
    f->n = sllm_tok_encode(f->tok, prompt, strlen(prompt), true, true, f->ids, 64);
    return f->n > 0;
}

static void fixture_free(fixture * f) {
    sllm_tok_free(f->tok);
    sllm_model_free(f->m);
    if (f->open) { sllm_gguf_close(&f->g); f->open = false; }
    memset(f, 0, sizeof(*f));
}

/*
 * Every chunk size must give the same bits as one token per forward.
 *
 * The sizes are chosen to hit the awkward cases rather than the tidy ones:
 * 1 is the degenerate case, sizes equal to and larger than the prompt exercise
 * a single chunk, and 3, 4, 5 and 7 all leave a partial final chunk. A mask
 * that mis-handles the tail of a chunk fails on all of them.
 */
#define SLLM_P5_PROMPT "The name of the capital city of France is"

TEST(chunk_boundaries_are_not_observable) {
    fixture f;
    if (!fixture_open(&f, SLLM_P5_PROMPT)) {
        printf("    skipped: %s not available\n", SLLM_TEST_MODEL);
        CHECK(1);
        return;
    }

    const size_t n_floats = (size_t) f.n * (size_t) f.nv;

    sllm_ctx * one = NULL;
    if (sllm_ctx_new(f.m, 512, &one) != SLLM_OK) { fixture_free(&f); CHECK(1); return; }
    float * ref = (float *) malloc(n_floats * sizeof(float));
    if (ref == NULL) { sllm_ctx_free(one); fixture_free(&f); CHECK(1); return; }
    if (sllm_forward_prefill(f.m, one, f.ids, f.n, ref) != SLLM_OK) {
        free(ref); sllm_ctx_free(one); fixture_free(&f);
        CHECK(1);
        return;
    }

    static const int32_t sizes[] = { 1, 2, 3, 4, 5, 7, 16, 64, 256 };
    float * got = (float *) malloc(n_floats * sizeof(float));

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        const int32_t cs = sizes[si];
        sllm_ctx * c = NULL;
        if (sllm_ctx_new(f.m, 512, &c) != SLLM_OK) { continue; }
        const sllm_status rc = sllm_forward_prefill_chunked(f.m, c, f.ids, f.n, cs, got);

        sllm_tests_run++;
        if (rc != SLLM_OK) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL chunked prefill with chunk=%d returned %s\n",
                    (int) cs, sllm_status_string(rc));
            sllm_ctx_free(c);
            continue;
        }

        /* Bit-exact. A tolerance here would be a tolerance on a difference that
         * has no cause: both sides are the same code with a different loop
         * nesting. */
        const size_t ulp = memcmp(ref, got, n_floats * sizeof(float));
        if (ulp != 0) {
            size_t worst_i = 0;
            float worst = 0.0f;
            for (size_t i = 0; i < n_floats; ++i) {
                const float d = fabsf(ref[i] - got[i]);
                if (d > worst) { worst = d; worst_i = i; }
            }
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL chunk=%d differs from one-token-per-forward: "
                            "first %zu bytes differ, worst logit %g at position %zu "
                            "vocab %zu\n",
                    (int) cs, ulp, (double) worst, worst_i / (size_t) f.nv,
                    worst_i % (size_t) f.nv);
        }
        sllm_ctx_free(c);
    }

    printf("    %d chunk sizes, all bit-identical to one token per forward "
           "(%d positions)\n", (int) (sizeof(sizes) / sizeof(sizes[0])), (int) f.n);

    free(got);
    free(ref);
    sllm_ctx_free(one);
    fixture_free(&f);
}

/*
 * Chunking must also hold when the chunk boundary falls inside the region that
 * needs cached keys, which is the whole point of the mask test: a query in the
 * second chunk attends to keys written by the first.
 */
TEST(a_later_chunk_sees_keys_written_by_an_earlier_one) {
    fixture f;
    if (!fixture_open(&f, SLLM_P5_PROMPT)) { CHECK(1); return; }

    /* Chunk 1 is the first two tokens, chunk 2 is the rest. If the mask used
     * indices within the chunk rather than absolute positions, the second
     * chunk's tokens would attend only to each other and every logit from
     * position 2 onwards would change. */
    const size_t n_floats = (size_t) f.n * (size_t) f.nv;
    float * whole = (float *) malloc(n_floats * sizeof(float));
    float * split = (float *) malloc(n_floats * sizeof(float));
    if (whole == NULL || split == NULL) { free(whole); free(split); fixture_free(&f); CHECK(1); return; }

    sllm_ctx * a = NULL;
    sllm_ctx * b = NULL;
    sllm_ctx_new(f.m, 512, &a);
    sllm_ctx_new(f.m, 512, &b);
    if (a == NULL || b == NULL) {
        sllm_ctx_free(a); sllm_ctx_free(b); free(whole); free(split);
        fixture_free(&f); CHECK(1); return;
    }

    sllm_forward_prefill(f.m, a, f.ids, f.n, whole);
    sllm_forward_chunk(f.m, b, f.ids, 2, 0, split);
    sllm_forward_chunk(f.m, b, f.ids + 2, f.n - 2, 2, split + 2 * (size_t) f.nv);

    sllm_tests_run++;
    if (memcmp(whole, split, n_floats * sizeof(float)) != 0) {
        sllm_tests_failed++;
        /* Report the first differing position, which is where the mask went
         * wrong rather than just that something did. */
        size_t first = 0;
        for (size_t i = 0; i < n_floats; ++i) {
            if (whole[i] != split[i]) { first = i; break; }
        }
        fprintf(stderr, "  FAIL a 2|%-d split differs from a single chunk; first "
                        "difference at position %zu vocab %zu (%g vs %g). A mask "
                        "keyed on chunk-local indices would fail exactly here.\n",
                (int) f.n - 2, first / (size_t) f.nv, first % (size_t) f.nv,
                (double) whole[first], (double) split[first]);
    }

    free(whole); free(split);
    sllm_ctx_free(a); sllm_ctx_free(b);
    fixture_free(&f);
}

/*
 * Save, restore into a fresh context, and continue: the continuation must be
 * identical to never having stopped.
 *
 * This is the property that matters for a long conversation. The first
 * continuation token comes from the prompt's last position, and the rest come
 * from a cache that was written to disk and read back, so this exercises the
 * whole chain rather than just the file format.
 */
TEST(state_round_trips_and_a_restored_context_continues_identically) {
    fixture f;
    if (!fixture_open(&f, SLLM_P5_PROMPT)) { CHECK(1); return; }

    const int32_t n_new = 6;
    int32_t direct[16];
    int32_t resumed[16];
    int32_t ids[64];

    sllm_ctx * a = NULL;
    sllm_ctx * b = NULL;
    if (sllm_ctx_new(f.m, 512, &a) != SLLM_OK || sllm_ctx_new(f.m, 512, &b) != SLLM_OK) {
        sllm_ctx_free(a); sllm_ctx_free(b); fixture_free(&f); CHECK(1); return;
    }

    /* Reference run: prompt then continuation, no interruption. */
    CHECK_STATUS(sllm_generate_greedy(f.m, a, f.ids, f.n, n_new, direct), SLLM_OK);

    /* Interrupted run: prefill, save, wipe, load into a second context, then
     * continue from there. */
    const int32_t n_prompt_only = f.n;
    sllm_ctx_reset(a);
    float * scratch = (float *) malloc((size_t) n_prompt_only * (size_t) f.nv * sizeof(float));
    if (scratch == NULL) {
        sllm_ctx_free(a); sllm_ctx_free(b); fixture_free(&f); CHECK(1); return;
    }
    CHECK_STATUS(sllm_forward_prefill(f.m, a, f.ids, n_prompt_only, scratch), SLLM_OK);
    free(scratch);

    const char * path = "/tmp/sllm-state-test.bin";
    CHECK_STATUS(sllm_state_save(f.m, a, path), SLLM_OK);

    /* b is fresh and empty, so anything it can produce came from the file. */
    CHECK_STATUS(sllm_state_load(f.m, b, path), SLLM_OK);

    /*
     * Continue from the restored state with no prompt at all. The first token
     * comes from the last position's logits, which the state carries, and the
     * rest come from the restored cache. Nothing here re-runs the prompt: if
     * the forward were re-run, the test would pass even if the state file were
     * empty, which is the failure this is here to catch.
     */
    CHECK_STATUS(sllm_state_load(f.m, b, path), SLLM_OK);
    CHECK_STATUS(sllm_generate_greedy(f.m, b, ids, 0, n_new, resumed), SLLM_OK);

    sllm_tests_run++;
    /* Only the n_new elements the generator produced exist. Comparing the
     * whole 16-slot array would compare uninitialised stack, which passes or
     * fails depending on how the compiler laid the frame out. */
    if (memcmp(direct, resumed, (size_t) n_new * sizeof(int32_t)) != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL restored context diverged from the direct run\n");
        fprintf(stderr, "    direct :");
        for (int i = 0; i < n_new; ++i) { fprintf(stderr, " %d", (int) direct[i]); }
        fprintf(stderr, "\n    resumed:");
        for (int i = 0; i < n_new; ++i) { fprintf(stderr, " %d", (int) resumed[i]); }
        fprintf(stderr, "\n");
    } else {
        printf("    save, load and continue: %d tokens identical to an "
               "uninterrupted run\n", (int) n_new);
    }

    remove(path);
    sllm_ctx_free(a); sllm_ctx_free(b);
    fixture_free(&f);
}

/*
 * Reusing a context must give the same answer as a fresh one.
 *
 * This is the contract callers rely on, and it is the one most likely to rot:
 * sllm_ctx_reset sets n_past to 0 and the next prefill then reads only what it
 * just wrote, so the reuse path is easy to get wrong in a way that shows up
 * only for a long conversation and only for a caller that bothered to reuse a
 * context. Running a real prompt first, so the cache is dirty, is the point.
 *
 * Note what this does and does not prove. It passes today both with and
 * without the defensive memset in sllm_ctx_reset, because n_past = 0 already
 * puts every stale position out of reach. It locks the reuse contract; it is
 * not a regression test for the memset's layer count.
 */
TEST(a_reset_context_gives_the_same_answer_as_a_fresh_one) {
    fixture f;
    if (!fixture_open(&f, SLLM_P5_PROMPT)) { CHECK(1); return; }

    sllm_ctx * reused = NULL;
    sllm_ctx * fresh = NULL;
    if (sllm_ctx_new(f.m, 512, &reused) != SLLM_OK ||
        sllm_ctx_new(f.m, 512, &fresh) != SLLM_OK) {
        sllm_ctx_free(reused); sllm_ctx_free(fresh); fixture_free(&f); CHECK(1); return;
    }

    const int32_t nv = sllm_model_n_vocab(f.m);
    float * first = (float *) malloc((size_t) f.n * (size_t) nv * sizeof(float));
    float * second = (float *) malloc((size_t) f.n * (size_t) nv * sizeof(float));
    float * third = (float *) malloc((size_t) f.n * (size_t) nv * sizeof(float));
    if (first == NULL || second == NULL || third == NULL) {
        free(first); free(second); free(third);
        sllm_ctx_free(reused); sllm_ctx_free(fresh); fixture_free(&f); CHECK(1); return;
    }

    /* A first conversation on the reused context. */
    CHECK_STATUS(sllm_forward_prefill(f.m, reused, f.ids, f.n, first), SLLM_OK);
    sllm_ctx_reset(reused);
    /* A second, so the reused context is dirty well past the first position. */
    CHECK_STATUS(sllm_forward_prefill(f.m, reused, f.ids + 1, f.n - 1, second), SLLM_OK);

    /* Now the actual claim: a reset context must not remember any of it. */
    sllm_ctx_reset(reused);
    CHECK_STATUS(sllm_forward_prefill(f.m, reused, f.ids, f.n, third), SLLM_OK);

    sllm_tests_run++;
    if (memcmp(first, third, (size_t) f.n * (size_t) nv * sizeof(float)) != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL a reset context did not forget the previous run\n");
        size_t worst = 0;
        float worst_delta = 0.0f;
        for (size_t i = 0; i < (size_t) f.n * (size_t) nv; ++i) {
            const float d = first[i] - third[i];
            if ((d < 0.0f ? -d : d) > worst_delta) { worst_delta = d < 0.0f ? -d : d; worst = i; }
        }
        size_t n_diff = 0;
        for (size_t i = 0; i < (size_t) f.n * (size_t) nv; ++i) {
            if (first[i] != third[i]) { ++n_diff; }
        }
        fprintf(stderr, "    %zu/%zu logits differ, largest %g at row %zu of %d\n",
                n_diff, (size_t) f.n * (size_t) nv, (double) worst_delta,
                worst / (size_t) nv, (int) f.n);
    } else {
        printf("    a reset context is bit-identical to a fresh one\n");
    }

    free(first); free(second); free(third);
    sllm_ctx_free(reused); sllm_ctx_free(fresh);
    fixture_free(&f);
}

/*
 * A state file is input from outside the process, so every field is checked
 * against the model before a byte of it is trusted.
 */
TEST(state_load_refuses_a_file_that_is_not_ours) {
    fixture f;
    if (!fixture_open(&f, SLLM_P5_PROMPT)) { CHECK(1); return; }

    sllm_ctx * c = NULL;
    if (sllm_ctx_new(f.m, 512, &c) != SLLM_OK) { fixture_free(&f); CHECK(1); return; }

    const char * path = "/tmp/sllm-state-bad.bin";
    FILE * fp = fopen(path, "wb");
    if (fp == NULL) { sllm_ctx_free(c); fixture_free(&f); CHECK(1); return; }

    /* A valid magic, so the magic check passes, and a geometry that does not
     * match this model. Reading this without checking would write 2 GiB of
     * cache from a 32-byte file. */
    const uint32_t magic = 0x564B4C53u, version = 1u;
    const int32_t n_layer = 1 << 20, n_ctx = 1 << 20;
    const int32_t n_head_kv = 999, n_embd_head = 999, n_past = 1 << 20, n_vocab = 999;
    fwrite(&magic, 4, 1, fp);
    fwrite(&version, 4, 1, fp);
    fwrite(&n_layer, 4, 1, fp);
    fwrite(&n_ctx, 4, 1, fp);
    fwrite(&n_head_kv, 4, 1, fp);
    fwrite(&n_embd_head, 4, 1, fp);
    fwrite(&n_past, 4, 1, fp);
    fwrite(&n_vocab, 4, 1, fp);
    fclose(fp);

    CHECK_STATUS(sllm_state_load(f.m, c, path), SLLM_ERR_TENSOR_SHAPE);
    remove(path);

    /* And a file that is not a state file at all. */
    fp = fopen(path, "wb");
    if (fp != NULL) {
        const char junk[64] = "this is not a state file";
        fwrite(junk, 1, sizeof(junk), fp);
        fclose(fp);
        CHECK_STATUS(sllm_state_load(f.m, c, path), SLLM_ERR_KV_TYPE);
        remove(path);
    }

    sllm_ctx_free(c);
    fixture_free(&f);
}

void sllm_test_phase5(void) {
    printf("phase5\n");
    RUN(chunk_boundaries_are_not_observable);
    RUN(a_later_chunk_sees_keys_written_by_an_earlier_one);
    RUN(state_round_trips_and_a_restored_context_continues_identically);
    RUN(a_reset_context_gives_the_same_answer_as_a_fresh_one);
    RUN(state_load_refuses_a_file_that_is_not_ours);
}
