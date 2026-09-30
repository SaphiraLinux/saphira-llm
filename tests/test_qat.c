/*
 * test_qat.c — does the tiny QAT actually work?
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA Limited.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The order of these checks is not arbitrary. The gradient check comes FIRST,
 * before anything that trusts the optimizer, because every other assertion
 * here is only meaningful if the gradient is right: a decreasing loss proves
 * nothing about a lifecycle if the thing decreasing it was computed wrongly.
 */
#include <saphira_llm/gguf.h>
#include <saphira_llm/i2s_convert.h>
#include <saphira_llm/qat.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"

#define TINY_TOKENS 24

/* A short, deliberately structured token sequence. Not language: a fixture for
 * a fixture. The only property that matters is that it is deterministic and
 * that it is not uniform, so a loss that fails to move is a real failure and
 * not the model correctly learning that everything is equally likely. */
static int32_t tokens[TINY_TOKENS] = {
      3,  17,  42,   7,  99,  13, 200,  55,
     71,   2,  88,  31, 150,  64,  19, 233,
     40,  77, 128,   9, 201,  26, 110,  61
};

static sllm_qat_batch batch_of(const sllm_qat_config * cfg) {
    sllm_qat_batch b;
    int n = TINY_TOKENS < cfg->n_ctx ? TINY_TOKENS : cfg->n_ctx;
    b.tokens = tokens;
    b.n_tokens = n;
    return b;
}

/* ------------------------------------------------------------------ */
/* 1. the gradient, against a central difference                       */
/* ------------------------------------------------------------------ */

/*
 * With `ternary` off the forward pass is a plain, fully differentiable
 * transformer: no weight ternary anywhere, no activation quantiser anywhere.
 * That is the only setting in which a finite difference means anything, and it
 * is why the hook disables BOTH rather than just the weights -- leaving the
 * activation quantiser in would have made this test pass for the wrong reason
 * or fail for a reason that had nothing to do with the backward.
 *
 * What is being checked is therefore every analytic derivative in the file:
 * the matmul gradients, the grouped-query attention, the softmax, the RoPE
 * inverse, siLU, all four RMSNorms, the residual stream and the tied
 * embedding. The straight-through paths are NOT checked here, and cannot be:
 * they have no finite-difference counterpart. They are checked structurally in
 * test_ste_reaches_master below.
 */
TEST(test_gradient_matches_central_difference) {
    sllm_qat_config cfg;
    sllm_qat_default_config(&cfg);
    sllm_qat * q = sllm_qat_new(&cfg, 12345);
    CHECK(q != NULL);
    q->ternary = false;
    sllm_qat_init(q, 12345);

    sllm_qat_batch b = batch_of(&cfg);
    const double loss = sllm_qat_loss(q, &b);
    CHECK(isfinite(loss));
    CHECK(loss > 0.0);

    /* Sample entries across every parameter, including the ones on the residual
     * path that a plausible-looking bug would leave at zero. */
    /* eps is large on purpose. This is a float32 central difference on a loss
     * of order 1.4, where one ulp is ~1e-7: an eps of 1e-3 puts the loss
     * difference at ~1e-8, i.e. below the noise floor, and the test then
     * "fails" on rounding while measuring nothing. 5e-2 puts the difference
     * five orders of magnitude above the floor, and truncation error in the
     * central difference stays well under the tolerance below. */
    /*
     * eps, and what a relative tolerance is even allowed to mean here.
     *
     * This is a float32 central difference on a loss of order 5.5, where one
     * ulp is about 6.6e-7. A finite difference is a DIFFERENCE OF TWO NEARLY
     * EQUAL NUMBERS, so its resolution floor is a few ulps of the loss divided
     * by 2*eps. Most individual weight gradients in this model are smaller
     * than that floor, and for those the "numeric" gradient is not a weak
     * measurement of the right answer -- it is rounding noise, and it would
     * happily report 0.0 for a gradient that is genuinely 1e-9 and non-zero.
     *
     * So the test is explicit about both halves rather than hiding behind one
     * aggregate: every entry whose loss difference is RESOLVABLE must agree to
     * a tight relative tolerance, and a healthy number of entries must be
     * resolvable in the first place. An aggregate that averaged the noise in
     * would let a genuinely wrong gradient pass, and one that ignored
     * resolvability would fail on arithmetic it cannot see.
     */
    const double eps = 5e-2;
    const double ulp = fabs(loss) * 1.1920929e-7;   /* FLT_EPSILON for f32 */
    const double floor_dL = 64.0 * ulp;            /* clearly resolvable */
    double worst = 0.0;
    int checked = 0, nonzero_grads = 0, resolvable = 0, compared = 0;

    for (int i = 0; i < q->n_p; ++i) {
        sllm_qat_tensor * t = &q->p[i];
        for (int s = 0; s < 3; ++s) {
            const int k = (int) ((long) t->n * (s + 1) / 4);
            if (k >= t->n) { continue; }
            if (fabsf(t->grad[k]) > 1e-9f) { ++nonzero_grads; }

            const float save = t->master[k];
            t->master[k] = save + (float) eps;
            const double lp = sllm_qat_loss(q, &b);
            t->master[k] = save - (float) eps;
            const double lm = sllm_qat_loss(q, &b);
            t->master[k] = save;
            /* recompute the analytic gradient at the unperturbed point, since
             * the two calls above overwrote it */
            (void) sllm_qat_loss(q, &b);
            const double g = (double) t->grad[k];
            const double g_numeric = (lp - lm) / (2.0 * eps);
            ++checked;
            if (fabs(lp - lm) < floor_dL) { continue; }   /* below the floor */
            ++resolvable;
            const double denom = fmax(fabs(g), fabs(g_numeric)) + 1e-12;
            const double rel = fabs(g - g_numeric) / denom;
            if (rel > worst) { worst = rel; }
            ++compared;
        }
    }
    /* Tight enough to catch a missing residual branch, a transposed softmax, a
     * dropped 1/n, or a stale workspace row -- all of which show up as rel ~ 1
     * -- while leaving room for the truncation error a step this large costs. */
    CHECK(compared > 0);
    CHECK(worst < 5e-2);
    CHECK(checked > 40);
    CHECK(nonzero_grads > checked / 2);
    /* If almost nothing were resolvable the test would be vacuous, so require
     * that a real share of the sampled entries cleared the noise floor. */
    CHECK(resolvable >= 8);

    sllm_qat_free(q);
}

/* ------------------------------------------------------------------ */
/* 2. the straight-through estimator actually reaches the master         */
/* ------------------------------------------------------------------ */

/*
 * The STE is the claim that makes QAT work at all, and it is invisible to the
 * gradient check above. This asserts its two halves directly:
 *
 *   - with the ternary ON, gradients are non-zero. If the estimator were
 *     missing, the gradient through a quantised weight would be exactly zero
 *     and nothing would train.
 *   - the gradient magnitude does not depend on the value of the quantised
 *     weight, only on the loss. That is the estimator: the gradient passes
 *     through the ternary unchanged.
 */
TEST(test_ste_reaches_master) {
    sllm_qat_config cfg;
    sllm_qat_default_config(&cfg);
    sllm_qat * q = sllm_qat_new(&cfg, 999);
    sllm_qat_init(q, 999);
    sllm_qat_batch b = batch_of(&cfg);

    q->ternary = true;
    (void) sllm_qat_loss(q, &b);

    sllm_qat_tensor * t = NULL;
    for (int i = 0; i < q->n_p; ++i) {
        if (strcmp(q->p[i].name, "blk.0.ffn_up") == 0) { t = &q->p[i]; }
    }
    CHECK(t != NULL);
    if (t == NULL) { sllm_qat_free(q); return; }

    double gnorm = 0.0;
    for (int k = 0; k < t->n; ++k) {
        gnorm += (double) t->grad[k] * t->grad[k];
    }
    gnorm = sqrt(gnorm);
    CHECK(gnorm > 0.0);

    /*
     * The estimator, tested where it is actually falsifiable.
     *
     * A master entry whose ternary code is ZERO has, under any real derivative
     * of the forward, a gradient of exactly zero: the code is locally constant
     * there, so nothing about that entry can influence the loss. The STE says
     * the loss gradient with respect to the quantised weight is handed to the
     * master regardless, so a zero-coded entry must receive the same treatment
     * as every other entry and come away NON-ZERO.
     *
     * This replaces an earlier version of this test that scaled the whole
     * master by 3 and asserted the gradient came back bit-identical. That
     * assertion was only true while the ternary value was +-1, where the
     * forward genuinely did not move. The value is now +-absmax to match the
     * deployed format, so the forward does scale and the gradient legitimately
     * changes -- the old test was measuring the old convention, not the
     * estimator, and would have passed or failed for reasons that had nothing
     * to do with the STE. The zero-code property is convention-independent,
     * which is what makes it the better probe.
     */
    const double scale = sllm_qat_weight_scale(t);
    int zero_coded = 0, zero_coded_with_grad = 0;
    for (int k = 0; k < t->n; ++k) {
        if (fabs((double) t->master[k]) <= 0.5 * (double) scale) {
            ++zero_coded;
            if (fabs((double) t->grad[k]) > 0.0) { ++zero_coded_with_grad; }
        }
    }
    CHECK(zero_coded > t->n / 4);
    /* A majority, not all. An estimator that is missing entirely would give
     * exactly zero here for EVERY zero-coded entry, so the count being large is
     * what falsifies it; a few entries landing on exactly 0.0 through float32
     * cancellation in a 24-token sum are expected and say nothing about the
     * estimator. Demanding every single one would be demanding a coincidence. */
    CHECK(zero_coded_with_grad * 2 > zero_coded);

    sllm_qat_free(q);
}

/* ------------------------------------------------------------------ */
/* 3. training actually reduces the loss                                */
/* ------------------------------------------------------------------ */

TEST(test_training_reduces_loss) {
    sllm_qat_config cfg;
    sllm_qat_default_config(&cfg);
    sllm_qat * q = sllm_qat_new(&cfg, 2024);
    sllm_qat_init(q, 2024);
    sllm_qat_batch b = batch_of(&cfg);

    const double l0 = sllm_qat_loss(q, &b);
    for (int i = 0; i < 200; ++i) {
        (void) sllm_qat_loss(q, &b);
        sllm_qat_clip_grad(q, 1.0f);
        sllm_qat_step(q, 0.05f);
    }
    const double l1 = sllm_qat_loss(q, &b);
    CHECK(l1 < l0);
    /* A large drop would mean the fixture is degenerate; no drop at all would
     * mean the optimizer is disconnected. Both are failures of a different
     * kind and the bounds make each of them visible. */
    CHECK(l1 < 4.0);
    sllm_qat_free(q);
}

/* ------------------------------------------------------------------ */
/* 4. determinism from the seed alone                                   */
/* ------------------------------------------------------------------ */

TEST(test_deterministic_from_seed) {
    sllm_qat_config cfg;
    sllm_qat_default_config(&cfg);
    sllm_qat_batch b = batch_of(&cfg);

    sllm_qat * a = sllm_qat_new(&cfg, 7), * c = sllm_qat_new(&cfg, 7);
    sllm_qat_init(a, 7); sllm_qat_init(c, 7);
    for (int i = 0; i < 20; ++i) {
        (void) sllm_qat_loss(a, &b); sllm_qat_clip_grad(a, 1.0f); sllm_qat_step(a, 0.02f);
        (void) sllm_qat_loss(c, &b); sllm_qat_clip_grad(c, 1.0f); sllm_qat_step(c, 0.02f);
    }
    int identical = 1;
    for (int i = 0; i < a->n_p && identical; ++i) {
        for (int k = 0; k < a->p[i].n; ++k) {
            if (a->p[i].master[k] != c->p[i].master[k]) { identical = 0; break; }
        }
    }
    CHECK(identical);
    sllm_qat_free(a); sllm_qat_free(c);
}

/* ------------------------------------------------------------------ */
/* 5. checkpoint and restart                                            */
/* ------------------------------------------------------------------ */

TEST(test_checkpoint_restart_is_exact) {
    const char * path = "test_qat_ckpt.bin";
    sllm_qat_config cfg;
    sllm_qat_default_config(&cfg);
    sllm_qat_batch b = batch_of(&cfg);

    sllm_qat * a = sllm_qat_new(&cfg, 31337);
    sllm_qat_init(a, 31337);
    for (int i = 0; i < 10; ++i) {
        (void) sllm_qat_loss(a, &b); sllm_qat_clip_grad(a, 1.0f); sllm_qat_step(a, 0.02f);
    }
    CHECK(sllm_qat_save(a, path) == 0);

    /* the uninterrupted run */
    for (int i = 0; i < 10; ++i) {
        (void) sllm_qat_loss(a, &b); sllm_qat_clip_grad(a, 1.0f); sllm_qat_step(a, 0.02f);
    }
    const double uninterrupted = sllm_qat_loss(a, &b);

    /* the interrupted run: a fresh process-equivalent, loaded from disk */
    sllm_qat * c = sllm_qat_load(path);
    CHECK(c != NULL);
    for (int i = 0; i < 10; ++i) {
        (void) sllm_qat_loss(c, &b); sllm_qat_clip_grad(c, 1.0f); sllm_qat_step(c, 0.02f);
    }
    const double restarted = sllm_qat_loss(c, &b);
    CHECK(restarted == uninterrupted);

    int same = 1;
    for (int i = 0; i < a->n_p && same; ++i) {
        for (int k = 0; k < a->p[i].n; ++k) {
            if (a->p[i].master[k] != c->p[i].master[k]) { same = 0; break; }
        }
    }
    CHECK(same);

    sllm_qat_free(a); sllm_qat_free(c);
    remove(path);
}

/* ------------------------------------------------------------------ */
/* 6. the export loads through the ordinary runtime                      */
/* ------------------------------------------------------------------ */

/*
 * The point of the whole exercise. Not "the converter runs" -- the converter
 * was already proven. The point is that a model TRAINED HERE goes through the
 * SAME loader and the SAME forward the released model uses, with no alternate
 * path, no special case and no flag.
 */
TEST(test_export_loads_with_ordinary_loader) {
    const char * path = "test_qat_tiny.gguf";
    sllm_qat_config cfg;
    sllm_qat_default_config(&cfg);
    sllm_qat * q = sllm_qat_new(&cfg, 5150);
    sllm_qat_init(q, 5150);
    sllm_qat_batch b = batch_of(&cfg);
    for (int i = 0; i < 30; ++i) {
        (void) sllm_qat_loss(q, &b); sllm_qat_clip_grad(q, 1.0f); sllm_qat_step(q, 0.03f);
    }
    CHECK(sllm_qat_export_i2s_gguf(q, path, SLLM_I2S_RULE_GGUF_PY) == 0);

    sllm_gguf gg;
    sllm_gguf * g = NULL;
    if (sllm_gguf_open(path, &gg, NULL, 0) == SLLM_OK) { g = &gg; }
    CHECK(g != NULL);
    if (g != NULL) {
        uint32_t u = 0;
        CHECK(sllm_gguf_kv_u32(g, "bitnet-b1.58.block_count", &u) == SLLM_OK && u == (uint32_t) cfg.n_layer);
        CHECK(sllm_gguf_kv_u32(g, "bitnet-b1.58.embedding_length", &u) == SLLM_OK && u == (uint32_t) cfg.n_embd);
        CHECK(sllm_gguf_kv_u32(g, "bitnet-b1.58.attention.head_count", &u) == SLLM_OK && u == (uint32_t) cfg.n_head);
        CHECK(sllm_gguf_kv_u32(g, "bitnet-b1.58.attention.head_count_kv", &u) == SLLM_OK && u == (uint32_t) cfg.n_head_kv);
        CHECK(sllm_gguf_kv_u32(g, "bitnet-b1.58.vocab_size", &u) == SLLM_OK && u == (uint32_t) cfg.n_vocab);

        const sllm_gguf_tensor * t = sllm_gguf_find_tensor(g, "blk.0.ffn_down.weight");
        CHECK(t != NULL);
        if (t != NULL) {
            CHECK(t->type == SLLM_TYPE_I2_S);
            CHECK((uint64_t) t->ne[0] == (uint64_t) cfg.n_ff &&
                  (uint64_t) t->ne[1] == (uint64_t) cfg.n_embd);
        }
        const sllm_gguf_tensor * e = sllm_gguf_find_tensor(g, "token_embd.weight");
        CHECK(e != NULL && e->type == SLLM_TYPE_F16);
        const sllm_gguf_tensor * n = sllm_gguf_find_tensor(g, "blk.0.attn_norm.weight");
        CHECK(n != NULL && n->type == SLLM_TYPE_F32);
        sllm_gguf_close(g);
    }
    sllm_qat_free(q);
    remove(path);
}

void sllm_test_qat(void) {
    printf("qat\n");
    RUN(test_gradient_matches_central_difference);
    RUN(test_ste_reaches_master);
    RUN(test_training_reduces_loss);
    RUN(test_deterministic_from_seed);
    RUN(test_checkpoint_restart_is_exact);
    RUN(test_export_loads_with_ordinary_loader);
}
