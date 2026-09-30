/*
 * qat.h — tiny-model quantisation-aware training, for proving the lifecycle.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA Limited.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * SCOPE, and it matters what this is not.
 *
 * This exists to prove a PIPELINE: evaluate, train something tiny, checkpoint,
 * convert, load through the ordinary runtime, evaluate again. It proves that
 * Saphira LLM owns a complete native lifecycle. It proves nothing about 2B
 * training, convergence, model quality, or production-scale QAT, and every
 * number it produces is from a deliberately tiny fixture.
 *
 * It shares tokenizer, operator and container code with the runtime. It does
 * NOT turn /usr/bin/saphira-llm into a training framework: saphira-llm remains
 * an inference binary and knows nothing about gradients.
 *
 * Everything here is RAM-resident. Dataset, master weights, gradients,
 * optimiser state and working activations are all in memory; disk is touched
 * only for deliberate checkpoints, exported models and measured results.
 */
#ifndef SAPHIRA_LLM_QAT_H
#define SAPHIRA_LLM_QAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <saphira_llm/i2s_convert.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * THE MODEL. Deliberately tiny, and chosen so the exported artefact passes
 * through the ordinary saphira-llm loader with no change at all: that loader
 * reads every dimension from GGUF metadata and only requires
 * n_head %% n_head_kv == 0 and n_embd %% n_head == 0.
 */
typedef struct sllm_qat_config {
    int n_layer;
    int n_embd;
    int n_head;
    int n_head_kv;
    int n_embd_head;
    int n_embd_gqa;
    int n_ff;
    int n_vocab;
    int n_ctx;
    float rms_eps;
    float rope_base;
} sllm_qat_config;

void sllm_qat_default_config(sllm_qat_config * cfg);

/* TRAINABLE PARAMETERS.
 *
 * Master weights are BF16, which is what the upstream -bf16 release publishes
 * and what the ternary is derived FROM. The ternary is a function of the
 * master, not a parameter: it is recomputed every forward and thrown away.
 *
 * Gradients and optimiser moments are fp32 regardless, because a BF16 gradient
 * cannot represent a small update: bf16 has 8 significand bits, so a step of
 * 1e-5 against a weight of 1.0 is below the representable resolution and would
 * silently do nothing.
 */
typedef struct sllm_qat_tensor {
    const char * name;      /* for diagnostics and export */
    float * master;         /* fp32 working copy of the BF16 master */
    float * grad;           /* fp32 */
    float * adam_m;         /* fp32, first moment */
    float * adam_v;         /* fp32, second moment */
    int n;                  /* element count */
    bool trainable;
} sllm_qat_tensor;

typedef struct sllm_qat {
    sllm_qat_config cfg;
    sllm_qat_tensor * p;
    char * names;      /* owned storage for p[].name */
    int n_p;
    int n_trainable;
    long long n_params;
    /* RNG state, so a run is reproducible from its seed alone. */
    uint64_t rng;
    uint32_t step;
    /* Cached workspace, allocated on first use and kept. All working tensors
     * live here, in RAM, for the life of the model. */
    void * ws;
    /* Test hook: when false the forward uses the masters directly instead of
     * their ternaries. It exists so the hand-written backward can be checked
     * against a central difference, which is only meaningful for a forward
     * that is actually differentiable. A production run leaves this true. */
    bool ternary;
} sllm_qat;

sllm_qat * sllm_qat_new(const sllm_qat_config * cfg, uint64_t seed);
void sllm_qat_free(sllm_qat * q);

/*
 * INITIALISATION. Deterministic from the seed, and deliberately small: the
 * ternary of a near-zero weight is 0, so a master that starts at 0 would give a
 * permanently zero gradient under the STE for the sign function. The master
 * therefore starts at a small nonzero value, which is why this function exists
 * rather than a calloc.
 */
void sllm_qat_init(sllm_qat * q, uint64_t seed);

const sllm_qat_tensor * sllm_qat_find(const sllm_qat * q, const char * name);

/*
 * LOSS. Next-token cross entropy over the vocabulary, summed and averaged over
 * the scored positions. This is the same quantity saphira-llm-eval measures
 * as perplexity, so the two are directly comparable: exp(mean NLL).
 *
 * No label smoothing, no auxiliary losses, no temperature. A quantity that
 * cannot be checked against an existing measurement is not worth introducing in
 * an experiment whose whole point is that the measurement is trustworthy.
 */
typedef struct sllm_qat_batch {
    int32_t * tokens;       /* [T] */
    int n_tokens;
} sllm_qat_batch;

/*
 * Score a batch. Returns the mean cross entropy in nats and leaves the
 * gradients in the model. This is the PRE-TRAINING EVALUATION path too: call it
 * without stepping and the number is the baseline.
 */
double sllm_qat_loss(sllm_qat * q, const sllm_qat_batch * batch);

/* One AdamW step. Explicit, not a black box: the betas, epsilon and weight
 * decay are all in the header so they can be stated rather than assumed. */
#define SLLM_QAT_ADAM_BETA1  0.9f
#define SLLM_QAT_ADAM_BETA2  0.999f
#define SLLM_QAT_ADAM_EPS    1e-8f
#define SLLM_QAT_ADAM_WD     0.0f

void sllm_qat_step(sllm_qat * q, float lr);

/* Gradient clip, in global norm. Returns the norm BEFORE clipping, so a run
 * can report whether it fired rather than silently changing the optimisation. */
double sllm_qat_clip_grad(sllm_qat * q, float max_norm);

/* The I2_S scale a weight would export with: absmax, floored at 1e-5. Exposed
 * so the export and the training forward cannot disagree about it. */
float sllm_qat_weight_scale(const sllm_qat_tensor * t);

/* Checkpoint. Deliberate and explicit: a magic header, a version, the config,
 * the seed, the step and every parameter. Binary, little-endian, no partial
 * writes -- the file is built in memory and written once. */
int sllm_qat_save(const sllm_qat * q, const char * path);
sllm_qat * sllm_qat_load(const char * path);

/* Export the trained masters to an I2_S GGUF that the ordinary saphira-llm
 * loader accepts, with no special path and no alternate reader. */
int sllm_qat_export_i2s_gguf(const sllm_qat * q, const char * path,
                              sllm_i2s_rule rule);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_QAT_H */
