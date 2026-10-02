/*
 * rope_contract.h — typed, Saphira-owned metadata that carries RoPE SEMANTICS
 *                   explicitly through conversion.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_ROPE_CONTRACT_H
#define SAPHIRA_LLM_ROPE_CONTRACT_H

#include <stddef.h>
#include <stdint.h>

#include <saphira_llm/gguf.h>
#include <saphira_llm/ops.h>
#include <saphira_llm/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * WHY THIS CONTRACT EXISTS
 *
 * A GGUF from a real converter records the rotary EXTENT and the frequency base,
 * but it does not record whether dimension pairs are adjacent (2i, 2i+1) or
 * half-split (i, i+n/2), and it does not record the position origin. Both are
 * required to compute the rotation, both are used in the wild, and they produce
 * DIFFERENT results that each look entirely reasonable.
 *
 * The tempting shortcut is to look up the convention by architecture name. That
 * is refused here for a concrete reason, not a stylistic one: architecture
 * identity is not evidence for mathematics. Worse, the shortcut was checked and
 * it is demonstrably unsafe. The vendored reference path in third_party maps
 * LLM_ARCH_QWEN3 to LLAMA_ROPE_TYPE_NORM (ggml documents NORMAL as adjacent,
 * "[cscscscs]"), while the model's own authoritative implementation, HF
 * transformers modeling_qwen3.py, applies rotate_half, which is unambiguously
 * half-split ("[ccccssss]"). A name-based lookup would have selected the wrong
 * pairing from the very artefact it was meant to describe.
 *
 * So the semantics are established at the SOURCE MODEL, carried in the artefact
 * as typed fields Saphira owns, and read back with typed reads. The runtime
 * refuses when a required field is absent. It never infers.
 * ------------------------------------------------------------------------- */

/* The contract's key names. Namespaced under saphira. so they cannot collide
 * with, and are not confused for, upstream GGUF conventions that were never
 * written down in the first place. */
#define SLLM_ROPE_K_PAIRING          "saphira.rope.pairing"           /* STRING  */
#define SLLM_ROPE_K_POSITION_ORIGIN  "saphira.rope.position_origin"   /* UINT32  */
#define SLLM_ROPE_K_ROTARY_DIM       "saphira.rope.rotary_dim"        /* UINT32  */
#define SLLM_ROPE_K_FREQ_BASE        "saphira.rope.freq_base"         /* FLOAT32 */
#define SLLM_ROPE_K_SCALING_MODE     "saphira.rope.scaling_mode"      /* STRING  */
#define SLLM_ROPE_K_SCALING_FACTOR   "saphira.rope.scaling_factor"    /* FLOAT32 */
#define SLLM_ROPE_K_SOURCE           "saphira.rope.source"            /* STRING  */

/* Canonical pairing spellings. Deliberately not the upstream "neox"/"normal"
 * vocabulary: those words have been remapped across implementations, which is
 * part of how this ambiguity survived. These two names mean what they say. */
#define SLLM_ROPE_PAIRING_HALF_SPLIT "half_split"  /* pairs (i, i + n/2) */
#define SLLM_ROPE_PAIRING_ADJACENT   "adjacent"    /* pairs (2i, 2i+1)   */

typedef struct sllm_rope_semantics {
    sllm_rope_type pairing;          /* from SLLM_ROPE_K_PAIRING           */
    uint32_t       position_origin;  /* 0 or 1, from SLLM_ROPE_K_POSITION_ORIGIN */
    uint32_t       rotary_dim;       /* from SLLM_ROPE_K_ROTARY_DIM        */
    float          freq_base;        /* from SLLM_ROPE_K_FREQ_BASE         */
    char           scaling_mode[64]; /* "none", or a named mode            */
    float          scaling_factor;   /* 1.0 when mode is "none"            */
    char           source[192];      /* provenance of the source-side proof */
} sllm_rope_semantics;

/*
 * Resolve the contract from an artefact. Returns SLLM_OK ONLY when every
 * required field is present with the correct type and a valid value; otherwise
 * returns an error and writes the name of the FIRST missing or invalid fact
 * into `missing`. A partially populated struct is never returned as success,
 * because half a convention is worse than none: it invites the caller to fill
 * the gap with a default.
 */
sllm_status sllm_rope_semantics_from_gguf(const sllm_gguf * g,
                                          sllm_rope_semantics * out,
                                          char * missing, size_t missing_len);

/* True when the artefact carries the Saphira RoPE contract at all. Cheap
 * presence test, used to keep the real-model negative fixture honest: it must
 * keep reporting ABSENT forever rather than quietly acquiring a default. */
int sllm_rope_contract_present(const sllm_gguf * g);

/*
 * Cross-check the contract against what the artefact independently states.
 * `attention.key_length` (or equivalent) and the contract's rotary_dim are two
 * separate pieces of evidence and must agree; if they disagree the caller has a
 * corrupt artefact and must not proceed. Returns SLLM_OK on agreement.
 */
sllm_status sllm_rope_semantics_check_extent(const sllm_gguf * g,
                                             const sllm_rope_semantics * s,
                                             uint32_t independent_extent,
                                             char * why, size_t why_len);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_ROPE_CONTRACT_H */