/*
 * rope_contract.c — resolve the Saphira RoPE semantics contract from an artefact.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#include "saphira_llm/rope_contract.h"
#include "saphira_llm/ops.h"
#include "saphira_llm/log.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* Every refusal names the fact that was missing, so a caller can act on it. The
 * helper is variadic because those messages carry values -- an absent key with no
 * value is less useful than "position_origin must be 0 or 1, got 7". */
static void say(char * buf, size_t n, const char * fmt, ...) {
    if (!buf || !n) { return; }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, n, fmt, ap);
    va_end(ap);
}

int sllm_rope_contract_present(const sllm_gguf * g) {
    const char * s = NULL;
    return g && sllm_gguf_kv_str(g, SLLM_ROPE_K_PAIRING, &s) == SLLM_OK && s != NULL;
}

sllm_status sllm_rope_semantics_from_gguf(const sllm_gguf * g,
                                          sllm_rope_semantics * out,
                                          char * missing, size_t missing_len) {
    if (!g || !out) {
        say(missing, missing_len, "%s", "no artefact or output supplied");
        return SLLM_ERR_ARG;
    }
    memset(out, 0, sizeof *out);

    /* Each field is read with the type it was declared with. A field present
     * under the wrong type is treated as MISSING, not coerced: a string where a
     * float belongs is an artefact defect, and coercing it would convert a loud
     * failure into a quiet wrong number. */

    const char * pairing = NULL;
    if (sllm_gguf_kv_str(g, SLLM_ROPE_K_PAIRING, &pairing) != SLLM_OK || !pairing) {
        say(missing, missing_len, "%s", SLLM_ROPE_K_PAIRING);
        return SLLM_ERR_KV_MISSING;
    }
    if (sllm_rope_type_parse(pairing, &out->pairing) != SLLM_OK) {
        say(missing, missing_len, "%s (unrecognised value '%s')", SLLM_ROPE_K_PAIRING, pairing);
        return SLLM_ERR_UNSUPPORTED;
    }

    if (sllm_gguf_kv_u32(g, SLLM_ROPE_K_POSITION_ORIGIN, &out->position_origin) != SLLM_OK) {
        say(missing, missing_len, "%s", SLLM_ROPE_K_POSITION_ORIGIN);
        return SLLM_ERR_KV_MISSING;
    }
    if (out->position_origin > 1u) {
        say(missing, missing_len, "%s (must be 0 or 1, got %u)",
            SLLM_ROPE_K_POSITION_ORIGIN, out->position_origin);
        return SLLM_ERR_UNSUPPORTED;
    }

    if (sllm_gguf_kv_u32(g, SLLM_ROPE_K_ROTARY_DIM, &out->rotary_dim) != SLLM_OK) {
        say(missing, missing_len, "%s", SLLM_ROPE_K_ROTARY_DIM);
        return SLLM_ERR_KV_MISSING;
    }
    if (out->rotary_dim == 0u || (out->rotary_dim & 1u)) {
        say(missing, missing_len, "%s (must be a positive even number, got %u)",
            SLLM_ROPE_K_ROTARY_DIM, out->rotary_dim);
        return SLLM_ERR_UNSUPPORTED;
    }

    if (sllm_gguf_kv_f32(g, SLLM_ROPE_K_FREQ_BASE, &out->freq_base) != SLLM_OK) {
        say(missing, missing_len, "%s", SLLM_ROPE_K_FREQ_BASE);
        return SLLM_ERR_KV_MISSING;
    }
    if (!(out->freq_base > 0.0f)) {
        say(missing, missing_len, "%s (must be positive, got %g)",
            SLLM_ROPE_K_FREQ_BASE, (double) out->freq_base);
        return SLLM_ERR_UNSUPPORTED;
    }

    const char * mode = NULL;
    if (sllm_gguf_kv_str(g, SLLM_ROPE_K_SCALING_MODE, &mode) != SLLM_OK || !mode) {
        say(missing, missing_len, "%s", SLLM_ROPE_K_SCALING_MODE);
        return SLLM_ERR_KV_MISSING;
    }
    snprintf(out->scaling_mode, sizeof out->scaling_mode, "%s", mode);

    if (sllm_gguf_kv_f32(g, SLLM_ROPE_K_SCALING_FACTOR, &out->scaling_factor) != SLLM_OK) {
        say(missing, missing_len, "%s", SLLM_ROPE_K_SCALING_FACTOR);
        return SLLM_ERR_KV_MISSING;
    }
    /* An explicit "none" mode carries a factor of exactly 1. Anything else in
     * combination with factor 1 is legal but suspicious, so it is noted rather
     * than refused: the check belongs to the caller who knows the source. */
    if (strcmp(out->scaling_mode, "none") == 0 && out->scaling_factor != 1.0f) {
        say(missing, missing_len, "%s contradicts scaling_mode 'none' (factor %g)",
            SLLM_ROPE_K_SCALING_FACTOR, (double) out->scaling_factor);
        return SLLM_ERR_UNSUPPORTED;
    }

    const char * src = NULL;
    if (sllm_gguf_kv_str(g, SLLM_ROPE_K_SOURCE, &src) != SLLM_OK || !src) {
        say(missing, missing_len, "%s", SLLM_ROPE_K_SOURCE);
        return SLLM_ERR_KV_MISSING;
    }
    snprintf(out->source, sizeof out->source, "%s", src);

    return SLLM_OK;
}

sllm_status sllm_rope_semantics_check_extent(const sllm_gguf * g,
                                             const sllm_rope_semantics * s,
                                             uint32_t independent_extent,
                                             char * why, size_t why_len) {
    (void) g;
    if (!s) {
        say(why, why_len, "%s", "no resolved semantics");
        return SLLM_ERR_ARG;
    }
    /* The extent appears twice, once in the contract and once in whatever the
     * artefact already carried (attention.key_length). They were established by
     * different routes, so agreement is real corroboration. Disagreement means
     * the artefact is inconsistent and no rotation should be computed from it. */
    if (independent_extent != 0u && independent_extent != s->rotary_dim) {
        if (why && why_len) {
            snprintf(why, why_len,
                     "rotary extent disagreement: contract %s=%u vs independent key %u",
                     SLLM_ROPE_K_ROTARY_DIM, s->rotary_dim, independent_extent);
        }
        return SLLM_ERR_ARG;
    }
    return SLLM_OK;
}