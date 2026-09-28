/*
 * status.h — saphira-llm status codes.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_STATUS_H
#define SAPHIRA_LLM_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Every fallible entry point returns one of these. Zero is success, so
 * `if (rc != SLLM_OK)` is the only test a caller ever needs. Negative values
 * are failures, positive values are non-fatal advisories that the caller may
 * choose to ignore.
 *
 * The intent is that an unsupported model or quantisation fails *cleanly and
 * specifically*. "BitNet weights but you asked for K-quants" must be
 * distinguishable from "file is truncated" and from "no such file".
 */
typedef enum sllm_status {
    SLLM_OK = 0,

    /* Caller and environment */
    SLLM_ERR_ARG        = -1,   /* a function argument was invalid        */
    SLLM_ERR_NOMEM      = -2,   /* allocation failed                      */
    SLLM_ERR_IO         = -3,   /* read/write/seek failed                 */
    SLLM_ERR_UNSUPPORTED = -4,  /* valid input we do not implement         */

    /* GGUF container */
    SLLM_ERR_GGUF_MAGIC     = -10, /* not a GGUF file                       */
    SLLM_ERR_GGUF_VERSION   = -11, /* GGUF major version we do not support  */
    SLLM_ERR_GGUF_TRUNCATED = -12, /* file ends inside a structure          */
    SLLM_ERR_GGUF_STRING    = -13, /* a string length is absurd or invalid  */
    SLLM_ERR_GGUF_TYPE      = -14, /* a metadata value type is unknown      */
    SLLM_ERR_GGUF_TENSOR    = -15, /* a tensor descriptor is invalid        */
    SLLM_ERR_GGUF_ALIGNMENT = -16, /* general.alignment is not a power of 2 */
    SLLM_ERR_GGUF_LAYOUT    = -17, /* tensor offsets overlap or overflow    */
    SLLM_ERR_GGUF_KEYDUP    = -18, /* the same metadata key appears twice   */

    /* Model */
    SLLM_ERR_ARCH_UNSUPPORTED = -30, /* no adapter for this architecture    */
    SLLM_ERR_KV_MISSING       = -31, /* required metadata key is absent     */
    SLLM_ERR_KV_TYPE          = -32, /* metadata key has the wrong type     */
    SLLM_ERR_TENSOR_MISSING   = -33, /* required tensor is absent           */
    SLLM_ERR_TENSOR_SHAPE     = -34, /* tensor has the wrong rank or dims   */
    SLLM_ERR_TYPE_UNSUPPORTED = -35, /* tensor type not implemented         */
    SLLM_ERR_TOO_LARGE        = -36, /* model exceeds a hard safety limit   */

    SLLM_WARN_TRUNCATED_KV = 1    /* a value was clipped; results may differ */
} sllm_status;

/* Stable, human-readable name for a status. Never returns NULL. */
const char * sllm_status_string(sllm_status status);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_STATUS_H */
