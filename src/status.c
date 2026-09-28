/*
 * status.c — status code names.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */

#include <saphira_llm/status.h>

const char * sllm_status_string(sllm_status status) {
    switch (status) {
        case SLLM_OK:                       return "ok";
        case SLLM_ERR_ARG:                  return "invalid-argument";
        case SLLM_ERR_NOMEM:                return "out-of-memory";
        case SLLM_ERR_IO:                   return "io-error";
        case SLLM_ERR_UNSUPPORTED:          return "unsupported";
        case SLLM_ERR_GGUF_MAGIC:           return "gguf-bad-magic";
        case SLLM_ERR_GGUF_VERSION:         return "gguf-unsupported-version";
        case SLLM_ERR_GGUF_TRUNCATED:       return "gguf-truncated";
        case SLLM_ERR_GGUF_STRING:          return "gguf-bad-string";
        case SLLM_ERR_GGUF_TYPE:            return "gguf-bad-value-type";
        case SLLM_ERR_GGUF_TENSOR:          return "gguf-bad-tensor";
        case SLLM_ERR_GGUF_ALIGNMENT:       return "gguf-bad-alignment";
        case SLLM_ERR_GGUF_LAYOUT:          return "gguf-bad-layout";
        case SLLM_ERR_GGUF_KEYDUP:          return "gguf-duplicate-key";
        case SLLM_ERR_ARCH_UNSUPPORTED:     return "unsupported-architecture";
        case SLLM_ERR_KV_MISSING:           return "missing-metadata-key";
        case SLLM_ERR_KV_TYPE:              return "wrong-metadata-type";
        case SLLM_ERR_TENSOR_MISSING:       return "missing-tensor";
        case SLLM_ERR_TENSOR_SHAPE:         return "wrong-tensor-shape";
        case SLLM_ERR_TYPE_UNSUPPORTED:     return "unsupported-tensor-type";
        case SLLM_ERR_TOO_LARGE:            return "exceeds-limit";
        case SLLM_WARN_TRUNCATED_KV:        return "warn-truncated-value";
        default:                            return "unknown-status";
    }
}
