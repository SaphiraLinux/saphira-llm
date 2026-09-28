/*
 * log.h — saphira-llm diagnostics.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_LOG_H
#define SAPHIRA_LLM_LOG_H

#include <saphira_llm/status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum sllm_log_level {
    SLLM_LOG_QUIET = 0,  /* errors only                                */
    SLLM_LOG_ERROR = 1,
    SLLM_LOG_WARN  = 2,
    SLLM_LOG_INFO  = 3,  /* the default for a normal run                */
    SLLM_LOG_DEBUG = 4   /* kernel selection, dispatch decisions, shapes */
} sllm_log_level;

/* Set the threshold. Messages below it are discarded before formatting. */
void sllm_log_set_level(sllm_log_level level);
sllm_log_level sllm_log_get_level(void);

/*
 * Parse a level name ("quiet", "error", "warn", "info", "debug") or a
 * non-negative integer. Rejects anything else rather than defaulting, so a
 * typo is visible.
 */
sllm_status sllm_log_level_parse(const char * name, sllm_log_level * out);

void sllm_log(sllm_log_level level, const char * fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/*
 * Report a failed status with context, then return it, so call sites read as
 * one statement:
 *
 *     return sllm_fail(SLLM_ERR_IO, "reading %s", path);
 */
sllm_status sllm_fail(sllm_status status, const char * fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_LOG_H */
