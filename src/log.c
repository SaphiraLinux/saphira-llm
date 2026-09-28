/*
 * log.c — saphira-llm diagnostics.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */

#include <saphira_llm/log.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static sllm_log_level g_level = SLLM_LOG_INFO;

void sllm_log_set_level(sllm_log_level level) {
    g_level = level;
}

sllm_log_level sllm_log_get_level(void) {
    return g_level;
}

sllm_status sllm_log_level_parse(const char * name, sllm_log_level * out) {
    if (name == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    if (strcmp(name, "quiet") == 0) { *out = SLLM_LOG_QUIET; return SLLM_OK; }
    if (strcmp(name, "error") == 0) { *out = SLLM_LOG_ERROR; return SLLM_OK; }
    if (strcmp(name, "warn")  == 0) { *out = SLLM_LOG_WARN;  return SLLM_OK; }
    if (strcmp(name, "info")  == 0) { *out = SLLM_LOG_INFO;  return SLLM_OK; }
    if (strcmp(name, "debug") == 0) { *out = SLLM_LOG_DEBUG; return SLLM_OK; }

    /* A bare number is accepted so scripts need not know the names. `end == name`
     * means no digits were consumed, which is what an empty string gives:
     * strtol returns 0 there, and accepting that would silently turn a typo
     * into "quiet". */
    char * end = NULL;
    const long v = strtol(name, &end, 10);
    if (end != NULL && end != name && *end == '\0' && v >= 0 && v <= SLLM_LOG_DEBUG) {
        *out = (sllm_log_level) v;
        return SLLM_OK;
    }
    return SLLM_ERR_ARG;
}

static const char * level_tag(sllm_log_level level) {
    switch (level) {
        case SLLM_LOG_ERROR: return "error";
        case SLLM_LOG_WARN:  return "warn";
        case SLLM_LOG_INFO:  return "info";
        case SLLM_LOG_DEBUG: return "debug";
        default:             return "quiet";
    }
}

void sllm_log(sllm_log_level level, const char * fmt, ...) {
    if (level > g_level || level == SLLM_LOG_QUIET) {
        return;
    }
    /* Diagnostics go to stderr so that generated text on stdout stays clean
     * enough to pipe. Nothing in the runtime prints model output except the
     * generation loop. */
    fprintf(stderr, "saphira-llm: %s: ", level_tag(level));

    va_list ap;
    va_start(ap, fmt);
    (void) vfprintf(stderr, fmt, ap);
    va_end(ap);

    fputc('\n', stderr);
}

sllm_status sllm_fail(sllm_status status, const char * fmt, ...) {
    if (g_level >= SLLM_LOG_ERROR) {
        fprintf(stderr, "saphira-llm: error: %s: ", sllm_status_string(status));
        va_list ap;
        va_start(ap, fmt);
        (void) vfprintf(stderr, fmt, ap);
        va_end(ap);
        fputc('\n', stderr);
    }
    return status;
}
