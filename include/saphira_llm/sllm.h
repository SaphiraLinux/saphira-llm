/*
 * sllm.h — saphira-llm umbrella header.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_SLLM_H
#define SAPHIRA_LLM_SLLM_H

#include <saphira_llm/status.h>
#include <saphira_llm/log.h>
#include <saphira_llm/isa.h>
#include <saphira_llm/gguf.h>
#include <saphira_llm/topology.h>
#include <saphira_llm/thread.h>
#include <saphira_llm/quant.h>
#include <saphira_llm/ops.h>

#define SLLM_VERSION_MAJOR 0
#define SLLM_VERSION_MINOR 1
#define SLLM_VERSION_PATCH 1
#define SLLM_VERSION_STRING "0.2.0"

/*
 * The ISA baseline this build assumes. Not a claim about the host: the host
 * may be better, and runtime dispatch will use what it finds. It is a claim
 * about the binary, and `make check-isa` proves it mechanically.
 */
#define SLLM_BASELINE_ISA "x86-64-v3"

#endif /* SAPHIRA_LLM_SLLM_H */
