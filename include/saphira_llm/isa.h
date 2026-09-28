/*
 * isa.h — CPU feature detection and kernel dispatch.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_ISA_H
#define SAPHIRA_LLM_ISA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <saphira_llm/status.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * saphira-llm targets x86-64-v3 as its *baseline*, not as a fallback tier.
 * v3 means AVX, AVX2, FMA, F16C, BMI, BMI2 are assumed present everywhere,
 * and the whole binary is compiled for v3. Above v3 we offer strictly more,
 * never less.
 *
 * The levels are ordered, and a level implies all lower levels:
 *
 *   SLLM_ISA_V3     x86-64-v3. Always assumed. This is the contract.
 *   SLLM_ISA_VNNI   AVX-VNNI: vpdpbusd/vpdpbusds. The ternary GEMM upgrade.
 *   SLLM_ISA_AVX512 AVX-512F/BW/VL/DQ + VNNI, i.e. the Skylake-X set.
 *   SLLM_ISA_AMX    AMX-INT8/AMX-BF16 (Sapphire Rapids and later).
 */
typedef enum sllm_isa_level {
    SLLM_ISA_V3     = 0,
    SLLM_ISA_VNNI   = 1,
    SLLM_ISA_AVX512 = 2,
    SLLM_ISA_AMX    = 3,

    SLLM_ISA_COUNT  = 4
} sllm_isa_level;

/* What the CPU actually reports, before any override. */
typedef struct sllm_isa_caps {
    bool avx;
    bool avx2;
    bool fma;
    bool f16c;
    bool bmi1;
    bool bmi2;
    bool avx_vnni;
    bool avx512f;
    bool avx512bw;
    bool avx512vl;
    bool avx512dq;
    bool avx512_vnni;
    bool amx_int8;
    bool amx_bf16;
    bool sse3;
    bool ssse3;
    bool sse41;
    bool sse42;
    bool popcnt;
    bool f16c_flag;
} sllm_isa_caps;

/* A dispatch table. Entries above the selected level are deliberately NULL. */
typedef struct sllm_isa_dispatch {
    sllm_isa_level selected;   /* level actually in force                     */
    sllm_isa_caps  caps;       /* what the hardware reported                 */
    bool           overridden; /* true if a floor was imposed by the caller  */
} sllm_isa_dispatch;

/*
 * Read the CPU feature bits. Never fails, never allocates, and is safe to
 * call more than once. Cheap enough to call anywhere.
 */
void sllm_isa_detect(sllm_isa_caps * out);

/*
 * The highest level the CPU genuinely supports, given what it reports. This
 * is the ceiling; it is not the level we will use.
 */
sllm_isa_level sllm_isa_ceiling(const sllm_isa_caps * caps);

/*
 * Build a dispatch table.
 *
 * `floor` forces a *minimum* capability set, which is how we get a portable
 * binary to take the safe path, and how the test suite reaches the low paths
 * on hardware that supports more. Pass SLLM_ISA_LEVEL_AUTO (below) to let
 * detection decide.
 *
 * Honours the SAPHIRA_LLM_ISA environment variable, which is the same thing
 * as `floor` and exists so a deployed binary can be pinned without a rebuild.
 * An unknown or malformed value is an error, never a silent fallback — a typo
 * in a deployment variable must not quietly change which kernels run.
 */
sllm_isa_dispatch sllm_isa_build(sllm_isa_level floor);

/*
 * Validate a floor against what the CPU reports. Returns
 * SLLM_ERR_UNSUPPORTED if the floor is above the hardware ceiling.
 *
 * sllm_isa_build() deliberately does not call this, because it must not fail:
 * callers that want a hard error ask for it explicitly. A build policy in
 * which an impossible floor is a hard error is a decision for the caller, and
 * in most deployments clamping is the friendlier behaviour.
 */
sllm_status sllm_isa_check_floor(sllm_isa_level floor, const sllm_isa_caps * caps);

/* Sentinel meaning "no floor, use detection". Not a real level. */
#define SLLM_ISA_LEVEL_AUTO ((sllm_isa_level) -1)

/* Parse a level name ("v3", "vnni", "avx512", "amx", "auto"). */
sllm_status sllm_isa_level_parse(const char * name, sllm_isa_level * out);

/* Canonical name for a level. Never returns NULL. */
const char * sllm_isa_level_name(sllm_isa_level level);

/* True if `level` is available given `caps`. */
bool sllm_isa_level_supported(sllm_isa_level level, const sllm_isa_caps * caps);

/* Human-readable capability summary, for --isa and the log. Writes into buf. */
void sllm_isa_describe(const sllm_isa_dispatch * d, char * buf, size_t buflen);

/*
 * The first kernel, and the dispatch pattern every later kernel follows.
 *
 * sllm_probe_init() installs a function pointer. The v3 path is always
 * installed, because v3 is the build's baseline and not a fallback. An
 * above-v3 path is installed only when detection found the hardware *and* the
 * selected level permits it; otherwise the pointer keeps the v3 version. The
 * test suite asserts both properties, so a forced-low run provably cannot
 * reach guarded code.
 *
 * sllm_probe_installed_for() reports which level is actually installed, which
 * is what makes "the dispatch decision was correct" checkable rather than
 * merely asserted.
 */
#include <stdint.h>

void     sllm_probe_init(const sllm_isa_dispatch * d);
int32_t  sllm_probe_dot(const int8_t * a, const int8_t * b, size_t n);
int32_t  sllm_probe_dot_scalar(const int8_t * a, const int8_t * b, size_t n);
sllm_isa_level sllm_probe_installed_for(void);

/*
 * Compile-time guard for kernel authors.
 *
 * A kernel that needs more than v3 must be placed in SLLM_ISA_EXT_SECTION, so
 * that `make check-isa` can prove mechanically that no instruction above the
 * v3 baseline exists anywhere else in the binary. The ISA scanner treats that
 * section as the only permitted home for above-baseline code.
 *
 * The section is named .sllm.isa.ext and deliberately NOT .text.sllm_isa_ext:
 * the default linker script folds every .text.* input section into a single
 * output .text, so a .text-prefixed name would be silently merged away and the
 * guard would become vacuous. A non-.text name survives linking intact, which
 * was verified before the name was chosen.
 */
#if defined(__x86_64__)
#  define SLLM_ISA_EXT_SECTION ".sllm.isa.ext"
#  define SLLM_ISA_EXT __attribute__((section(SLLM_ISA_EXT_SECTION)))
#  define SLLM_ISA_TARGET_VNNI   __attribute__((target("avx2,avxvnni")))
#  define SLLM_ISA_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avxvnni")))
#  define SLLM_ISA_TARGET_AMX    __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avxvnni,amx-int8,amx-bf16")))
#else
/* Non-x86 builds have no v3 baseline to protect; the attributes are inert. */
#  define SLLM_ISA_EXT
#  define SLLM_ISA_TARGET_VNNI
#  define SLLM_ISA_TARGET_AVX512
#  define SLLM_ISA_TARGET_AMX
#endif

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_ISA_H */
