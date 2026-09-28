/*
 * isa.c — CPU feature detection and kernel dispatch.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * x86-64-v3 is the baseline and the whole binary is compiled for it, so this
 * file never needs to avoid v3 instructions. What it must do is decide, once,
 * whether anything *above* v3 is allowed, and make that decision inspectable.
 */

#include <saphira_llm/isa.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if defined(__x86_64__)
#  define SLLM_HAVE_X86 1
#endif

void sllm_isa_detect(sllm_isa_caps * out) {
    if (out == NULL) {
        return;
    }
    memset(out, 0, sizeof(*out));

#if defined(SLLM_HAVE_X86)
    /*
     * __builtin_cpu_supports is a GCC/Clang builtin. It is called through a
     * small helper so the argument is always a compile-time constant, which is
     * what the builtin requires; folding the calls by hand would be both
     * noisier and easier to get subtly wrong.
     */
#  define SLLM_SUPPORTS(x) (__builtin_cpu_supports(x) != 0)

    out->sse3        = SLLM_SUPPORTS("sse3");
    out->ssse3       = SLLM_SUPPORTS("ssse3");
    out->sse41       = SLLM_SUPPORTS("sse4.1");
    out->sse42       = SLLM_SUPPORTS("sse4.2");
    out->popcnt      = SLLM_SUPPORTS("popcnt");
    out->avx         = SLLM_SUPPORTS("avx");
    out->avx2        = SLLM_SUPPORTS("avx2");
    out->fma         = SLLM_SUPPORTS("fma");
    out->bmi1        = SLLM_SUPPORTS("bmi");
    out->bmi2        = SLLM_SUPPORTS("bmi2");
    out->avx_vnni    = SLLM_SUPPORTS("avxvnni");
    out->avx512f     = SLLM_SUPPORTS("avx512f");
    out->avx512bw    = SLLM_SUPPORTS("avx512bw");
    out->avx512vl    = SLLM_SUPPORTS("avx512vl");
    out->avx512dq    = SLLM_SUPPORTS("avx512dq");
    out->avx512_vnni = SLLM_SUPPORTS("avx512vnni");
    out->amx_int8    = SLLM_SUPPORTS("amx-int8");
    out->amx_bf16    = SLLM_SUPPORTS("amx-bf16");

    /*
     * F16C is part of v3. The field name carries _flag so it cannot be
     * confused with the FMA instruction set. If it is missing, the binary
     * cannot honour its own baseline and the caller is warned rather than
     * left to misexecute.
     */
    out->f16c_flag   = SLLM_SUPPORTS("f16c");

#  undef SLLM_SUPPORTS
#endif
}

sllm_isa_level sllm_isa_ceiling(const sllm_isa_caps * caps) {
    if (caps == NULL) {
        return SLLM_ISA_V3;
    }
    if (caps->amx_int8 && caps->amx_bf16 && caps->avx512_vnni) {
        return SLLM_ISA_AMX;
    }
    if (caps->avx512f && caps->avx512bw && caps->avx512vl && caps->avx512dq && caps->avx512_vnni) {
        return SLLM_ISA_AVX512;
    }
    if (caps->avx_vnni) {
        return SLLM_ISA_VNNI;
    }
    return SLLM_ISA_V3;
}

bool sllm_isa_level_supported(sllm_isa_level level, const sllm_isa_caps * caps) {
    switch (level) {
        case SLLM_ISA_V3:
            return true;  /* the baseline is assumed, by contract */
        case SLLM_ISA_VNNI:
            return caps != NULL && caps->avx_vnni;
        case SLLM_ISA_AVX512:
            return caps != NULL && caps->avx512f && caps->avx512bw &&
                   caps->avx512vl && caps->avx512dq && caps->avx512_vnni;
        case SLLM_ISA_AMX:
            return caps != NULL && caps->amx_int8 && caps->amx_bf16 && caps->avx512_vnni;
        default:
            return false;
    }
}

const char * sllm_isa_level_name(sllm_isa_level level) {
    /* SLLM_ISA_LEVEL_AUTO is a sentinel outside the enumeration, so it is
     * compared before the switch rather than as a case label. */
    if (level == SLLM_ISA_LEVEL_AUTO) {
        return "auto";
    }
    switch (level) {
        case SLLM_ISA_V3:     return "v3";
        case SLLM_ISA_VNNI:   return "vnni";
        case SLLM_ISA_AVX512: return "avx512";
        case SLLM_ISA_AMX:    return "amx";
        default:              return "invalid";
    }
}

sllm_status sllm_isa_level_parse(const char * name, sllm_isa_level * out) {
    if (name == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    if (strcmp(name, "v3") == 0 || strcmp(name, "baseline") == 0) {
        *out = SLLM_ISA_V3;
    } else if (strcmp(name, "vnni") == 0) {
        *out = SLLM_ISA_VNNI;
    } else if (strcmp(name, "avx512") == 0 || strcmp(name, "avx-512") == 0) {
        *out = SLLM_ISA_AVX512;
    } else if (strcmp(name, "amx") == 0) {
        *out = SLLM_ISA_AMX;
    } else if (strcmp(name, "auto") == 0) {
        *out = SLLM_ISA_LEVEL_AUTO;
    } else {
        return SLLM_ERR_ARG;
    }
    return SLLM_OK;
}

sllm_isa_dispatch sllm_isa_build(sllm_isa_level floor) {
    sllm_isa_dispatch d;
    memset(&d, 0, sizeof(d));

    sllm_isa_detect(&d.caps);

    const sllm_isa_level ceiling = sllm_isa_ceiling(&d.caps);

    /*
     * An explicit floor wins over detection. That is the point: it is how a
     * portable binary takes a conservative path, and how the test suite
     * exercises the lower paths on capable hardware. Asking for a level the
     * CPU cannot do is an error condition, reported through SLLM_ERR_ARG by
     * sllm_isa_check_floor, not silently downgraded here.
     */
    if (floor != SLLM_ISA_LEVEL_AUTO) {
        d.selected = floor;
        d.overridden = true;
    } else {
        d.selected = ceiling;
    }

    return d;
}

void sllm_isa_describe(const sllm_isa_dispatch * d, char * buf, size_t buflen) {
    if (buf == NULL || buflen == 0) {
        return;
    }
    if (d == NULL) {
        buf[0] = '\0';
        return;
    }
    (void) snprintf(buf, buflen,
        "selected=%s ceiling=%s%s%s%s%s%s%s",
        sllm_isa_level_name(d->selected),
        sllm_isa_level_name(sllm_isa_ceiling(&d->caps)),
        d->caps.avx        ? " avx"      : "",
        d->caps.avx2       ? " avx2"     : "",
        d->caps.fma        ? " fma"      : "",
        d->caps.f16c_flag  ? " f16c"     : "",
        d->caps.avx_vnni   ? " avx_vnni" : "",
        d->caps.avx512f    ? " avx512f"  : "");
}

/*
 * Kept separate so the tests can assert the contract: a floor above the
 * ceiling is refused, never honoured by hoping.
 */
sllm_status sllm_isa_check_floor(sllm_isa_level floor, const sllm_isa_caps * caps) {
    if (floor == SLLM_ISA_LEVEL_AUTO || caps == NULL) {
        return SLLM_OK;
    }
    if (floor < SLLM_ISA_V3 || floor >= SLLM_ISA_COUNT) {
        return SLLM_ERR_ARG;
    }
    if (!sllm_isa_level_supported(floor, caps)) {
        return SLLM_ERR_UNSUPPORTED;
    }
    return SLLM_OK;
}
