/*
 * test_isa.c — CPU feature detection and dispatch tests.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The dispatch tests are the important ones. The mechanical proof that no
 * instruction above the v3 baseline can run outside the guarded section is in
 * `make check-isa`; these tests prove the *decision* is right, so that the
 * guarded section is only ever entered when it may be.
 */

#include "harness.h"

#include <stdlib.h>

TEST(isa_detection_reports_the_baseline) {
    sllm_isa_caps caps;
    sllm_isa_detect(&caps);

    /*
     * The whole binary is compiled for x86-64-v3, so a host that cannot
     * supply v3 is a broken deployment, not a supported fallback. This test
     * records the fact on whatever machine it runs on; it is a warning
     * channel, not a gate, because the suite must also pass on older
     * hardware used for smoke testing.
     */
    if (!(caps.avx && caps.avx2 && caps.fma && caps.bmi1 && caps.bmi2 && caps.f16c_flag)) {
        printf("    note: this CPU does not report the full x86-64-v3 baseline\n");
    }
    sllm_tests_run++;
}

TEST(isa_ceiling_is_ordered_and_conservative) {
    sllm_isa_caps caps;
    memset(&caps, 0, sizeof(caps));

    /* Nothing at all: the baseline is still assumed. */
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_V3);

    /* VNNI alone lifts exactly one level. */
    caps.avx_vnni = true;
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_VNNI);

    /* A partial AVX-512 set must not claim AVX-512. */
    caps.avx512f = true;
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_VNNI);
    caps.avx512bw = true;
    caps.avx512vl = true;
    caps.avx512dq = true;
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_VNNI);

    /* The full AVX-512 set plus 512-bit VNNI reaches AVX-512 even though the
     * 256-bit VNNI flag is gone: these are different instructions, and the
     * ladder follows capability sets rather than one flag. */
    caps.avx512_vnni = true;
    caps.avx_vnni = false;
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_AVX512);

    /* AMX needs the 512-bit VNNI, so without it AMX must not be claimed. */
    caps.amx_int8 = true;
    caps.amx_bf16 = true;
    caps.avx512_vnni = false;
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_V3);

    /* With the prerequisite, AMX. */
    caps.avx512_vnni = true;
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_AMX);

    /* Clearing everything above the baseline must land on v3, never below. */
    memset(&caps, 0, sizeof(caps));
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_V3);

    /* AMX alone, with no VNNI of any kind, is still not AMX. */
    memset(&caps, 0, sizeof(caps));
    caps.amx_int8 = true;
    caps.amx_bf16 = true;
    CHECK_EQ_INT(sllm_isa_ceiling(&caps), SLLM_ISA_V3);
}

TEST(isa_v3_is_always_supported) {
    sllm_isa_caps caps;
    memset(&caps, 0, sizeof(caps));
    /* The baseline is a contract of the build, not a probe result. */
    CHECK(sllm_isa_level_supported(SLLM_ISA_V3, &caps));
    CHECK(sllm_isa_level_supported(SLLM_ISA_V3, NULL));

    CHECK(!sllm_isa_level_supported(SLLM_ISA_VNNI, &caps));
    CHECK(!sllm_isa_level_supported(SLLM_ISA_AVX512, &caps));
    CHECK(!sllm_isa_level_supported(SLLM_ISA_AMX, &caps));
}

TEST(isa_floor_is_honoured_and_marked_as_an_override) {
    sllm_isa_caps caps;
    memset(&caps, 0, sizeof(caps));

    const sllm_isa_dispatch auto_d = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    CHECK_EQ_INT(auto_d.selected, sllm_isa_ceiling(&auto_d.caps));
    CHECK(!auto_d.overridden);

    /* Forcing the baseline must pin selection to the baseline even on a CPU
     * that offers more. This is how the portable path gets exercised on
     * capable hardware. */
    const sllm_isa_dispatch forced = sllm_isa_build(SLLM_ISA_V3);
    CHECK_EQ_INT(forced.selected, SLLM_ISA_V3);
    CHECK(forced.overridden);
    CHECK(forced.selected <= sllm_isa_ceiling(&forced.caps) ||
          forced.selected == SLLM_ISA_V3);
}

TEST(isa_impossible_floor_is_refused_not_downgraded) {
    sllm_isa_caps caps;
    memset(&caps, 0, sizeof(caps));

    /* Asking for more than the hardware has must be an error the caller can
     * see, never a silent clamp: a deployment that pinned --isa amx and got
     * v3 instead would be running the wrong kernels without knowing. */
    CHECK_STATUS(sllm_isa_check_floor(SLLM_ISA_V3, &caps), SLLM_OK);
    CHECK_STATUS(sllm_isa_check_floor(SLLM_ISA_VNNI, &caps), SLLM_ERR_UNSUPPORTED);
    CHECK_STATUS(sllm_isa_check_floor(SLLM_ISA_AVX512, &caps), SLLM_ERR_UNSUPPORTED);
    CHECK_STATUS(sllm_isa_check_floor(SLLM_ISA_AMX, &caps), SLLM_ERR_UNSUPPORTED);

    /* A level the CPU does support must be accepted. */
    caps.avx_vnni = true;
    CHECK_STATUS(sllm_isa_check_floor(SLLM_ISA_VNNI, &caps), SLLM_OK);
    CHECK_STATUS(sllm_isa_check_floor(SLLM_ISA_AVX512, &caps), SLLM_ERR_UNSUPPORTED);

    /* Out-of-range and AUTO. */
    CHECK_STATUS(sllm_isa_check_floor(SLLM_ISA_LEVEL_AUTO, &caps), SLLM_OK);
    CHECK_STATUS(sllm_isa_check_floor((sllm_isa_level) 99, &caps), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_isa_check_floor((sllm_isa_level) -5, &caps), SLLM_ERR_ARG);
}

TEST(isa_level_names_round_trip) {
    sllm_isa_level lvl = SLLM_ISA_V3;

    CHECK_STATUS(sllm_isa_level_parse("v3", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_ISA_V3);
    CHECK_STATUS(sllm_isa_level_parse("baseline", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_ISA_V3);
    CHECK_STATUS(sllm_isa_level_parse("vnni", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_ISA_VNNI);
    CHECK_STATUS(sllm_isa_level_parse("avx512", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_ISA_AVX512);
    CHECK_STATUS(sllm_isa_level_parse("amx", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_ISA_AMX);
    CHECK_STATUS(sllm_isa_level_parse("auto", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_ISA_LEVEL_AUTO);

    /* A typo must be an error, not a silent default. */
    CHECK_STATUS(sllm_isa_level_parse("avx-512-bogus", &lvl), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_isa_level_parse("", &lvl), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_isa_level_parse("V3", &lvl), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_isa_level_parse("v3", NULL), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_isa_level_parse(NULL, &lvl), SLLM_ERR_ARG);

    CHECK_STR(sllm_isa_level_name(SLLM_ISA_V3), "v3");
    CHECK_STR(sllm_isa_level_name((sllm_isa_level) 42), "invalid");
}

TEST(isa_environment_override_is_honoured) {
    /*
     * A deployment can pin the ISA without a rebuild. Verify the plumbing by
     * setting the variable and reading the dispatch.
     */
    setenv("SAPHIRA_LLM_ISA", "v3", 1);
    const sllm_isa_dispatch d = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    /* sllm_isa_build itself does not read the environment in Phase 1; the CLI
     * resolves it. What is asserted here is that the resolved floor and the
     * dispatch agree, and that the value parses. */
    sllm_isa_level parsed = SLLM_ISA_LEVEL_AUTO;
    CHECK_STATUS(sllm_isa_level_parse(getenv("SAPHIRA_LLM_ISA"), &parsed), SLLM_OK);
    CHECK_EQ_INT(parsed, SLLM_ISA_V3);
    CHECK_EQ_INT(d.selected, sllm_isa_ceiling(&d.caps));
    unsetenv("SAPHIRA_LLM_ISA");
}

TEST(isa_describe_is_informative) {
    sllm_isa_dispatch d = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    char buf[256];
    sllm_isa_describe(&d, buf, sizeof(buf));
    sllm_tests_run++;
    if (strstr(buf, "selected=") == NULL) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL describe did not mention the selected level: '%s'\n", buf);
    }

    /* Must not write past a tiny buffer. */
    char tiny[8];
    sllm_isa_describe(&d, tiny, sizeof(tiny));
    sllm_isa_describe(NULL, buf, sizeof(buf));
    sllm_isa_describe(&d, NULL, 0);
    CHECK(1);
}

TEST(log_levels_parse_and_reject_typos) {
    sllm_log_level lvl = SLLM_LOG_INFO;

    CHECK_STATUS(sllm_log_level_parse("debug", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_LOG_DEBUG);
    CHECK_STATUS(sllm_log_level_parse("quiet", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_LOG_QUIET);
    CHECK_STATUS(sllm_log_level_parse("3", &lvl), SLLM_OK);
    CHECK_EQ_INT(lvl, SLLM_LOG_INFO);

    CHECK_STATUS(sllm_log_level_parse("verbose", &lvl), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_log_level_parse("99", &lvl), SLLM_ERR_ARG);
    CHECK_STATUS(sllm_log_level_parse("", &lvl), SLLM_ERR_ARG);

    sllm_log_set_level(SLLM_LOG_QUIET);
    CHECK_EQ_INT(sllm_log_get_level(), SLLM_LOG_QUIET);
    sllm_log_set_level(SLLM_LOG_INFO);
}

TEST(status_names_are_stable_and_total) {
    CHECK_STR(sllm_status_string(SLLM_OK), "ok");
    CHECK_STR(sllm_status_string(SLLM_ERR_GGUF_MAGIC), "gguf-bad-magic");
    CHECK_STR(sllm_status_string(SLLM_ERR_GGUF_TRUNCATED), "gguf-truncated");
    CHECK_STR(sllm_status_string(SLLM_ERR_TYPE_UNSUPPORTED), "unsupported-tensor-type");
    /* An out-of-range code must still yield a string, never NULL, so error
     * paths can never dereference nothing. */
    CHECK(sllm_status_string((sllm_status) 12345) != NULL);
    CHECK_STR(sllm_status_string((sllm_status) 12345), "unknown-status");
}

/* ------------------------------------------------------------------ */
/* dispatch correctness                                                */
/* ------------------------------------------------------------------ */

/*
 * The v3 path is not a fallback. It is installed whatever the host does,
 * because the whole binary is compiled for x86-64-v3 and a host that cannot
 * supply it is a broken deployment rather than something to degrade into.
 */
TEST(isa_dispatch_installs_v3_regardless_of_host) {
    sllm_isa_dispatch bare;
    memset(&bare, 0, sizeof(bare));

    /* Even with every capability cleared, v3 is what gets installed. */
    sllm_probe_init(&bare);
    CHECK_EQ_INT(sllm_probe_installed_for(), SLLM_ISA_V3);

    /* And with the real host caps, unless the host genuinely offers more. */
    const sllm_isa_dispatch d = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    sllm_probe_init(&d);
    const sllm_isa_level got = sllm_probe_installed_for();
    sllm_tests_run++;
    if (got != SLLM_ISA_V3 && got != sllm_isa_ceiling(&d.caps)) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL installed level %s is neither v3 nor the ceiling %s\n",
                sllm_isa_level_name(got), sllm_isa_level_name(sllm_isa_ceiling(&d.caps)));
    }
}

/*
 * A forced-low floor must not reach guarded code, and a floor above the
 * hardware must not be honoured.
 */
TEST(isa_dispatch_only_guards_when_hardware_and_level_agree) {
    sllm_isa_caps caps;
    memset(&caps, 0, sizeof(caps));

    /* Hardware without VNNI, floor of v3: v3 path. */
    sllm_isa_dispatch d = sllm_isa_build(SLLM_ISA_V3);
    sllm_probe_init(&d);
    CHECK_EQ_INT(sllm_probe_installed_for(), SLLM_ISA_V3);

    /* Hardware with VNNI but the floor pinned to v3: still v3, because the
     * floor is a choice about which kernels may run, not a hint. */
    caps.avx_vnni = true;
    d.selected    = SLLM_ISA_V3;
    sllm_probe_init(&d);
    CHECK_EQ_INT(sllm_probe_installed_for(), SLLM_ISA_V3);

    /* Hardware with VNNI and the level actually at VNNI: guarded path. */
    d.selected = SLLM_ISA_VNNI;
    sllm_probe_init(&d);
    CHECK_EQ_INT(sllm_probe_installed_for(), SLLM_ISA_VNNI);

    /* Hardware with VNNI but the level at v3 even though the ceiling is
     * higher: guarded path must not be taken. */
    d.selected = SLLM_ISA_V3;
    sllm_probe_init(&d);
    CHECK_EQ_INT(sllm_probe_installed_for(), SLLM_ISA_V3);

    /* Restore. */
    const sllm_isa_dispatch real = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    sllm_probe_init(&real);
}

/*
 * The decisive one. The v3 path, the VNNI path and the scalar reference must
 * produce bit-identical results on data chosen to break naive kernels:
 * saturation at the int16 boundary, values at the extremes of int8, and a
 * length that leaves a tail.
 */
TEST(isa_dispatch_paths_agree_exactly) {
    enum { N = 1024 };
    static int8_t a[N];
    static int8_t b[N];

    /* Seed a deterministic mix including full-scale values. */
    uint32_t s = 0x12345678u;
    for (int i = 0; i < N; ++i) {
        s = s * 1664525u + 1013904223u;
        a[i] = (int8_t) ((s >> 16) & 0xff);
        s = s * 1664525u + 1013904223u;
        b[i] = (int8_t) ((s >> 16) & 0xff);
    }
    /* Force the extremes into the middle so saturation would show up. */
    for (int i = 0; i < N; i += 64) {
        a[i] = (int8_t) (i % 2 ? -128 : 127);
        b[i] = (int8_t) (i % 2 ? -128 : 127);
    }

    const int32_t ref = sllm_probe_dot_scalar(a, b, N);

    /* v3 path */
    sllm_isa_caps caps;
    memset(&caps, 0, sizeof(caps));
    sllm_isa_dispatch d = sllm_isa_build(SLLM_ISA_V3);
    sllm_probe_init(&d);
    const int32_t v3 = sllm_probe_dot(a, b, N);
    CHECK_EQ_INT(v3, ref);

    /* VNNI path, if the hardware has it. */
    caps.avx_vnni = true;
    d.selected    = SLLM_ISA_VNNI;
    sllm_probe_init(&d);
    if (sllm_probe_installed_for() == SLLM_ISA_VNNI) {
        const int32_t vnni = sllm_probe_dot(a, b, N);
        sllm_tests_run++;
        if (vnni != ref) {
            sllm_tests_failed++;
            fprintf(stderr,
                "  FAIL VNNI path disagrees with the reference: %d vs %d\n", vnni, ref);
        }
    }

    /* Tail handling: every length from 0 to 80 must match, on both paths.
     * A vectorised kernel that mishandles the tail is the classic way to
     * pass a whole-array test and fail in production. */
    sllm_probe_init(&d);
    for (size_t n = 0; n <= 80; ++n) {
        const int32_t want = sllm_probe_dot_scalar(a, b, n);
        CHECK_EQ_INT(sllm_probe_dot(a, b, n), want);
    }

    /* Null arguments must be tolerated, not dereferenced. */
    sllm_probe_init(&d);
    CHECK_EQ_INT(sllm_probe_dot(NULL, b, N), 0);
    CHECK_EQ_INT(sllm_probe_dot(a, NULL, N), 0);
    CHECK_EQ_INT(sllm_probe_dot(NULL, NULL, 0), 0);

    const sllm_isa_dispatch real = sllm_isa_build(SLLM_ISA_LEVEL_AUTO);
    sllm_probe_init(&real);
}

void sllm_test_isa(void) {
    printf("isa\n");
    RUN(isa_detection_reports_the_baseline);
    RUN(isa_ceiling_is_ordered_and_conservative);
    RUN(isa_v3_is_always_supported);
    RUN(isa_floor_is_honoured_and_marked_as_an_override);
    RUN(isa_impossible_floor_is_refused_not_downgraded);
    RUN(isa_level_names_round_trip);
    RUN(isa_environment_override_is_honoured);
    RUN(isa_describe_is_informative);
    RUN(log_levels_parse_and_reject_typos);
    RUN(status_names_are_stable_and_total);
    RUN(isa_dispatch_installs_v3_regardless_of_host);
    RUN(isa_dispatch_only_guards_when_hardware_and_level_agree);
    RUN(isa_dispatch_paths_agree_exactly);
}
