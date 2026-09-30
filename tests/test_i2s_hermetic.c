/*
 * test_i2s_hermetic.c — the hermetic round-trip gate for the converter.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * THE GATE. Take the shipped I2_S model, dequantise each tensor to BF16, run
 * the native converter on that BF16, and require the output to be BYTE-IDENTICAL
 * to the original packed tensor.
 *
 * Why this gate and not "the converted model loads": a file that loads proves
 * nothing about its contents. Bit patterns can be wrong in a way that produces
 * perfectly well-formed bytes and a model that generates fluent nonsense. This
 * is the only first gate that can fail for the right reason, and it needs no
 * network and no external master model.
 *
 * WHY THE SCALE IS SUPPLIED RATHER THAN RECOVERED, which is the one thing that
 * had to be reasoned out rather than assumed:
 *
 * The I2_S scale is an arbitrary f32. BF16 keeps 7 explicit mantissa bits, and
 * an arbitrary f32 is not representable in BF16 -- truncation perturbs it. So a
 * dequantise-to-BF16 step CANNOT preserve the scale, and byte identity of the
 * scale field is unachievable by construction.
 *
 * What the round trip must preserve is the CODES, and codes are decided by sign,
 * which BF16 preserves exactly. So the gate supplies the ORIGINAL scale through
 * sllm_i2s_quantize_scaled -- the same override_scale path the upstream python
 * converter exposes -- and requires everything else to match byte for byte.
 * That is the strongest criterion the format permits, and the reason it IS
 * attainable is that the codes and the scale are separate fields.
 *
 * If this ever fails, that is either a converter defect or a fact about the
 * model, and either way it is not something to paper over by loosening the
 * comparison.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "harness.h"

#include <saphira_llm/gguf.h>
#include <saphira_llm/i2s_convert.h>
#include <saphira_llm/quant.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#ifndef SLLM_TEST_MODEL
#define SLLM_TEST_MODEL "/var/lib/spoon/models/ggml-model-i2_s.gguf"
#endif

/*
 * Which rule to use for the round trip. Both must give identical results here,
 * because the dequantised values are exactly {-s, 0, +s} and on such input the
 * two rules were measured to agree completely (0 disagreements). The test
 * asserts BOTH, which is a stronger statement than picking one.
 */
/*
 * Upper bound on a tensor the gate will dequantise. It has to be generous
 * enough for the real model: the FFN tensors are 6912 x 2560 = 17,694,720
 * elements, which is ABOVE 1<<24. An earlier 1<<24 guard silently refused every
 * one of them and reported a failure, so the gate looked exercised while never
 * having run on the largest tensors in the model. A guard that excludes the
 * interesting cases is worse than no guard.
 *
 * 1<<27 elements is 134M, or 512 MB as f32 plus 32 MB packed, which is well
 * inside RAM and far above anything this model contains.
 */
#define I2S_RT_MAX_ELEMS ((size_t) 1u << 27)

static int roundtrip_one(const sllm_gguf * g, const sllm_gguf_tensor * t,
                         sllm_i2s_rule rule, int * out_n) {
    const size_t n = (size_t) t->nbytes;
    /* I2_S stores n_elements/4 packed bytes plus a 32-byte tail. Recover the
     * element count from that rather than trusting a metadata key, because the
     * scale offset is derived from it. */
    if (n < 36u) { return 0; }
    const size_t n_elems = (n - 32u) * 4u;
    if ((n_elems % 4u) != 0u) { return 0; }
    if (n_elems == 0u || n_elems > I2S_RT_MAX_ELEMS) { return 0; }

    const uint8_t * packed = (const uint8_t *) g->file + g->data_offset + t->offset;
    const float original_scale = sllm_i2s_block_scale(packed, n_elems);

    /* Dequantise to BF16, which is what a real master weight is. */
    float * src = (float *) malloc(n_elems * sizeof(float));
    if (src == NULL) { return -1; }
    sllm_i2s_dequant(packed, src, n_elems);
    for (size_t i = 0; i < n_elems; ++i) {
        src[i] = sllm_bf16_to_f32(sllm_f32_to_bf16(src[i]));
    }

    /* Reconvert with the original scale supplied. */
    const size_t pb = sllm_i2s_packed_size(n_elems);
    uint8_t * out = (uint8_t *) calloc(pb, 1);
    if (out == NULL) { free(src); return -1; }
    if (sllm_i2s_quantize_scaled(src, n_elems, out, original_scale, rule) != 0) {
        free(out); free(src);
        return -1;
    }

    /*
     * The criterion, made precise.
     *
     * An I2_S payload is n/4 packed code bytes, then the 4-byte f32 scale, then
     * 28 bytes of ALIGNMENT PADDING. The first n/4 + 4 bytes are the data and
     * must match exactly. The last 28 are not data: nothing reads them.
     *
     *   - our decoder reads exactly four bytes at offset n/4 (src/quant.c:104)
     *   - the upstream dequantiser takes the scale as a PARAMETER
     *     (quants.c:1335) and never touches the tail
     *
     * and the shipped file has RESIDUAL DATA in them, because whatever wrote it
     * did not clear them. Requiring byte identity there would be requiring
     * identity of memory nobody defined or reads, which is not a property any
     * implementation can satisfy. That is why the first version of this gate
     * failed on all 24 tensors it touched, every one of them in the last 28
     * bytes and nowhere else.
     *
     * So: the codes and the scale are compared byte for byte, and the padding is
     * instead REQUIRED to be zero. Requiring determinism where the format is
     * silent is a stronger claim than inheriting whatever was in the buffer,
     * and it is the one a shipped artefact should be held to.
     */
    const size_t code_bytes = n_elems / 4u;
    int ok = 1;
    if (pb != n) { ok = 0; }
    if (ok && memcmp(out, packed, code_bytes + 4u) != 0) {
        ok = 0;
        size_t first = 0;
        const size_t limit = code_bytes + 4u;
        while (first < limit && out[first] == packed[first]) { ++first; }
        const char * which = (first < code_bytes) ? "packed codes" : "scale";
        fprintf(stderr, "    roundtrip differs in the %s at byte %zu of %zu "
                        "(rule %d)\n", which, first, limit, (int) rule);
    }
    if (ok) {
        /* The scale came back exactly, because it was supplied and the format
         * stores it verbatim. */
        if (sllm_i2s_block_scale(out, n_elems) != original_scale) { ok = 0; }
    }
    if (ok) {
        for (size_t i = code_bytes + 4u; i < pb; ++i) {
            if (out[i] != 0u) { ok = 0;
                fprintf(stderr, "    padding byte %zu is %02x, not zero\n", i, out[i]); }
        }
    }
    if (ok) {
        /* Recorded rather than asserted: how much residual data the shipped
         * file carries in the padding. If this is ever zero, the comparison
         * above and a whole-buffer one would be equivalent again. */
        size_t residual = 0;
        for (size_t i = code_bytes + 4u; i < n; ++i) {
            if (packed[i] != 0u) { ++residual; }
        }
        *out_n = (int) n_elems;
        if (residual) { fprintf(stderr, "    (shipped padding carries %zu residual bytes)\n", residual); }
    }

    free(out);
    free(src);
    if (!ok) { *out_n = (int) n_elems; }
    return ok;
}


TEST(i2s_convert_hermetic_roundtrip_is_byte_identical) {
    if (access(SLLM_TEST_MODEL, R_OK) != 0) {
        printf("    skipped: %s not available\n", SLLM_TEST_MODEL);
        CHECK(1);
        return;
    }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) {
        printf("    skipped: %s (%s)\n", SLLM_TEST_MODEL, err);
        CHECK(1);
        return;
    }

    int i2s_tensors = 0, tested = 0, failed = 0;
    long long elems_total = 0;

    for (uint64_t i = 0; i < g.n_tensors; ++i) {
        const sllm_gguf_tensor * t = &g.tensors[i];
        if (t->type != SLLM_TYPE_I2_S) { continue; }
        ++i2s_tensors;
        if (t->nbytes < 36u) { continue; }
        /* Keep the gate quick but broad: every I2_S tensor of a sensible size,
         * capped so the suite stays fast on a large model. */
        if (tested >= 24) { continue; }

        int n_elems = 0;
        for (int rule = 0; rule < 2; ++rule) {
            const sllm_i2s_rule r = (rule == 0) ? SLLM_I2S_RULE_GGUF_PY
                                                 : SLLM_I2S_RULE_BITNET_C;
            const int rc = roundtrip_one(&g, t, r, &n_elems);
            if (rc < 0) { fprintf(stderr, "    allocation failure on %s\n", t->name); ++failed; break; }
            if (rc == 0) {
                fprintf(stderr, "  FAIL roundtrip tensor '%s' (%llu bytes) rule %d\n",
                        t->name, (unsigned long long) t->nbytes, (int) r);
                ++failed;
                break;
            }
        }
        elems_total += n_elems;
        ++tested;
    }

    printf("    %d I2_S tensors in the model, %d round-tripped, %lld elements, "
           "%d failures\n", i2s_tensors, tested, elems_total, failed);
    /* The gate is only meaningful if it actually ran on real tensors. */
    CHECK(tested > 0);
    CHECK_EQ_INT(failed, 0);
    CHECK(i2s_tensors > 100);   /* the 2B model has 210 weight tensors */

    sllm_gguf_close(&g);
}

/*
 * The gate above is only as good as its coverage claim, so state it: it must
 * actually reach real tensors, and it must not be vacuously passing on zero.
 * This asserts the shape of the run rather than the numbers.
 */
TEST(i2s_convert_hermetic_gate_is_not_vacuous) {
    if (access(SLLM_TEST_MODEL, R_OK) != 0) { CHECK(1); return; }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) {
        CHECK(1); return;
    }
    int i2s = 0;
    uint64_t largest = 0;
    for (uint64_t i = 0; i < g.n_tensors; ++i) {
        if (g.tensors[i].type != SLLM_TYPE_I2_S) { continue; }
        ++i2s;
        if (g.tensors[i].nbytes > largest) { largest = g.tensors[i].nbytes; }
    }
    /* 210 of 332 tensors are I2_S in this model, per PHASE4-CONTRACT. If that
     * ever changes, the reason should be written down rather than the test
     * quietly loosening. */
    CHECK(i2s >= 200);
    CHECK(largest > 1000000u);   /* the token embedding is 128256 x 2560 */
    printf("    %d I2_S tensors, largest %llu bytes\n", i2s,
           (unsigned long long) largest);
    sllm_gguf_close(&g);
}

void sllm_test_i2s_hermetic(void) {
    printf("i2s-hermetic\n");
    RUN(i2s_convert_hermetic_gate_is_not_vacuous);
    RUN(i2s_convert_hermetic_roundtrip_is_byte_identical);
}
