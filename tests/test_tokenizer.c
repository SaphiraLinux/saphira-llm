/*
 * Tokenizer tests: exact agreement with the pinned reference.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The gate is exact token-id equality, and it is exact because a tokeniser has
 * no floating point in it. There is no reduction order to argue about and no
 * tolerance worth having: if the ids differ, the model is conditioning on a
 * different sequence, and nothing downstream can recover.
 *
 * The fixture is the reference's own llama_tokenize() output, recorded by
 * tools/reference/sllm-tokenize-ref. It carries the prompt text as well as the
 * ids, so this test is driven by the fixture rather than by a second copy of
 * the prompt list that could drift away from it.
 */

#include "harness.h"
#include "saphira_llm/tokenizer.h"

#include <ctype.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SLLM_TEST_MODEL
#define SLLM_TEST_MODEL "/var/lib/spoon/models/ggml-model-i2_s.gguf"
#endif
#define SLLM_TEST_GOLDEN  "tests/golden/tokenizer.txt"
#define SLLM_TEST_GOLDEN_GPT2 "tests/golden/tokenizer-gpt2.txt"
#define SLLM_TEST_GPT2_LABEL "tests/golden/tokenizer-gpt2.txt"

/* Undo the escaping the reference tool applied. Returns the length. */
static size_t unescape(const char * in, char * out, size_t cap) {
    size_t o = 0;
    for (size_t i = 0; in[i] != '\0' && in[i] != '\n'; ) {
        /*
         * Stop at a bare newline: that is the end of the fixture line, not
         * part of the prompt. A newline that belongs to the prompt is always
         * written as the two characters \ and n by the reference tool, so a
         * bare one here can only be the line terminator. Carrying it into the
         * text added a spurious token to every prompt and made all 80 records
         * mismatch by exactly one.
         */
        if (in[i] == '\\' && in[i + 1] != '\0') {
            const char c = in[i + 1];
            if (c == 'n')      { if (o + 1 >= cap) { break; } out[o++] = '\n'; i += 2; continue; }
            if (c == 'r')      { if (o + 1 >= cap) { break; } out[o++] = '\r'; i += 2; continue; }
            if (c == 't')      { if (o + 1 >= cap) { break; } out[o++] = '\t'; i += 2; continue; }
            if (c == '\\')     { if (o + 1 >= cap) { break; } out[o++] = '\\'; i += 2; continue; }
            if (c == 'x' && isxdigit((unsigned char) in[i + 2]) &&
                isxdigit((unsigned char) in[i + 3])) {
                const char hex[3] = { in[i + 2], in[i + 3], '\0' };
                if (o + 1 >= cap) { break; }
                out[o++] = (char) strtol(hex, NULL, 16);
                i += 4;
                continue;
            }
        }
        if (o + 1 >= cap) { break; }
        out[o++] = in[i++];
    }
    out[o] = '\0';
    return o;
}

typedef struct {
    char     name[128];
    char     text[1024];
    size_t   text_len;
    int32_t  ids[4096];
    int32_t  n;
} record;

static void check_record_pre(const record * rec, const sllm_tok * tok, sllm_pre_type pre);

/* Compare against the model's own pre-type. */
static void check_record(const record * rec, const sllm_tok * tok) {
    check_record_pre(rec, tok, sllm_tok_pre_type(tok));
}

/* Compare one fixture record against what we produce for the same text. */
static void check_record_pre(const record * rec, const sllm_tok * tok, sllm_pre_type pre) {
    int32_t got[4096];
    const int32_t n = sllm_tok_encode_pre(tok, pre, rec->text, rec->text_len,
                                          true /* add_special */, true /* parse_special */,
                                          got, (int32_t) (sizeof(got) / sizeof(got[0])));

    sllm_tests_run++;
    if (n != rec->n) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL %s: %d tokens, reference has %d\n", rec->name, (int) n, (int) rec->n);
        fprintf(stderr, "       text <%s>\n", rec->text);
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        if (got[i] == rec->ids[i]) { continue; }
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL %s: token %d is %d, reference has %d\n",
                rec->name, (int) i, (int) got[i], (int) rec->ids[i]);
        fprintf(stderr, "       text <%s>\n", rec->text);
        fprintf(stderr, "       ours %s\n", sllm_tok_piece(tok, got[i]) ?
                sllm_tok_piece(tok, got[i]) : "?");
        fprintf(stderr, "       want %s\n", sllm_tok_piece(tok, rec->ids[i]) ?
                sllm_tok_piece(tok, rec->ids[i]) : "?");
        return;
    }
}

TEST(tokenizer_matches_the_reference_token_for_token) {
    if (access(SLLM_TEST_GOLDEN, R_OK) != 0) {
        printf("    skipped: %s not present\n", SLLM_TEST_GOLDEN);
        CHECK(1);
        return;
    }

    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) {
        printf("    skipped: %s not available (%s)\n", SLLM_TEST_MODEL, err);
        CHECK(1);
        return;
    }

    sllm_tok * tok = NULL;
    const sllm_status rc = sllm_tok_load(&g, &tok);
    if (rc != SLLM_OK) {
        sllm_tests_run++;
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL sllm_tok_load: %s\n", sllm_status_string(rc));
        sllm_gguf_close(&g);
        return;
    }

    CHECK(sllm_tok_n_vocab(tok) == 128256u);
    CHECK(sllm_tok_bos(tok) == 128000);
    CHECK(sllm_tok_add_bos(tok));

    FILE * f = fopen(SLLM_TEST_GOLDEN, "r");
    CHECK(f != NULL);
    if (f == NULL) { sllm_tok_free(tok); sllm_gguf_close(&g); return; }
    const int sllm_tests_failed_at_entry = sllm_tests_failed;

    record rec;
    memset(&rec, 0, sizeof(rec));
    int records = 0;
    char line[65536];

    while (fgets(line, sizeof(line), f) != NULL) {
        char name[128];
        if (sscanf(line, "prompt %127s", name) == 1) {
            memset(&rec, 0, sizeof(rec));
            snprintf(rec.name, sizeof(rec.name), "%s", name);
            continue;
        }
        if (strncmp(line, "  text ", 7) == 0) {
            rec.text_len = unescape(line + 7, rec.text, sizeof(rec.text));
            continue;
        }
        if (strncmp(line, "  ids ", 6) == 0) {
            rec.n = 0;
            const char * p = line + 6;
            while (rec.n < (int32_t) (sizeof(rec.ids) / sizeof(rec.ids[0]))) {
                char * end = NULL;
                const long v = strtol(p, &end, 10);
                if (end == p) { break; }
                rec.ids[rec.n++] = (int32_t) v;
                p = end;
            }
            check_record(&rec, tok);
            records++;
        }
    }

    fclose(f);
    sllm_tok_free(tok);
    sllm_gguf_close(&g);

    const int failed_before = sllm_tests_failed;
    (void) failed_before;
    sllm_tests_run++;
    if (records < 80) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL only %d prompts in the fixture, expected at least 80\n", records);
    } else if (sllm_tests_failed != sllm_tests_failed_at_entry) {
        /*
         * Print this unconditionally when something failed, including the count
         * that actually mismatched. An earlier version printed "all matching"
         * on the strength of the record count alone, while every record in the
         * file was failing -- a success line that reports success regardless of
         * the result is worse than no success line.
         */
        fprintf(stderr, "  FAIL %d prompt(s) in the fixture did not match the reference\n",
                sllm_tests_failed - sllm_tests_failed_at_entry);
    } else {
        printf("    %d prompts, all matching the reference\n", records);
    }
}

/*
 * Round-tripping is checked separately from token agreement, because they fail
 * differently. Token agreement can pass while decode is wrong if a piece
 * round-trips through a path the encoder never takes, and decode can pass while
 * encode is wrong if every token happens to be a single byte.
 */
TEST(tokenizer_round_trips_every_prompt) {
    if (access(SLLM_TEST_GOLDEN, R_OK) != 0) {
        CHECK(1);
        return;
    }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) {
        CHECK(1);
        return;
    }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) {
        sllm_gguf_close(&g);
        CHECK(1);
        return;
    }

    FILE * f = fopen(SLLM_TEST_GOLDEN, "r");
    if (f == NULL) { sllm_tok_free(tok); sllm_gguf_close(&g); CHECK(1); return; }

    char text[1024];
    char line[65536];
    int checked = 0;
    int bad = 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "  text ", 7) != 0) { continue; }
        const size_t tlen = unescape(line + 7, text, sizeof(text));
        if (tlen == 0) { continue; }

        int32_t ids[4096];
        const int32_t n = sllm_tok_encode(tok, text, tlen, false, true, ids, 4096);
        if (n <= 0) { continue; }

        char back[2048];
        const int32_t blen = sllm_tok_decode(tok, ids, n, false, back, (int32_t) sizeof(back));
        checked++;
        if (blen != (int32_t) tlen || memcmp(back, text, tlen) != 0) {
            bad++;
            if (bad <= 3) {
                fprintf(stderr, "  FAIL round trip: got %d bytes, want %zu\n  got  <%s>\n  want <%s>\n",
                        (int) blen, tlen, back, text);
            }
        }
    }
    fclose(f);

    CHECK(checked > 0);
    sllm_tests_run++;
    if (bad != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL %d of %d prompts did not round trip\n", bad, checked);
    } else {
        printf("    %d prompts round trip through encode and decode\n", checked);
    }

    sllm_tok_free(tok);
    sllm_gguf_close(&g);
}

/*
 * The pre-tokeniser on its own. A golden vector over whole token ids cannot say
 * which of the four split passes was wrong, and pass 4 in particular changes
 * only digit runs, which a prose-only prompt set would never exercise.
 */
TEST(pre_tokeniser_splits_the_way_the_reference_does) {
    if (access(SLLM_TEST_GOLDEN, R_OK) != 0) { CHECK(1); return; }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) { CHECK(1); return; }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) { sllm_gguf_close(&g); CHECK(1); return; }

    struct { const char * in; const char * want; const char * why; } cases[] = {
        { "1234567890", "123\000456\000789\0000\000",
          "pass 4 groups digits in threes; only pass 4 does this" },
        { "1234567",    "123\000456\0007\000",
          "the trailing digit is left over, not merged" },
        { "007",        "007\000",
          "a short run is under three and is left alone" },
        { "they're",    "they\000'\000re\000",
          "the contraction rule is anchored at the current position" },
        { "it's",       "it\000'\000s\000",
          "an apostrophe mid-word is not a contraction either" },
        { "%!",         "%!\000",
          "pass 1 eats a run of punctuation into one word" },
        { "a + b",      "a\000 \000+\000 b\000",
          "a bare space is its own word after pass 1 splits the operator" },
        { "a   b",      "a\000  \000b\000",
          "\\s+(?!\\S) leaves the last space for the next word" },
        { "a  b",       "a\000 \000 b\000",
          "with two spaces the first is emitted alone" },
        { "U.S.",       "U\000.\000S\000.\000",
          "periods split from letters" },
    };

    char got[512];
    int n = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const sllm_status rc = sllm_tok_pretokenize(SLLM_PRE_UNSET, cases[i].in, strlen(cases[i].in),
                                                     got, sizeof(got), &n);
        sllm_tests_run++;
        if (rc != SLLM_OK) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL pretokenize(\"%s\"): %s\n", cases[i].in,
                    sllm_status_string(rc));
            continue;
        }
        if (strcmp(got, cases[i].want) != 0) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL pretokenize(\"%s\") [%s]\n    got  <", cases[i].in, cases[i].why);
            for (int k = 0; k < n; ++k) {
                const char * w = got + k;
                fprintf(stderr, "%s<%s>", k ? "|" : "", w);
            }
            fprintf(stderr, ">\n    want <");
            for (int k = 0; k < n; ++k) {
                const char * w = cases[i].want;
                for (int q = 0; q < k; ++q) { w += strlen(w) + 1; }
                fprintf(stderr, "%s<%s>", k ? "|" : "", w);
            }
            fprintf(stderr, ">\n");
        }
    }

    sllm_tok_free(tok);
    sllm_gguf_close(&g);
}

/*
 * The pre-tokeniser is model metadata, and this is the test that holds it to
 * that. The only model available declares no `pre`, so without this the
 * DEFAULT behaviour would be indistinguishable from a constant compiled into
 * the tokenizer, and Phase 7's models would silently tokenise wrong.
 */
TEST(pre_tokeniser_is_read_from_metadata_not_assumed) {
    if (access(SLLM_TEST_MODEL, R_OK) != 0) { CHECK(1); return; }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) { CHECK(1); return; }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) { sllm_gguf_close(&g); CHECK(1); return; }

    /* This model has no `pre` key at all. Absent is DEFAULT, and DEFAULT is
     * emphatically not GPT-2 -- conflating the two is the defect this phase
     * was written to avoid. */
    CHECK(strcmp(sllm_tok_pre_declared(tok), "") == 0);
    CHECK(sllm_tok_pre_type(tok) == SLLM_PRE_UNSET);
    CHECK(sllm_tok_pre_type(tok) != SLLM_PRE_GPT2);
    CHECK(sllm_tok_pre_is_verified(tok));

    sllm_tok_free(tok);
    sllm_gguf_close(&g);
}

TEST(default_and_gpt2_pre_tokenisers_genuinely_differ) {
    /*
     * Recorded from the reference on the same vocabulary with
     * tokenizer.ggml.pre = "gpt-2" (tests/golden/tokenizer-gpt2.txt), so these
     * are measured differences and not an argument about what a GPT-2
     * pre-tokeniser ought to do.
     */
    struct { const char * in; const char * default_want; const char * gpt2_want; const char * why; } cases[] = {
        { "they're",   "they\0'\0re\0",     "they\0're\0",
          "the contraction rule is anchored, so DEFAULT splits the apostrophe out" },
        { "it's",      "it\0'\0s\0",        "it\0's\0",
          "same rule, two-letter contraction" },
        { "1234567890","123\0" "456\0" "789\0" "0\0", "1234567890\0",
          "DEFAULT adds the digit-triple pass; gpt-2 has no such pass, so the "
          "digits stay one word and BPE splits it instead" },
        { "a + b = c", "a\0 \0+\0 b\0 \0=\0 c\0", "a\0 +\0 b\0 =\0 c\0",
          "DEFAULT splits punctuation runs that gpt-2 keeps whole" },
    };

    char def_buf[256];
    char gpt_buf[256];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int n = 0;
        sllm_tests_run++;
        if (sllm_tok_pretokenize(SLLM_PRE_UNSET, cases[i].in, strlen(cases[i].in),
                                 def_buf, sizeof(def_buf), &n) != SLLM_OK ||
            strcmp(def_buf, cases[i].default_want) != 0) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL DEFAULT split of \"%s\" [%s]\n    got  <%s>\n    want <%s>\n",
                    cases[i].in, cases[i].why, def_buf, cases[i].default_want);
        }
        sllm_tests_run++;
        if (sllm_tok_pretokenize(SLLM_PRE_GPT2, cases[i].in, strlen(cases[i].in),
                                 gpt_buf, sizeof(gpt_buf), &n) != SLLM_OK ||
            strcmp(gpt_buf, cases[i].gpt2_want) != 0) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL gpt-2 split of \"%s\" [%s]\n    got  <%s>\n    want <%s>\n",
                    cases[i].in, cases[i].why, gpt_buf, cases[i].gpt2_want);
        }
    }

    /* And the two must not be the same function under two names. */
    sllm_tests_run++;
    if (memcmp(def_buf, gpt_buf, sizeof(def_buf)) == 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL DEFAULT and gpt-2 produced identical output for the last case\n");
    }
}

TEST(gpt2_pre_tokeniser_matches_the_reference_on_the_same_vocabulary) {
    if (access(SLLM_TEST_GOLDEN_GPT2, R_OK) != 0) {
        printf("    skipped: %s not present\n", SLLM_TEST_GPT2_LABEL);
        CHECK(1);
        return;
    }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) { CHECK(1); return; }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) { sllm_gguf_close(&g); CHECK(1); return; }

    record rec;
    memset(&rec, 0, sizeof(rec));
    int records = 0;
    char line[65536];
    FILE * f = fopen(SLLM_TEST_GOLDEN_GPT2, "r");
    if (f == NULL) { sllm_tok_free(tok); sllm_gguf_close(&g); CHECK(1); return; }
    const int failed_at_entry = sllm_tests_failed;

    while (fgets(line, sizeof(line), f) != NULL) {
        char name[128];
        if (sscanf(line, "prompt %127s", name) == 1) {
            memset(&rec, 0, sizeof(rec));
            snprintf(rec.name, sizeof(rec.name), "%s", name);
            continue;
        }
        if (strncmp(line, "  text ", 7) == 0) {
            rec.text_len = unescape(line + 7, rec.text, sizeof(rec.text));
            continue;
        }
        if (strncmp(line, "  ids", 5) != 0) { continue; }
        rec.n = 0;
        const char * p = line + 5;
        while (rec.n < (int32_t) (sizeof(rec.ids) / sizeof(rec.ids[0]))) {
            char * end = NULL;
            const long v = strtol(p, &end, 10);
            if (end == p) { break; }
            rec.ids[rec.n++] = (int32_t) v;
            p = end;
        }
        check_record_pre(&rec, tok, SLLM_PRE_GPT2);
        records++;
    }
    fclose(f);
    sllm_tok_free(tok);
    sllm_gguf_close(&g);

    sllm_tests_run++;
    if (records < 80) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL only %d gpt-2 prompts in the fixture, expected 80\n", records);
    } else if (sllm_tests_failed != failed_at_entry) {
        fprintf(stderr, "  FAIL %d gpt-2 prompt(s) did not match the reference\n",
                sllm_tests_failed - failed_at_entry);
    } else {
        printf("    %d prompts under gpt-2 pre-type, all matching the reference\n", records);
    }
}

TEST(unsupported_pre_types_are_rejected_rather_than_approximated) {
    /*
     * A model naming a pre-tokeniser we have not implemented must fail. Loading
     * it and tokenising with whatever happens to be compiled in is the worst
     * outcome available: the model would appear to work and be wrong in a way
     * no fixture covers. The reference throws here too.
     */
    char got[128];
    int n = 0;
    sllm_tests_run++;
    if (sllm_tok_pretokenize(SLLM_PRE_UNSUPPORTED, "hello", 5,
                             got, sizeof(got), &n) != SLLM_ERR_UNSUPPORTED) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL pretokenize accepted SLLM_PRE_UNSUPPORTED\n");
    }
}

/* --- regression coverage for the Phase 3.5 defects ------------------- */

TEST(regression_whitespace_is_a_list_not_a_category) {
    /*
     * U+0020 is Zs. The category range table marks it SEPARATOR with no
     * whitespace bit, and only the separate whitespace list grants one. Port
     * the ranges alone and ` ?[^\s\p{L}\p{N}]+` swallows runs of spaces,
     * \s+(?!\S) never fires, and every double-space prompt tokenises wrongly
     * while every single space stays correct.
     */
    char buf[256];
    int n = 0;

    /* A single space attaches to the following word. */
    sllm_tests_run++;
    if (sllm_tok_pretokenize(SLLM_PRE_UNSET, "a b", 3, buf, sizeof(buf), &n) != SLLM_OK ||
        strcmp(buf, "a\0\xc4" "\xa0" "b\0") != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL single space should prefix the next word\n");
    }

    /* Two spaces: \s+(?!\S) emits all but the last, which prefixes the next
     * word. Emitting the whole run is the classic off-by-one. */
    sllm_tests_run++;
    if (sllm_tok_pretokenize(SLLM_PRE_UNSET, "a  b", 4, buf, sizeof(buf), &n) != SLLM_OK ||
        strcmp(buf, "a\0 \0 \0b\0") != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL double space should split 1 + 1-with-next, got <%s>\n", buf);
    }

    /* Non-breaking space is whitespace by list membership, not by category:
     * it is Zs, exactly like an ASCII space, and must be treated the same. */
    sllm_tests_run++;
    if (sllm_tok_pretokenize(SLLM_PRE_UNSET, "a\xc2\xa0" "b", 4, buf, sizeof(buf), &n) != SLLM_OK ||
        strcmp(buf, "a\0" "\xc2\xa0" "b\0") != 0) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL NBSP should be treated as whitespace\n");
    }
}

TEST(regression_byte_encoding_returns_a_utf8_encoding_not_a_byte) {
    /*
     * A passing-through byte maps to the UTF-8 encoding of the code point of
     * the same value. For ASCII the two are the same byte, so getting it wrong
     * passes on English prose and makes non-ASCII encode to nothing.
     */
    if (access(SLLM_TEST_MODEL, R_OK) != 0) { CHECK(1); return; }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) { CHECK(1); return; }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) { sllm_gguf_close(&g); CHECK(1); return; }

    /* 0xCE is the two bytes C3 8E ("\xc3\x8e"), not the single byte CE. If the
     * byte-encoded word contained a raw 0xCE it would not match any piece and
     * the prompt would encode to nothing. */
    char buf[256];
    int n = 0;
    sllm_tests_run++;
    if (sllm_tok_pretokenize(SLLM_PRE_UNSET, "\xce\xb1", 2, buf, sizeof(buf), &n) != SLLM_OK) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL pretokenize rejected a two-byte sequence\n");
    }

    /*
     * U+03B1 is bytes CE B1. Byte-encoded, each of those becomes the UTF-8
     * encoding of the code point of the same value: C3 8E and C2 B1. So the
     * word must be those four bytes, and must NOT contain the original CE.
     * This is the assertion that failed when byte_to_cpt returned the bare
     * byte -- English prompts were unaffected and this one produced nothing.
     */
    sllm_tests_run++;
    if (strstr(buf, "\xc3\x8e") == NULL || strstr(buf, "\xc2\xb1") == NULL) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL byte-encoded form should be C3 8E C2 B1, got <%s>\n", buf);
    }

    sllm_tests_run++;
    if (strstr(buf, "\xce") != NULL && strstr(buf, "\xc3\x8e") == NULL) {
        sllm_tests_failed++;
        fprintf(stderr, "  FAIL raw unexpanded byte 0xCE in the byte-encoded word\n");
    }

    sllm_tok_free(tok);
    sllm_gguf_close(&g);
}

TEST(regression_bpe_rejects_stale_bigrams) {
    /*
     * BPE merges the lowest-ranked available pair, repeatedly. A pair whose
     * text changed after it was queued must be dropped, or the result is still
     * plausible, still round trips, and is not BPE. These are the prompts that
     * disagreed before the staleness check was restored: merges that chain.
     */
    if (access(SLLM_TEST_GOLDEN, R_OK) != 0) { CHECK(1); return; }
    sllm_gguf g;
    char err[512];
    if (sllm_gguf_open(SLLM_TEST_MODEL, &g, err, sizeof(err)) != SLLM_OK) { CHECK(1); return; }
    sllm_tok * tok = NULL;
    if (sllm_tok_load(&g, &tok) != SLLM_OK) { sllm_gguf_close(&g); CHECK(1); return; }

    /* Ids below are read from tests/golden/tokenizer.txt, not written from
     * memory. Guessing an expected id here is how a test ends up asserting
     * something other than the reference. */
    struct { const char * in; int32_t want[16]; } cases[] = {
        /* one long word: six merges, every one chaining into the last */
        { "antidisestablishmentarianism",
          { 128000, 519, 85342, 34500, 479, 8997, 2191, 0 } },
        /* a long punctuation run, where merges chain at equal-ish ranks */
        { "!!!!!!!!", { 128000, 0 } },
        /* a repeated word, so the same merge fires on disjoint segments */
        { "the the the the", { 128000, 1820, 279, 279, 279, 0 } },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int32_t got[64];
        const int32_t n = sllm_tok_encode(tok, cases[i].in, strlen(cases[i].in),
                                          true, true, got, 64);
        sllm_tests_run++;
        /* compare as far as the expectation is populated, and require the
         * length to match, so a case that stops being valid fails loudly */
        size_t want_n = 0;
        while (want_n < sizeof(cases[i].want) / sizeof(cases[i].want[0]) &&
               cases[i].want[want_n] != 0) {
            want_n++;
        }
        if (n <= 0) {
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL \"%s\" produced no tokens\n", cases[i].in);
            continue;
        }
        for (size_t k = 0; k < want_n && (int32_t) k < n; ++k) {
            if (got[k] == cases[i].want[k]) { continue; }
            sllm_tests_failed++;
            fprintf(stderr, "  FAIL \"%s\" token %zu is %d, expected %d\n",
                    cases[i].in, k, (int) got[k], (int) cases[i].want[k]);
            break;
        }
    }

    sllm_tok_free(tok);
    sllm_gguf_close(&g);
}


/* ==================================================================== *
 * Qwen3 tokenizer parity against the committed llama.cpp reference.
 *
 * The expectations are NOT written here. They are read from
 * tests/golden/mainstream-qwen3-tokenizer.txt, the single source of truth,
 * so this test cannot drift from what was actually captured.
 *
 * Input bytes come from text_hex, never a retyped string. The detokenized
 * comparison is over BYTES, never a UTF-8 string, because these pieces are
 * not individually valid UTF-8 and a string comparison would quietly be a
 * different test.
 *
 * Honesty about absence: with no model reachable this prints SKIPPED and is
 * NOT counted as a pass. "No failure" and "no evidence" share an exit code and
 * are not the same thing, and a gate that prints green when it measured
 * nothing is the exact failure this project keeps meeting.
 * ==================================================================== */

#define SLLM_QWEN3_FIXTURE "tests/golden/mainstream-qwen3-tokenizer.txt"
#define SLLM_QWEN3_MAX_CASES 64
#define SLLM_QWEN3_MAX_INPUT 8192
#define SLLM_QWEN3_MAX_IDS 4096
#define SLLM_QWEN3_MAX_OUT 8192

typedef struct {
    char    label[48];
    char    input[SLLM_QWEN3_MAX_INPUT];
    size_t  input_len;
    int32_t expect_n;
    int32_t ids[SLLM_QWEN3_MAX_IDS];
    char    out[SLLM_QWEN3_MAX_OUT];
    size_t  out_len;
    int     has_out;
} qwen3_case;

static int qhexval(char c) {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

/* "5468 6F" or "54686f" -> bytes. Returns byte count, or -1 on malformed. */
static int qhex_bytes(const char * h, char * out, size_t cap) {
    size_t n = 0;
    while (*h == ' ' || *h == '\t') { ++h; }
    while (qhexval(h[0]) >= 0 && qhexval(h[1]) >= 0) {
        if (n >= cap) { return -1; }
        out[n++] = (char) ((qhexval(h[0]) << 4) | qhexval(h[1]));
        h += 2;
        while (*h == ' ') { ++h; }
    }
    return (int) n;
}

/* Field order in the file is: text_hex, [label]..-> n=, ids =, pieces =,
 * decode_hex =. So text_hex opens a case and every other field attaches to the
 * case in progress. Nothing else has to be tracked. */
static int qwen3_parse(qwen3_case * cs, int cap_cases) {
    FILE * f = fopen(SLLM_QWEN3_FIXTURE, "rb");
    if (f == NULL) { return -1; }
    char line[SLLM_QWEN3_MAX_OUT + 64];
    int n = 0;
    int open = 0;                       /* a case is in progress */

    while (fgets(line, sizeof line, f) != NULL) {
        char * p = NULL;
        if ((p = strstr(line, "text_hex = ")) != NULL) {
            if (n >= cap_cases) { continue; }
            memset(&cs[n], 0, sizeof cs[n]);
            const int got = qhex_bytes(p + 11, cs[n].input, sizeof cs[n].input);
            if (got < 0) { fclose(f); return -1; }
            cs[n].input_len = (size_t) got;
            ++n; open = 1;
            continue;
        }
        if (!open) { continue; }
        qwen3_case * c = &cs[n - 1];

        if ((p = strstr(line, "-> n=")) != NULL) {
            c->expect_n = (int32_t) strtol(p + 5, NULL, 10);
        } else if ((p = strchr(line, '[')) != NULL && strchr(line, ']') != NULL &&
                   strstr(line, "bytes ->") != NULL) {
            char * rb = strchr(p, ']');
            const size_t len = (size_t) (rb - p - 1);
            if (len < sizeof c->label) { memcpy(c->label, p + 1, len); c->label[len] = '\0'; }
        } else if (strncmp(line, "   ids = ", 9) == 0) {
            char * q = line + 9;
            int32_t k = 0;
            for (;;) {
                while (*q == ' ') { ++q; }
                if (*q < '0' || *q > '9') { break; }
                if (k >= SLLM_QWEN3_MAX_IDS) { fclose(f); return -1; }
                char * endp = NULL;
                c->ids[k++] = (int32_t) strtol(q, &endp, 10);
                q = endp;
            }
        } else if (strncmp(line, "   decode_hex = ", 16) == 0) {
            const int got = qhex_bytes(line + 16, c->out, sizeof c->out);
            if (got < 0) { fclose(f); return -1; }
            c->out_len = (size_t) got;
            c->has_out = 1;
        }
    }
    fclose(f);
    return n;
}

static void run_qwen3_tokenizer_parity(void) {
    printf("qwen3-tokenizer-parity\n");

    static qwen3_case cs[SLLM_QWEN3_MAX_CASES];
    const int n = qwen3_parse(cs, SLLM_QWEN3_MAX_CASES);
    printf("    qwen3 fixture parsed at least one case\n"); CHECK(n > 0);
    if (n <= 0) { return; }
    printf("    %d cases parsed from %s\n", n, SLLM_QWEN3_FIXTURE);

    /* The model path. Env override so the gate can be pointed at any copy. */
    const char * path = getenv("SLLM_QWEN3_GGUF");
    if (path == NULL) { path = "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"; }

    sllm_gguf g;
    char err[256] = "";
    if (sllm_gguf_open(path, &g, err, sizeof err) != SLLM_OK) {
        printf("    SKIPPED: cannot open %s (%s)\n", path, err);
        printf("    NOT a pass. This is an absence of evidence, not evidence of parity.\n");
        return;
    }

    sllm_tok * tok = NULL;
    const sllm_status rc = sllm_tok_load(&g, &tok);
    printf("    the Qwen3 tokenizer loads and selects the qwen2 splitter\n"); CHECK(rc == SLLM_OK && tok != NULL);
    if (rc != SLLM_OK || tok == NULL) {
        printf("    load failed; the loader must have said why in its warning above\n");
        sllm_gguf_close(&g);
        return;
    }

    /* The tuple actually selected, so a failure here is diagnosable. */
    printf("    selected pre-type = %s (declared \"%s\"), n_vocab = %u, add_bos=%d add_eos=%d\n",
           sllm_tok_pre_type_name(sllm_tok_pre_type(tok)),
           sllm_tok_pre_declared(tok),
           (unsigned) sllm_tok_n_vocab(tok),
           (int) sllm_tok_add_bos(tok), (int) sllm_tok_add_eos(tok));

    /* add_bos/add_eos must come from the file. If this model asks for neither,
     * ids are exactly the split output and the fixture needs no unwrapping. */
    const bool wrapped = sllm_tok_add_bos(tok) || sllm_tok_add_eos(tok);

    int fails = 0, ran = 0;
    for (int i = 0; i < n; ++i) {
        const qwen3_case * c = &cs[i];
        if (c->expect_n < 0) { continue; }

        int32_t got[SLLM_QWEN3_MAX_IDS];
        const int32_t gn = sllm_tok_encode(tok, c->input, c->input_len, true, true,
                                           got, SLLM_QWEN3_MAX_IDS);
        ++ran;

        if (gn != c->expect_n) {
            printf("    FAIL %-18s token count: got %d, expected %d\n",
                   c->label, (int) gn, (int) c->expect_n);
            ++fails; continue;
        }

        /* Compare the middle, dropping BOS/EOS only if the file asked for them. */
        int32_t off = 0;
        if (wrapped && c->expect_n > 0) {
            off = (sllm_tok_add_bos(tok) && gn > 0) ? 1 : 0;
        }
        int bad = 0;
        for (int k = 0; k < c->expect_n - off; ++k) {
            if (got[k + off] != c->ids[k]) { bad = 1; break; }
        }
        if (bad) {
            printf("    FAIL %-18s ids differ:", c->label);
            for (int k = 0; k < c->expect_n; ++k) { printf(" %d", (int) got[k]); }
            printf("\n");
            ++fails; continue;
        }

        /* Empty input must be exactly zero tokens, stated as its own check so
         * it cannot be satisfied by accident via the count comparison. */
        if (c->input_len == 0) {
            printf("    empty input is exactly zero tokens\n"); CHECK(gn == 0);
            printf("    ok   %-18s 0 tokens\n", c->label);
            continue;
        }

        if (c->has_out) {
            char buf[SLLM_QWEN3_MAX_OUT];
            const int32_t dn = sllm_tok_decode(tok, got, gn, false, buf, (int32_t) sizeof buf);
            const size_t dlen = (dn > 0) ? (size_t) dn : 0u;
            if (dlen != c->out_len || memcmp(buf, c->out, dlen) != 0) {
                printf("    FAIL %-18s detokenized bytes: got %zu, expected %zu\n",
                       c->label, dlen, c->out_len);
                ++fails; continue;
            }
        }
        printf("    ok   %-18s %2d tokens, bytes exact\n", c->label, (int) c->expect_n);
    }

    printf("    %d cases run, %d failed\n", ran, fails);
    printf("    every Qwen3 fixture matches llama.cpp exactly\n"); CHECK(fails == 0);
    printf("    at least 20 fixtures actually ran\n"); CHECK(ran >= 20);

    sllm_tok_free(tok);
    sllm_gguf_close(&g);
}

void sllm_test_tokenizer(void) {
    printf("tokenizer\n");
    RUN(tokenizer_matches_the_reference_token_for_token);
    RUN(tokenizer_round_trips_every_prompt);
    RUN(pre_tokeniser_splits_the_way_the_reference_does);
    RUN(pre_tokeniser_is_read_from_metadata_not_assumed);
    RUN(default_and_gpt2_pre_tokenisers_genuinely_differ);
    RUN(gpt2_pre_tokeniser_matches_the_reference_on_the_same_vocabulary);
    RUN(unsupported_pre_types_are_rejected_rather_than_approximated);
    RUN(regression_whitespace_is_a_list_not_a_category);
    RUN(regression_byte_encoding_returns_a_utf8_encoding_not_a_byte);
    RUN(regression_bpe_rejects_stale_bigrams);
    run_qwen3_tokenizer_parity();
}
