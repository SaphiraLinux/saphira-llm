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

/* Compare one fixture record against what we produce for the same text. */
static void check_record(const record * rec, const sllm_tok * tok) {
    int32_t got[4096];
    const int32_t n = sllm_tok_encode(tok, rec->text, rec->text_len,
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
        const sllm_status rc = sllm_tok_pretokenize(tok, cases[i].in, strlen(cases[i].in),
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

void sllm_test_tokenizer(void) {
    printf("tokenizer\n");
    RUN(tokenizer_matches_the_reference_token_for_token);
    RUN(tokenizer_round_trips_every_prompt);
    RUN(pre_tokeniser_splits_the_way_the_reference_does);
}
