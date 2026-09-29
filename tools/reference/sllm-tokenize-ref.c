/*
 * sllm-tokenize-ref — reference tokenizer dumper for saphira-llm.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * THIS IS A REFERENCE TOOL. It is not part of the saphira-llm runtime, it is
 * not installed, and it is never shipped. It links against the *upstream*
 * BitNet-capable llama.cpp and records exactly how that implementation
 * tokenises, so the golden vectors in tests/golden/ come from the reference
 * and not from us.
 *
 * The prompt set is chosen to attack the pre-tokeniser rather than to be
 * representative prose. The acceptance model has no `tokenizer.ggml.pre` key,
 * so the reference falls back to its DEFAULT pre-type, which is four regexes
 * rather than the single canonical GPT-2 pattern, and splits punctuation and
 * long digit runs on its own. Punctuation, digits, contractions, whitespace
 * runs, newlines and non-ASCII are therefore not incidental coverage here;
 * they are the cases where a hand-written "standard GPT-2 BPE" diverges.
 *
 * Output: one record per prompt, with the token ids, the reference's own
 * rendered piece for each token, and an FNV-1a 64 over the ids.
 */

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct llama_model          llama_model;
typedef struct llama_vocab          llama_vocab;
typedef struct llama_model_params   llama_model_params;

static void die(const char * msg) {
    fprintf(stderr, "sllm-tokenize-ref: %s\n", msg);
    exit(1);
}

/*
 * Escape to unambiguous ASCII.
 *
 * Prompts and pieces contain raw bytes -- UTF-8 sequences, and byte-encoded
 * pieces that are not UTF-8 at all -- so a fixture written verbatim is not
 * line-oriented, is not diffable, and cannot be round-tripped through a text
 * tool without a decoding decision nobody wrote down. Everything outside
 * printable ASCII is written as \xNN.
 *
 * ASCII prompts stay readable, which is the point: the fixture is read by
 * people reviewing a change, not only by the test.
 */
static void escape(FILE * f, const char * s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = (unsigned char) s[i];
        if      (c == '\\') { fprintf(f, "\\\\"); }
        else if (c == '\n')  { fprintf(f, "\\n"); }
        else if (c == '\r')  { fprintf(f, "\\r"); }
        else if (c == '\t')  { fprintf(f, "\\t"); }
        else if (c < 0x20 || c >= 0x7F) { fprintf(f, "\\x%02x", c); }
        else                 { fputc(c, f); }
    }
}

static uint64_t fnv1a64(const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/*
 * The attack set. Written out rather than read from a file so that the
 * fixture is self-describing and the ordering cannot drift.
 *
 * \x escapes are used so the awkward cases (embedded newlines, tabs, NUL-free
 * control bytes) survive editing, review and diffing intact.
 */
typedef struct {
    const char * name;
    const char * text;
} ref_prompt;

static const ref_prompt prompts[] = {
    { "plain",        "The quick brown fox jumps over the lazy dog." },
    { "leading_sp",   "  leading spaces" },
    { "trailing_sp",  "trailing spaces  " },
    { "single_sp",    " one leading space" },
    { "nl",           "first line\nsecond line" },
    { "nl_blank",     "a\n\n\nb" },
    { "nl_crlf",      "a\r\nb" },
    { "tab",          "a\tb" },
    { "sp_run",       "a   b" },
    { "sp_run_end",   "a   " },
    { "sp_run_start", "   a" },
    { "punct_light",  "Hello, world!" },
    { "punct_full",   "Hello, world! (Test) [x] {y}" },
    { "punct_math",   "a + b = c" },
    { "punct_dollar", "costs $5 or $10" },
    { "punct_caret",  "a<b>c^d~e|f" },
    { "punct_gt",     "x => y <= z" },
    { "punct_run",    "!!!!!!???..." },
    { "digit_1",      "7" },
    { "digit_3",      "123" },
    { "digit_4",      "1234" },
    { "digit_6",      "123456" },
    { "digit_7",      "1234567" },
    { "digit_10",     "1234567890" },
    { "digit_lead0",  "007" },
    { "digit_mixed",  "abc123def456" },
    { "digit_sent",   "In 2024, GDP was 3.5 billion." },
    { "apos_s",       "it's" },
    { "apos_t",       "don't" },
    { "apos_re",      "they're" },
    { "apos_ve",      "we've" },
    { "apos_m",       "I'm" },
    { "apos_ll",      "we'll" },
    { "apos_d",       "he'd" },
    { "apos_upper",   "IT'S DON'T THEY'RE" },
    { "apos_bare",    "the ' symbol" },
    { "unicode_acc",  "naive cafe resume" },
    { "unicode_2byte","café naïve" },
    { "unicode_3byte","日本語のテキスト" },
    { "unicode_4byte","emoji \xF0\x9F\x8E\x89" " here" },
    { "unicode_math", "\xCE\xA9 and \xCE\x91" },
    { "unicode_mix",  "caf\xC3\xA9" " 123 \xE6\x97\xA5\xE6\x9C\xAC" },
    { "mixed_hard",   "In 2024, the U.S. spent $3.5B on AI -- up 40%!" },
    { "capitol",      "The Capitol" },
    { "save",         "save" },
    { "one_char",     "a" },
    { "one_space",    " " },
    { "only_nl",      "\n" },
    { "digit_emoji",  "1\xF0\x9F\x8E\x89" },
    { "long_word",    "antidisestablishmentarianism" },
    { "repeat_word",  "the the the the" },
    { "code_ish",     "def f(x):\n    return x**2\n" },
    { "url_ish",      "https://example.com/a/b?c=d&e=f" },
    { "quotes",       "\"quoted\" and 'single'" },
    { "slash",        "a/b/c" },
    { "backslash",    "a\\b" },
    { "plus_plus",    "a++ + ++b" },
    { "star_star",    "x = a ** b" },
    { "at_amp",       "@user & more" },
    { "percent",      "50% of 100%" },
    { "colon_semi",   "key: value; other" },
    { "comma_run",    "1,000,000" },
    { "dash_run",     "a---b" },
    { "underscore",   "snake_case_name" },
    { "camel",        "camelCaseName" },
    { "dot_num",      "1.5.2.3" },
    { "trail_nl",     "text\n" },
    { "dbl_space_nl", "a  \n  b" },
    { "tab_after_sp", "a \t b" },
    { "nbsp_like",    "a\xc2" "\xa0" "b" },
    { "zero_width",   "a\xe2\x80\x8b" "b" },
    { "combining",    "e\xCC\x81" "cole" },
    { "cyrillic",     "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82" },
    { "greek_run",    "\xCE\xB1\xCE\xB2\xCE\xB3" },
    { "digit_letter",  "a1b2c3" },
    { "big_int",      "123456789012345678901234567890" },
    { "neg_num",      "-42 +17" },
    { "sp_punct_word","hello , world ! next" },
    { "caps_run",     "ABCDEFGHIJKLMNOP" },
    { "mixed_ws",     " \t\n mixed \n\t " },
};

int main(int argc, char ** argv) {
    if (argc < 2) { die("usage: sllm-tokenize-ref <model.gguf> [out.txt]"); }
    const char * model_path = argv[1];
    const char * out_path   = (argc > 2) ? argv[2] : "tests/golden/tokenizer.txt";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = true;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) { die("could not load model"); }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == NULL) { die("model has no vocab"); }

    FILE * f = fopen(out_path, "w");
    if (f == NULL) { die("could not open output"); }

    fprintf(f, "# saphira-llm tokenizer golden vectors\n");
    fprintf(f, "# Captured from the REFERENCE llama_tokenize(), never from us.\n");
    fprintf(f, "#\n");
    fprintf(f, "# The acceptance model has no tokenizer.ggml.pre key, so the reference\n");
    fprintf(f, "# uses its DEFAULT pre-type: FOUR regexes, not the single canonical\n");
    fprintf(f, "# GPT-2 pattern. Punctuation and digit runs are split separately. A\n");
    fprintf(f, "# hand-written standard GPT-2 BPE diverges on most of these prompts.\n");
    fprintf(f, "#\n");
    fprintf(f, "# add_special is 1: the model sets add_bos_token, so every sequence\n");
    fprintf(f, "# here begins with BOS. The renderer below is the reference's own\n");
    fprintf(f, "# llama_token_to_piece, so a decode mismatch is visible per token.\n");
    fprintf(f, "model %s\n", model_path);
    fprintf(f, "n_vocab %d\n", llama_vocab_n_tokens(vocab));
    fprintf(f, "n_prompts %zu\n\n", sizeof(prompts) / sizeof(prompts[0]));

    const int32_t n_prompts = (int32_t) (sizeof(prompts) / sizeof(prompts[0]));
    int failures = 0;

    for (int32_t pi = 0; pi < n_prompts; ++pi) {
        const char * text = prompts[pi].text;
        const int32_t text_len = (int32_t) strlen(text);

        /* Two passes: size, then fill. add_special = true, parse_special = true
         * so that a literal <|bos|> in a prompt is honoured exactly as the
         * reference honours it. */
        int32_t need = -llama_tokenize(vocab, text, text_len, NULL, 0, true, true);
        if (need < 0) { need = -need; }
        if (need < 1) {
            fprintf(stderr, "sllm-tokenize-ref: tokenize failed for %s\n", prompts[pi].name);
            failures++;
            continue;
        }

        llama_token * toks = calloc((size_t) need, sizeof(llama_token));
        if (toks == NULL) { die("out of memory"); }
        int32_t got = llama_tokenize(vocab, text, text_len, toks, need, true, true);
        if (got < 0) { die("second tokenize pass failed"); }

        fprintf(f, "prompt %s\n", prompts[pi].name);
        fprintf(f, "  bytes %zu\n", strlen(text));
        /* The text itself, escaped, so the fixture drives the test rather than
         * the test carrying a second copy of the prompt list that can drift. */
        fprintf(f, "  text ");
        escape(f, text, (size_t) text_len);
        fprintf(f, "\n");
        fprintf(f, "  n %d\n", got);
        fprintf(f, "  fnv1a64 %016llx\n",
                (unsigned long long) fnv1a64(toks, (size_t) got * sizeof(llama_token)));
        fprintf(f, "  ids");
        for (int32_t i = 0; i < got; ++i) { fprintf(f, " %d", (int) toks[i]); }
        fprintf(f, "\n");

        /* The reference's own rendering of each token, which is what decode
         * must reproduce. control=false: special tokens are not rendered as
         * their readable form, matching an ordinary detokenisation. */
        fprintf(f, "  pieces");
        for (int32_t i = 0; i < got; ++i) {
            char buf[256];
            int32_t len = llama_token_to_piece(vocab, toks[i], buf, (int32_t) sizeof(buf), 0, false);
            if (len < 0) { len = 0; }
            buf[len < (int32_t) sizeof(buf) ? len : (int32_t) sizeof(buf) - 1] = '\0';
            fprintf(f, " <");
            escape(f, buf, (size_t) len);
            fprintf(f, ">");
        }
        fprintf(f, "\n\n");

        free(toks);
    }

    fclose(f);
    llama_model_free(model);
    llama_backend_free();

    if (failures) { die("one or more prompts failed to tokenize"); }
    return 0;
}
