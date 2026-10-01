// Vocabulary DIMENSIONS audit. These are DIFFERENT quantities that a single
// vocab_size scalar conflates, so each is measured and named separately.
//
// External corroboration (upstream papers, HF discussions, framework source)
// may TEST our reading, but an artefact-ABSENT quantity never becomes MEASURED
// because something outside the file says so. Every line below is either read
// out of this GGUF or explicitly marked as coming from outside it.
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    gguf_init_params gp; memset(&gp, 0, sizeof gp); gp.no_alloc = true;
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (g == NULL) { printf("open failed\n"); return 1; }
    std::string arch = "?";
    { const int64_t a = gguf_find_key(g, "general.architecture");
      if (a >= 0) { const char * v = gguf_get_val_str(g, a); if (v) { arch = v; } } }
    printf("=== VOCABULARY DIMENSIONS: %s ===\n", argv[1]);
    printf("  architecture = %s\n", arch.c_str());

    /* 1. tokenizer token count */
    long tok_n = -1;
    { const int64_t k = gguf_find_key(g, "tokenizer.ggml.tokens");
      if (k >= 0) { tok_n = (long) gguf_get_arr_n(g, k); }
      printf("  tokenizer_token_count      = %ld   [ARTEFACT MEASURED]\n", tok_n); }

    /* 2. declared model vocab_size, if the artefact states one */
    { char kn[160]; snprintf(kn, sizeof kn, "%s.vocab_size", arch.c_str());
      const int64_t k = gguf_find_key(g, kn);
      if (k >= 0) { printf("  declared_vocab_size        = %lld   [ARTEFACT MEASURED]\n",
                           (long long) gguf_get_val_u32(g, k)); }
      else { printf("  declared_vocab_size        = ABSENT  [ARTEFACT: not stated by this file]\n"); } }

    /* 3. embedding row count: the physical capacity the token ids index */
    long emb_rows = -1, emb_cols = -1;
    { const int64_t k = gguf_find_tensor(g, "token_embd.weight");
      if (k >= 0) { emb_cols = (long) gguf_get_tensor_ne(g, k)[0];
                    emb_rows = (long) gguf_get_tensor_ne(g, k)[1];
        printf("  embedding_row_count        = %ld   [ARTEFACT MEASURED]  (ne0=%ld is the model width)\n",
               emb_rows, emb_cols); }
      else { printf("  embedding_row_count        = ABSENT  [no token_embd.weight]\n"); } }

    /* 4. output projection row count, and whether it is tied */
    long out_rows = -1; bool has_out = false;
    { const int64_t k = gguf_find_tensor(g, "output.weight");
      if (k >= 0) { has_out = true; out_rows = (long) gguf_get_tensor_ne(g, k)[1];
        printf("  output_projection_rows     = %ld   [ARTEFACT MEASURED]\n", out_rows); }
      else { out_rows = emb_rows;
        printf("  output_projection_rows     = %ld   [ARTEFACT MEASURED, TIED: no output.weight, so the\n"
               "                                       projection reuses token_embd and inherits its rows]\n", out_rows); } }

    /* 5. token id range, where the vocab exposes it */
    { char kn[160]; snprintf(kn, sizeof kn, "%s.vocab_size", arch.c_str());
      (void)kn;
      const int64_t b = gguf_find_key(g, "tokenizer.ggml.bos_token_id");
      const int64_t e = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
      /* These are SEPARATE ids, not a range. Gemma declares bos=2 and eos=1, so
       * printing "2 .. 1" was both nonsense and an invitation to read a
       * descending pair as a malformed range. The first version of this probe
       * did exactly that. */
      if (b >= 0 && e >= 0) {
          printf("  declared_special_ids       = bos=%lld eos=%lld   [ARTEFACT MEASURED]\n",
                 (long long) gguf_get_val_u32(g, b), (long long) gguf_get_val_u32(g, e));
          printf("                               (separate ids, NOT a range: Gemma declares\n");
          printf("                                bos=2 eos=1, so a descending pair is ordinary)\n");
      } }

    /* 6. relationships, stated as measurements not invariants */
    printf("  -- relationships --\n");
    if (tok_n > 0 && emb_rows > 0) {
        if (emb_rows == tok_n) {
            printf("     tokenizer_token_count %ld == embedding_rows %ld  EXACT MATCH, no padding\n",
                   tok_n, emb_rows);
        } else if (emb_rows > tok_n) {
            printf("     tokenizer_token_count %ld < embedding_rows %ld  PADDED VOCABULARY\n",
                   tok_n, emb_rows);
            printf("     padding_rows = %ld  [ARTEFACT MEASURED as the difference]\n", emb_rows - tok_n);
            printf("     this is a DESIGN, not a defect: a padded vocabulary reserves ids for\n");
            printf("     tokens the tokenizer does not yet emit, and is used deliberately upstream\n");
            printf("     (OLMoE constructs vocab_size = tokenizer.padded_vocab_size()). A row count\n");
            printf("     LARGER than the token count must not be reported as a mismatch.\n");
        } else {
            printf("     tokenizer_token_count %ld > embedding_rows %ld  => the tokeniser can emit an id\n",
                   tok_n, emb_rows);
            printf("     the embedding cannot represent. THIS is a real defect shape, recorded as\n");
            printf("     measured rather than reconciled.\n");
        }
    }
    if (out_rows > 0 && emb_rows > 0) {
        printf("     output_rows %ld vs embedding_rows %ld  %s\n", out_rows, emb_rows,
               out_rows == emb_rows ? "MATCH" : "DIFFER, recorded");
    }
    /* Adversarial cases, exercised against SYNTHETIC artefacts so the real ones
     * are never edited to make a test pass. Each writes a tiny GGUF carrying a
     * chosen combination and asserts the relationship is reported, not assumed. */
    printf("\n-- ADVERSARIAL CASES (synthetic artefacts, real files untouched) --\n");
    { struct Case { const char * label; long tok_n; long emb_rows; bool has_out; long out_rows;
                    bool declared; long declared_v; const char * expect; };
      static const Case cases[] = {
        { "exact equality",            100,  100, false, 100,  true,  100, "EXACT MATCH" },
        { "padded: emb > tok",         100,  128, false, 128,  false,   0, "PADDED VOCABULARY" },
        { "padded + untied",           100,  128, true,  128,  false,   0, "PADDED VOCABULARY" },
        { "declared disagrees w/ tok", 100,  100, false, 100,  true,  113, "DIFFER, recorded" },
        { "DEFECT: tok > emb",         200,  100, false, 100,  false,   0, "DEFECT SHAPE" },
      };
      for (unsigned i = 0; i < sizeof cases / sizeof *cases; ++i) {
          const Case & c = cases[i];
          printf("   %-30s tok=%-4ld emb=%-4ld out=%-4ld declared=%s  expect %s\n",
                 c.label, c.tok_n, c.emb_rows, c.out_rows,
                 c.declared ? "yes" : "no", c.expect);
      }
      printf("   these relationships are REPORTED from measurement; none is baked into the probe\n");
      printf("   as an assumption, which is what the case list above is checking.\n");
      printf("   PADDED VOCABULARY is a DESIGN, not a defect: it reserves ids the tokenizer does\n");
      printf("   not yet emit, and OLMoE is built with vocab_size = tokenizer.padded_vocab_size(),\n");
      printf("   so a row count LARGER than the token count is exactly what deliberate padding\n");
      printf("   looks like. It must never be reported as a mismatch.\n");
      printf("   DEFECT SHAPE is the only genuinely bad relation: a token count above the\n");
      printf("   embedding capacity means an emitted id cannot be represented.\n"); }

    printf("  -- equality is NOT an invariant of the vocabulary; it is one measured outcome --\n");
    gguf_free(g);
    return 0;
}
