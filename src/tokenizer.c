/*
 * BPE tokenizer, matching the pinned reference exactly.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * ---------------------------------------------------------------------------
 * Read this before changing anything in here.
 *
 * This is not "a BPE tokenizer". It is a reimplementation of what one specific
 * pinned build does, and the acceptance model's tokenizer is configured in a
 * way that separates "a correct GPT-2 BPE" from "the reference" within the
 * first ten tokens of ordinary text.
 *
 * The model has no `tokenizer.ggml.pre` key. The reference therefore falls back
 * to its DEFAULT pre-type, logs "GENERATION QUALITY WILL BE DEGRADED", and that
 * default is FOUR split passes rather than the single canonical GPT-2 pattern:
 *
 *   1.  [\p{P}\$\+<=>\^~\|]+
 *   2.  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
 *   3.  \p{N}+
 *   4.  [0-9][0-9][0-9]
 *
 * Applied in sequence, each re-splitting the previous result. Pass 2 is a
 * hand-written splitter in the reference; passes 1, 3 and 4 run through
 * std::regex over a "collapsed" text in which every non-ASCII code point is
 * replaced by a single byte naming its general category.
 *
 * The consequences are the reason the fixture exists:
 *
 *   "1234567890"  ->  123 456 789 0     pass 4, and only pass 4
 *   "they're"     ->  they ' re         pass 2's apostrophe rule is anchored
 *   "%!"          ->  %!                pass 1 eats punctuation runs
 *   "a + b"       ->  a ' ' + ' b'      a bare space is its own word
 *
 * A textbook GPT-2 BPE gives different tokens for all four. Matching the
 * reference is the gate, so the reference is the specification, including the
 * parts that look like mistakes.
 * ---------------------------------------------------------------------------
 */

#include "saphira_llm/tokenizer.h"
#include "saphira_llm/log.h"
#include "saphira_llm/status.h"
#include "saphira_llm/unicode.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* small utilities                                                      */
/* ------------------------------------------------------------------ */

#define FNV_SEED 1469598103934665603ULL

static uint64_t fnv1a(const char * s, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) {
        h ^= (unsigned char) s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static size_t next_pow2(size_t n) {
    size_t p = 16;
    while (p < n && p < (SIZE_MAX / 2)) {
        p <<= 1;
    }
    return p;
}

/* Bump allocator. Everything the tokenizer owns lives in two arenas released
 * by two frees, so there is no ownership question to get wrong. */
typedef struct arena {
    char  * base;
    size_t  used;
    size_t  cap;
    bool    oom;
} arena;

/* Everything stored is either a char or a uint32_t, so 4 is the only alignment
 * that is actually needed. Rounding to 16 doubled the merge pool: there are
 * 280147 merges, each of which allocates two short strings, and 16 bytes of
 * slack on each is several megabytes of pure padding. */
#define ARENA_ALIGN 4u
#define ARENA_ROUND(n) (((n) + (ARENA_ALIGN - 1u)) & ~(size_t) (ARENA_ALIGN - 1u))

static void * arena_alloc(arena * a, size_t n) {
    n = ARENA_ROUND(n);
    if (a->oom || a->base == NULL || n > a->cap - a->used) {
        a->oom = true;
        return NULL;
    }
    void * p = a->base + a->used;
    a->used += n;
    return p;
}

/* ------------------------------------------------------------------ */
/* string -> int32 hash table                                           */
/* ------------------------------------------------------------------ */

typedef struct { const char * key; uint32_t hash; int32_t val; } slot_str;

typedef struct { slot_str * slots; size_t mask; } map_str;

static bool map_str_init(map_str * m, size_t want) {
    const size_t cap = next_pow2(want * 2);
    m->slots = (slot_str *) calloc(cap, sizeof(slot_str));
    if (m->slots == NULL) { return false; }
    m->mask = cap - 1;
    return true;
}

static void map_str_free(map_str * m) { free(m->slots); m->slots = NULL; }

/*
 * Insert only if absent, so the first insert wins. The reference fills its
 * token map with emplace(), which likewise keeps the earliest entry for a
 * duplicated piece; letting a later insert win would change which id a piece
 * maps to, and the vocab here contains duplicates.
 */
static void map_str_put_first(map_str * m, const char * key, int32_t val) {
    const uint32_t h = (uint32_t) fnv1a(key, strlen(key), FNV_SEED);
    size_t i = (size_t) h & m->mask;
    while (m->slots[i].key != NULL) {
        if (m->slots[i].hash == h && strcmp(m->slots[i].key, key) == 0) {
            return;
        }
        i = (i + 1) & m->mask;
    }
    m->slots[i].key  = key;
    m->slots[i].hash = h;
    m->slots[i].val  = val;
}

static int32_t map_str_get_n(const map_str * m, const char * key, size_t n) {
    if (m->slots == NULL) { return -1; }
    const uint32_t h = (uint32_t) fnv1a(key, n, FNV_SEED);
    size_t i = (size_t) h & m->mask;
    while (m->slots[i].key != NULL) {
        if (m->slots[i].hash == h && strlen(m->slots[i].key) == n &&
            memcmp(m->slots[i].key, key, n) == 0) {
            return m->slots[i].val;
        }
        i = (i + 1) & m->mask;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* (left, right) -> rank hash table                                     */
/* ------------------------------------------------------------------ */

/*
 * The reference keys merges by the PAIR and splits each stored merge at the
 * first space from index 1:
 *
 *   "Ġ ĠĠĠ"  ->  first "Ġ", second "ĠĠĠ"
 *
 * Both details matter. Splitting from index 0 yields an empty `first` for any
 * merge whose left side is empty, and keying on the concatenation would make
 * ("ab","c") collide with ("a","bc").
 */
typedef struct { const char * l; const char * r; uint32_t hash; int32_t rank; } slot_pair;

typedef struct { slot_pair * slots; size_t mask; } map_pair;

static bool map_pair_init(map_pair * m, size_t want) {
    const size_t cap = next_pow2(want * 2);
    m->slots = (slot_pair *) calloc(cap, sizeof(slot_pair));
    if (m->slots == NULL) { return false; }
    m->mask = cap - 1;
    return true;
}

static void map_pair_free(map_pair * m) { free(m->slots); m->slots = NULL; }

static uint32_t pair_hash(const char * l, size_t ln, const char * r, size_t rn) {
    const uint64_t a = fnv1a(l, ln, FNV_SEED);
    const uint64_t b = fnv1a(r, rn, FNV_SEED);
    return (uint32_t) (a ^ (b * 1099511628211ULL));
}

static void map_pair_put_first(map_pair * m, const char * l, const char * r, int32_t rank) {
    const uint32_t h = pair_hash(l, strlen(l), r, strlen(r));
    size_t i = (size_t) h & m->mask;
    while (m->slots[i].l != NULL) {
        if (m->slots[i].hash == h &&
            strcmp(m->slots[i].l, l) == 0 && strcmp(m->slots[i].r, r) == 0) {
            return;
        }
        i = (i + 1) & m->mask;
    }
    m->slots[i].l    = l;
    m->slots[i].r    = r;
    m->slots[i].hash = h;
    m->slots[i].rank = rank;
}

static int32_t map_pair_get_n(const map_pair * m, const char * l, size_t ln,
                              const char * r, size_t rn) {
    if (m->slots == NULL) { return -1; }
    const uint32_t h = pair_hash(l, ln, r, rn);
    size_t i = (size_t) h & m->mask;
    while (m->slots[i].l != NULL) {
        if (m->slots[i].hash == h &&
            strlen(m->slots[i].l) == ln && memcmp(m->slots[i].l, l, ln) == 0 &&
            strlen(m->slots[i].r) == rn && memcmp(m->slots[i].r, r, rn) == 0) {
            return m->slots[i].rank;
        }
        i = (i + 1) & m->mask;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* growable uint32 vector                                               */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t * v; size_t n; size_t cap; bool oom; } u32vec;

static void uv_init(u32vec * x) { x->v = NULL; x->n = 0; x->cap = 0; x->oom = false; }
static void uv_free(u32vec * x) { free(x->v); x->v = NULL; x->n = 0; x->cap = 0; }

static void uv_push(u32vec * x, uint32_t v) {
    if (x->oom) { return; }
    if (x->n == x->cap) {
        const size_t nc = x->cap ? x->cap * 2 : 64;
        uint32_t * nv = (uint32_t *) realloc(x->v, nc * sizeof(uint32_t));
        if (nv == NULL) { x->oom = true; return; }
        x->v = nv;
        x->cap = nc;
    }
    x->v[x->n++] = v;
}

/* ------------------------------------------------------------------ */
/* the tokenizer                                                        */
/* ------------------------------------------------------------------ */

struct sllm_tok {
    arena     pool;        /* vocab, offsets, special text   */
    arena     merge_pool;  /* merge left/right halves        */

    char    * id_to_piece;  /* n_vocab NUL-terminated strings, packed */
    uint32_t* piece_off;    /* byte offset of each id's piece         */
    uint32_t  n_vocab;

    uint8_t  * special;     /* one flag per id                        */

    map_str   piece_to_id;
    map_pair  merge_rank;

    int32_t   bos;
    int32_t   eos;
    bool      add_bos;
    bool      add_eos;

    sllm_pre_type pre_type;
    char       pre_declared[64];   /* "" when the key is absent */

    char    * special_text; /* NUL-separated special piece strings */
};

static int32_t tok_piece_id(const sllm_tok * t, const char * s, size_t n) {
    return map_str_get_n(&t->piece_to_id, s, n);
}

/*
 * Resolve `tokenizer.ggml.pre` to a pre-tokeniser.
 *
 * The reference maps a long list of `pre` strings onto a smaller set of
 * pre-types, and treats an unrecognised value as a hard error rather than a
 * fallback. We reproduce that shape: a table of the names we accept, DEFAULT
 * when the key is absent, and a specific rejection naming the value otherwise.
 *
 * Only the two pre-types whose behaviour is fully understood are accepted.
 * Adding a name here without implementing its split passes would be the worst
 * possible outcome -- the model would load, tokenise, and be wrong in a way no
 * fixture covers. Unsupported is the honest answer and it is also the
 * reference's.
 */
static sllm_status resolve_pre_type(const sllm_gguf * g, sllm_pre_type * out,
                                    char * declared, size_t declared_cap) {
    const char * pre = NULL;
    const sllm_status rc = sllm_gguf_kv_str(g, "tokenizer.ggml.pre", &pre);

    if (rc == SLLM_ERR_KV_MISSING) {
        /* Absent is DEFAULT, and DEFAULT is not GPT-2. */
        *out = SLLM_PRE_UNSET;
        if (declared_cap > 0) { declared[0] = '\0'; }
        return SLLM_OK;
    }
    if (rc != SLLM_OK) {
        return rc;
    }

    snprintf(declared, declared_cap, "%s", pre);

    /* The reference groups many names onto GPT2. Only the names that describe
     * GPT-2's own tokeniser are listed; the vendor-specific aliases in the
     * reference are not, because a model carrying one of those has not been
     * shown to behave like GPT-2 here and we would be guessing. */
    static const char * const gpt2_names[] = {
        "gpt-2", "gpt2", "phi-2", "jina-es", "jina-de", "gigachat",
        "jina-v2-es", "jina-v2-de", "a.x-4.0", "mellum", "modern-bert",
        "jina-v1-en", "jina-v2-code", "roberta-bpe", "exaone4",
        NULL
    };
    for (size_t i = 0; gpt2_names[i] != NULL; ++i) {
        if (strcmp(pre, gpt2_names[i]) == 0) {
            *out = SLLM_PRE_GPT2;
            return SLLM_OK;
        }
    }
    if (strcmp(pre, "default") == 0) {
        *out = SLLM_PRE_UNSET;
        return SLLM_OK;
    }
    /* Qwen2 and Qwen3 both declare pre = "qwen2". */
    if (strcmp(pre, "qwen2") == 0) {
        *out = SLLM_PRE_QWEN2;
        return SLLM_OK;
    }

    *out = SLLM_PRE_UNSUPPORTED;
    return SLLM_ERR_UNSUPPORTED;
}

sllm_status sllm_tok_load(const sllm_gguf * g, sllm_tok ** out) {
    if (g == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    *out = NULL;

    /*
     * The tokenizer is selected from a TUPLE, not from one field.
     *
     * `tokenizer.ggml.model` says only which BYTE-LEVEL BPE family the vocab
     * uses, and on its own it is not sufficient: Qwen3-8B declares
     * model = "gpt2" while pre = "qwen2" and n_vocab = 151936. A backend keyed
     * on model alone would take the right family with the wrong parameters --
     * GPT-2's pre-tokeniser pattern and a 50257 vocab, applied to Qwen3
     * weights. That fails as a FALSE GREEN: ordinary prose tokenises correctly
     * and the error only shows up on spacing, punctuation and non-ASCII input.
     *
     * So the three legs are read together and all three must agree:
     *   model  the family, which is the ONLY thing this leg is allowed to say
     *   pre    the pre-tokeniser, which is what actually selects the splitter
     *   vocab   the size, which is validation evidence and never a shortcut
     *
     * `general.architecture` is deliberately not consulted anywhere in this
     * function. Architecture names what computes; a tokenizer is a property of
     * the artefact and must be read off the artefact.
     */
    const char * model = NULL;
    if (sllm_gguf_kv_str(g, "tokenizer.ggml.model", &model) != SLLM_OK) {
        sllm_log(SLLM_LOG_WARN, "no tokenizer.ggml.model declared; refusing to guess a family");
        return SLLM_ERR_UNSUPPORTED;
    }
    if (strcmp(model, "gpt2") != 0) {
        sllm_log(SLLM_LOG_WARN, "tokenizer.ggml.model = \"%s\" is not the BPE family this build "
                    "implements; refusing to tokenise rather than guess", model);
        return SLLM_ERR_UNSUPPORTED;   /* family leg, and it is not sufficient on its own */
    }

    sllm_pre_type pre_type = SLLM_PRE_UNSUPPORTED;
    char pre_declared[64] = "";
    {
        const sllm_status prc = resolve_pre_type(g, &pre_type, pre_declared, sizeof(pre_declared));
        if (prc != SLLM_OK) {
            /*
             * Name the value in the log rather than only in a status code: a
             * rejection the operator cannot act on is barely better than a
             * wrong answer, and the fix is to implement that pre-tokeniser or
             * convert the model.
             */
            sllm_log(SLLM_LOG_WARN, "tokenizer.ggml.pre = \"%s\" names a pre-tokeniser this build does "
                          "not implement; refusing to tokenise rather than guess", pre_declared);
            return prc;
        }
    }

    char * const * tokens = NULL;
    uint64_t n_tokens = 0;
    if (sllm_gguf_kv_str_array(g, "tokenizer.ggml.tokens", &tokens, &n_tokens) != SLLM_OK ||
        n_tokens == 0) {
        return SLLM_ERR_KV_MISSING;
    }

    /* The vocab-size leg. This VALIDATES rather than selects: a byte-level BPE
     * vocab cannot be smaller than one entry per byte value, so anything under
     * 256 means the alphabet is incomplete and every id derived from it would be
     * suspect. It is not used to pick a backend, and no specific size is
     * hardcoded for any pre-type, because that would be a filename-style
     * shortcut wearing a numeric disguise. */
    if (n_tokens < 256u) {
        sllm_log(SLLM_LOG_WARN, "tokenizer.ggml.tokens has only %llu entries; a byte-level BPE "
                    "needs at least one per byte value, so the alphabet is incomplete",
                 (unsigned long long) n_tokens);
        return SLLM_ERR_KV_MISSING;
    }

    /* Where the file states a vocab size, it must agree with the array we just
     * read. A disagreement means the two were written at different times and we
     * cannot say which one the model was trained against. */
    {
        uint32_t declared_n_vocab = 0;
        if (sllm_gguf_kv_u32(g, "qwen3.vocab_size", &declared_n_vocab) == SLLM_OK ||
            sllm_gguf_kv_u32(g, "llama.vocab_size", &declared_n_vocab) == SLLM_OK) {
            if ((uint64_t) declared_n_vocab != n_tokens) {
                sllm_log(SLLM_LOG_WARN, "vocab size metadata says %u but tokenizer.ggml.tokens has "
                            "%llu entries; refusing to pick a side", declared_n_vocab,
                         (unsigned long long) n_tokens);
                return SLLM_ERR_KV_MISSING;
            }
        }
    }

    char * const * merges = NULL;
    uint64_t n_merges = 0;
    (void) sllm_gguf_kv_str_array(g, "tokenizer.ggml.merges", &merges, &n_merges);

    const int32_t * token_type = NULL;
    uint64_t n_type = 0;
    (void) sllm_gguf_kv_i32_array(g, "tokenizer.ggml.token_type", &token_type, &n_type);

    sllm_tok * t = (sllm_tok *) calloc(1, sizeof(sllm_tok));
    if (t == NULL) {
        return SLLM_ERR_NOMEM;
    }
    t->bos = -1;
    t->eos = -1;
    t->n_vocab = (uint32_t) n_tokens;
    t->pre_type = pre_type;
    snprintf(t->pre_declared, sizeof(t->pre_declared), "%s", pre_declared);

    size_t piece_bytes = 0;
    size_t spec_need   = 1;
    for (uint64_t i = 0; i < n_tokens; ++i) {
        const size_t l = strlen(tokens[i]);
        piece_bytes += l + 1;
        const int32_t ty = (token_type != NULL && i < n_type) ? token_type[i] : 1;
        if (ty == 3 || ty == 4) {         /* CONTROL, USER_DEFINED */
            spec_need += l + 1;
        }
    }

    const size_t pool_guess = 4096
        + ARENA_ROUND(piece_bytes)
        + ARENA_ROUND(spec_need)
        + ARENA_ROUND((size_t) n_tokens * sizeof(uint32_t))
        + ARENA_ROUND((size_t) n_tokens);
    /* Two allocations per merge, each rounded exactly as arena_alloc rounds.
     * Budgeting the raw string length under-counted by 2x and every load failed
     * with ENOMEM, which is a confusing way to learn about a rounding constant. */
    size_t mpool_guess = 4096;
    for (uint64_t i = 0; i < n_merges; ++i) {
        const char * word = merges[i];
        const char * sp   = strchr(word + 1, ' ');
        if (sp == NULL) { continue; }
        mpool_guess += ARENA_ROUND((size_t) (sp - word) + 1);
        mpool_guess += ARENA_ROUND(strlen(sp + 1) + 1);
    }

    t->pool.base = (char *) malloc(pool_guess);
    t->pool.cap  = t->pool.base != NULL ? pool_guess : 0;
    t->merge_pool.base = (char *) malloc(mpool_guess);
    t->merge_pool.cap  = t->merge_pool.base != NULL ? mpool_guess : 0;

    if (t->pool.base == NULL || t->merge_pool.base == NULL ||
        !map_str_init(&t->piece_to_id, (size_t) n_tokens) ||
        !map_pair_init(&t->merge_rank, (size_t) (n_merges ? n_merges : 16))) {
        sllm_tok_free(t);
        return SLLM_ERR_NOMEM;
    }

    t->id_to_piece = (char *) arena_alloc(&t->pool, piece_bytes);
    t->piece_off   = (uint32_t *) arena_alloc(&t->pool, (size_t) n_tokens * sizeof(uint32_t));
    t->special     = (uint8_t *) arena_alloc(&t->pool, (size_t) n_tokens);
    t->special_text = (char *) arena_alloc(&t->pool, spec_need);
    if (t->id_to_piece == NULL || t->piece_off == NULL ||
        t->special == NULL || t->special_text == NULL) {
        sllm_tok_free(t);
        return SLLM_ERR_NOMEM;
    }

    char * cursor = t->id_to_piece;
    char * wspec  = t->special_text;
    for (uint64_t i = 0; i < n_tokens; ++i) {
        const size_t l = strlen(tokens[i]) + 1;
        t->piece_off[i] = (uint32_t) (cursor - t->id_to_piece);
        memcpy(cursor, tokens[i], l);
        map_str_put_first(&t->piece_to_id, cursor, (int32_t) i);
        cursor += l;

        const int32_t ty = (token_type != NULL && i < n_type) ? token_type[i] : 1;
        t->special[i] = (ty == 3 || ty == 4) ? 1u : 0u;
        if (t->special[i]) {
            memcpy(wspec, tokens[i], l);
            wspec += l;
        }
    }

    /* Merges, split at the first space from index 1. A merge with no space is
     * skipped, as the reference's substrings stay empty for those. */
    for (uint64_t i = 0; i < n_merges; ++i) {
        const char * word = merges[i];
        const char * sp   = strchr(word + 1, ' ');
        if (sp == NULL) {
            continue;
        }
        const size_t ll = (size_t) (sp - word);
        const size_t rl = strlen(sp + 1);
        char * l = (char *) arena_alloc(&t->merge_pool, ll + 1);
        char * r = (char *) arena_alloc(&t->merge_pool, rl + 1);
        if (l == NULL || r == NULL) {
            sllm_tok_free(t);
            return SLLM_ERR_NOMEM;
        }
        memcpy(l, word, ll);   l[ll] = '\0';
        memcpy(r, sp + 1, rl); r[rl] = '\0';
        map_pair_put_first(&t->merge_rank, l, r, (int32_t) i);
    }

    uint32_t u32 = 0;
    if (sllm_gguf_kv_u32(g, "tokenizer.ggml.bos_token_id", &u32) == SLLM_OK) {
        t->bos = (int32_t) u32;
    }
    if (sllm_gguf_kv_u32(g, "tokenizer.ggml.eos_token_id", &u32) == SLLM_OK) {
        t->eos = (int32_t) u32;
    }
    const sllm_gguf_kv * k = sllm_gguf_find_kv(g, "tokenizer.ggml.add_bos_token");
    if (k != NULL && k->type == SLLM_VT_BOOL) {
        t->add_bos = *(const bool *) k->data;
    }
    k = sllm_gguf_find_kv(g, "tokenizer.ggml.add_eos_token");
    if (k != NULL && k->type == SLLM_VT_BOOL) {
        t->add_eos = *(const bool *) k->data;
    }

    *out = t;
    return SLLM_OK;
}

void sllm_tok_free(sllm_tok * t) {
    if (t == NULL) { return; }
    map_str_free(&t->piece_to_id);
    map_pair_free(&t->merge_rank);
    free(t->pool.base);
    free(t->merge_pool.base);
    free(t);
}

uint32_t sllm_tok_n_vocab(const sllm_tok * t) { return t != NULL ? t->n_vocab : 0; }
int32_t  sllm_tok_bos(const sllm_tok * t)     { return t != NULL ? t->bos : -1; }
int32_t  sllm_tok_eos(const sllm_tok * t)     { return t != NULL ? t->eos : -1; }
bool     sllm_tok_add_bos(const sllm_tok * t) { return t != NULL && t->add_bos; }
bool     sllm_tok_add_eos(const sllm_tok * t) { return t != NULL && t->add_eos; }

sllm_pre_type sllm_tok_pre_type(const sllm_tok * t) {
    return t != NULL ? t->pre_type : SLLM_PRE_UNSUPPORTED;
}

const char * sllm_tok_pre_type_name(sllm_pre_type pre) {
    switch (pre) {
        case SLLM_PRE_UNSET:       return "default";
        case SLLM_PRE_GPT2:        return "gpt-2";
        case SLLM_PRE_QWEN2:       return "qwen2";
        case SLLM_PRE_UNSUPPORTED: return "unsupported";
        default:                   return "unknown";
    }
}

const char * sllm_tok_pre_declared(const sllm_tok * t) {
    return t != NULL ? t->pre_declared : "";
}

bool sllm_tok_pre_is_verified(const sllm_tok * t) {
    /*
     * DEFAULT is verified by tests/golden/tokenizer.txt, captured from this
     * model. GPT2 is implemented because it is the same single pass the
     * DEFAULT pipeline applies as its second stage, and it is verified by a
     * synthetic-vocabulary fixture, but no real model carrying `pre` = "gpt-2"
     * was available. Both facts are reported rather than collapsed into a
     * single "supported" answer.
     */
    if (t == NULL) { return false; }
    return t->pre_type == SLLM_PRE_UNSET || t->pre_type == SLLM_PRE_GPT2;
}

bool sllm_tok_is_special(const sllm_tok * t, int32_t id) {
    if (t == NULL || id < 0 || (uint32_t) id >= t->n_vocab) { return false; }
    return t->special[id] != 0;
}

const char * sllm_tok_piece(const sllm_tok * t, int32_t id) {
    if (t == NULL || id < 0 || (uint32_t) id >= t->n_vocab) { return NULL; }
    return t->id_to_piece + t->piece_off[(uint32_t) id];
}

/* ------------------------------------------------------------------ */
/* UTF-8                                                                */
/* ------------------------------------------------------------------ */

/*
 * Deliberately lenient, and deliberately so.
 *
 * The reference does not reject overlong encodings, does not reject surrogate
 * code points and does not range-check the result. On anything it cannot
 * decode, the caller advances one byte and emits U+FFFD. Tightening any of that
 * would be a correctness improvement and a parity regression, so it is not done
 * here. A fixture prompt exercises the invalid path.
 */
static uint32_t utf8_next(const char * s, size_t len, size_t * pos) {
    const unsigned char c0 = (unsigned char) s[*pos];
    if (!(c0 & 0x80u)) { *pos += 1; return c0; }
    if (!(c0 & 0x40u)) { return 0xFFFFFFFFu; }

    unsigned need, mask;
    if      (!(c0 & 0x20u)) { need = 1; mask = 0x1Fu; }
    else if (!(c0 & 0x10u)) { need = 2; mask = 0x0Fu; }
    else if (!(c0 & 0x08u)) { need = 3; mask = 0x07u; }
    else                    { return 0xFFFFFFFFu; }

    if (*pos + need >= len) { return 0xFFFFFFFFu; }
    uint32_t cp = (uint32_t) (c0 & mask);
    for (unsigned k = 1; k <= need; ++k) {
        const unsigned char cn = (unsigned char) s[*pos + k];
        if ((cn & 0xC0u) != 0x80u) { return 0xFFFFFFFFu; }
        cp = (cp << 6) | (uint32_t) (cn & 0x3Fu);
    }
    *pos += need + 1;
    return cp;
}

static size_t utf8_encode(uint32_t cp, char * out) {
    if (cp < 0x80u) {
        out[0] = (char) cp;
        return 1;
    }
    if (cp < 0x800u) {
        out[0] = (char) (0xC0u | (cp >> 6));
        out[1] = (char) (0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        out[0] = (char) (0xE0u | (cp >> 12));
        out[1] = (char) (0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (char) (0x80u | (cp & 0x3Fu));
        return 3;
    }
    out[0] = (char) (0xF0u | (cp >> 18));
    out[1] = (char) (0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = (char) (0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = (char) (0x80u | (cp & 0x3Fu));
    return 4;
}

/* Length in bytes of the UTF-8 sequence starting at s[i]. */
static size_t utf8_len(const char * s, size_t remaining) {
    const unsigned char c0 = (unsigned char) s[0];
    unsigned need;
    if      (!(c0 & 0x80u)) { return 1; }
    else if (!(c0 & 0x40u)) { return 1; }
    else if (!(c0 & 0x20u)) { need = 1; }
    else if (!(c0 & 0x10u)) { need = 2; }
    else if (!(c0 & 0x08u)) { need = 3; }
    else                    { return 1; }
    if (1 + need > remaining) { return 1; }
    for (unsigned k = 1; k <= need; ++k) {
        if (((unsigned char) s[k] & 0xC0u) != 0x80u) { return 1; }
    }
    return 1 + need;
}

static sllm_status to_cpts(const char * text, size_t len, uint32_t ** out, size_t * n_out) {
    uint32_t * cp = (uint32_t *) malloc((len + 1) * sizeof(uint32_t));
    if (cp == NULL) { return SLLM_ERR_NOMEM; }
    size_t n = 0;
    size_t pos = 0;
    while (pos < len) {
        const uint32_t c = utf8_next(text, len, &pos);
        if (c == 0xFFFFFFFFu) {
            ++pos;                        /* exactly the reference's path */
            cp[n++] = 0xFFFDu;
        } else {
            cp[n++] = c;
        }
    }
    *out   = cp;
    *n_out = n;
    return SLLM_OK;
}

/* ------------------------------------------------------------------ */
/* byte-level encoding, GPT-2 style                                    */
/* ------------------------------------------------------------------ */

/*
 * The reference builds this mapping by rule, not from a table: printable ASCII
 * and most of Latin-1 pass through unchanged, and every remaining byte maps to
 * U+0100 upwards in increasing byte order. Implementing the rule rather than
 * copying 256 strings means the 68 remapped bytes cannot drift out of step.
 */
static bool byte_passes_through(uint8_t b) {
    return (b >= 0x21u && b <= 0x7Eu) ||
           (b >= 0xA1u && b <= 0xACu) ||
           b >= 0xAEu;                 /* 0xAE..0xFF */
}

static size_t byte_to_cpt(uint8_t b, char * out) {
    if (byte_passes_through(b)) {
        /*
         * A passing-through byte becomes the UTF-8 encoding of the code point
         * of the same value, NOT the bare byte. For ASCII those coincide, which
         * is why the distinction is easy to miss and why it only shows up on
         * non-ASCII input: byte 0xCE is the two bytes C3 8E, the character
         * "Î", not the single byte CE.
         *
         * Getting this wrong leaves every word's byte-encoded form equal to
         * its original bytes, no piece is ever found in the vocabulary, and
         * non-ASCII text encodes to nothing at all -- while ASCII, which is
         * most of the fixture, passes throughout.
         */
        if (b < 0x80u) {
            out[0] = (char) b;
            return 1;
        }
        return utf8_encode(b, out);
    }
    uint32_t n = 0;
    for (uint32_t x = 0x00u; x <= 0x20u; ++x, ++n) {
        if (x == b) { return utf8_encode(0x0100u + n, out); }
    }
    for (uint32_t x = 0x7Fu; x <= 0xA0u; ++x, ++n) {
        if (x == b) { return utf8_encode(0x0100u + n, out); }
    }
    return utf8_encode(0x0100u + n, out);   /* b == 0xAD */
}

static int cpt_to_byte(uint32_t cp) {
    if (cp <= 0xFFu && byte_passes_through((uint8_t) cp)) {
        return (int) cp;
    }
    if (cp < 0x0100u || cp > 0x0143u) { return -1; }
    uint32_t n = 0;
    for (uint32_t x = 0x00u; x <= 0x20u; ++x, ++n) {
        if (cp == 0x0100u + n) { return (int) x; }
    }
    for (uint32_t x = 0x7Fu; x <= 0xA0u; ++x, ++n) {
        if (cp == 0x0100u + n) { return (int) x; }
    }
    if (cp == 0x0100u + n) { return 0xAD; }
    return -1;
}

/* ------------------------------------------------------------------ */
/* the pre-tokeniser: four passes, in the reference's order             */
/* ------------------------------------------------------------------ */

/*
 * Passes 1 and 3 run against the reference's "collapsed" text, where every
 * code point at or above 128 has been replaced by a single byte naming its
 * general category. ASCII is kept verbatim.
 *
 * The category test below is an EXACT comparison against a single category,
 * not a bit test, and that is not a stylistic choice. category_flag() returns
 * the low byte of the flags unmasked-by-category, which may hold several bits
 * at once. The reference looks that combined value up in a table keyed by
 * single categories, so a code point that is LETTER|SEPARATOR matches nothing
 * and falls through to its own fallback byte. A bit test would match it and
 * silently disagree.
 */
#define SLLM_COLLAPSE_PUNCT   0xD3u
#define SLLM_COLLAPSE_NUMBER  0xD1u

/* [ \xD3 !-\#%-\*,\-/:-\;?-\@\[-\]_\{\} $ + < = > ^ ~ | ] */
static bool pass1_punct(uint32_t c) {
    if (c < 0x80u) {
        return (c >= 0x21u && c <= 0x23u) || (c >= 0x25u && c <= 0x2Au) ||
               (c >= 0x2Cu && c <= 0x2Fu) || (c >= 0x3Au && c <= 0x3Bu) ||
               (c >= 0x3Fu && c <= 0x40u) || (c >= 0x5Bu && c <= 0x5Du) ||
               c == 0x5Fu || c == 0x7Bu || c == 0x7Du ||
               c == '$' || c == '+' || c == '<' || c == '=' || c == '>' ||
               c == '^' || c == '~' || c == '|';
    }
    return sllm_uni_category(c) == SLLM_UNI_PUNCTUATION;
}

/* [ \xD1 0-9 ] */
static bool pass3_number(uint32_t c) {
    if (c < 0x80u) {
        return c >= '0' && c <= '9';
    }
    return sllm_uni_category(c) == SLLM_UNI_NUMBER;
}

static bool is_ascii_digit(uint32_t c) { return c >= '0' && c <= '9'; }

/* Split `segs` on maximal runs where `pred` holds, within each segment. This is
 * the shape of every std::regex pass in the reference: find the leftmost
 * longest run, emit the gap before it if non-empty, emit the run, repeat, and
 * emit the trailing gap if non-empty. */
static void split_runs(const uint32_t * cp, const u32vec * segs, u32vec * out,
                       bool (*pred)(uint32_t)) {
    size_t base = 0;
    for (size_t s = 0; s < segs->n; ++s) {
        const size_t len = segs->v[s];
        const size_t e   = base + len;
        size_t last     = base;
        size_t i        = base;
        while (i < e) {
            if (pred(cp[i])) {
                size_t j = i;
                while (j < e && pred(cp[j])) { ++j; }
                if (i > last) { uv_push(out, (uint32_t) (i - last)); }
                uv_push(out, (uint32_t) (j - i));
                last = j;
                i    = j;
            } else {
                ++i;
            }
        }
        if (e > last) { uv_push(out, (uint32_t) (e - last)); }
        base = e;
    }
}

/* Pass 4, [0-9][0-9][0-9]: exactly three ASCII digits, leftmost, not
 * overlapping. A run of seven yields 123 456 7, which is why the last digit is
 * left as its own segment. */
static void split_digit_triples(const uint32_t * cp, const u32vec * segs, u32vec * out) {
    size_t base = 0;
    for (size_t s = 0; s < segs->n; ++s) {
        const size_t len = segs->v[s];
        const size_t e   = base + len;
        size_t last     = base;
        size_t i        = base;
        while (i + 2 < e) {
            if (is_ascii_digit(cp[i]) && is_ascii_digit(cp[i + 1]) && is_ascii_digit(cp[i + 2])) {
                if (i > last) { uv_push(out, (uint32_t) (i - last)); }
                uv_push(out, 3u);
                last = i + 3;
                i    = last;
            } else {
                ++i;
            }
        }
        if (e > last) { uv_push(out, (uint32_t) (e - last)); }
        base = e;
    }
}

/* Pass 2, the hand-written GPT-2 splitter, transcribed from the reference.
 *
 * Two details carry most of the behaviour:
 *
 *  - the apostrophe rules are anchored at the CURRENT position, so an
 *    apostrophe in the middle of a word does not start a contraction. That is
 *    why "they're" is they / ' / re.
 *
 *  - \s+(?!\S) emits all but the last whitespace of a run, leaving the final
 *    one for the next token's optional leading space. Emitting the whole run
 *    instead is the classic off-by-one that changes every token after a
 *    double space.
 */
static void split_gpt2(const uint32_t * cp, const u32vec * segs, u32vec * out) {
    const uint32_t OUT_OF_RANGE = 0xFFFFFFFFu;

    size_t base = 0;
    for (size_t s = 0; s < segs->n; ++s) {
        const size_t ini = base;
        const size_t end = base + segs->v[s];
        base = end;

        size_t prev_end = ini;

        /* the reference's _add_token: append a non-empty span and advance */
        #define ADD_TOKEN(e_)                                                  \
            do {                                                               \
                const size_t e__ = (e_);                                       \
                if (e__ > prev_end) { uv_push(out, (uint32_t) (e__ - prev_end)); } \
                prev_end = e__;                                                \
            } while (0)

        for (size_t pos = ini; pos < end; ) {
            const uint32_t cpt = cp[pos];
            const uint16_t fl  = sllm_uni_flags(cpt);

            /* 's|'t|'re|'ve|'m|'ll|'d */
            if (cpt == '\'' && pos + 1 < end) {
                const uint32_t n1 = cp[pos + 1];
                if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                    pos += 2;
                    ADD_TOKEN(pos);
                    continue;
                }
                if (pos + 2 < end) {
                    const uint32_t n2 = cp[pos + 2];
                    if ((n1 == 'r' && n2 == 'e') ||
                        (n1 == 'v' && n2 == 'e') ||
                        (n1 == 'l' && n2 == 'l')) {
                        pos += 3;
                        ADD_TOKEN(pos);
                        continue;
                    }
                }
            }

            /* the optional leading space is part of the same token */
            uint16_t fl2 = (cpt == ' ') ? (pos + 1 < end ? sllm_uni_flags(cp[pos + 1]) : 0u) : fl;

            if (fl2 & SLLM_UNI_LETTER) {
                if (cpt == ' ') { ++pos; }
                while (fl2 & SLLM_UNI_LETTER) {
                    ++pos;
                    fl2 = (pos < end) ? sllm_uni_flags(cp[pos]) : 0u;
                }
                ADD_TOKEN(pos);
                continue;
            }
            if (fl2 & SLLM_UNI_NUMBER) {
                if (cpt == ' ') { ++pos; }
                while (fl2 & SLLM_UNI_NUMBER) {
                    ++pos;
                    fl2 = (pos < end) ? sllm_uni_flags(cp[pos]) : 0u;
                }
                ADD_TOKEN(pos);
                continue;
            }
            /* ?[^\s\p{L}\p{N}]+ , guarded on any flag being set at all */
            if (!(fl2 & (SLLM_UNI_WHITESPACE | SLLM_UNI_LETTER | SLLM_UNI_NUMBER)) && fl2 != 0) {
                if (cpt == ' ') { ++pos; }
                while (!(fl2 & (SLLM_UNI_WHITESPACE | SLLM_UNI_LETTER | SLLM_UNI_NUMBER)) &&
                       fl2 != 0) {
                    ++pos;
                    fl2 = (pos < end) ? sllm_uni_flags(cp[pos]) : 0u;
                }
                ADD_TOKEN(pos);
                continue;
            }

            size_t nws = 0;
            while (pos + nws < end && (sllm_uni_flags(cp[pos + nws]) & SLLM_UNI_WHITESPACE)) {
                ++nws;
            }

            /* \s+(?!\S): leave the last whitespace for whatever follows */
            if (nws > 1 && pos + nws < end) {
                pos += nws - 1;
                ADD_TOKEN(pos);
                continue;
            }
            if (nws > 0) {
                pos += nws;
                ADD_TOKEN(pos);
                continue;
            }

            (void) OUT_OF_RANGE;
            ADD_TOKEN(++pos);
        }
        #undef ADD_TOKEN
    }
}

/*
 * The qwen2 pre-tokeniser.
 *
 * Same byte alphabet as GPT-2 -- proven against the file's raw
 * tokenizer.ggml.tokens, see tests/golden/mainstream-qwen3-tokenizer.txt -- so
 * the vocabulary, the merge loop and the decoder are shared unchanged. Only the
 * SPLIT differs, and it differs in five ways that each change real tokens:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)   case-INsensitive, and tried FIRST
 *   [^\r\n\p{L}\p{N}]?\p{L}+        the optional leading char is any char that
 *                                   is not a letter, digit, CR or LF -- so a
 *                                   TAB may lead a word, not only a space
 *   \p{N}{1,3}                      digits, at most three, and no leading space
 *   ?[^\s\p{L}\p{N}]+[\r\n]*        punctuation, then any trailing newlines
 *   \s*[\r\n]+                      whitespace ending in a newline run
 *
 * plus the two whitespace rules GPT-2 also has. Order is the alternation order
 * above and is significant: contractions must beat the letter rule, and the
 * newline rule must beat the generic whitespace rules.
 *
 * Hand-written like split_gpt2 rather than driven by a regex engine, for the
 * same reason: the engine would have to be a dependency, and the split is the
 * part most worth being able to read.
 */
static void split_qwen2(const uint32_t * cp, const u32vec * segs, u32vec * out) {
    const size_t n_in = cp == NULL ? 0 : segs->n;

    size_t base = 0;
    for (size_t s = 0; s < n_in; ++s) {
        const size_t ini = base;
        const size_t end = base + segs->v[s];
        base = end;

        size_t prev_end = ini;

        #define ADD_TOKEN(e_)                                                  \
            do {                                                               \
                const size_t e__ = (e_);                                       \
                if (e__ > prev_end) { uv_push(out, (uint32_t) (e__ - prev_end)); } \
                prev_end = e__;                                                \
            } while (0)

        /* ASCII case fold, and only for the ASCII letters the contractions use.
         * Folding the whole of Unicode here would change which words the
         * contraction rule fires on, and the reference folds ASCII only. */
        #define LOWER(c_) ( ((c_) >= 'A' && (c_) <= 'Z') ? (c_) + 32 : (c_) )

        for (size_t pos = ini; pos < end; ) {
            const uint32_t cpt = cp[pos];
            const uint16_t fl  = sllm_uni_flags(cpt);

            /* (?i:'s|'t|'re|'ve|'m|'ll|'d) -- first, and case-insensitive */
            if (cpt == '\'' && pos + 1 < end) {
                const uint32_t n1 = LOWER(cp[pos + 1]);
                if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                    pos += 2; ADD_TOKEN(pos); continue;
                }
                if (pos + 2 < end) {
                    const uint32_t n2 = LOWER(cp[pos + 2]);
                    if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') ||
                        (n1 == 'l' && n2 == 'l')) {
                        pos += 3; ADD_TOKEN(pos); continue;
                    }
                }
            }

            /* [^\r\n\p{L}\p{N}]?\p{L}+ */
            {
                const bool is_nl = (cpt == '\r' || cpt == '\n');
                const bool prefixable =
                    !is_nl && !(fl & (SLLM_UNI_LETTER | SLLM_UNI_NUMBER));
                size_t at = pos;
                if (prefixable) {
                    if (pos + 1 >= end || !(sllm_uni_flags(cp[pos + 1]) & SLLM_UNI_LETTER)) {
                        goto not_letters;   /* the optional part is not optional enough */
                    }
                    at = pos + 1;
                } else if (!(fl & SLLM_UNI_LETTER)) {
                    goto not_letters;
                }
                while (at < end && (sllm_uni_flags(cp[at]) & SLLM_UNI_LETTER)) { ++at; }
                pos = at;
                ADD_TOKEN(pos);
                continue;
            }
        not_letters:;

            /* \p{N}{1,3} -- no leading space, and a hard cap of three. GPT-2
             * allows an optional space and an unbounded run, so this is the
             * difference that makes "0123456789" group differently. */
            if (fl & SLLM_UNI_NUMBER) {
                size_t at = pos;
                while (at < end && at < pos + 3 &&
                       (sllm_uni_flags(cp[at]) & SLLM_UNI_NUMBER)) { ++at; }
                pos = at;
                ADD_TOKEN(pos);
                continue;
            }

            /* ?[^\s\p{L}\p{N}]+[\r\n]* */
            if (!(fl & (SLLM_UNI_WHITESPACE | SLLM_UNI_LETTER | SLLM_UNI_NUMBER)) && fl != 0) {
                size_t at = (cpt == ' ') ? pos + 1 : pos;
                while (at < end) {
                    const uint16_t f2 = sllm_uni_flags(cp[at]);
                    if (f2 & (SLLM_UNI_WHITESPACE | SLLM_UNI_LETTER | SLLM_UNI_NUMBER)) { break; }
                    ++at;
                }
                /* the trailing newline run belongs to this token */
                while (at < end && (cp[at] == '\r' || cp[at] == '\n')) { ++at; }
                pos = at;
                ADD_TOKEN(pos);
                continue;
            }

            /* \s*[\r\n]+ : whitespace ending in a newline run. Greedy \s* then
             * backtrack, so the match runs up to and including the LAST CR or LF
             * in the whitespace run. Without this a blank line would split
             * into two tokens instead of one. */
            {
                size_t nws = 0;
                while (pos + nws < end &&
                       (sllm_uni_flags(cp[pos + nws]) & SLLM_UNI_WHITESPACE)) { ++nws; }
                if (nws > 0) {
                    size_t last_nl = 0;
                    bool have_nl = false;
                    for (size_t i = 0; i < nws; ++i) {
                        const uint32_t w = cp[pos + i];
                        if (w == '\r' || w == '\n') { last_nl = i + 1; have_nl = true; }
                    }
                    if (have_nl) {
                        pos += last_nl;
                        ADD_TOKEN(pos);
                        continue;
                    }
                    /* \s+(?!\S): keep the last whitespace back for whatever
                     * follows, but only if something non-space does follow. */
                    if (pos + nws < end && nws > 1) {
                        pos += nws - 1;
                        ADD_TOKEN(pos);
                        continue;
                    }
                    pos += nws;
                    ADD_TOKEN(pos);
                    continue;
                }
            }

            ADD_TOKEN(++pos);
        }
        #undef ADD_TOKEN
        #undef LOWER
    }
}

/*
 * Run the split passes for the model's pre-type, in the reference's order.
 *
 * GPT2 is exactly pass 2 and nothing else. DEFAULT is passes 1, 2, 3 and 4.
 * QWEN2 is a single pass of its own. Both funnel through the same hand-written
 * GPT-2 splitter, which is why the second stage is factored out: the difference
 * between the two pre-types is precisely the passes wrapped around it, and
 * expressing it that way keeps the shared stage in one place.
 */
static sllm_status pretokenize_cpts(sllm_pre_type pre, const uint32_t * cp,
                                    size_t n, u32vec * out) {
    u32vec a, b, c, d;
    uv_init(&a); uv_init(&b); uv_init(&c); uv_init(&d);
    uv_push(&a, (uint32_t) n);

    if (pre == SLLM_PRE_GPT2) {
        split_gpt2(cp, &a, out);
        const bool bad = a.oom || out->oom;
        uv_free(&a);
        return bad ? SLLM_ERR_NOMEM : SLLM_OK;
    }

    if (pre == SLLM_PRE_QWEN2) {
        split_qwen2(cp, &a, out);
        const bool bad = a.oom || out->oom;
        uv_free(&a);
        return bad ? SLLM_ERR_NOMEM : SLLM_OK;
    }

    split_runs(cp, &a, &b, pass1_punct);   /* [\p{P}\$\+<=>\^~\|]+  */
    split_gpt2(cp, &b, &c);                /* the GPT-2 pattern      */
    split_runs(cp, &c, &d, pass3_number);  /* \p{N}+                 */
    split_digit_triples(cp, &d, out);      /* [0-9][0-9][0-9]        */

    const bool bad = a.oom || b.oom || c.oom || d.oom || out->oom;
    uv_free(&a); uv_free(&b); uv_free(&c); uv_free(&d);
    return bad ? SLLM_ERR_NOMEM : SLLM_OK;
}

/* ------------------------------------------------------------------ */
/* BPE merge                                                            */
/* ------------------------------------------------------------------ */

typedef struct { int32_t left; int32_t right; int32_t rank; size_t text_len; } bigram;

/* Min-heap on (rank, left), matching the reference's comparator:
 *   l.rank > r.rank || (l.rank == r.rank && l.left > r.left)
 * A live bigram is uniquely identified by its left index, so the tiebreak makes
 * the pop order total and this reproduces the reference's order. */
static bool bg_less(const bigram * a, const bigram * b) {
    if (a->rank != b->rank) { return a->rank < b->rank; }
    return a->left < b->left;
}

typedef struct { bigram * v; size_t n; size_t cap; } bgheap;

static void bg_push(bgheap * h, bigram b) {
    if (h->n == h->cap) {
        const size_t nc = h->cap ? h->cap * 2 : 64;
        bigram * nv = (bigram *) realloc(h->v, nc * sizeof(bigram));
        if (nv == NULL) { return; }
        h->v = nv;
        h->cap = nc;
    }
    size_t i = h->n++;
    h->v[i] = b;
    while (i > 0) {
        const size_t par = (i - 1) / 2;
        if (bg_less(&h->v[i], &h->v[par])) {
            const bigram t = h->v[i]; h->v[i] = h->v[par]; h->v[par] = t;
            i = par;
        } else {
            break;
        }
    }
}

static bool bg_pop(bgheap * h, bigram * out) {
    if (h->n == 0) { return false; }
    *out = h->v[0];
    h->n--;
    if (h->n == 0) { return true; }
    h->v[0] = h->v[h->n];
    size_t i = 0;
    for (;;) {
        const size_t l = 2 * i + 1;
        const size_t r = l + 1;
        size_t m = i;
        if (l < h->n && bg_less(&h->v[l], &h->v[m])) { m = l; }
        if (r < h->n && bg_less(&h->v[r], &h->v[m])) { m = r; }
        if (m == i) { break; }
        const bigram t = h->v[i]; h->v[i] = h->v[m]; h->v[m] = t;
        i = m;
    }
    return true;
}

typedef struct { size_t off; size_t len; int32_t prev; int32_t next; } sym;

/* Build the tokens for one byte-encoded word. */
static sllm_status bpe_word(const sllm_tok * t, const char * word, size_t word_len,
                            int32_t * out, int32_t cap, int32_t * n_out) {
    /* symbols, one per UTF-8 character of the byte-encoded word */
    size_t nsym = 0;
    for (size_t i = 0; i < word_len; ) {
        i += utf8_len(word + i, word_len - i);
        ++nsym;
    }
    if (nsym == 0) { return SLLM_OK; }

    sym * syms = (sym *) malloc(nsym * sizeof(sym));
    if (syms == NULL) { return SLLM_ERR_NOMEM; }

    size_t k = 0;
    for (size_t i = 0; i < word_len; ++k) {
        const size_t l = utf8_len(word + i, word_len - i);
        syms[k].off  = i;
        syms[k].len  = l;
        syms[k].prev = (int32_t) k - 1;
        syms[k].next = (i + l >= word_len) ? -1 : (int32_t) k + 1;
        i += l;
    }

    bgheap heap = { NULL, 0, 0 };

    #define ADD_BIGRAM(l_, r_)                                                  \
        do {                                                                     \
            if ((l_) >= 0 && (r_) >= 0) {                                        \
                const int32_t rank_ = map_pair_get_n(                             \
                    &t->merge_rank,                                              \
                    word + syms[(l_)].off, syms[(l_)].len,                        \
                    word + syms[(r_)].off, syms[(r_)].len);                       \
                if (rank_ >= 0) {                                                \
                    const bigram b_ = { (int32_t) (l_), (int32_t) (r_), rank_,   \
                                        syms[(l_)].len + syms[(r_)].len };        \
                    bg_push(&heap, b_);                                          \
                }                                                                \
            }                                                                    \
        } while (0)

    for (size_t i = 1; i < nsym; ++i) {
        ADD_BIGRAM((int32_t) i - 1, (int32_t) i);
    }

    bigram b;
    while (bg_pop(&heap, &b)) {
        if (syms[b.left].len == 0 || syms[b.right].len == 0) { continue; }
        /*
         * Staleness check. A queued bigram records the combined length its two
         * halves had when it was pushed. If either half has since been merged
         * into, that length no longer matches and the bigram must be dropped.
         *
         * Without this, a pair whose left half had already absorbed a
         * neighbour gets merged anyway, and BPE -- which is defined as "always
         * merge the lowest-ranked available pair" -- quietly stops being BPE.
         * The results still look like tokens and still round trip; they are
         * simply not the reference's, which is the failure this project exists
         * to prevent. The reference compares the concatenated text; since both
         * halves are contiguous slices of the same word, comparing the combined
         * length against the recorded one is equivalent.
         */
        if (syms[b.left].len + syms[b.right].len != b.text_len) { continue; }
        /* skip a bigram whose right half has been absorbed since it was queued */
        syms[b.left].len += syms[b.right].len;
        syms[b.right].len = 0;
        syms[b.left].next = syms[b.right].next;
        if (syms[b.right].next >= 0) {
            syms[syms[b.right].next].prev = b.left;
        }
        ADD_BIGRAM(syms[b.left].prev, b.left);
        ADD_BIGRAM(b.left, syms[b.left].next);
    }
    #undef ADD_BIGRAM

    free(heap.v);

    /* emit, in order, resolving any piece as one token and falling back to
     * per-byte ids exactly as the reference does */
    for (size_t i = 0; i < nsym; ++i) {
        if (syms[i].len == 0) { continue; }
        const char * piece = word + syms[i].off;
        const size_t   plen = syms[i].len;
        int32_t id = tok_piece_id(t, piece, plen);
        if (id >= 0) {
            if (*n_out < cap) { out[*n_out] = id; }
            (*n_out)++;
        } else {
            for (size_t bpos = 0; bpos < plen; ) {
                const size_t l = utf8_len(piece + bpos, plen - bpos);
                const int32_t bid = tok_piece_id(t, piece + bpos, l);
                if (bid >= 0) {
                    if (*n_out < cap) { out[*n_out] = bid; }
                    (*n_out)++;
                }
                bpos += l;
            }
        }
    }

    free(syms);
    return SLLM_OK;
}

/* ------------------------------------------------------------------ */
/* public entry points                                                  */
/* ------------------------------------------------------------------ */

/*
 * Build the byte-encoded form of each segment, recording where each one lands.
 *
 * A word is byte-encoded one BYTE at a time, not one code point at a time. The
 * reference re-encodes each code point to UTF-8 and then byte-encodes each of
 * the resulting bytes, so U+00E9 becomes the two bytes C3 A9 and then the two
 * pieces "Ã" and "©". Collapsing that to a single step would merge tokens and
 * silently change the output for every non-ASCII character.
 */
static sllm_status encode_words(const uint32_t * cp, const u32vec * segs,
                                char * buf, size_t buf_cap,
                                size_t ** off, size_t ** len, size_t * n_out) {
    *n_out = segs->n;
    *off = (size_t *) calloc(segs->n ? segs->n : 1, sizeof(size_t));
    *len = (size_t *) calloc(segs->n ? segs->n : 1, sizeof(size_t));
    if (*off == NULL || *len == NULL) {
        free(*off); free(*len);
        *off = NULL; *len = NULL;
        return SLLM_ERR_NOMEM;
    }

    size_t used = 0;
    size_t base = 0;
    for (size_t s = 0; s < segs->n; ++s) {
        const size_t nseg = segs->v[s];
        size_t u = 0;
        for (size_t i = base; i < base + nseg; ++i) {
            char utf8[4];
            const size_t nu = utf8_encode(cp[i], utf8);
            for (size_t k = 0; k < nu; ++k) {
                char piece[4];
                const size_t np = byte_to_cpt((uint8_t) utf8[k], piece);
                if (used + u + np + 1 > buf_cap) {
                    free(*off); free(*len);
                    *off = NULL; *len = NULL;
                    return SLLM_ERR_TOO_LARGE;
                }
                memcpy(buf + used + u, piece, np);
                u += np;
            }
        }
        buf[used + u] = '\0';
        (*off)[s] = used;
        (*len)[s] = u;
        used += u + 1;
        base += nseg;
    }
    return SLLM_OK;
}

/* Encode a run of ordinary text, no special-token handling. */
static sllm_status encode_plain(const sllm_tok * t, sllm_pre_type pre,
                                const char * text, size_t len,
                                int32_t * out, int32_t cap, int32_t * n_out) {
    uint32_t * cp = NULL;
    size_t n = 0;
    sllm_status rc = to_cpts(text, len, &cp, &n);
    if (rc != SLLM_OK) { return rc; }

    u32vec segs;
    uv_init(&segs);
    rc = pretokenize_cpts(pre, cp, n, &segs);
    if (rc != SLLM_OK) { free(cp); uv_free(&segs); return rc; }

    size_t * woff = NULL;
    size_t * wlen = NULL;
    size_t nwords = 0;
    const size_t buf_cap = n * 8 + 64;
    char * buf = (char *) malloc(buf_cap);
    if (buf == NULL) { free(cp); uv_free(&segs); return SLLM_ERR_NOMEM; }
    rc = encode_words(cp, &segs, buf, buf_cap, &woff, &wlen, &nwords);
    if (rc != SLLM_OK) { free(buf); free(woff); free(wlen); free(cp); uv_free(&segs); return rc; }

    for (size_t w = 0; w < nwords; ++w) {
        rc = bpe_word(t, buf + woff[w], wlen[w], out, cap, n_out);
        if (rc != SLLM_OK) { break; }
    }

    free(buf); free(woff); free(wlen); free(cp); uv_free(&segs);
    return rc;
}

static sllm_status emit(int32_t id, int32_t * out, int32_t cap, int32_t * n) {
    if (*n < cap) { out[*n] = id; }
    (*n)++;
    return SLLM_OK;
}

int32_t sllm_tok_encode(const sllm_tok * t, const char * text, size_t text_len,
                        bool add_special, bool parse_special,
                        int32_t * out, int32_t cap) {
    if (t == NULL) { return SLLM_ERR_ARG; }
    return sllm_tok_encode_pre(t, t->pre_type, text, text_len,
                               add_special, parse_special, out, cap);
}

int32_t sllm_tok_encode_pre(const sllm_tok * t, sllm_pre_type pre,
                            const char * text, size_t text_len,
                            bool add_special, bool parse_special,
                            int32_t * out, int32_t cap) {
    if (t == NULL || (text == NULL && text_len > 0)) { return SLLM_ERR_ARG; }
    if (pre != SLLM_PRE_UNSET && pre != SLLM_PRE_GPT2) { return SLLM_ERR_UNSUPPORTED; }
    if (cap <= 0) { return 0; }

    int32_t n = 0;
    if (add_special && t->add_bos && t->bos >= 0) {
        (void) emit(t->bos, out, cap, &n);
    }

    if (parse_special) {
        /*
         * Literal special-token text in the input is emitted as one token. The
         * leftmost match wins, and ties go to the longer special so a token
         * that is a prefix of another cannot shadow it.
         */
        size_t pos = 0;
        while (pos < text_len) {
            const char * best = NULL;
            size_t best_len = 0;
            for (const char * p = t->special_text; p != NULL && *p != '\0'; ) {
                const size_t l = strlen(p);
                if (l > best_len && pos + l <= text_len && memcmp(text + pos, p, l) == 0) {
                    best = p;
                    best_len = l;
                }
                p += l + 1;
            }
            if (best == NULL) {
                if (encode_plain(t, pre, text + pos, text_len - pos, out, cap, &n) != SLLM_OK) {
                    return SLLM_ERR_TOO_LARGE;
                }
                break;
            }
            if (pos > 0 || best_len > 0) {
                if (encode_plain(t, pre, text + pos, 0, out, cap, &n) != SLLM_OK) { break; }
            }
            const int32_t id = tok_piece_id(t, best, best_len);
            if (id >= 0) { (void) emit(id, out, cap, &n); }
            pos += best_len;
        }
    } else {
        if (encode_plain(t, pre, text, text_len, out, cap, &n) != SLLM_OK) {
            return SLLM_ERR_TOO_LARGE;
        }
    }

    if (add_special && t->add_eos && t->eos >= 0) {
        (void) emit(t->eos, out, cap, &n);
    }
    return n;
}

int32_t sllm_tok_encode_len(const sllm_tok * t, size_t text_len, bool add_special) {
    if (t == NULL) { return SLLM_ERR_ARG; }
    /* One token per byte is the worst case: the fallback path emits a token per
     * byte of a word when the merged form is not in the vocab. */
    int32_t n = (int32_t) (text_len + 1);
    if (add_special) {
        if (t->add_bos && t->bos >= 0) { n += 1; }
        if (t->add_eos && t->eos >= 0) { n += 1; }
    }
    return n;
}

int32_t sllm_tok_decode(const sllm_tok * t, const int32_t * tokens, int32_t n,
                        bool remove_special, char * out, int32_t cap) {
    if (t == NULL || (tokens == NULL && n > 0)) { return SLLM_ERR_ARG; }
    if (cap <= 0) { return SLLM_ERR_ARG; }

    int32_t used = 0;
    for (int32_t i = 0; i < n; ++i) {
        const int32_t id = tokens[i];
        if (id < 0 || (uint32_t) id >= t->n_vocab) { continue; }
        if (remove_special && t->special[id]) { continue; }

        const char * piece = t->id_to_piece + t->piece_off[id];
        size_t plen = strlen(piece);
        size_t b = 0;
        while (b < plen) {
            size_t adv = 0;
            const uint32_t cp = utf8_next(piece + b, plen - b, &adv);
            if (cp == 0xFFFFFFFFu) {
                /* A piece that is not valid UTF-8 cannot be a byte-encoded
                 * token. Copy the byte through rather than dropping it, so a
                 * corrupt vocabulary degrades visibly instead of silently
                 * losing bytes. */
                if (used + 1 >= cap) { out[used] = '\0'; return SLLM_ERR_TOO_LARGE; }
                out[used++] = piece[b];
                b += 1;
                continue;
            }
            const int by = cpt_to_byte(cp);
            if (by >= 0) {
                if (used + 1 >= cap) { out[used] = '\0'; return SLLM_ERR_TOO_LARGE; }
                out[used++] = (char) by;
            } else {
                /* Same reasoning: pass the code point through unchanged. */
                if (used + 4 >= cap) { out[used] = '\0'; return SLLM_ERR_TOO_LARGE; }
                used += (int32_t) utf8_encode(cp, out + used);
            }
            b += adv;
        }
    }
    out[used] = '\0';
    return used;
}

/*
 * Split `text` the way the pre-tokeniser does, without running BPE, so a
 * failing golden vector can be attributed to a split pass rather than to the
 * merge loop. Each word is written byte-encoded and NUL-terminated, words
 * separated by a single NUL, so the result is a run of C strings and the caller
 * needs no allocation.
 */
sllm_status sllm_tok_pretokenize(sllm_pre_type pre, const char * text, size_t text_len,
                                 char * out, size_t cap, int32_t * n_out) {
    if (out == NULL || n_out == NULL) { return SLLM_ERR_ARG; }
    if (pre != SLLM_PRE_UNSET && pre != SLLM_PRE_GPT2) { return SLLM_ERR_UNSUPPORTED; }
    *n_out = 0;

    uint32_t * cp = NULL;
    size_t n = 0;
    sllm_status rc = to_cpts(text, text_len, &cp, &n);
    if (rc != SLLM_OK) { return rc; }

    u32vec segs;
    uv_init(&segs);
    rc = pretokenize_cpts(pre, cp, n, &segs);
    if (rc != SLLM_OK) { free(cp); uv_free(&segs); return rc; }

    size_t * woff = NULL;
    size_t * wlen = NULL;
    size_t nwords = 0;
    const size_t buf_cap = n * 8 + 64;
    char * buf = (char *) malloc(buf_cap);
    if (buf == NULL) { free(cp); uv_free(&segs); return SLLM_ERR_NOMEM; }
    rc = encode_words(cp, &segs, buf, buf_cap, &woff, &wlen, &nwords);
    if (rc != SLLM_OK) {
        free(buf); free(woff); free(wlen); free(cp); uv_free(&segs);
        return rc;
    }

    size_t wrote = 0;
    for (size_t w = 0; w < nwords; ++w) {
        if (wrote + wlen[w] + 1 > cap) {
            free(buf); free(woff); free(wlen); free(cp); uv_free(&segs);
            return SLLM_ERR_TOO_LARGE;
        }
        memcpy(out + wrote, buf + woff[w], wlen[w]);
        wrote += wlen[w];
        out[wrote++] = '\0';
        (*n_out)++;
    }

    free(buf); free(woff); free(wlen); free(cp); uv_free(&segs);
    return SLLM_OK;
}
