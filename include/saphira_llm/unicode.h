/*
 * Unicode character classes, for the BPE pre-tokeniser.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * The pre-tokeniser needs to know one thing about every code point: which of
 * the general categories it belongs to. That is Unicode data, and it is taken
 * from the pinned reference so that our classification and theirs cannot drift.
 * See src/unicode_data.c for the table and its provenance.
 */

#ifndef SAPHIRA_LLM_UNICODE_H
#define SAPHIRA_LLM_UNICODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* One (first code point, flags) pair. The entry covers from `first` to one
 * before the next entry's `first`. */
typedef struct sllm_uni_range {
    uint32_t first;
    uint16_t flags;
} sllm_uni_range;

extern const sllm_uni_range sllm_uni_ranges[];
extern const size_t          sllm_uni_range_count;
extern const uint32_t       sllm_uni_whitespace[];
extern const size_t          sllm_uni_whitespace_count;

/* Flag bits, matching the reference's unicode_cpt_flags so the two agree. */
#define SLLM_UNI_UNDEFINED    0x0001u
#define SLLM_UNI_NUMBER       0x0002u  /* \p{N} */
#define SLLM_UNI_LETTER       0x0004u  /* \p{L} */
#define SLLM_UNI_SEPARATOR    0x0008u  /* \p{Z} */
#define SLLM_UNI_ACCENT_MARK  0x0010u  /* \p{M} */
#define SLLM_UNI_PUNCTUATION  0x0020u  /* \p{P} */
#define SLLM_UNI_SYMBOL       0x0040u  /* \p{S} */
#define SLLM_UNI_CONTROL      0x0080u  /* \p{C} */
#define SLLM_UNI_MASK_CATEG   0x00FFu
#define SLLM_UNI_WHITESPACE   0x0100u  /* \s   */
#define SLLM_UNI_LOWERCASE    0x0200u
#define SLLM_UNI_UPPERCASE    0x0400u
#define SLLM_UNI_NFD          0x0800u

/* Is the code point in the separate whitespace list? */
static inline bool sllm_uni_in_whitespace_set(uint32_t cpt) {
    size_t lo = 0;
    size_t hi = sllm_uni_whitespace_count;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (sllm_uni_whitespace[mid] < cpt) {
            lo = mid + 1;
        } else if (sllm_uni_whitespace[mid] > cpt) {
            hi = mid;
        } else {
            return true;
        }
    }
    return false;
}

/*
 * Flags for a code point, or 0 when it is past the end of the table.
 *
 * A binary search for the last range starting at or below the code point, then
 * the whitespace bit ORed in from the separate whitespace list. Both steps are
 * needed: the range table alone does not mark a space as whitespace, and
 * omitting the second step silently mis-tokenises every prompt containing one.
 */
static inline uint16_t sllm_uni_flags(uint32_t cpt) {
    size_t lo = 0;
    size_t hi = sllm_uni_range_count;
    size_t found = 0;
    int    have  = 0;

    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (sllm_uni_ranges[mid].first <= cpt) {
            found = mid;
            have  = 1;
            lo    = mid + 1;
        } else {
            hi = mid;
        }
    }
    uint16_t f = have ? sllm_uni_ranges[found].flags : (uint16_t) 0;
    if (sllm_uni_in_whitespace_set(cpt)) {
        f = (uint16_t) (f | SLLM_UNI_WHITESPACE);
    }
    return f;
}

/*
 * The masked category value, which is what the reference's category_flag()
 * returns: the low byte of the flags, unfiltered.
 *
 * It is a mask rather than a tag, and that matters. The pre-tokeniser's
 * collapse step looks this value up in a table of single categories, so a code
 * point whose masked value is a *combination* -- LETTER|SEPARATOR, say --
 * matches no single category and falls through to the fallback byte. Reading
 * it as "is it a letter" instead would silently disagree with the reference on
 * every such code point.
 */
static inline uint16_t sllm_uni_category(uint32_t cpt) {
    return (uint16_t) (sllm_uni_flags(cpt) & SLLM_UNI_MASK_CATEG);
}

static inline bool sllm_uni_is_whitespace(uint32_t cpt) {
    return (sllm_uni_flags(cpt) & SLLM_UNI_WHITESPACE) != 0;
}

#endif /* SAPHIRA_LLM_UNICODE_H */
