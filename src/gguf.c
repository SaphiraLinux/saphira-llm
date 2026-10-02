/*
 * gguf.c — GGUF container parsing.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * A GGUF file is untrusted input. Everything here is written on the assumption
 * that the header is hostile: every read is bounds-checked against the mapped
 * length, every length taken from the file is range-checked before it is used
 * to size an allocation, and every rejection carries a specific status so the
 * user is told what is actually wrong.
 */

#include <saphira_llm/gguf.h>
#include <saphira_llm/log.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* bounds-checked reader                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t * base;
    size_t          size;
    size_t          pos;
    char          * err;
    size_t          err_len;
} sllm_reader;

/*
 * Record why a read ran off the end. Kept to the first message so the earliest
 * and most specific reason survives: a later failure is usually a consequence
 * of an earlier one, and reporting the consequence is useless.
 */
static void rerr(sllm_reader * r, const char * what) {
    if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
        (void) snprintf(r->err, r->err_len,
            "file is %zu bytes but a %s extends past the end (at offset %zu)",
            r->size, what, r->pos);
    }
}

static bool rneed(sllm_reader * r, size_t n) {
    if (n > r->size || r->pos > r->size - n) {
        rerr(r, "structure");
        return false;
    }
    return true;
}

static bool rbytes(sllm_reader * r, void * dst, size_t n) {
    if (!rneed(r, n)) {
        return false;
    }
    if (dst != NULL && n > 0) {
        memcpy(dst, r->base + r->pos, n);
    }
    r->pos += n;
    return true;
}

static bool ru32(sllm_reader * r, uint32_t * v) {
    uint8_t b[4];
    if (!rbytes(r, b, 4)) {
        return false;
    }
    *v = (uint32_t) b[0] | ((uint32_t) b[1] << 8) |
         ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
    return true;
}

static bool ru64(sllm_reader * r, uint64_t * v) {
    uint8_t b[8];
    if (!rbytes(r, b, 8)) {
        return false;
    }
    uint64_t x = 0;
    for (int i = 7; i >= 0; --i) {
        x = (x << 8) | (uint64_t) b[i];
    }
    *v = x;
    return true;
}

static bool rstr(sllm_reader * r, char ** out) {
    uint64_t len = 0;
    if (!ru64(r, &len)) {
        return false;
    }
    if (len > SLLM_GGUF_MAX_STRING) {
        if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
            (void) snprintf(r->err, r->err_len,
                "string length %llu exceeds the %u byte limit",
                (unsigned long long) len, (unsigned) SLLM_GGUF_MAX_STRING);
        }
        return false;
    }
    if (!rneed(r, (size_t) len)) {
        return false;
    }
    char * s = malloc((size_t) len + 1);
    if (s == NULL) {
        return false;
    }
    if (len > 0) {
        memcpy(s, r->base + r->pos, (size_t) len);
    }
    s[len] = '\0';
    r->pos += (size_t) len;
    *out = s;
    return true;
}

/* ------------------------------------------------------------------ */
/* metadata value types                                                */
/* ------------------------------------------------------------------ */

static size_t vtype_size(sllm_gguf_vtype t) {
    switch (t) {
        case SLLM_VT_UINT8:
        case SLLM_VT_INT8:
        case SLLM_VT_BOOL:    return 1;
        case SLLM_VT_UINT16:
        case SLLM_VT_INT16:   return 2;
        case SLLM_VT_UINT32:
        case SLLM_VT_INT32:
        case SLLM_VT_FLOAT32: return 4;
        case SLLM_VT_UINT64:
        case SLLM_VT_INT64:
        case SLLM_VT_FLOAT64: return 8;
        case SLLM_VT_STRING:  return 0; /* variable */
        default:              return 0;
    }
}


/* ------------------------------------------------------------------ */
/* tensor type traits                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    const char * name;
    uint32_t    blck_size;
    uint32_t    type_size;
    bool        supported;   /* we have a kernel for it */
} type_info;

static const type_info g_types[SLLM_TYPE_COUNT] = {
    [SLLM_TYPE_F32]    = { "F32",    1, 4, true  },
    [SLLM_TYPE_F16]    = { "F16",    1, 2, true  },
    [SLLM_TYPE_BF16]   = { "BF16",   1, 2, true  },
    [SLLM_TYPE_Q8_0]   = { "Q8_0",   32, 34, true  },
    [SLLM_TYPE_Q4_0]   = { "Q4_0",   32, 18, true  },
    /* supported == "sllm_dequant_row can decode this". It is NOT "we know the
     * byte layout". Those are different questions and Q4_K/Q6_K were answering
     * the first one as though it were the second: the block and type sizes are
     * real, so sllm_gguf_type_nbytes sizes them correctly and a Q4_K_M file
     * opens and parses fine, but sllm_dequant_row has no case for either and
     * returns SLLM_ERR_TYPE_UNSUPPORTED. sllm_gguf_type_is_supported() was
     * therefore reporting a capability that did not exist.
     *
     * Measured on a real ordinary model (Qwen3-8B-Q4_K_M.gguf): 217 Q4_K and
     * 37 Q6_K tensors size correctly, zero of them decode. Same species as the
     * 64-wide I2_S defect -- a claim that a consumer could trust and that was
     * not true. Corrected to false; the size and layout recognition above is
     * deliberately left intact, because that part was always right and is what
     * lets the container remain general.
     *
     * The dequantisers are NOT implemented by this change. */
    /* Kernels exist and are gated against the REFERENCE golden in
     * tests/test_quant_k.c. The flag was flipped only after that gate passed;
     * until then it stays false and sllm_dequant_row keeps returning
     * SLLM_ERR_TYPE_UNSUPPORTED rather than a wrong answer. */
    [SLLM_TYPE_Q4_K]   = { "Q4_K",   256, 144, true  },
    [SLLM_TYPE_Q6_K]   = { "Q6_K",   256, 210, true  },
    [SLLM_TYPE_I2_S]   = { "I2_S",   1, 0, true },  /* 0 == variable, see nbytes */

    [SLLM_TYPE_Q4_1]   = { "Q4_1",   32, 20, false },
    [SLLM_TYPE_Q5_0]   = { "Q5_0",   32, 22, false },
    [SLLM_TYPE_Q5_1]   = { "Q5_1",   32, 24, false },
    [SLLM_TYPE_Q2_K]   = { "Q2_K",   256, 84, false },
    [SLLM_TYPE_Q3_K]   = { "Q3_K",   256, 110, false },
    [SLLM_TYPE_Q5_K]   = { "Q5_K",   256, 176, false },
    [SLLM_TYPE_IQ4_NL] = { "IQ4_NL", 32, 18, false },
    [SLLM_TYPE_IQ2_S]  = { "IQ2_S",  256, 82, false },
    [SLLM_TYPE_IQ4_XS] = { "IQ4_XS", 256, 136, false },
    [SLLM_TYPE_I8]     = { "I8",     1, 1, false },
    [SLLM_TYPE_I16]    = { "I16",    1, 2, false },
    [SLLM_TYPE_I32]    = { "I32",    1, 4, false },
    [SLLM_TYPE_I64]    = { "I64",    1, 8, false },
    [SLLM_TYPE_F64]    = { "F64",    1, 8, false },
    [SLLM_TYPE_TQ1_0]  = { "TQ1_0",  256, 54, false },
    [SLLM_TYPE_TQ2_0]  = { "TQ2_0",  256, 66, false },
    [SLLM_TYPE_MXFP4]  = { "MXFP4",  32, 17, false },
    [SLLM_TYPE_NVFP4]  = { "NVFP4",  64, 36, false },
    [SLLM_TYPE_I8_S]   = { "I8_S",   1, 0, false },
    [SLLM_TYPE_Q1_0]   = { "Q1_0",   1, 0, false },
    [SLLM_TYPE_TL2]    = { "TL2",    1, 0, false },
};

/*
 * The I2_S tail. Upstream computes an I2_S (or TL1) tensor's size as
 * nbytes/4 + 32, and the mul_mat path then reads the f32 scale at
 * data + n_elements/4. The 32 bytes are alignment slack around that scale.
 */
#define SLLM_I2_S_TAIL_BYTES 32u

/*
 * Sizes above were verified by calling ggml_blck_size()/ggml_type_size() on the
 * pinned reference build, not by transcription, because a wrong block size is
 * a silent data-corruption bug rather than a failure.
 */

const char * sllm_gguf_type_name(sllm_ggml_type type) {
    static char unknown[32];
    if ((int) type < 0 || (int) type >= SLLM_TYPE_COUNT || g_types[type].name == NULL) {
        (void) snprintf(unknown, sizeof(unknown), "UNKNOWN(%d)", (int) type);
        return unknown;
    }
    return g_types[type].name;
}

bool sllm_gguf_type_is_known(uint32_t type) {
    return type < SLLM_TYPE_COUNT && g_types[type].name != NULL;
}

bool sllm_gguf_type_is_supported(sllm_ggml_type type) {
    if ((int) type < 0 || (int) type >= SLLM_TYPE_COUNT) {
        return false;
    }
    return g_types[type].supported;
}

sllm_status sllm_gguf_type_traits(sllm_ggml_type type,
                                  uint32_t * blck_size, uint32_t * type_size) {
    if ((int) type < 0 || (int) type >= SLLM_TYPE_COUNT || g_types[type].name == NULL) {
        return SLLM_ERR_TYPE_UNSUPPORTED;
    }
    /* Type TRAITS are about layout, and are deliberately independent of whether
     * we have a kernel. sllm_gguf_type_nbytes already relies on that separation:
     * it sizes a Q4_K tensor correctly even when no kernel exists, which is what
     * lets a container be parsed and sized without being decodable. Gating traits
     * on `supported` collapsed those two questions into one and made a correctly
     * sized-but-unsupported type unmeasurable, which is how a caller ends up
     * unable to even report the truth about a file. Use
     * sllm_gguf_type_is_supported() to ask about a kernel. */
    if (blck_size != NULL) {
        *blck_size = g_types[type].blck_size;
    }
    if (type_size != NULL) {
        /* A type_size of 0 means "not a fixed block size"; callers must use
         * sllm_gguf_type_nbytes for the true payload size. */
        *type_size = g_types[type].type_size;
    }
    return SLLM_OK;
}

sllm_status sllm_gguf_type_nbytes(sllm_ggml_type type, uint64_t n_elements,
                                  uint64_t * out) {
    if (out == NULL) {
        return SLLM_ERR_ARG;
    }
    if ((int) type < 0 || (int) type >= SLLM_TYPE_COUNT || g_types[type].name == NULL) {
        return SLLM_ERR_TYPE_UNSUPPORTED;
    }

    if (type == SLLM_TYPE_I2_S) {
        /* Four weights per byte, then the scale tail. */
        if (n_elements % 4 != 0) {
            return SLLM_ERR_GGUF_TENSOR;
        }
        *out = n_elements / 4 + SLLM_I2_S_TAIL_BYTES;
        return SLLM_OK;
    }

    const uint32_t blck = g_types[type].blck_size;
    const uint32_t tsz  = g_types[type].type_size;
    if (blck == 0 || tsz == 0) {
        /* A known type whose layout we have not established. The caller must
         * not guess a size. */
        return SLLM_ERR_TYPE_UNSUPPORTED;
    }
    if (n_elements % blck != 0) {
        return SLLM_ERR_GGUF_TENSOR;
    }
    *out = (n_elements / blck) * (uint64_t) tsz;
    return SLLM_OK;
}

/* ------------------------------------------------------------------ */
/* kv parsing                                                          */
/* ------------------------------------------------------------------ */

static void kv_free(sllm_gguf_kv * kv) {
    if (kv == NULL) {
        return;
    }
    free(kv->key);
    if (kv->data != NULL) {
        if (kv->type == SLLM_VT_ARRAY && kv->arr_type == SLLM_VT_STRING) {
            /* A string array owns `n` individually allocated strings. Slots
             * never read are NULL, and free(NULL) is defined, so a partially
             * built array needs no separate cleanup path. */
            char ** strs = (char **) kv->data;
            for (uint64_t i = 0; i < kv->n; ++i) {
                free(strs[i]);
            }
        }
        free(kv->data);
    }
    kv->key  = NULL;
    kv->data = NULL;
}

static sllm_status read_kv(sllm_reader * r, sllm_gguf_kv * kv) {
    memset(kv, 0, sizeof(*kv));

    if (!rstr(r, &kv->key)) {
        return SLLM_ERR_GGUF_TRUNCATED;
    }

    uint32_t vtype = 0;
    if (!ru32(r, &vtype)) {
        kv_free(kv);
        return SLLM_ERR_GGUF_TRUNCATED;
    }
    if (vtype >= SLLM_VT_COUNT) {
        if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
            (void) snprintf(r->err, r->err_len,
                "metadata key '%s' has unknown value type %u", kv->key, vtype);
        }
        kv_free(kv);
        return SLLM_ERR_GGUF_TYPE;
    }

    if (vtype == SLLM_VT_ARRAY) {
        uint32_t atype = 0;
        uint64_t n     = 0;
        if (!ru32(r, &atype) || !ru64(r, &n)) {
            kv_free(kv);
            return SLLM_ERR_GGUF_TRUNCATED;
        }
        if (atype >= SLLM_VT_COUNT || atype == SLLM_VT_ARRAY) {
            if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                (void) snprintf(r->err, r->err_len,
                    "metadata key '%s' is an array of invalid element type %u", kv->key, atype);
            }
            kv_free(kv);
            return SLLM_ERR_GGUF_TYPE;
        }
        if (n > SLLM_GGUF_MAX_ARRAY) {
            if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                (void) snprintf(r->err, r->err_len,
                    "metadata key '%s' declares %llu array elements, over the %u limit",
                    kv->key, (unsigned long long) n, (unsigned) SLLM_GGUF_MAX_ARRAY);
            }
            kv_free(kv);
            return SLLM_ERR_TOO_LARGE;
        }
        kv->type     = SLLM_VT_ARRAY;
        kv->arr_type = (sllm_gguf_vtype) atype;
        kv->n        = n;

        if (atype == SLLM_VT_STRING) {
            /*
             * String arrays, needed by the tokenizer: the vocabulary is one,
             * and it is the largest in practice at 128256 entries.
             *
             * Each element occupies at least 8 bytes in the file, because that
             * is the size of its length prefix. That gives a bound on the
             * pointer array we are about to allocate which comes from the file
             * rather than from the declared count: a header claiming 2^28
             * strings would otherwise ask for 2 GiB of pointers out of a file
             * that has no room for them.
             */
            if (n > (uint64_t) (r->size / sizeof(uint64_t))) {
                if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                    (void) snprintf(r->err, r->err_len,
                        "metadata key '%s' declares %llu strings, more than the %zu bytes left in the file can hold",
                        kv->key, (unsigned long long) n, r->size);
                }
                kv_free(kv);
                return SLLM_ERR_TOO_LARGE;
            }
            if (n > 0) {
                /* Overflow-safe: n is already bounded above by r->size/8. */
                kv->data = calloc((size_t) n, sizeof(char *));
                if (kv->data == NULL) {
                    if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                        (void) snprintf(r->err, r->err_len,
                            "out of memory reading %llu strings for '%s'",
                            (unsigned long long) n, kv->key);
                    }
                    kv_free(kv);
                    return SLLM_ERR_TOO_LARGE;
                }
                for (uint64_t i = 0; i < n; ++i) {
                    char * tmp = NULL;
                    if (!rstr(r, &tmp)) {
                        kv->n = i;   /* free only what was filled in */
                        kv_free(kv);
                        return SLLM_ERR_GGUF_TRUNCATED;
                    }
                    ((char **) kv->data)[i] = tmp;
                }
            }
            kv->elem_size = sizeof(char *);
            return SLLM_OK;
        }

        const size_t esz = vtype_size((sllm_gguf_vtype) atype);
        if (esz == 0) {
            kv_free(kv);
            return SLLM_ERR_GGUF_TYPE;
        }
        /* Overflow-safe size check before the multiply. */
        if (n > (uint64_t) (SIZE_MAX / esz)) {
            kv_free(kv);
            return SLLM_ERR_TOO_LARGE;
        }
        const size_t total = (size_t) n * esz;
        if (!rneed(r, total)) {
            kv_free(kv);
            return SLLM_ERR_GGUF_TRUNCATED;
        }
        kv->elem_size = esz;
        if (n > 0) {
            kv->data = malloc(total);
            if (kv->data == NULL) {
                kv_free(kv);
                return SLLM_ERR_NOMEM;
            }
            if (!rbytes(r, kv->data, total)) {
                kv_free(kv);
                return SLLM_ERR_GGUF_TRUNCATED;
            }
        }
        return SLLM_OK;
    }

    kv->type     = (sllm_gguf_vtype) vtype;
    kv->arr_type = (sllm_gguf_vtype) vtype;
    kv->n        = 1;

    if (vtype == SLLM_VT_STRING) {
        char * s = NULL;
        if (!rstr(r, &s)) {
            kv_free(kv);
            return SLLM_ERR_GGUF_TRUNCATED;
        }
        kv->data      = s;
        kv->elem_size = 1;
        return SLLM_OK;
    }

    const size_t esz = vtype_size(kv->type);
    if (esz == 0) {
        kv_free(kv);
        return SLLM_ERR_GGUF_TYPE;
    }
    kv->data = malloc(esz);
    if (kv->data == NULL) {
        kv_free(kv);
        return SLLM_ERR_NOMEM;
    }
    kv->elem_size = esz;
    if (!rbytes(r, kv->data, esz)) {
        kv_free(kv);
        return SLLM_ERR_GGUF_TRUNCATED;
    }
    return SLLM_OK;
}

/* ------------------------------------------------------------------ */
/* container parsing                                                   */
/* ------------------------------------------------------------------ */

static sllm_status parse(sllm_reader * r, sllm_gguf * g) {
    char magic[4];
    if (!rbytes(r, magic, 4)) {
        return SLLM_ERR_GGUF_TRUNCATED;
    }
    if (memcmp(magic, SLLM_GGUF_MAGIC, 4) != 0) {
        if (r->err != NULL && r->err_len > 0) {
            (void) snprintf(r->err, r->err_len,
                "not a GGUF file: magic is %02x %02x %02x %02x, expected 'G' 'G' 'U' 'F'",
                magic[0], magic[1], magic[2], magic[3]);
        }
        return SLLM_ERR_GGUF_MAGIC;
    }

    if (!ru32(r, &g->version)) {
        return SLLM_ERR_GGUF_TRUNCATED;
    }
    if (g->version != SLLM_GGUF_VERSION) {
        if (r->err != NULL && r->err_len > 0) {
            (void) snprintf(r->err, r->err_len,
                "GGUF version %u is not supported (this build reads version %u)",
                g->version, SLLM_GGUF_VERSION);
        }
        return SLLM_ERR_GGUF_VERSION;
    }

    if (!ru64(r, &g->n_tensors) || !ru64(r, &g->n_kv)) {
        return SLLM_ERR_GGUF_TRUNCATED;
    }
    if (g->n_tensors > SLLM_GGUF_MAX_TENSORS) {
        if (r->err != NULL && r->err_len > 0) {
            (void) snprintf(r->err, r->err_len,
                "header declares %llu tensors, over the %u limit",
                (unsigned long long) g->n_tensors, (unsigned) SLLM_GGUF_MAX_TENSORS);
        }
        return SLLM_ERR_TOO_LARGE;
    }
    if (g->n_kv > SLLM_GGUF_MAX_KV) {
        if (r->err != NULL && r->err_len > 0) {
            (void) snprintf(r->err, r->err_len,
                "header declares %llu metadata entries, over the %u limit",
                (unsigned long long) g->n_kv, (unsigned) SLLM_GGUF_MAX_KV);
        }
        return SLLM_ERR_TOO_LARGE;
    }

    /* --- metadata --- */
    if (g->n_kv > 0) {
        g->kv = calloc((size_t) g->n_kv, sizeof(sllm_gguf_kv));
        if (g->kv == NULL) {
            return SLLM_ERR_NOMEM;
        }
    }
    for (uint64_t i = 0; i < g->n_kv; ++i) {
        const sllm_status st = read_kv(r, &g->kv[i]);
        if (st != SLLM_OK) {
            return st;
        }
        /* Duplicate keys are legal-ish in the wild but ambiguous for us, and
         * silently taking the first or last would be a correctness trap. */
        for (uint64_t j = 0; j < i; ++j) {
            if (strcmp(g->kv[j].key, g->kv[i].key) == 0) {
                if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                    (void) snprintf(r->err, r->err_len,
                        "metadata key '%s' appears more than once", g->kv[i].key);
                }
                return SLLM_ERR_GGUF_KEYDUP;
            }
        }
    }

    /* --- tensors --- */
    if (g->n_tensors > 0) {
        g->tensors = calloc((size_t) g->n_tensors, sizeof(sllm_gguf_tensor));
        if (g->tensors == NULL) {
            return SLLM_ERR_NOMEM;
        }
    }
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        sllm_gguf_tensor * t = &g->tensors[i];
        if (!rstr(r, &t->name)) {
            return SLLM_ERR_GGUF_TRUNCATED;
        }
        if (!ru32(r, &t->n_dims)) {
            return SLLM_ERR_GGUF_TRUNCATED;
        }
        if (t->n_dims == 0 || t->n_dims > SLLM_GGUF_MAX_DIMS) {
            if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                (void) snprintf(r->err, r->err_len,
                    "tensor '%s' has %u dimensions, expected 1..%d",
                    t->name, t->n_dims, SLLM_GGUF_MAX_DIMS);
            }
            return SLLM_ERR_GGUF_TENSOR;
        }
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < t->n_dims; ++d) {
            if (!ru64(r, &t->ne[d])) {
                return SLLM_ERR_GGUF_TRUNCATED;
            }
            if (t->ne[d] == 0 || nelem > UINT64_MAX / t->ne[d]) {
                if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                    (void) snprintf(r->err, r->err_len,
                        "tensor '%s' has an empty or overflowing dimension", t->name);
                }
                return SLLM_ERR_GGUF_TENSOR;
            }
            nelem *= t->ne[d];
        }
        for (uint32_t d = t->n_dims; d < SLLM_GGUF_MAX_DIMS; ++d) {
            t->ne[d] = 1;
        }

        uint32_t type = 0;
        if (!ru32(r, &type)) {
            return SLLM_ERR_GGUF_TRUNCATED;
        }
        if (!ru64(r, &t->offset)) {
            return SLLM_ERR_GGUF_TRUNCATED;
        }
        if (!sllm_gguf_type_is_known(type)) {
            if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                (void) snprintf(r->err, r->err_len,
                    "tensor '%s' has type %u, which is not a GGUF tensor type", t->name, type);
            }
            return SLLM_ERR_GGUF_TENSOR;
        }
        t->type = (sllm_ggml_type) type;
        if (sllm_gguf_type_nbytes(t->type, nelem, &t->nbytes) == SLLM_OK) {
            t->nbytes_known = true;
        } else {
            /*
             * A valid GGUF type whose layout we have not established. The
             * file is not malformed and we will say so precisely, but we
             * cannot bounds-check the payload, so extent validation is
             * skipped for it. The model is rejected later by type support,
             * not silently misread here.
             */
            t->nbytes_known = false;
            t->nbytes = 0;
        }
    }

    /* --- alignment --- */
    g->alignment = SLLM_GGUF_DEFAULT_ALIGNMENT;
    const sllm_gguf_kv * align_kv = sllm_gguf_find_kv(g, "general.alignment");
    if (align_kv != NULL && align_kv->type == SLLM_VT_UINT32 && align_kv->data != NULL) {
        uint32_t a = 0;
        memcpy(&a, align_kv->data, sizeof(a));
        if (a == 0 || (a & (a - 1)) != 0) {
            if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                (void) snprintf(r->err, r->err_len,
                    "general.alignment is %u, which is not a non-zero power of two", a);
            }
            return SLLM_ERR_GGUF_ALIGNMENT;
        }
        g->alignment = a;
    }

    /* --- data blob placement --- */
    const uint64_t meta_end = (uint64_t) r->pos;
    g->data_offset = (meta_end + g->alignment - 1) & ~(g->alignment - 1);
    if (g->data_offset > g->file_size) {
        if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
            (void) snprintf(r->err, r->err_len,
                "aligned data offset %llu is past the end of the %llu byte file",
                (unsigned long long) g->data_offset, (unsigned long long) g->file_size);
        }
        return SLLM_ERR_GGUF_TRUNCATED;
    }
    g->data_size = g->file_size - g->data_offset;

    /* --- validate every tensor extent against the blob --- */
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        sllm_gguf_tensor * t = &g->tensors[i];
        if (!t->nbytes_known) {
            continue;
        }
        if (t->offset > g->data_size || t->nbytes > g->data_size - t->offset) {
            if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                (void) snprintf(r->err, r->err_len,
                    "tensor '%s' spans data bytes [%llu, %llu) but the blob is only %llu bytes",
                    t->name,
                    (unsigned long long) t->offset,
                    (unsigned long long) (t->offset + t->nbytes),
                    (unsigned long long) g->data_size);
            }
            return SLLM_ERR_GGUF_LAYOUT;
        }
        t->data = r->base + g->data_offset + t->offset;
    }

    /* --- extents must not overlap --- */
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        if (!g->tensors[i].nbytes_known) {
            continue;
        }
        for (uint64_t j = i + 1; j < g->n_tensors; ++j) {
            if (!g->tensors[j].nbytes_known) {
                continue;
            }
            const sllm_gguf_tensor * a = &g->tensors[i];
            const sllm_gguf_tensor * b = &g->tensors[j];
            if (a->offset < b->offset + b->nbytes && b->offset < a->offset + a->nbytes) {
                if (r->err != NULL && r->err_len > 0 && r->err[0] == '\0') {
                    (void) snprintf(r->err, r->err_len,
                        "tensors '%s' and '%s' claim overlapping data extents", a->name, b->name);
                }
                return SLLM_ERR_GGUF_LAYOUT;
            }
        }
    }

    return SLLM_OK;
}

sllm_status sllm_gguf_open_memory(const void * data, size_t size, sllm_gguf * out,
                                  char * error, size_t error_len) {
    if (data == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    memset(out, 0, sizeof(*out));
    if (error != NULL && error_len > 0) {
        error[0] = '\0';
    }

    sllm_reader r = {
        .base    = (const uint8_t *) data,
        .size    = size,
        .pos     = 0,
        .err     = error,
        .err_len = error_len,
    };

    out->file      = (void *) (uintptr_t) data;
    out->file_size = size;
    out->mapped    = false;

    const sllm_status st = parse(&r, out);
    if (st != SLLM_OK) {
        sllm_gguf_close(out);
        return st;
    }
    return SLLM_OK;
}

sllm_status sllm_gguf_open(const char * path, sllm_gguf * out,
                           char * error, size_t error_len) {
    if (path == NULL || out == NULL) {
        return SLLM_ERR_ARG;
    }
    memset(out, 0, sizeof(*out));
    if (error != NULL && error_len > 0) {
        error[0] = '\0';
    }

    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return sllm_fail(SLLM_ERR_IO, "cannot open '%s': %s", path, strerror(errno));
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        const int e = errno;
        (void) close(fd);
        return sllm_fail(SLLM_ERR_IO, "cannot stat '%s': %s", path, strerror(e));
    }
    if (!S_ISREG(st.st_mode)) {
        (void) close(fd);
        return sllm_fail(SLLM_ERR_IO, "'%s' is not a regular file", path);
    }
    if (st.st_size <= 0) {
        (void) close(fd);
        return sllm_fail(SLLM_ERR_GGUF_TRUNCATED, "'%s' is empty", path);
    }

    const size_t size = (size_t) st.st_size;
    void * base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int e = errno;
    (void) close(fd);
    if (base == MAP_FAILED) {
        return sllm_fail(SLLM_ERR_IO, "cannot map '%s': %s", path, strerror(e));
    }

    sllm_reader r = {
        .base    = (const uint8_t *) base,
        .size    = size,
        .pos     = 0,
        .err     = error,
        .err_len = error_len,
    };

    out->file      = base;
    out->file_size = size;
    out->mapped    = true;

    const sllm_status rc = parse(&r, out);
    if (rc != SLLM_OK) {
        sllm_gguf_close(out);
        return rc;
    }

    sllm_log(SLLM_LOG_DEBUG, "gguf %s: %llu tensors, %llu metadata entries, alignment %llu, data at %llu",
             path, (unsigned long long) out->n_tensors, (unsigned long long) out->n_kv,
             (unsigned long long) out->alignment, (unsigned long long) out->data_offset);
    return SLLM_OK;
}

void sllm_gguf_close(sllm_gguf * g) {
    if (g == NULL) {
        return;
    }
    if (g->kv != NULL) {
        for (uint64_t i = 0; i < g->n_kv; ++i) {
            kv_free(&g->kv[i]);
        }
        free(g->kv);
    }
    if (g->tensors != NULL) {
        for (uint64_t i = 0; i < g->n_tensors; ++i) {
            free(g->tensors[i].name);
        }
        free(g->tensors);
    }
    if (g->mapped && g->file != NULL) {
        (void) munmap(g->file, g->file_size);
    }
    memset(g, 0, sizeof(*g));
}

/* ------------------------------------------------------------------ */
/* lookup                                                              */
/* ------------------------------------------------------------------ */

const sllm_gguf_kv * sllm_gguf_find_kv(const sllm_gguf * g, const char * key) {
    if (g == NULL || key == NULL || g->kv == NULL) {
        return NULL;
    }
    for (uint64_t i = 0; i < g->n_kv; ++i) {
        if (strcmp(g->kv[i].key, key) == 0) {
            return &g->kv[i];
        }
    }
    return NULL;
}

const sllm_gguf_tensor * sllm_gguf_find_tensor(const sllm_gguf * g, const char * name) {
    if (g == NULL || name == NULL || g->tensors == NULL) {
        return NULL;
    }
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        if (strcmp(g->tensors[i].name, name) == 0) {
            return &g->tensors[i];
        }
    }
    return NULL;
}

/* Read element `i` of a numeric kv as a 64-bit value, sign-extending as the
 * declared type requires.
 *
 * For an array the element type is `arr_type`, not `type`: `type` is
 * SLLM_VT_ARRAY and would fall through to the default case, so reading element
 * zero of any array failed. No test covered it, because the only arrays in the
 * acceptance model are the tokenizer's, and the tokenizer was not built yet. */
static bool kv_as_i64(const sllm_gguf_kv * kv, uint64_t i, int64_t * out) {
    if (kv->data == NULL || i >= kv->n) {
        return false;
    }
    const sllm_gguf_vtype et = (kv->type == SLLM_VT_ARRAY) ? kv->arr_type : kv->type;
    const uint8_t * p = (const uint8_t *) kv->data + i * kv->elem_size;
    switch (et) {
        case SLLM_VT_UINT8:  { uint8_t  v; memcpy(&v, p, 1); *out = (int64_t) v; return true; }
        case SLLM_VT_INT8:   { int8_t   v; memcpy(&v, p, 1); *out = v; return true; }
        case SLLM_VT_BOOL:   { uint8_t  v; memcpy(&v, p, 1); *out = v ? 1 : 0; return true; }
        case SLLM_VT_UINT16: { uint16_t v; memcpy(&v, p, 2); *out = (int64_t) v; return true; }
        case SLLM_VT_INT16:  { int16_t  v; memcpy(&v, p, 2); *out = v; return true; }
        case SLLM_VT_UINT32: { uint32_t v; memcpy(&v, p, 4); *out = (int64_t) v; return true; }
        case SLLM_VT_INT32:  { int32_t  v; memcpy(&v, p, 4); *out = v; return true; }
        case SLLM_VT_UINT64: { uint64_t v; memcpy(&v, p, 8); *out = (int64_t) v; return true; }
        case SLLM_VT_INT64:  { int64_t  v; memcpy(&v, p, 8); *out = v; return true; }
        case SLLM_VT_FLOAT32:{ float    v; memcpy(&v, p, 4); *out = (int64_t) v; return true; }
        case SLLM_VT_FLOAT64:{ double   v; memcpy(&v, p, 8); *out = (int64_t) v; return true; }
        default: return false;
    }
}

sllm_status sllm_gguf_kv_u32(const sllm_gguf * g, const char * key, uint32_t * out) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type == SLLM_VT_ARRAY) {
        if (kv->arr_type != SLLM_VT_UINT32 || kv->n < 1) {
            return SLLM_ERR_KV_TYPE;
        }
    } else if (kv->type != SLLM_VT_UINT32) {
        return SLLM_ERR_KV_TYPE;
    }
    int64_t v = 0;
    if (!kv_as_i64(kv, 0, &v) || v < 0 || v > (int64_t) UINT32_MAX) {
        return SLLM_ERR_KV_TYPE;
    }
    *out = (uint32_t) v;
    return SLLM_OK;
}

sllm_status sllm_gguf_kv_u64(const sllm_gguf * g, const char * key, uint64_t * out) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type == SLLM_VT_ARRAY) {
        if (kv->arr_type != SLLM_VT_UINT64 || kv->n < 1) {
            return SLLM_ERR_KV_TYPE;
        }
    } else if (kv->type != SLLM_VT_UINT64) {
        return SLLM_ERR_KV_TYPE;
    }
    int64_t v = 0;
    if (!kv_as_i64(kv, 0, &v) || v < 0) {
        return SLLM_ERR_KV_TYPE;
    }
    *out = (uint64_t) v;
    return SLLM_OK;
}

/*
 * String and typed-array access, for the tokenizer.
 *
 * These hand back a pointer into the parsed kv rather than copying, so the
 * caller must not free it and must not assume it outlives the sllm_gguf. The
 * element count is returned separately because a NULL data pointer is legal
 * for an empty array and is not the same as a missing key.
 */
sllm_status sllm_gguf_kv_str_array(const sllm_gguf * g, const char * key,
                                   char * const * * out, uint64_t * n) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type != SLLM_VT_ARRAY || kv->arr_type != SLLM_VT_STRING) {
        return SLLM_ERR_KV_TYPE;
    }
    *out = (char * const *) kv->data;
    *n   = kv->n;
    return SLLM_OK;
}

sllm_status sllm_gguf_kv_i32_array(const sllm_gguf * g, const char * key,
                                   const int32_t ** out, uint64_t * n) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type != SLLM_VT_ARRAY || kv->arr_type != SLLM_VT_INT32) {
        return SLLM_ERR_KV_TYPE;
    }
    *out = (const int32_t *) kv->data;
    *n   = kv->n;
    return SLLM_OK;
}

sllm_status sllm_gguf_kv_f32_array(const sllm_gguf * g, const char * key,
                                   const float ** out, uint64_t * n) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type != SLLM_VT_ARRAY || kv->arr_type != SLLM_VT_FLOAT32) {
        return SLLM_ERR_KV_TYPE;
    }
    *out = (const float *) kv->data;
    *n   = kv->n;
    return SLLM_OK;
}

sllm_status sllm_gguf_kv_i32(const sllm_gguf * g, const char * key, int32_t * out) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type == SLLM_VT_ARRAY) {
        if (kv->arr_type != SLLM_VT_INT32 || kv->n < 1) {
            return SLLM_ERR_KV_TYPE;
        }
    } else if (kv->type != SLLM_VT_INT32) {
        return SLLM_ERR_KV_TYPE;
    }
    int64_t v = 0;
    if (!kv_as_i64(kv, 0, &v)) {
        return SLLM_ERR_KV_TYPE;
    }
    *out = (int32_t) v;
    return SLLM_OK;
}

sllm_status sllm_gguf_kv_f32(const sllm_gguf * g, const char * key, float * out) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type == SLLM_VT_ARRAY) {
        if (kv->arr_type != SLLM_VT_FLOAT32 || kv->n < 1) {
            return SLLM_ERR_KV_TYPE;
        }
    } else if (kv->type != SLLM_VT_FLOAT32) {
        return SLLM_ERR_KV_TYPE;
    }
    if (kv->data == NULL) {
        return SLLM_ERR_KV_TYPE;
    }
    memcpy(out, kv->data, sizeof(float));
    return SLLM_OK;
}

sllm_status sllm_gguf_kv_str(const sllm_gguf * g, const char * key, const char ** out) {
    const sllm_gguf_kv * kv = sllm_gguf_find_kv(g, key);
    if (kv == NULL) {
        return SLLM_ERR_KV_MISSING;
    }
    if (kv->type != SLLM_VT_STRING || kv->data == NULL) {
        return SLLM_ERR_KV_TYPE;
    }
    *out = (const char *) kv->data;
    return SLLM_OK;
}

void sllm_gguf_describe(const sllm_gguf * g, char * buf, size_t buflen) {
    if (buf == NULL || buflen == 0) {
        return;
    }
    if (g == NULL) {
        buf[0] = '\0';
        return;
    }
    const char * arch = NULL;
    (void) sllm_gguf_kv_str(g, "general.architecture", &arch);
    const char * name = NULL;
    (void) sllm_gguf_kv_str(g, "general.name", &name);
    (void) snprintf(buf, buflen, "gguf v%u arch=%s name=%s tensors=%llu kv=%llu align=%llu blob=%llu",
                    g->version, arch ? arch : "?", name ? name : "?",
                    (unsigned long long) g->n_tensors, (unsigned long long) g->n_kv,
                    (unsigned long long) g->alignment, (unsigned long long) g->data_size);
}
