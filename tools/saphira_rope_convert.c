/*
 * saphira_rope_convert.c — emit the Saphira RoPE semantics contract into a GGUF.
 *
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 *
 * THE CONVERTER REFUSES TO EMIT A FIELD IT CANNOT ESTABLISH.
 *
 * The input is a source-side EVIDENCE file: the facts established from the source
 * model, by hand, from the model's own configuration and implementation code. It
 * is deliberately NOT a filename and deliberately NOT an architecture name. A
 * converter that could name a file and receive conventions would reintroduce
 * exactly the inference this contract was created to forbid.
 *
 * Every required field must be present and parse. A missing field is a refusal
 * naming that field, not a default and not an omission. An artefact that lacks
 * one of these keys will be refused at runtime, so a converter that quietly skips
 * a field it could not determine produces a model that cannot be run -- which is
 * the correct outcome, and visible rather than latent.
 *
 * Usage:
 *   saphira-rope-convert <evidence.txt> <out.gguf>
 *
 * Evidence file format, one "key = value" per line, '#' comments allowed:
 *   pairing = half_split
 *   position_origin = 0
 *   rotary_dim = 128
 *   freq_base = 1000000
 *   scaling_mode = none
 *   scaling_factor = 1.0
 *   source = <provenance string, required>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "saphira_llm/rope_contract.h"

/* -- minimal GGUF writer: magic, version, tensor count, KV table ------------- */

typedef struct { const char * k; uint32_t t; uint64_t u; double f; const char * s; } kv;

/* GGUF value type ids, matching the format. */
enum { T_UINT32 = 4, T_FLOAT32 = 6, T_STRING = 8 };

static void put_str(FILE * f, const char * s, int with_len) {
    const uint64_t n = (uint64_t) strlen(s);
    if (with_len) { fwrite(&n, sizeof n, 1, f); }
    fwrite(s, 1, (size_t) n, f);
}

static void write_gguf(const char * path, const kv * kvs, int n_kv) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s for writing\n", path); exit(2); }
    fwrite("GGUF", 1, 4, f);
    const uint32_t ver = 3;
    const uint64_t n_tensors = 0;
    fwrite(&ver, 4, 1, f);
    fwrite(&n_tensors, 8, 1, f);
    const uint64_t nkv = (uint64_t) n_kv;
    fwrite(&nkv, 8, 1, f);
    for (int i = 0; i < n_kv; ++i) {
        put_str(f, kvs[i].k, 1);                       /* key, length-prefixed */
        const uint32_t t = kvs[i].t;
        fwrite(&t, 4, 1, f);
        switch (t) {
        case T_UINT32: { const uint32_t v = (uint32_t) kvs[i].u; fwrite(&v, 4, 1, f); break; }
        case T_FLOAT32: { const float v = (float) kvs[i].f; fwrite(&v, 4, 1, f); break; }
        case T_STRING: put_str(f, kvs[i].s, 1); break; /* value, length-prefixed */
        default: break;
        }
    }
    /* Pad to a generous boundary. A GGUF reader aligns the tensor-data section
     * that follows the KV table, and rejects the file when that offset runs past
     * the end -- which a metadata-only artefact would otherwise do, since with no
     * tensors there is nothing to fill the gap. Padding past the default 32-byte
     * alignment keeps the artefact readable for any sane alignment setting. */
    {   const long pos = ftell(f);
        const long pad_to = 4096;
        if (pos >= 0 && pos < pad_to) {
            static const char zeros[256] = { 0 };
            long need = pad_to - pos;
            while (need > 0) {
                const long chunk = need < (long) sizeof zeros ? need : (long) sizeof zeros;
                fwrite(zeros, 1, (size_t) chunk, f);
                need -= chunk;
            }
        }
    }
    if (fclose(f) != 0) { fprintf(stderr, "write failed\n"); exit(2); }
}

/* -- evidence parsing -------------------------------------------------------- */

#define MAXV 512
static char ev[MAXV][MAXV];

static const char * lookup(const char * key) {
    for (int i = 0; ev[i][0]; ++i) {
        const char * eq = strchr(ev[i], '=');
        if (!eq) continue;
        size_t klen = (size_t) (eq - ev[i]);
        while (klen && (ev[i][klen - 1] == ' ' || ev[i][klen - 1] == '\t')) klen--;
        if (strlen(key) == klen && strncmp(ev[i], key, klen) == 0) {
            const char * v = eq + 1;
            while (*v == ' ' || *v == '\t') v++;
            return v;
        }
    }
    return NULL;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <evidence.txt> <out.gguf>\n", argv[0]);
        return 2;
    }
    FILE * in = fopen(argv[1], "rb");
    if (!in) { fprintf(stderr, "cannot read evidence file %s\n", argv[1]); return 2; }
    char line[MAXV];
    int n_ev = 0;
    while (fgets(line, sizeof line, in)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0 || line[0] == '#') continue;
        const int cap = (int) (sizeof ev / sizeof ev[0]);
        if (n_ev >= cap - 1) {
            fprintf(stderr, "REFUSED: evidence file has more lines than the %d supported.\n", cap);
            fclose(in);
            return 3;
        }
        snprintf(ev[n_ev], MAXV, "%s", line);
        n_ev++;
    }
    fclose(in);
    ev[n_ev][0] = '\0';   /* explicit terminator; lookup stops here */

    /* Resolve every required field or REFUSE, naming exactly what is missing. */
    const char * pairing   = lookup("pairing");
    const char * origin_s  = lookup("position_origin");
    const char * rdim_s    = lookup("rotary_dim");
    const char * fbase_s   = lookup("freq_base");
    const char * smode     = lookup("scaling_mode");
    const char * sfac_s    = lookup("scaling_factor");
    const char * source    = lookup("source");

    struct { const char * name; const char * val; } req[] = {
        { "pairing",         pairing  },
        { "position_origin", origin_s },
        { "rotary_dim",      rdim_s   },
        { "freq_base",       fbase_s  },
        { "scaling_mode",    smode    },
        { "scaling_factor",  sfac_s   },
        { "source",          source   },
    };
    int missing = 0;
    for (unsigned i = 0; i < sizeof req / sizeof req[0]; ++i) {
        if (!req[i].val || !*req[i].val) {
            fprintf(stderr, "REFUSED: source-side evidence does not establish '%s'.\n", req[i].name);
            missing++;
        }
    }
    if (missing) {
        fprintf(stderr, "REFUSED: %d required field(s) unestablished. No artefact emitted, because an\n"
                        "artefact missing these keys would simply be refused again at runtime.\n", missing);
        return 3;
    }

    /* Validate rather than trust: an unestablished value is not made valid by
     * being present. */
    sllm_rope_type t;
    if (sllm_rope_type_parse(pairing, &t) != SLLM_OK) {
        fprintf(stderr, "REFUSED: pairing '%s' is neither half_split nor adjacent.\n", pairing);
        return 3;
    }
    const long origin = strtol(origin_s, NULL, 10);
    if (origin != 0 && origin != 1) {
        fprintf(stderr, "REFUSED: position_origin must be 0 or 1, got %ld.\n", origin);
        return 3;
    }
    const long rdim = strtol(rdim_s, NULL, 10);
    if (rdim <= 0 || (rdim % 2) != 0) {
        fprintf(stderr, "REFUSED: rotary_dim must be a positive even number, got %ld.\n", rdim);
        return 3;
    }
    const double fbase = strtod(fbase_s, NULL);
    if (!(fbase > 0.0)) {
        fprintf(stderr, "REFUSED: freq_base must be positive, got %g.\n", fbase);
        return 3;
    }
    const double sfac = strtod(sfac_s, NULL);
    if (strcmp(smode, "none") == 0 && sfac != 1.0) {
        fprintf(stderr, "REFUSED: scaling_mode 'none' contradicts scaling_factor %g.\n", sfac);
        return 3;
    }
    if (t == SLLM_ROPE_COUNT) {
        fprintf(stderr, "REFUSED: unusable pairing.\n");
        return 3;
    }

    /* Canonicalise the pairing on the way out. The evidence file may say
     * "neox"; what lands in the artefact is the unambiguous name. */
    const char * canonical = (t == SLLM_ROPE_NEOX) ? SLLM_ROPE_PAIRING_HALF_SPLIT
                                                  : SLLM_ROPE_PAIRING_ADJACENT;

    kv kvs[] = {
        { SLLM_ROPE_K_PAIRING,         T_STRING,  0, 0, canonical },
        { SLLM_ROPE_K_POSITION_ORIGIN, T_UINT32,  (uint64_t) origin, 0, NULL },
        { SLLM_ROPE_K_ROTARY_DIM,      T_UINT32,  (uint64_t) rdim,   0, NULL },
        { SLLM_ROPE_K_FREQ_BASE,       T_FLOAT32, 0, fbase, NULL },
        { SLLM_ROPE_K_SCALING_MODE,    T_STRING,  0, 0, smode },
        { SLLM_ROPE_K_SCALING_FACTOR,  T_FLOAT32, 0, sfac,  NULL },
        { SLLM_ROPE_K_SOURCE,          T_STRING,  0, 0, source },
    };
    const int n = (int) (sizeof kvs / sizeof kvs[0]);
    write_gguf(argv[2], kvs, n);

    printf("  emitted %s: pairing=%s position_origin=%ld rotary_dim=%ld freq_base=%g "
           "scaling_mode=%s scaling_factor=%g\n",
           argv[2], canonical, origin, rdim, fbase, smode, sfac);
    printf("  source: %s\n", source);
    return 0;
}