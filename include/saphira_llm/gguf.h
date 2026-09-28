/*
 * gguf.h — GGUF container parsing.
 * Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
 * https://www.akadata.co.uk
 *
 * Licensed under the MIT License — see LICENSE.
 * Part of Saphira Linux (https://saphira.vm2.uk).
 */
#ifndef SAPHIRA_LLM_GGUF_H
#define SAPHIRA_LLM_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <saphira_llm/status.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SLLM_GGUF_MAGIC        "GGUF"
#define SLLM_GGUF_VERSION      3u
#define SLLM_GGUF_DEFAULT_ALIGNMENT 32u

/*
 * Hard ceilings. A GGUF file is untrusted input; these bounds stop a corrupt
 * or hostile header from turning into a huge allocation or an overflow before
 * anything has been validated.
 */
#define SLLM_GGUF_MAX_TENSORS      (1u << 20)
#define SLLM_GGUF_MAX_KV           (1u << 22)
#define SLLM_GGUF_MAX_STRING       (1u << 30)
#define SLLM_GGUF_MAX_DIMS         4
#define SLLM_GGUF_MAX_ARRAY        (1u << 28)

/* ggml tensor type numbers, as stored on disk. */
typedef enum sllm_ggml_type {
    SLLM_TYPE_F32     = 0,
    SLLM_TYPE_F16     = 1,
    SLLM_TYPE_Q4_0    = 2,
    SLLM_TYPE_Q4_1    = 3,
    SLLM_TYPE_Q5_0    = 6,
    SLLM_TYPE_Q5_1    = 7,
    SLLM_TYPE_Q8_0    = 8,
    SLLM_TYPE_Q2_K    = 10,
    SLLM_TYPE_Q3_K    = 11,
    SLLM_TYPE_Q4_K    = 12,
    SLLM_TYPE_Q5_K    = 13,
    SLLM_TYPE_Q6_K    = 14,
    SLLM_TYPE_IQ4_NL  = 20,
    SLLM_TYPE_IQ2_S   = 22,
    SLLM_TYPE_IQ4_XS  = 23,
    SLLM_TYPE_I8      = 24,
    SLLM_TYPE_I16     = 25,
    SLLM_TYPE_I32     = 26,
    SLLM_TYPE_I64     = 27,
    SLLM_TYPE_F64     = 28,
    SLLM_TYPE_BF16    = 30,
    SLLM_TYPE_TQ1_0   = 34,
    SLLM_TYPE_TQ2_0   = 35,
    SLLM_TYPE_I2_S    = 36,
    SLLM_TYPE_I8_S    = 37,
    SLLM_TYPE_MXFP4   = 39,
    SLLM_TYPE_NVFP4   = 40,
    SLLM_TYPE_Q1_0    = 41,
    SLLM_TYPE_TL2     = 42,
    SLLM_TYPE_COUNT   = 43
} sllm_ggml_type;

/* Metadata value type, as stored on disk. */
typedef enum sllm_gguf_vtype {
    SLLM_VT_UINT8   = 0,
    SLLM_VT_INT8    = 1,
    SLLM_VT_UINT16  = 2,
    SLLM_VT_INT16   = 3,
    SLLM_VT_UINT32  = 4,
    SLLM_VT_INT32   = 5,
    SLLM_VT_FLOAT32 = 6,
    SLLM_VT_BOOL    = 7,
    SLLM_VT_STRING  = 8,
    SLLM_VT_ARRAY   = 9,
    SLLM_VT_UINT64  = 10,
    SLLM_VT_INT64   = 11,
    SLLM_VT_FLOAT64 = 12,
    SLLM_VT_COUNT   = 13
} sllm_gguf_vtype;

typedef struct sllm_gguf_kv {
    char           * key;      /* owned, NUL-terminated            */
    sllm_gguf_vtype type;      /* element type, or VTYPE_ARRAY    */
    sllm_gguf_vtype arr_type;  /* element type when type==ARRAY    */
    uint64_t        n;         /* element count (array length, 1 otherwise) */
    void           * data;     /* owned                            */
    size_t          elem_size; /* bytes per element in `data`     */
} sllm_gguf_kv;

typedef struct sllm_gguf_tensor {
    char           * name;     /* owned                             */
    uint32_t        n_dims;
    uint64_t        ne[SLLM_GGUF_MAX_DIMS];
    sllm_ggml_type  type;
    uint64_t        offset;    /* from the start of the data blob   */
    uint64_t        nbytes;    /* size of the tensor payload        */
    bool            nbytes_known; /* false when the layout is not one
                                     we have established, in which case
                                     extent validation was skipped     */
    const void    * data;     /* into the mapped file, set by open  */
} sllm_gguf_tensor;

typedef struct sllm_gguf {
    void           * file;         /* mmap base, or NULL when from memory  */
    size_t           file_size;
    bool             mapped;

    uint32_t         version;
    uint64_t         alignment;
    uint64_t         data_offset;  /* absolute file offset of the blob    */
    uint64_t         data_size;

    sllm_gguf_kv   * kv;
    uint64_t         n_kv;

    sllm_gguf_tensor * tensors;
    uint64_t         n_tensors;
} sllm_gguf;

/*
 * Map a file read-only and parse its header, metadata and tensor table. The
 * tensor payloads are NOT copied: `tensor.data` points into the mapping.
 *
 * Returns a specific status for every rejection reason — bad magic, unsupported
 * version, truncation, absurd string length, unknown value type, overlapping
 * tensor extents, and so on — because "your file is broken" is useless and
 * "tensor 41 'blk.3.attn_q.weight' claims an offset that runs past the end of
 * the file" is not.
 *
 * A NUL-terminated `error` buffer, when supplied, receives a human-readable
 * explanation suitable for printing directly to the user.
 */
sllm_status sllm_gguf_open(const char * path, sllm_gguf * out,
                           char * error, size_t error_len);

sllm_status sllm_gguf_open_memory(const void * data, size_t size, sllm_gguf * out,
                                  char * error, size_t error_len);

void sllm_gguf_close(sllm_gguf * gguf);

/* Metadata lookup. Returns NULL when absent. */
const sllm_gguf_kv * sllm_gguf_find_kv(const sllm_gguf * gguf, const char * key);

/*
 * Typed metadata accessors. Each returns SLLM_ERR_KV_MISSING or
 * SLLM_ERR_KV_TYPE rather than guessing, so a model with a string where we
 * expect a uint32 fails cleanly at load time.
 */
sllm_status sllm_gguf_kv_u32(const sllm_gguf * gguf, const char * key, uint32_t * out);
sllm_status sllm_gguf_kv_u64(const sllm_gguf * gguf, const char * key, uint64_t * out);
sllm_status sllm_gguf_kv_i32(const sllm_gguf * gguf, const char * key, int32_t * out);
sllm_status sllm_gguf_kv_f32(const sllm_gguf * gguf, const char * key, float * out);
sllm_status sllm_gguf_kv_str(const sllm_gguf * gguf, const char * key, const char ** out);

/* Tensor lookup by name. Returns NULL when absent. */
const sllm_gguf_tensor * sllm_gguf_find_tensor(const sllm_gguf * gguf, const char * name);

/*
 * Storage traits for a tensor type.
 *
 *   blck_size  elements per stored block (1 for plain types)
 *   type_size  bytes per block
 *
 * Returns SLLM_ERR_TYPE_UNSUPPORTED for a type we do not implement. Note that
 * "known to the format" and "implemented by us" are different questions; see
 * sllm_gguf_type_is_known.
 */
sllm_status sllm_gguf_type_traits(sllm_ggml_type type,
                                  uint32_t * blck_size, uint32_t * type_size);

/*
 * Payload size in bytes for `n_elements` values of `type`.
 *
 * This is the function callers want. sllm_gguf_type_traits cannot express
 * every type, because not all of them are fixed-size blocks. The BitNet
 * ternary type I2_S packs four weights per byte and then appends a 32-byte
 * tail holding the f32 scale, so its size is `n_elements/4 + 32` and depends
 * on the element count. Treating it as a fixed 1-byte-per-element block, as
 * ggml's own type_traits table nominally does, is how a loader ends up
 * reading a scale from the wrong offset.
 *
 * Returns SLLM_ERR_TYPE_UNSUPPORTED for a type whose layout we have not
 * established. A container may still parse; the tensor is then flagged
 * size-unknown and the model is rejected later with a specific message.
 */
sllm_status sllm_gguf_type_nbytes(sllm_ggml_type type, uint64_t n_elements,
                                  uint64_t * out);

/* True if the number is a type the GGUF format defines at all. */
bool sllm_gguf_type_is_known(uint32_t type);

/* True if we have a kernel for it. */
bool sllm_gguf_type_is_supported(sllm_ggml_type type);

/* Canonical name, e.g. "I2_S". "UNKNOWN(123)" for unrecognised. Never NULL. */
const char * sllm_gguf_type_name(sllm_ggml_type type);

/* Stable one-line summary of the container, for the log. */
void sllm_gguf_describe(const sllm_gguf * gguf, char * buf, size_t buflen);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_GGUF_H */
