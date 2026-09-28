# Formats

Layouts are recorded here as they were **verified against the pinned reference**,
not transcribed from memory. A wrong block size is a silent data-corruption bug
rather than a failure, so every number here was checked against
`ggml_blck_size()` / `ggml_type_size()` on the reference build, or read out of
the pinned source.

## GGUF v3 container

```
1. magic            "GGUF"                       4 bytes
2. version          uint32                       4 bytes   (3)
3. tensor_count     uint64                       8 bytes
4. kv_count         uint64                       8 bytes
5. kv_count entries of:
     key            uint64 length + bytes        (no NUL)
     value type     uint32
     value          depends on type; arrays carry an element type (uint32)
                    and a length (uint64) first
6. tensor_count entries of:
     name           uint64 length + bytes
     n_dims         uint32                       1..4
     dims           uint64[n_dims]
     type           uint32
     offset         uint64   from the data blob
7. data blob, aligned to general.alignment (default 32)
```

All integers little-endian. All enums are int32. Bools are int8.

### Metadata value types

| # | Name | Bytes |
| ---: | --- | ---: |
| 0 | UINT8 | 1 |
| 1 | INT8 | 1 |
| 2 | UINT16 | 2 |
| 3 | INT16 | 2 |
| 4 | UINT32 | 4 |
| 5 | INT32 | 4 |
| 6 | FLOAT32 | 4 |
| 7 | BOOL | 1 |
| 8 | STRING | uint64 + bytes |
| 9 | ARRAY | uint32 element type + uint64 count + payload |
| 10 | UINT64 | 8 |
| 11 | INT64 | 8 |
| 12 | FLOAT64 | 8 |

### What the parser enforces

Every read is bounds-checked against the mapped length. Before any allocation is
sized from a file-supplied length, that length is range-checked. Rejections are
specific:

| Condition | Status |
| --- | --- |
| wrong magic | `SLLM_ERR_GGUF_MAGIC` |
| version != 3 | `SLLM_ERR_GGUF_VERSION` |
| ends inside a structure | `SLLM_ERR_GGUF_TRUNCATED` |
| string length over 2^30 | `SLLM_ERR_GGUF_TRUNCATED` |
| unknown metadata value type | `SLLM_ERR_GGUF_TYPE` |
| same metadata key twice | `SLLM_ERR_GGUF_KEYDUP` |
| 0 or > 4 dims, or a zero dim | `SLLM_ERR_GGUF_TENSOR` |
| element count not a whole number of blocks | `SLLM_ERR_GGUF_TENSOR` |
| type number not in the format | `SLLM_ERR_GGUF_TENSOR` |
| `general.alignment` not a power of two | `SLLM_ERR_GGUF_ALIGNMENT` |
| tensor extent past the blob | `SLLM_ERR_GGUF_LAYOUT` |
| two tensors claiming overlapping extents | `SLLM_ERR_GGUF_LAYOUT` |
| counts over the safety limits | `SLLM_ERR_TOO_LARGE` |

Duplicate metadata keys are refused rather than resolved by taking the first or
last occurrence: either choice is a correctness trap waiting for a model that
happens to disagree.

## Tensor storage traits

Verified against the pinned ggml. `blck` is elements per stored block.

| Type | # | blck | bytes | supported |
| --- | ---: | ---: | ---: | --- |
| F32 | 0 | 1 | 4 | yes |
| F16 | 1 | 1 | 2 | yes |
| BF16 | 30 | 1 | 2 | yes |
| Q8_0 | 8 | 32 | 34 | yes |
| Q4_0 | 2 | 32 | 18 | yes |
| Q4_K | 12 | 256 | 144 | yes |
| Q6_K | 14 | 256 | 210 | yes |
| **I2_S** | **36** | **1** | **variable** | **yes** |
| Q4_1 | 3 | 32 | 20 | no |
| Q5_0 | 6 | 32 | 22 | no |
| Q5_1 | 7 | 32 | 24 | no |
| Q2_K | 10 | 256 | 84 | no |
| Q3_K | 11 | 256 | 110 | no |
| Q5_K | 13 | 256 | 176 | no |
| IQ4_NL | 20 | 32 | 18 | no |
| IQ2_S | 22 | 256 | 82 | no |
| IQ4_XS | 23 | 256 | 136 | no |
| I8 | 24 | 1 | 1 | no |
| I16 | 25 | 1 | 2 | no |
| I32 | 26 | 1 | 4 | no |
| I64 | 27 | 1 | 8 | no |
| F64 | 28 | 1 | 8 | no |
| TQ1_0 | 34 | 256 | 54 | no |
| TQ2_0 | 35 | 256 | 66 | no |
| MXFP4 | 39 | 32 | 17 | no |
| NVFP4 | 40 | 64 | 36 | no |
| I8_S | 37 | 1 | variable | no |
| Q1_0 | 41 | 1 | variable | no |
| TL2 | 42 | 1 | variable | no |

NVFP4 is 64/36, not 32/17 — that is MXFP4. Getting it wrong is the kind of
error that reads valid data from the wrong offset.

**Known is not the same as supported.** A model using Q5_K is a valid GGUF that
we have no kernel for, and the error says exactly that. It is not reported as a
malformed file, and the model is never silently misread.

## I2_S — the BitNet ternary type

The one non-fixed-size format, and the most important one.

```
payload = n_elements / 4  +  32 bytes
```

* Four weights per byte, two bits each.
* The trailing 32 bytes hold the per-tensor `f32` scale in the first four. The
  scale therefore lives at `data + n_elements/4`, which is exactly where
  upstream's mul_mat reads it.
* `n_elements` must be a multiple of 4.

Two traps, both of which produce *plausible wrong numbers* rather than crashes:

1. ggml's own `type_traits` table gives I2_S a nominal `type_size` of 1 and
   `blck_size` of 1, which is a fiction. It is compensated for by a special
   case in the tensor `nbytes` computation. A loader that trusts the traits
   table reads the scale from the wrong offset and produces confident garbage.
2. `n_elements/4` must use the *element* count, not the byte count of a
   sub-block, or the offset drifts by a factor of four.

### Value encoding

The converter states it directly: ternary `{-1, 0, +1}` maps to I2_S codes
`{0, 1, 2}`. Codes are therefore **non-negative**, which matters for the
kernel: `dpbusd` takes an unsigned first operand, so the weights can be passed
straight in and only the activations are signed. A signed-by-signed port would
need a bias-and-correct and would read every negative code as a large positive
one. `src/kernel_probe.c` carries that lesson deliberately.

### Layout inside the packed data

The BLAST layout is a transposed tile. Each 32-byte chunk holds 128 weights
split into four colour groups of 32 by bit shifts 0/2/4/6, each matched against
32 int8 activations. A group spans 2048 weights against 4096 activations. This
is recorded here because it is a *storage* decision made at conversion time, not
a kernel decision, and getting it wrong cannot be detected by a shape check.

### Scale selection

The converter uses "the first nonzero absolute value", matching C's
`quantize_i2_s`, not the maximum. Recorded because max-versus-first-nonzero is
an easy divergence that produces subtly different output.

## Runtime output epilogue

Reproduced verbatim, including the questionable part:

```
dst = (dot - act_sums[i]) / act_scales[i] * i2_scale
```

The `- act_sums` term assumes `sum(w_k) == 0` over K, which is not generally
true. Upstream applies it, so parity requires applying it. Correcting it would
be a divergence from the reference, not an improvement, and would fail the
parity gate for a reason that has nothing to do with the gate.

## Reference model facts

`/var/lib/spoon/models/ggml-model-i2_s.gguf`

| | |
| --- | --- |
| container | GGUF v3 |
| architecture | `bitnet-b1.58` |
| name | `bitnet2b` |
| tensors | 332 |
| metadata entries | 24 |
| alignment | 32 |
| data blob | 1,179,449,920 bytes |
| parameters | 2,412,820,480 |
| tokenizer | `gpt2` byte-level BPE, 128256 tokens, BOS 128000 |

The blob size is the number to trust: saphira-llm's parser and llama-bench
arrive at 1,179,449,920 bytes independently, from the same file. That
cross-check is why the container layer is considered correct.
