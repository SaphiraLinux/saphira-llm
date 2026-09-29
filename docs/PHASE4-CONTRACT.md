# Phase 4: the forward pass, as read from the reference

Everything here was read out of the pinned tree at the revision in
`PROVENANCE.md`, not inferred and not recalled. Where a value could plausibly be
something else, the file and line are given. This exists so the implementation
can be checked against a specification rather than against a memory of one.

Model: `/var/lib/spoon/models/ggml-model-i2_s.gguf`, `bitnet-b1.58`.

## Hyperparameters

Read from the model's own metadata, not assumed:

| key | value |
| --- | ---: |
| `bitnet-b1.58.block_count` | 30 |
| `bitnet-b1.58.embedding_length` | 2560 |
| `bitnet-b1.58.feed_forward_length` | 6912 |
| `bitnet-b1.58.attention.head_count` | 20 |
| `bitnet-b1.58.attention.head_count_kv` | 5 |
| `bitnet-b1.58.rope.dimension_count` | 128 |
| `bitnet-b1.58.rope.freq_base` | 500000.0 |
| `bitnet-b1.58.attention.layer_norm_rms_epsilon` | 1e-5 |
| `bitnet-b1.58.context_length` | 4096 |
| `bitnet-b1.58.vocab_size` | 128256 |

So `n_embd_head = 2560/20 = 128`, and `n_embd_gqa = 128*5 = 640`. Grouped-query
attention with a ratio of 4 query heads per KV head. `n_rot = 128` equals the
head dimension, so RoPE covers the whole head and there is no partial rotation
and no pass-through tail.

## Tensor inventory

332 tensors, and the split matters because it says which arithmetic is exact:

| count | type | what |
| ---: | --- | --- |
| 210 | I2_S | the seven projections per layer: `wq` `wk` `wv` `wo` `ffn_up` `ffn_gate` `ffn_down` (7 x 30) |
| 121 | F32 | the norms: `attn_norm` `attn_sub_norm` `ffn_norm` `ffn_sub_norm` per layer plus one `output_norm` (4 x 30 + 1) |
| 1 | F16 | `token_embd`, which is also the `lm_head` |

Consequence for the parity gate: **every per-layer projection is integer
arithmetic** (I2_S weight times an int8-quantised activation, accumulated in
int32), so 210 of the 211 weight tensors contribute no float error at all. The
only float accumulation in the model is the tied `lm_head` over 2560 elements,
and the attention's QK and softmax-V products.

There are no `*_scale` and no `*_bias` tensors: `llm_model_bitnet` declares them
`TENSOR_NOT_REQUIRED` and this model omits them, so `build_lora_mm` is a plain
matmul. There is no `swiglu_clamp_shexp` metadata either, so the FFN is not
clamped.

## The graph

`llama_model_bitnet::graph::graph`, `src/models/bitnet.cpp:60-171`.

```
inpL = tok_embd[token]                                   # F16, 2560

for il in 0 .. 29:
    inpSA = inpL

    cur = rmsnorm(inpL, attn_norm[il], eps)              # F32 weight
    Q = wq  @ cur                                        # I2_S, 2560 -> 2560
    K = wk  @ cur                                        # I2_S, 2560 ->  640
    V = wv  @ cur                                        # I2_S, 2560 ->  640
    Q = rope_neox(Q, pos, n_rot=128, freq_base=500000)
    K = rope_neox(K, pos, n_rot=128, freq_base=500000)

    cur = attention(Q, K, V, scale = 1/sqrtf(128))       # 20 q heads, 5 kv heads
    cur = rmsnorm(cur, attn_sub_norm[il], eps)
    cur = wo @ cur                                        # I2_S, 2560 -> 2560

    ffn_inp = cur + inpSA
    cur = rmsnorm(ffn_inp, ffn_norm[il], eps)
    cur = swiglu(ffn_gate(x), ffn_up(x))                 # both from the same x
    cur = rmsnorm(cur, ffn_sub_norm[il], eps)            # note: width 6912
    cur = ffn_down @ cur                                  # I2_S, 6912 -> 2560
    cur = cur + ffn_inp

    inpL = cur

cur = rmsnorm(inpL, output_norm, eps)
logits = tok_embd @ cur                                   # F16, tied
```

Three details in that listing are easy to get wrong and each one changes every
logit:

**The FFN gate and up are parallel, not sequential.**
`build_ffn(cur, ffn_up, NULL, ffn_up_s, ffn_gate, NULL, ffn_gate_s, ...,
LLM_FFN_SILU, LLM_FFN_PAR, il)` at `llama-graph.cpp:119`. With `LLM_FFN_PAR`
the gate is computed from the block's *input* `cur`, not from the up
projection's output `tmp` (`llama-graph.cpp:1613-1617`). Then
`LLM_FFN_SILU` with a parallel gate calls `ggml_swiglu_split(ctx0, cur, tmp)`
(`llama-graph.cpp:1638`), which is `silu(gate) * up`. So both projections read
the same normalised input.

**`ffn_sub_norm` has width 6912, not 2560.** It normalises the FFN's hidden
width between the SwiGLU and the down projection. `llm_model_bitnet` declares
it `{n_ff}`. Applying a 2560-wide norm there, or skipping it, changes the
output.

**There is no `ffn_inp` residual naming collision.** `ffn_inp = cur + inpSA` is
formed *after* the attention block and is then the input to `ffn_norm`, and the
same tensor is the residual added after `ffn_down`. The attention output is not
added to `inpSA` separately; it is added once, as `ffn_inp`.

## Op contracts

### RMSNorm

`ggml_compute_forward_rms_norm_f32`, `ggml/src/ggml-cpu/ops.cpp:3793`.

```c
ggml_float sum = 0.0;                    // ggml_float is double
for (i = 0; i < ne00; ++i) sum += (ggml_float)(x[i] * x[i]);
const float mean  = sum / ne00;          // narrows to f32 HERE
const float scale = 1.0f / sqrtf(mean + eps);
y[i] = x[i] * scale * w[i];              // the weight multiply is a separate op
```

Two things are pinned by that snippet. The sum accumulates in double and the
mean is narrowed to `float` *before* the epsilon is added and *before* the
`sqrtf`. Computing the square root in double and narrowing afterwards is more
accurate and about an ulp away, and an ulp is enough to land on the other side
of a rounding boundary when the next layer quantises to int8. The reference's
grouping is the specification, so ours matches it.

`build_norm` multiplies by the weight in a separate `ggml_mul`, so the
association is `(x[i] * scale) * w[i]`.

### I2_S matmul

`ggml_compute_forward_mul_mat`, `ggml/src/ggml-cpu/ggml-cpu.c:1382-1560`.

An F32 activation row is quantised to int8 by `quantize_row_i8_s`, which also
produces a per-row scale and a per-row sum of the quantised values. The integer
dot product is then computed, and the epilogue is applied per output element:

```c
const float   post_scale = ws / act_scales[col];   // ONCE per activation column
const int32_t asum       = act_sums[col];
dst_row[row] = (tmp[row] - asum) * post_scale;
```

`ws` is the weight matrix's single I2_S scale, at `data + ne00*ne01/4`.

The division happens **once per activation column**, not per element, and the
per-element work is a subtraction and a multiply. There is a second, different
form of this epilogue in the tree, at `ggml/src/ggml-bitnet-compute.c:164`:

```c
tmp[row] = (tmp[row] - act_sums[i1]) / (act_scales[i1]) * (*scale);
```

which divides and multiplies per element and groups differently, so it rounds
differently. **That file is not the path the model's forward pass runs.** It
belongs to the standalone BitNet compute path. The graph goes through
`ggml_mul_mat`, so `ggml-cpu.c` is the contract. Recording this because both
forms are in the tree, they are mathematically equal, and picking the wrong one
is a silent ulp-level divergence.

`- act_sums` remains unjustified on its face: the correction one would expect
depends on the sum of the *weights*, and a ternary weight sum is zero only if
the values happen to balance. The reference subtracts it unconditionally, so we
do too, and Phase 3's `i2s_epilogue_is_reproduced_verbatim` test exists to stop
a later reader "fixing" it.

### Activation quantiser

`quantize_row_i8_s`, `ggml/src/ggml-cpu/quants.c`. Implemented in Phase 3 and
gated: `amax = max|x|`, no epsilon floor, `scale = amax > 0 ? 127/amax : 0`,
`q = clamp((int)roundf(x*scale), -128, 127)`, `sum = sum(q)`. An all-zero row
gives scale 0 and an all-zero row, and the epilogue then divides by zero, which
the reference also does. Reproduced, not fixed.

### RoPE

**NeoX**, not NORM. `llama_model_rope_type` puts `LLM_ARCH_BITNET` and
`LLM_ARCH_BITNET_B158` in the group returning `LLAMA_ROPE_TYPE_NEOX`
(`src/llama-model.cpp:2481-2482`, the group's return at `:2538`). NeoX pairs
dimension `i` with `i + n_rot/2`, and the golden vector caught our first
implementation pairing `x[i]` with itself.

`n_ctx_orig = 4096` and `freq_scale = 1.0`. `ext_factor`, `attn_factor`,
`beta_fast` and `beta_slow` are the defaults, so no scaling correction and no
attention scaling beyond the constant.

### SwiGLU

`silu(gate) * up`, with `silu(x) = x / (1 + exp(-x))`. Phase 2's kernel is
gated against the reference and the golden vector caught its first version
computing `x*exp(x)`.

### Attention

`build_attn` with `1.0f/sqrtf(n_embd_head)` as the scale, so `1/sqrt(128)`. GQA
with 4 query heads per KV head, causal masking, and the KV cache keyed on
position. `build_attn_inp_kv` supplies the mask.

The reference may dispatch prefill to a fused attention kernel. The logit
tolerance is measured at Phase 4 for exactly this reason, and the hard gate is
the greedy token sequence, not bit-equal logits.

## What the gate is

Hard: the greedy token sequence at temperature 0 must be **identical** to the
reference's on the frozen prompt set. That is the product requirement and it is
not subject to a tolerance, because a different argmax is a different model.

Secondary: logits within a tolerance **measured here**, not guessed. Our lm_head
reduces 2560 F16 products in our own order and the reference reduces them in
its own, so bit equality is not available without copying upstream's thread
partitioning. The measured figure goes in `BENCHMARKS.md`.

## Not claimed

`tg128 @ 28 threads`. Phase 4 is the first point at which an end-to-end
generation measurement is possible, and the Phase 2 scheduler result is a
microbench until then. When the forward pass runs, the measurement is made with
the identical model, prompt, thread count, affinity, context and batch as the
reference table, and the answer is reported whichever way it comes out.
