# BitNet lifecycle: what the vendored upstream actually contains

Findings from auditing `third_party/` at the revisions pinned in
`PROVENANCE.md` (BitNet `0b341e5`, isHuangXin/llama.cpp `390c3077`). Nothing
here is implemented yet. This records what the source says, so the next stage
does not have to re-derive it or guess.

---

## 1. Activation quantisation — already implemented, exactly

**Not missing. Already done, and ported line for line.**

`sllm_i2s_quant_act` in `src/i2s_gemm.c:45` is an exact port of the reference's
`quantize_row_i8_s` at
`third_party/BitNet/3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c:1311`:

```c
float amax = 0.0f;
for (i) amax = MAX(amax, fabsf(x[i]));
float s = (amax > 0.0f) ? 127.0f / amax : 0.0f;   // RECIPROCAL scale
for (i) {
    int v = (int) roundf(x[i] * s);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    dst[i] = (int8_t) v;
    sum += v;
}
```

and the epilogue matches the reference's optimized form,
`ggml-bitnet-compute.c:164`:

```c
tmp[row] = (tmp[row] - act_sums[i1]) / act_scales[i1] * (*scale);
```

ours being `(float)(dot - act_sum) * (w_scale / act_scale)` — one division
hoisted out of the inner loop, as the reference does it.

So the established facts, all from source:

| Property | Value |
| --- | --- |
| Granularity | per row (per token), over K elements. Not per block. |
| absmax | computed in **float**, `fabsf`, over the row |
| Zero row | `s = 0` when `amax == 0`. **Not** floored at 1e-5. Deliberate: the reference does the same, and the parity gate depends on reproducing it. |
| Scale stored | the **reciprocal** `127/absmax`, not the scale |
| Rounding | `roundf`, i.e. **round-half-away-from-zero** |
| Clamp | `[-128, 127]` after rounding |
| Extra output | `act_sums` = integer sum of the quantised row |
| Use of the sum | `dot - act_sums` zero-centres the integer dot: the packed codes are `{0,1,2}` and the kernel needs `code - 1`. Code 3 is unused and never occurs, which is what makes the shortcut valid. |
| Accumulation | int32 |

`docs/PHASE4-GAP.md` already identified this: *"210 of the 211 weight tensors
are I2_S. Every per-layer projection quantises its activation to int8 and
accumulates in int32."* It also explains the consequence that matters: the
residual stream is a chain of 30 quantisers, so a one-ulp float difference
crosses a rounding boundary and becomes a full quantisation step, repeated at
every layer. That is the "chaotic with respect to float reordering" finding,
and it is why the forward gate is an argmax margin rather than a byte compare.

### The one genuinely new finding: train and inference disagree

There are **two** activation quantisers upstream, and they are not the same
function.

| | inference (`quantize_row_i8_s`) | training / BLAST (`float_act_quant`) |
| --- | --- | --- |
| absmax | float | **double** |
| zero-row floor | `s = 0` | `max(1e-5, ...)` |
| rounding | `roundf` (half away from zero) | `nearest_int` magic constant (half to **even**) |
| emits `act_sums` | yes | no |

`float_act_quant` is at `ggml-bitnet-compute.c:25`. The torch reference
confirms the training convention — `third_party/BitNet/gpu/model.py:70` and `:80`:

```python
s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
return (input * s).round().clamp(-128, 127) / s
```

which is the 1e-5 floor and half-to-even, i.e. `float_act_quant`.

**Consequence for any future QAT work:** a training run must use the
*training* convention during training and the *inference* convention at export,
or accept a systematic train/inference mismatch. The two differ in absmax
precision, in rounding mode on exact halves, and in zero-row handling. This is
recorded now so it is a decision rather than a bug found later.

---

## 2. BF16 master → I2_S conversion — feasible, spec fully recoverable

**Yes, and the specification is completely recoverable from the pinned tree.**
The authoritative definition is
`third_party/BitNet/utils/convert-hf-to-gguf-bitnet.py:666`,
`quantize_to_i2_s`.

### The algorithm, in full

1. **Scale.** Two paths:
   - `override_scale` given (an already offline-quantised model): use it
     directly; the values are already ternary and are used as-is.
   - otherwise: `scale = first |w| where |w| > 1e-6`, else `1e-5`.
2. **Ternarise.** The reference computes
   `q_float = np.round(w * inv_scale).clip(-1, 1)` and then applies
   `q[q_float > 0.5] = 2`, `q[q_float < -0.5] = 0`, default `1`.
3. **Pad** to a multiple of 128 elements with the zero code (`1`).
4. **Pack** 2 bits per element, 4 per byte, field 0 in the **top** bits:
   `packed = (q0 << 6) | (q1 << 4) | (q2 << 2) | q3`
5. **Tail**: the f32 scale in the first 4 bytes after the packed data, in a
   32-byte-aligned region.

### The round and the clip are provably redundant

Step 2 looks like it needs floating-point rounding care. It does not. The
`±0.5` threshold already decides every case, so a C port needs no rounding-mode
subtlety at all:

```
t = w * inv_scale
code = 1            /* ternary zero  */
if (t >  0.5) code = 2
if (t < -0.5) code = 0
```

Verified rather than argued: transcribed the reference's `round`+`clip`+`threshold`
logic with stdlib only (numpy is not installed here, and `Python round()` is
half-to-even exactly like `np.round`) and swept **56,014 values** across the
threshold neighbourhoods and exact half-steps. **Zero mismatches.**

This matters because it removes the single most likely source of a silent
tensor-by-tensor mismatch between a C converter and the Python one.

### The layout already agrees with our decoder

`src/quant.c:98` reads `byte >> (6 - 2*field)`, matching the `(q0<<6)|(q1<<4)|
(q2<<2)|q3` packing. The comment there records that the first version read it as
`2*field` — reversing the four fields in every byte — and that the golden test
did **not** catch it, because the test built the buffer and read it back with the
same inverted assumption. The fix was a golden vector captured from the
reference's own dequantiser.

**That methodology is the answer to "how do we prove a converter".** It already
exists in this project: `tests/golden/i2s-reference.txt`, captured from the
reference's dequantiser on a real tensor. A native converter is validated by
comparing its output against that, byte for byte, not by checking the result
loads.

### The constraint on proving it

**There is no BF16/master BitNet model on this machine.** A search finds only the
three I2_S GGUFs and Qwen3. So the natural demo — convert the real upstream
BF16 release and diff against the shipped I2_S — cannot run here without a
network fetch.

There is a self-contained alternative that needs no network and is arguably a
better test anyway:

> **Round-trip proof.** Dequantise the shipped I2_S model to BF16, run the new
> C converter on it, and require the output to be **byte-identical to the
> original I2_S file**. Any packing, scale or threshold error shows up
> immediately, and the test is hermetic.

Caveat to check when building it: the scale is taken from the *first* nonzero
magnitude, so a dequantise→requantise round trip only reproduces the original
if the converter's scale recovery is fed the original scale (the
`override_scale` path) or if first-nonzero is genuinely scale-invariant for
these tensors. Worth testing rather than assuming.

---

## 3. Training / QAT — nothing trainable is vendored

**The vendored tree contains no training code at all.** This was checked before
assuming anything from the project's BitNet lineage.

`third_party/BitNet/gpu/model.py` looks like a training model and is not. It is
an inference harness with pre-quantised weights:

```python
self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features//4,
                                              dtype=torch.int8),
                                 requires_grad=False)
```

`requires_grad=False` and `dtype=torch.int8`. The weight cannot be trained
because it is not a float and does not require grad. The forward calls a custom
CUDA kernel (`bitnet_int8xint2_linear`) under `@torch.compile`, and
`.round()` is a hard non-differentiable op with **no straight-through estimator
anywhere** — no custom autograd Function, no surrogate, no `round_pass`.

A grep across the whole vendored tree for `optimizer`, `backward(`,
`requires_grad`, `nn.Module` training constructs returns only the inference
harness. There is no loss, no optimiser, no loss curve, no checkpoint of
trained state.

### What a QAT experiment would actually have to provide

Established from the source and the architecture, item by item as requested:

| Requirement | Status | Source of truth |
| --- | --- | --- |
| Trainable/master weight precision | **must be introduced** — BF16 latent weights, the thing the ternary is derived from. The `-bf16` HF release is these masters. | `PROVENANCE.md`, upstream release naming |
| Ternarisation function | available, and now proven to be a plain ±0.5 threshold | `quantize_to_i2_s`, §2 above |
| Activation quantisation (training-time) | available, **differs from inference-time** | `float_act_quant` / `model.py:70`, §1 above |
| Gradient treatment | **must be introduced** — straight-through estimator | paper; absent from the tree |
| Loss | **must be introduced** — next-token cross-entropy | — |
| Optimiser | **must be introduced** — AdamW on latent weights | paper; absent |
| Backward operators | **must be introduced** — backward through attention, RMSNorm, RoPE, softmax, the FFN | — |
| Optimiser state memory | AdamW keeps 2 fp32 moments: 8 bytes/param, plus BF16 master (2) and BF16/fp32 grad (2–4) | — |
| Checkpoint representation | **must be chosen** — the latent BF16 masters are the checkpoint, *not* the ternary. Ternary is derived and disposable. | — |

**Can an existing BF16 model be fine-tuned without reproducing pre-training?**
Yes, in principle — that is the entire point of the latent-weight formulation,
and it is why the `-bf16` release is published alongside the packed one. But
two things are unavoidable:

1. The whole QAT training stack must exist first (STE, loss, optimiser,
   backward). None of it is vendored, and none of it is in this repository.
2. Doing it on the 2B model is not viable on this hardware. See the sizing
   below.

### The finding that makes a tiny experiment possible

**The model loader reads every dimension from GGUF metadata.** `sllm_model_load`
at `src/forward.c:315` requires `general.architecture == "bitnet-b1.58"` and
then reads `block_count`, `embedding_length`, `feed_forward_length`,
`attention.head_count`, `attention.head_count_kv`, `rope.dimension_count`,
`context_length` and `vocab_size` from the file. The only shape constraints are

```
n_head % n_head_kv == 0
n_embd % n_head  == 0
```

Every hardcoded 2560 / 6912 / 128256 in this tree is in a **comment**. So a
tiny BitNet-shaped model — a few layers, a small embedding, a small vocabulary —
loads and runs in `saphira-llm` and is scored by `saphira-llm-eval` with **no
runtime change whatsoever**.

That closes the loop the stage was aiming at:

```
tiny dataset → baseline (saphira-llm-eval) → QAT run → improvement measured
             → export to I2_S GGUF → saphira-llm loads it → eval confirms
```

and every stage is exercised, including the two that were genuinely at risk —
the converter and the load-back.

---

## 4. Sizing the smallest credible experiment

For a model of `P` parameters trained QAT with AdamW on BF16 latent weights:

| Item | Bytes/param |
| --- | --- |
| latent master, BF16 | 2 |
| gradient, BF16 | 2 |
| AdamW m, fp32 | 4 |
| AdamW v, fp32 | 4 |
| **total** | **12** |

Activations dominate instead, and scale with `batch × n_ctx × n_embd`, not with
`P`. Forward activations must be retained for the backward pass, which is the
real memory driver for a deep model and negligible for a shallow one.

A deliberately tiny model, e.g. 2 layers, `n_embd=64`, `n_head=4`,
`n_head_kv=2`, `n_ff=128`, vocabulary 256, sequence 128:

- `P` is on the order of 10⁵, so weights plus optimiser state is **~1–2 MB**.
- Activations for a 128-token sequence: 2 layers × 128 × 64 × (a handful of
  intermediates) ≈ **1 MB**.
- Training steps on a few hundred sequences: **seconds to a couple of minutes**
  on this CPU, single-threaded or with the existing pool.
- Export and re-eval: seconds.

That is well inside what Egg can do, and it is a *complete* lifecycle proof
rather than a toy: real quantisation, real STE, real AdamW, real I2_S export,
real load-back, real measured perplexity change.

What it would **not** prove: anything about 2B-scale behaviour, long-context
stability, or the quality of a useful model. It proves the *pipeline*, which is
precisely the thing that does not exist yet.

---

## What is still dependent on upstream Python/PyTorch

Stated plainly, because it is the main reason a native converter is worth
writing:

- **Today**, producing an I2_S model from BF16 masters requires the upstream
  Python converter and `gguf-py`, plus numpy, plus torch for the BF16 release.
  None of those are on this machine, and the project policy is no Python in the
  build or runtime.
- A native C11 converter removes that for the conversion step. The **training**
  step would still need the BF16 masters as input, but nothing else.
- The full QAT stack — STE, AdamW, backward — has no upstream C to port, so
  that is new work in this repository rather than a port.
