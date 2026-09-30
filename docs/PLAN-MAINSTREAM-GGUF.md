# Plan: execute one mainstream GGUF architecture (Qwen3 or Llama)

**Planning only. Nothing in this document has been implemented.** It is written
so the work can be scoped, ordered and argued with before any of it exists.
Every "current state" claim below is measured or read from source, not assumed.

Date: 2026-09-30. Baseline commit for all measurements: `fdd4730`.

---

## 1. Where the runtime actually stands

Measured, not estimated, using production entry points only against a real
ordinary model at `/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf`:

```
container open              : ok, 399 tensors parsed, 145 F32, 0 without a known extent
sllm_model_load             : -30 unsupported-architecture
types with a real dequantiser: F32, F16, BF16, Q8_0, Q4_0, I2_S
```

### The three layers, which are not the same thing

This distinction drives the whole plan, and conflating it is what produced the
last two defects.

| layer | component | today | general? |
|---|---|---|---|
| **container** | `src/gguf.c` — header, KV, tensor table, extents | works for any GGUF v3; sizes every type it knows | **yes** |
| **model** | `sllm_model_load`, `sllm_ctx_new`, `sllm_forward_chunk` | `bitnet-b1.58` only; architecture string is rejected otherwise | no |
| **compute** | `sllm_i2s_*`, `sllm_dot_f16_f32`, `sllm_gemv` | ternary GEMV + F16 lm_head | no |

GGUF is a general container today and this plan does not change that. BitNet's
I2_S is an optimisation path inside it. The missing piece is layers 2 and 3.

### What a mainstream architecture would actually require

Not "add Qwen support". The distinct gaps, in dependency order:

1. **Weight dequantisation for the types a mainstream model ships in.** Qwen3
   Q4_K_M is Q4_K (217 tensors) and Q6_K (37). Llama-family GGUF commonly adds
   Q4_K, Q6_K, Q5_K, Q4_0, F16, and BF16. Only Q4_0 and F16/BF16 have kernels
   today. `fdd4730` stopped *claiming* Q4_K/Q6_K; implementing them is the next
   honest step.
2. **A non-ternary matmul.** The forward's arithmetic assumes ternary weights
   and int8 activations with the `(dot - act_sum) * w_scale/act_scale` epilogue.
   Every mainstream architecture needs F16/F32/BF16/Q4_K dequantise-to-f32 then
   a plain `f32 × f32` GEMV/GEMM. This does not exist and is the largest single
   piece of work.
3. **A second graph.** BitNet's graph is fixed: four pre-norms per layer, a
   ternary attention output, SwiGLU FFN, `output_norm`, tied-or-untied
   lm_head. Qwen3 and Llama have a pre-norm attention block, GQA, RoPE,
   SwiGLU, a post-attention norm, an untied `lm_head`, and per-layer KV cache.
   `sllm_forward_chunk` is written against the BitNet shape at every level.
4. **Tokenizer generality.** 128k GPT-2 BPE today. Qwen3 uses a 151k BPE with a
   different pre-tokenizer and chat template; Llama 3 uses 128k SentencePiece
   BPE. Neither is the same format.
5. **RoPE variants.** NEOX pairing is implemented and used. Qwen3 and Llama 3
   use NEOX, but with per-dimension frequency layout and, for some checkpoints,
   scaling (YaRN/linear). The `rope.dimension_count` handling is a partial
   rotation and would need the frequency layout to be parameterised.
6. **KV cache and context.** BitNet's cache is `n_head_kv × n_ctx ×
   n_embd_head` per layer, which is the right *shape* for GQA. Mainstream
   models need a much larger default context and, for correctness at long
   context, a working sliding-window variant.

### Two architectural decisions to settle before writing code

**A. One forward function, or a dispatch table?**

The current `sllm_forward_chunk` is a single BitNet-shaped function. Options:
(i) branch on `general.architecture` inside it; (ii) an architecture vtable with
one entry per family; (iii) a separate entry point per family.

*Recommendation: (ii), a vtable.* (i) is how the code grows a second arithmetic
path in one function and how the two paths start disagreeing; (iii) duplicates
the chunking, caching and pool plumbing that both families need. The vtable
keeps `sllm_ctx` and the pool shared, and it makes "which architectures are
supported" a queryable list rather than a scattered `if`.

The risk to name up front: a vtable is more structure than one family needs, and
if only one mainstream family is ever supported it will look like over-engineering.
That is the cost of not having the first family decide the shape for the second.

**B. Does the I2_S contract generalise?**

The row-width rule (`ne[0]` a multiple of 128 for I2_S only) is enforced in
`sllm_model_load` via `need_i2s`, which re-tests the type internally and is
called at exactly seven I2_S sites. A vtable makes this a per-type property:
each architecture declares the types it consumes and the per-type dimensional
contract. **The invariant to preserve is that this is a property of the type,
not of the model** — a Qwen3 Q4_K tensor is a legal, common thing and the
current check must never see it. That is already true and must stay true.

---

## 2. Ordered plan, with the gate that authorises each step

Nothing starts until the previous gate is green. No step is authorised by this
document.

**Step 0 — measurement harness for a mainstream model.** Extend the parity
probe (the one repaired for the K cache) to run a *reference* forward
(llama.cpp, already vendored under `third_party/`) beside a Saphira forward for
the same GGUF, and report per-tensor cosine and max error. *No new compute.*
This is the harness that makes every later claim falsifiable, and it is the
step whose absence produced both false greens this stage.
**Gate:** on a reference llama.cpp run of `Qwen3-8B-Q4_K_M.gguf`, the harness
reproduces the reference logits to float32 tolerance on the tensors Saphira can
already decode (the 145 F32), and reports `unsupported` honestly on the rest.
*If this cannot be built, no later step should be started.*

**Step 1 — dequantisers for the types a target family ships in.** `Q4_K`,
`Q6_K` first (Qwen3 Q4_K_M needs exactly these). Port from the vendored
reference, transcribe the block layouts, and gate each against a golden file
captured from the reference build — the same discipline as the I2_S converter.
Flip `g_types[].supported` to true **only** once the kernel exists and its golden
passes, so the flag can never again be a promise without a kernel.
**Gate:** every enabled type has a passing golden; the Qwen3 file reports
`decodes OK` for all 399 tensors; `fdd4730`'s correction is not reverted by
accident.

**Step 2 — f32 weight matmul.** Dequantise a block to f32 and a plain
`f32 × f32` GEMV over it, using the existing `SLLM_MAX_CHUNK` chunking and the
existing thread pool. This is the compute layer every mainstream family shares.
**Gate:** against the reference, one matrix-vector product agrees to float32
tolerance; the ISA guard is clean; ASan/UBSan clean.

**Step 3 — a second graph behind a vtable.** Introduce the architecture
dispatch, with `bitnet-b1.58` moved onto it **first** and with no behavioural
change, so the existing 35904 checks are the regression gate. Then add one
mainstream family. Only the pieces that family needs: pre-norm attention, GQA,
NEOX RoPE, SwiGLU, post-attention norm, untied lm_head, KV cache.
**Gate:** BitNet's existing suite still passes unchanged; the new family
reproduces the reference logits for a short prompt.

**Step 4 — tokenizer.** Only for the chosen family, and against its real
vocabulary file. This is deceptively large: BPE merges, pre-tokenizer regex, and
for Qwen3 a chat template. Byte-exact tokenisation is a hard gate — a tokeniser
that is *nearly* right produces a plausible model that scores badly, which is
the same failure class as everything else found today.
**Gate:** tokenise a reference corpus and get byte-identical ids.

**Step 5 — evaluation and perplexity on a real model.** Point `saphira-llm-eval`
at the mainstream model. The tool is already architecture-agnostic in its
scoring path; what is unproven is the loader and forward beneath it.
**Gate:** perplexity on a small fixed corpus is within a stated tolerance of the
reference, and the delta is explained rather than merely bounded.

---

## 3. What is deliberately out of scope

- **Training or fine-tuning** a mainstream model. QAT is for BitNet; nothing
  here changes the trainer.
- **Reopening the absmax/master-weight runaway.** Separate, still unauthorised.
- **Speculative decoding, tool use, or chat templating beyond what the target
  family needs to reproduce reference logits.**
- **Widening the I2_S contract** to support widths that are not a multiple of
  128. Shipped models do not need it; it remains a separate runtime feature.
- **Any change to the public GGUF container's generality.** It works. Leave it.

---

## 4. Risks, stated before the work rather than after

1. **The harness (Step 0) is the real work.** Everything else is mechanical once
   a value-level comparison against the reference exists. Skipping it is how
   this stage produced two false greens.
2. **Tokenizer drift is silent.** Perplexity will look merely bad rather than
   wrong, and it is tempting to attribute it to the forward.
3. **The vtable is a structural bet.** If a second family is never added, it was
   the wrong call. It is reversible at Step 3 and expensive after.
4. **Threading and chunking already have a documented order sensitivity** in
   this codebase (Phase 4 established that a different accumulation order moves
   logits *away* from the reference). Step 2 inherits that constraint and must
   match the reference's reduction order rather than merely being faster.
5. **Context length.** BitNet's default is small; a mainstream model's KV cache
   is orders of magnitude larger. Memory behaviour under a real context has not
   been measured and the existing `swap must stay 0` discipline applies.

---

## 5. Decision needed before Step 0

**Which family, and which single concrete model file, is the target?** Qwen3
Q4_K_M is on this machine and is a reasonable first target: it exercises Q4_K,
Q6_K and F16, it is GQA, it uses NEOX RoPE, and its tokenizer is a known,
bounded problem. Llama-family is a reasonable second.
This choice is not made in this document and should not be inferred from it.
