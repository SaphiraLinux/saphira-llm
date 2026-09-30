# Stage decisions

Durable record of the decisions taken at stage boundaries, so the next stage
starts from a contract rather than from chat history. Ordered newest first.

---

## Stage: evaluation and BitNet lifecycle — ACCEPTED

Commits `49e2652` (evaluation tool) and `5bc66d4` (lifecycle audit).
`v0.0.1` remains sealed at `7dbcc67` and is not reopened.

Accepted as the stage result: **a quality receipt exists.** Token-level
cross-entropy and perplexity, measured from the same tokenizer and the same
forward pass the runtime uses for generation, with no second inference stack.

The current figure, `perplexity 145.083620` on `tests/golden/eval-corpus.txt`,
is **a fixed regression corpus, not a quality claim and not a published-model
comparison.** It is 386 tokens of in-house technical prose, out of distribution
for a model trained on web text, and it is not comparable to any published
BitNet figure. It earns its place by being the same 386 tokens forever, so that
a change in the number means the model changed. That description must travel
with the number.

---

## Decision 1 — arithmetic contract (`-ffp-contract=off`)

**Rule: quality experiments need an arithmetic contract before they need
another 0.3%.**

`mean_nll` is not portable across compilers. Measured in the accepted stage by
building one source six ways: gcc and clang agree **bit-exactly** with
`-ffp-contract=off`, and disagree by `0.0034` in `mean_nll` (perplexity 145.0836
against 145.5773, a 0.34% gap) when FMA contraction is enabled. It is not the
sanitizers and not the optimisation level.

Binding rules from here on:

1. `v0.0.1` is not touched. Its arithmetic is whatever it shipped with.
2. **The evaluation and training validation path builds with
   `-ffp-contract=off`.** Regression arithmetic is deterministic across
   compilers.
3. A **fresh canonical baseline** is established under that arithmetic. The
   value recorded in `tests/test_eval.c` today was captured under the default
   (contracted) build and is **provisional until re-recorded**; the tolerance
   that currently absorbs the FMA spread becomes unnecessary and should be
   tightened to exact equality once the contract is in force.
4. **A contracted baseline is never compared against a non-contracted build,
   and the difference is never reported as a model improvement.** The two are
   different quantities.
5. If production inference later deliberately uses FMA for performance, that is
   a separate, separately benchmarked decision. It does not become the quality
   baseline by accident.

Not yet applied. See "Next actions" below.

---

## Decision 2 — rolling-window evaluation is its own stage

The current refusal for an over-context corpus is **correct and stays** until
rolling-window scoring exists. No silent truncation, ever: a corpus too large to
fit is rejected with an explanation, not scored as a prefix.

Rolling-window evaluation is the next evaluator capability, and it is a
**separate stage** from this one. Required of it:

- Scoring semantics documented precisely — window size, stride, which positions
  are scored, and why.
- Tested against a corpus that **fits in one context**, so that single-window
  and rolling-window results can be **proven equivalent** there. This is the
  acceptance test: if they disagree on a corpus that fits, the stride logic is
  wrong, and no larger-corpus number can be trusted until that is fixed.
- No larger-corpus claim before this exists.

---

## Decision 3 — do not make inference look like training

There are **two** activation quantisers upstream and they are not the same
function. The inference path (`quantize_row_i8_s`, which ours ports exactly)
uses `roundf` (half away from zero) and leaves a zero row at `s = 0`. The
training path (`float_act_quant`, and `gpu/model.py:70`) uses half-to-**even**
rounding and floors absmax at `1e-5`.

**Rule: inference is not changed to resemble training.** The two conventions
stay distinct, and the QAT/export boundary states explicitly which is used on
which side of it. Concretely, a future QAT run must:

- use the **training** convention for the forward pass during training, and
- export an artefact consumed under the **inference** convention,

and must acknowledge the systematic difference in rounding on exact halves and
in zero-row handling rather than papering over it.

---

## Decision 4 — native BF16 → I2_S is the approved conversion direction

Approved, with one proof required before relying on any external master model:

> **Hermetic round-trip.** Dequantise the shipped I2_S model to BF16, run the
> native converter on it, and require the output to be **byte-identical to the
> original I2_S file**. Any packing, scale or threshold error surfaces
> immediately, with no network and no external master.

This is required *before* the real upstream BF16 release is used, not after.
Verification method is already established in this project: golden vectors
captured from the reference's own dequantiser, not from our own encoder
round-tripping its own assumption. See `src/quant.c` for why that distinction
was learned the hard way.

Open caveat to test rather than assume: the scale is taken from the first
nonzero magnitude, so a dequantise/requantise round trip only reproduces the
original if scale recovery is fed the original scale (the `override_scale`
path) or if first-nonzero is genuinely scale-invariant for these tensors.

---

## Decision 5 — the tiny-model lifecycle experiment, scoped

Approved in principle. Shape: a **2-layer, 64-wide, 4-head (2 KV), 128 FFN,
256-vocab** BitNet, trained QAT on a few hundred tiny sequences.

Feasibility rests on a real finding: `sllm_model_load` reads every dimension
from GGUF metadata, and the only shape constraints are
`n_head % n_head_kv == 0` and `n_embd % n_head == 0`. Every hardcoded 2560 /
6912 / 128256 in the tree is in a comment. So a tiny model exercises the real
GGUF loader, the QAT path, the converter, the I2_S runtime and the evaluator
with **no runtime change whatsoever**.

The chain it closes:

```
tiny dataset -> baseline (saphira-llm-eval) -> QAT -> improvement measured
             -> export to I2_S GGUF -> saphira-llm loads it -> eval confirms
```

**What it does not prove:** anything about 2B-scale behaviour, long-context
stability, or the quality of a useful model. It proves the *pipeline*, which is
the thing that does not yet exist. It is not to be described as a model
improvement at 2B.

**No training run has been authorised and none has begun.**

---

## Next actions, in order

1. Apply the arithmetic contract (Decision 1): add `-ffp-contract=off` to the
   validation path, re-record the canonical baseline, and tighten the golden
   tolerance to exact equality. Small and unblocks everything else.
2. Rolling-window evaluation as its own stage (Decision 2), with the
   single-window equivalence test as its acceptance criterion.
3. Native BF16 → I2_S converter, gated on the hermetic round-trip proof
   (Decision 4).
4. Only then the QAT stack and the tiny experiment (Decision 5).

Items 1 and 2 are small and independent. Item 3 needs item 1 first, so that the
converter's output is compared under a known arithmetic. Item 4 needs all three.

---

## Decision 6 — the I2_S row-width contract, and a correction to Step 5

**The contract.** `sllm_i2s_dot` processes whole 128-element blocks and returns
0 outright when `n < SLLM_I2S_QK`; `sllm_i2s_gemv` strides rows by `n/4`, which
is the block layout's row size only when `n` is a whole number of blocks. A
matrix whose *contracted* width (`ne[0]`, the dimension actually passed to the
GEMV) is not a multiple of 128 therefore cannot be computed: below 128 every
projection returns zero, and at 192 and similar the rows are read from the wrong
offsets. Every shipped model is 2048-wide, so nothing in the project could
distinguish the two cases.

**Enforced in two places, on purpose.** `sllm_qat_export_i2s_gguf` refuses an
unsupported width loudly rather than emitting a model that loads and computes
nothing; and `sllm_model_load` applies the same rule to any I2_S tensor it
reads, so a GGUF from outside this project gets the same answer. The loader
checks `ne[0]` alone and not every dimension: `ffn_down` legitimately has an
output width of 128 while its contracted width is 2048. A format's contract
belongs at the boundary that enforces it, not in every producer that has to
remember it.

**CORRECTION to Step 5.** Commit `02e6e00` recorded Step 5 as proven: the
exported model loaded through the ordinary `sllm_model_load` and executed an
ordinary `sllm_forward_chunk`. That was true and it was not enough. The tiny
fixture was `n_embd = 64`, so every projection in it returned a zero dot, and
the "finite logit" that was checked was `(0 - act_sum) * (w_scale/act_scale)`.
The model loaded, the API executed, the sanitizers were clean, and the answer
was mathematically meaningless. Step 5 proved *loadability and API execution*;
it did not prove meaningful I2_S network execution, and should not be cited as
though it did.

**What replaces it.** With the fixture at `n_embd = 128`, `n_ff = 256`, every
contracted width is a multiple of 128, and the exported model executes the same
quantised network the trainer evaluated. The layer-by-layer trace shows the
pre-attention RMSNorm exact and Q, K and V post-RoPE at cosine 1.000000, and the
trainer/runtime NLL now tracks instead of diverging (0 steps 5.54464 vs
5.54464; 50 steps 0.14789 vs 0.147683; 200 steps 0.0335698 vs 0.0323623; 600
steps 0.00378052 vs 0.00376585, top-1 23/23). The earlier divergence — trainer
5.54 to 0.000005 while the runtime rose to 18.35 — was this, and not the absmax
runaway, which remains an open training-quality question.
