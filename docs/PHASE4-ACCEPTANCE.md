# Phase 4 acceptance

Phase 4 is **sealed**. This file records the boundary that was accepted, what
was deliberately left outside it, and the conditions under which it must be
reopened.

Sealed on the evidence in this file plus `BENCHMARKS.md`, `PHASE4-CONTRACT.md`
and `PHASE4-GAP.md`. 24,058 assertions, 0 failed, sanitizers clean, ISA gate
clean.

## The accepted boundary

**1. Forward semantics corrected against the reference.**
The graph is transcribed from `llama_model_bitnet` with file and line for every
point that could plausibly be otherwise. Six semantic errors were found and
fixed, two of which had survived earlier phases and were only caught by
comparing real end-to-end output against the reference:

- the RoPE layouts were **swapped** -- NEOX pairs `(k, k+n/2)`, NORMAL pairs
  `(2k, 2k+1)`, and ours had them the other way round, for three phases
- the RoPE angle was computed with a single `powf` per index instead of the
  reference's running multiplication
- SiLU and softmax used libm `expf` instead of the reference's AVX2
  polynomial `ggml_v_expf`
- `freq_scale` divided the angle where the reference multiplies it
- the RMSNorm mean was narrowed to f32 after the `sqrt` rather than before
- the RMSNorm scalar tail squared in double where the reference squares in f32

**2. Prompt argmax 16 / 16.**
Every prompt position of both frozen prompts takes the same argmax as the
reference. Read out of the raw golden `.f32`, not the manifest prose, so the
test and the fixture cannot disagree about formatting.

**3. Every prompt top-1 deviation is inside the measured reference top-2
margin.**
The gate is not a tolerance. It is: *at every position, our top-1 deviation is
smaller than the reference's top-1-to-top-2 margin.* True at 16 of 16. The
tightest case is position 1, where the deviation is 0.338 against a margin of
0.484 -- 70 percent of the margin, 30 percent headroom. The test asserts the
comparison per position, so a future change that crosses a margin fails
naming the position responsible.

**4. Generated continuation 7 / 8, from one near-tie.**
Our third continuation token is 6424 where the reference's is 3363 -- "a small
town" against "a small city" -- and the two sequences re-converge immediately
afterwards:

    ours      a small town, and the capital of France is a small
    reference a small city, and the capital of France is a small

This is a **named known divergence**, recorded as
`SLLM_KNOWN_DIVERGENCE` in `tests/test_forward.c` and covered by a test that
fails in **both** directions:

- if a future change makes the divergence worse, or the first two tokens stop
  matching, the test fails
- if a future change closes the gap and the token starts matching, the test
  **also** fails, and says the expectation is now stale

The second direction is the point. A boundary that can only move one way is not
a characterisation, it is a ratchet. This one forces a human to look at what
changed and decide whether the floor is now worth revisiting, instead of
letting the limit drift quietly.

**5. The remaining float floor is attributable to the reference's tinyBLAS
path.**
`GGML_LLAMAFILE=ON`. The reference's QK^T, P@V and lm_head all run through
tinyBLAS's blocked, register-tiled SIMD GEMM. This model amplifies any float
difference -- including a single ulp -- to O(0.1), because it is a chain of 30
int8 quantisers and a perturbation that crosses a rounding boundary moves a
whole quantisation step. So the deviation cannot be driven below that floor
without reproducing tinyBLAS's exact tile shape and FMA accumulation order.

**6. tinyBLAS is intentionally out of architecture scope.**
`ARCHITECTURE.md` declines to copy upstream's implementation for the same
reason it declines to copy the thread partitioning: the parity rule is about
the model, not about the implementation. Bit-exact logit equality would make the
tolerance question settled by transcription rather than by argument.

One consequence worth keeping: **wider accumulators are wrong here.** Running
the attention products and the lm_head in double is *worse* at four of six
positions, and much worse at two of them. The reference's own f32 blocked
accumulation carries error, and that error is part of the value the gate
compares against. "More accurate" is not "closer to the reference" in this
project. Measured numbers in `PHASE4-GAP.md`.

**7. No `tg128 @ 28 threads` claim.**
None is made and none will be until Phase 6, where the forward pass actually
uses the thread pool. Phase 4's end-to-end generation is single-threaded --
2.74 to 2.83 t/s -- and the reference's 28-thread figure of 1.16 t/s is not
comparable to it and is not compared. The Phase 2 scheduler result remains a
microbench.

## When this seal must be reopened

Only on a **semantic** failure attributable to the float floor. A logit
deviation, a near-tie, or a continuation token differing is **not** such a
failure: those are the floor behaving as characterised above, and "fixing" them
means porting tinyBLAS, which is out of scope.

The floor becomes a real problem if, for example:

* a prompt where the deviation exceeds the margin and the argmax disagrees --
  that would mean the margin argument does not generalise, and the gate itself
  needs rethinking
* a model where the I2_S projections are not the dominant tensor type, so the
  chaos amplifier described in `PHASE4-GAP.md` does not apply and the floor is
  no longer a defensible bound
* any correctness failure traceable to the attention or lm_head reductions
  rather than to their ordering

Phase 5 does not reopen it unless one of those occurs.
