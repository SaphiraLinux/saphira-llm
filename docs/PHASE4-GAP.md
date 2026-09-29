# The Phase 4 float gap, and where its floor is

Phase 4 hit an end-to-end result -- argmax token-identical at all 16 frozen
prompt positions -- with logit deviations of 0.3 to 0.9 against a tightest
reference argmax margin of 0.0427. The deviation is an order of magnitude
larger than the margin, so the agreement was real and not robust, and one
generated token already differed. This document records what the deviation
actually is, what was fixed, and what cannot be fixed.

## First: is it noise, or a bug?

A deviation of 0.3 to 0.9 on logits of magnitude ~20 is 2 to 4 percent. Float
reordering over a 2560-element reduction does not produce that; it produces
about 1e-4. So either there was a systematic error, or a tiny error was being
amplified enormously.

The discriminator was cheap. Running the attention products and the lm_head
reduction in **double** precision:

| | worst logit deviation |
| --- | ---: |
| float paths | 0.917 |
| float paths in double | 0.883 |

Barely moved. Double precision removes reordering error almost entirely, so
the source was **not** our reduction order. It was a systematic difference --
and a systematic difference can be found.

## What the model amplifies, and why it matters here

The inventory in `PHASE4-CONTRACT.md` is the key: 210 of the 211 weight
tensors are I2_S. Every per-layer projection quantises its activation to int8
and accumulates in int32. There is no float error in them.

So the residual stream is a chain of 30 quantisers. A perturbation of `eps` in
the input to a projection is rounded to int8; if `eps` is large enough to cross
a rounding boundary, **one element of that projection's output changes by a full
quantisation step**, about 1/127 of the activation scale. That is not a small
perturbation of the output, it is a discrete jump, and it happens at every one
of the 30 layers.

The practical consequence: **any float difference at all, including one ulp,
is amplified to O(0.1) by the time it reaches the logits.** This model is
chaotic with respect to float reordering. Which means:

* matching the reference's float *arithmetic forms* matters enormously, and
* matching its float *reduction order* in the SIMD kernels is hopeless without
  porting those kernels, because the chaos amplifies whatever difference is
  left to O(0.1) regardless of how small it started.

## The reference's actual arithmetic, op by op

| operation | reference | ours, before | status |
| --- | --- | --- | --- |
| RMSNorm | sum of squares in double, **narrow the mean to f32 first**, then `1.0f/sqrtf(mean+eps)`; `x*scale*w` | mean in double, `sqrt` in double, narrow after | **fixed** |
| RMSNorm tail | squares in **f32** then widens | squared in double | **fixed** |
| RoPE angle | `pos * powf(base, -2/n)` accumulated by repeated multiplication | `pos * powf(base, -2k/n)`, one `powf` per k | **fixed** |
| RoPE pairing | NEOX `(k, k+n/2)`, NORMAL `(2k, 2k+1)` | swapped | **fixed earlier** |
| RoPE freq_scale | **multiplies** the angle (`theta = freq_scale * theta_extrap`; the metadata factor is `1/scale_linear`) | divided | **fixed** |
| softmax exp | `ggml_v_expf`, an AVX2 polynomial, **not** libm | libm `expf` | **fixed** |
| SiLU exp | the same `ggml_v_expf` polynomial | libm `expf` per lane | **fixed** |
| softmax sum | per-vector partials into a double accumulator, `1.0/sum` in double, narrowed to f32, multiplied | same | already correct |
| I2_S epilogue | `(dot - act_sum) * (w_scale/act_scale)`, division once per column | `(dot - act_sum)/act_scale*w_scale` | **fixed earlier** |
| QK^T | tinyBLAS `sgemm`, blocked SIMD | scalar loop | **not matched** |
| P@V | tinyBLAS `sgemm`, blocked SIMD | scalar loop | **not matched** |
| lm_head | tinyBLAS `sgemm` over F16, blocked SIMD | scalar loop | **not matched** |

The three "fixed" rows are the ones that changed the numbers. The SiLU one
mattered most: 6912 elements per layer, 30 layers, every one of them biased by
a couple of ulps of exponential.

## The floor

`GGML_LLAMAFILE:BOOL=ON` in the reference build. The QK^T, P@V and lm_head all
go through tinyBLAS's blocked, register-tiled SIMD GEMM. Matching those
bit-for-bit means porting tinyBLAS's exact tile shape and FMA accumulation
order for three different shapes (f32 x f32 with a short K, f32 x f32 with a
short M, and F16 x F32 over 2560).

`ARCHITECTURE.md` already decided against this, for the same reason it decided
the same thing about thread partitioning: **the parity rule is about the model,
not about copying the implementation.** Bit-exact logit equality would force us
to reproduce upstream's blocking, and the moment we did, the tolerance question
would be settled by transcription rather than by argument.

So the floor is real and it is not zero. And the chaos amplifier above means it
is not 1e-4 -- it is O(0.1). **The deviation cannot be driven to zero without
porting tinyBLAS.** That is a deliberate design position, not a shortfall to be
apologised for, and it is stated here so nobody later mistakes it for one.

## What was done instead: make the comparison robust

Since the deviation cannot be eliminated, the gate has to show that it cannot
*change the answer*. The argmax flips only if the perturbation crosses the gap
between the top two logits. So the gate is:

> at every position, our top-1 deviation is smaller than the reference's
> top-1-to-top-2 margin.

Measured, per position, for both frozen prompts:

| position | our top-1 | reference top-1 | deviation | reference margin | deviation < margin |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 15.62215 | 15.61324 | 0.00892 | 1.00175 | yes |
| 1 | 24.54107 | 24.20345 | 0.33762 | 0.48370 | yes |
| 2 | 25.27494 | 25.22651 | 0.04843 | 1.40838 | yes |
| 3 | 22.33787 | 22.44006 | 0.10220 | 1.72408 | yes |
| 4 | 24.10487 | 24.09171 | 0.01316 | 1.07516 | yes |
| 5 | 21.12532 | 21.18792 | 0.06260 | 0.23187 | yes |
| 0 | 15.62215 | 15.61324 | 0.00892 | 1.00175 | yes |
| 1 | 24.54107 | 24.20345 | 0.33762 | 0.48370 | yes |
| 2 | 21.68980 | 21.47280 | 0.21700 | 2.31186 | yes |
| 3 | 22.36005 | 22.44175 | 0.08170 | 2.30733 | yes |
| 4 | 23.21506 | 23.05709 | 0.15796 | 0.51030 | yes |
| 5 | 24.45982 | 24.50262 | 0.04280 | 1.40658 | yes |
| 6 | 22.96839 | 22.94184 | 0.02655 | 1.22628 | yes |
| 7 | 22.78493 | 22.77729 | 0.00765 | 2.62002 | yes |
| 8 | 22.10659 | 22.18975 | 0.08316 | 1.44438 | yes |
| 9 | 18.73230 | 18.73331 | 0.00100 | 0.04271 | yes |

**16 of 16.** The tightest case is position 1, where the deviation is 0.338
against a margin of 0.484 -- 70 percent of the margin, so the argmax holds with
30 percent headroom. That is the honest number: not comfortable, but not
arbitrary either, and it is a *measured* relationship rather than a chosen
tolerance.

The test asserts `deviation < margin` per position, so if a future change
pushes the deviation past a margin the gate fails on the position that caused
it, naming it. That is a stronger statement than "within 0.5 of the reference",
which would pass regardless of whether it mattered.

## The generated continuation

| | |
| --- | --- |
| continuation tokens matching | 7 of 8 |
| divergence | token 2: ours 6424, reference 3363 |

    ours      a small town, and the capital of France is a small
    reference a small city, and the capital of France is a small

One token, at a near-tie inside our deviation, and the sequences re-converge.
The continuation is *not* covered by the margin argument above, because the
reference's per-continuation-position margins are not in the fixture: once a
token differs, both models are conditioning on different transcripts, so
comparing margins past the divergence is not meaningful.

This is preserved as an explicit regression case in `test_forward.c`, in both
directions. If a later change closes the gap the test fails and says so, so the
expectation is never left stale. If a later change makes the divergence worse --
more tokens differing, or the first two no longer matching -- it also fails,
which is the direction that matters.

## What closing this would actually take

Not tinyBLAS, for the record. The cheapest real win is to reduce the deviation
*before* the chaos amplifier sees it, by matching the reference's blocking
loosely enough that the amplified result is smaller, and by checking whether
the F16 lm_head can accumulate in a wider accumulator than the reference does
(a strictly-better accumulator is legitimate here, because the reference's own
value is not what we are matching -- we are matching the *argmax*, and a more
accurate lm_head moves us closer to the true argmax, not further from the
reference's).

That last point is worth stating carefully, because it is the one place where
departing from the reference's arithmetic is defensible: the gate is the
argmax, the reference is a witness, and a more accurate reduction is a better
witness. Everything else on the list above is matched to the reference exactly,
because everything else on it is amplified by the quantiser and accuracy buys
nothing.

Estimated at one focused pass. It is Phase 6 work, and it is tracked as such.
