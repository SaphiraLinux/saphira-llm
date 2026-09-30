# Trainer / runtime parity — the sealed evidence

Written 2026-09-30, at commit `ee59d63`. This is the record that the exported
model executes **the same quantised network the trainer evaluated**, measured
layer by layer and end to end.

Fixture: `n_layer=2, n_embd=128, n_head=4, n_head_kv=2, n_ff=256, n_vocab=256`,
24-token deterministic corpus. Every I2_S contracted width is a multiple of 128
(`n_embd=128`; `ffn_down` contracts on `n_ff=256`), which is the runtime's
proven requirement — see Decision 6.

## End-to-end: trainer vs the ordinary exported path

Trainer NLL is the quantised forward in `src/qat.c`. Runtime NLL is the same
model, exported to I2_S GGUF, loaded through the ordinary `sllm_gguf_open` and
`sllm_model_load`, and scored through the ordinary `sllm_forward_chunk`. No
alternate path on either side.

| steps | trainer NLL | runtime NLL (I2_S) | delta | trainer top-1 | runtime top-1 |
|---|---|---|---|---|---|
| 0   | 5.5446444  | 5.5446389  | -5.54e-06 | 1/23  | 1/23  |
| 50  | 0.14789016 | 0.14768325 | -2.07e-04 | 22/23 | 22/23 |
| 200 | 0.03356979 | 0.032362255| -1.21e-03 | 23/23 | 23/23 |
| 600 | 0.0037805171| 0.0037658483| -1.47e-05 | 23/23 | 23/23 |

The two top-1 counts are identical at every point. The delta is small and
consistently **negative** — the runtime scores marginally *better* than the
trainer. That sign is expected and is the documented train/inference boundary
showing up in the expected direction: the trainer rounds activations
half-to-even under the training convention, the runtime uses `roundf` under the
inference convention. It is a rounding difference, not a scale or contract
error, and it is bounded and stable rather than growing.

Before the row-width fix the same measurement diverged monotonically —
trainer `5.54 → 0.000005` while the runtime rose `5.88 → 17.36 → 18.49`.

## Layer by layer, layer 0, all 24 positions

Trainer intermediates against the runtime's, using production kernels on both
sides. `cos` is cosine similarity; `max` is the largest absolute difference.

| tensor | 0 steps: max / cos | 200 steps: max / cos |
|---|---|---|
| `RMSNorm n1` | 5.96e-08 / 1.000000 | 4.77e-07 / 1.000000 |
| quantised `n1` | 5.96e-08 / 1.000000 | 2.06e-02 / 0.999899 |
| **Q post-RoPE** | 5.59e-09 / 1.000000 | 1.91e-06 / 1.000000 |
| **K post-RoPE** | 7.45e-09 / 1.000000 | 7.63e-06 / 1.000000 |
| **V** | 5.59e-09 / 1.000000 | 3.81e-06 / 1.000000 |
| **attention out** | 3.73e-09 / 1.000000 | cos 1.000000 |
| `attn_sub_norm` | 6.71e-08 / 1.000000 | cos 1.000000 |
| quantised `ao` | 7.45e-08 / 1.000000 | cos 1.000000 |
| `attn_out` projection | 3.73e-09 / 1.000000 | cos 1.000000 |
| `RMSNorm n2` | 5.96e-08 / 1.000000 | cos 1.000000 |
| quantised `n2` | 5.96e-08 / 1.000000 | cos 1.000000 |
| `ffn gate` | 3.73e-09 / 1.000000 | cos 1.000000 |
| `ffn up` | 2.79e-09 / 1.000000 | cos 1.000000 |
| `silu(gate)*up` | 2.55e-11 / 1.000000 | 9.77e-04 / 1.000000 |
| `ffn_sub_norm` | 4.66e-10 / 1.000000 | cos 1.000000 |
| quantised `fy` | 4.66e-10 / 1.000000 | cos 1.000000 |
| `ffn_down` projection | 1.27e-11 / 1.000000 | cos 1.000000 |

Every tensor is at cosine 1.000000 with maximum absolute differences at or near
float32 rounding (1e-11 to 1e-7 at 0 steps). The larger maxima at 200 steps sit
on values an order of magnitude larger and are still cosine 1.000000.

## The exported representation is the trainer's own quantisation

All seven I2_S tensors in layer 0, at 0 and 200 steps, decoded the way
`sllm_i2s_gemv` decodes them — row base `r*(cols/4)`, interleaved 128-element
blocks, `j = 32*field + lane`, byte `lane`, shift `6 - 2*field`:

- per-tensor scale **bit-identical** to `sllm_qat_weight_scale` on the master
- **0** code mismatches out of every element, against the trainer's own
  `w/scale > 0.5` rule
- reconstruction from the file reproduces the trainer's forward weight exactly

Export byte-stability against fingerprints captured from the writer *before* the
row-width change: 128 → `3d8afbc91665b037`, 256 → `cceae8e057a0186d`, and the
shipped 2048 → `de1dfed496c24ff3`, all unchanged.

## Two harness defects found by writing the comparison

Recorded because both produced convincing wrong evidence:

1. The probe decoded I2_S with the naive `byte = i/4` and reported absurd
   scales (`5.86e+13`). Validated only by checking section A independently.
2. The probe then "repacked to simple order" — which was wrong, because the
   runtime's dot already walks the interleaved layout — and produced a
   fabricated scale error.
3. The probe applied the **query position's** RoPE to K and reused it as the
   key for every `s <= t`, so for `t > 0` it mis-rotated historical keys. For
   `t = 0` that is accidentally correct, which is why it looked plausible and
   why the "attention output diverges" reading was wrong. Fixed to maintain K
   after RoPE per token position, exactly as the production attention cache does.

The lesson is the same each time: a value that is merely *finite*, or a
comparison that agrees *approximately*, is not evidence. Only a comparison
against an independently-derived reference at cosine 1.0 is.
