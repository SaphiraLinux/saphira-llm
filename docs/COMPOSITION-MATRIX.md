# Composition matrix

What the discovery IR has actually PROVEN, per composition. A cell is filled only
when a real artefact forced it. An empty cell is not a claim that the composition
is impossible; it is a claim that nothing has been measured for it yet.

| Composition                          | Model         | Proved at | Distinguishing evidence |
|--------------------------------------|---------------|-----------|-------------------------|
| Attention only                       | Qwen3         | 91c9bcc   | 36 layers, all identical body |
| Attention only (GQA, untied out)      | Llama-3.2     | 91c9bcc   | 16 layers, GPT/adjacent pairing, tied output |
| Attention only (SWA + SoftCap)        | Gemma-2       | 9a931ea   | 26 layers, alternating sliding/full |
| MoE only (dense attention + experts)  | OLMoE         | 26312c9   | 64 experts top-8, router from graph op order |
| SSM only, NO attention at all        | Mamba-2       | af37e6e   | 24 layers, head_count 0 |
| Attention + SSM in ONE layer          | Falcon-H1     | 6c64778   | all 36 layers carry BOTH bodies |
| Heterogeneous bodies across depth    | Nemotron-H    | 6c64778   | 52 layers, 3 distinct bodies (4 ATTN / 24 SSM / 24 FFN) |
| Attention + MoE                      | **EMPTY**     | --        | MoE proved only with uniform layers |
| SSM + MoE                            | **EMPTY**     | --        | not attempted |
| Attention + SSM + MoE                | **EMPTY**     | --        | the target for architecture eight |

## Why the empty cells matter

OLMoE proved experts exist and how to read a router. It did NOT prove that experts
can coexist with a state-space recurrence, or that they can appear in only SOME
layers. Nemotron-H proved layer-local bodies but is the DENSE variant: no experts.

The composition `Attention + SSM + MoE` with a non-uniform schedule is therefore
the one remaining unproven corner, and it is the corner most likely to require a
new operator rather than a new parameter.

## Layer-locality status per op

Every op below was verified to be gated by `AT(role, layer)`, not by
`dims.count(role)`. See the audit in the commit message.

| Op                | Layer-local gate | Negative fixture |
|-------------------|------------------|------------------|
| Attention         | yes              | P5, P8           |
| SelectiveScan     | yes              | P3, P4, P9       |
| DepthwiseConv1D   | yes              | covered by SSM   |
| GatedMLP          | yes              | P6, P7, P10      |
| RMSNorm (all)     | yes              | P1 (ffn_norm absent at layer 7) |
| RoPE              | yes              | P5 (no Q/K at layer 0) |
| Linear (proj)     | yes              | layer 1 emits 4 ABSENT projections |
| ExpertRouter      | yes              | Q3, Q5            |
| ExpertGatedFFN    | yes              | Q4, Q6            |

## GLU ownership on Nemotron-H (resolved by attribution)

24 `GLU` ops appeared to contradict the ungated classification, since a GLU is a
multiplicative gate. Counts could never have settled it — GLU, SSM and FFN layers
are all 24 here, so every count-based story fits either answer.

| Witness | Says |
|---------|------|
| Tensor inventory | no `ffn_gate.weight` at the FFN layers |
| Reference graph | all 24 GLU nodes fed by `mamba2_y_add_d-N` — the **state-space** path |
| Reference graph | **zero** GLU nodes fed by an `ffn_*` tensor |

Neither measurement overrules the other; they corroborate. `DenseMLP` now rests on
**two independent witnesses**. Had the GLU been attributed to the FFN, the correct
response would have been a classification *change* to `GatedMLP` with the gate
tensor `ABSENT` — not a refinement of `DenseMLP`.

## Correctly model-global (NOT layer-owned)

These are single tensors with no layer identity, so a model-global test is correct
for them and they are deliberately not layer-local:

- `token_embd.weight`, `output.weight`, `output_norm.weight`
- tokenizer selector legs (`tokenizer.ggml.{model,pre,tokens}`)
- scalar metadata (`block_count`, `embedding_length`, `vocab_size`)

A per-layer ARRAY (`feed_forward_length`, `attention.head_count_kv` on Nemotron-H)
is neither: it is layer-owned AND array-valued, and is reported as such.
