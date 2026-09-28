# Architecture

## The problem

One runtime that runs BitNet 1.58 ternary weights properly *and* ordinary
GGUF models, in C11, on x86-64-v3, with x86-64-v3 as a genuinely optimised
path rather than a generic fallback. No Python at runtime.

## Layer chain

```
GGUF loader (mmap, metadata, tensor table, per-tensor extents)
      ↓
tensor description {ptr, ne[], type, nbytes, strides}
      ↓
arch adapter (vtable: bitnet_b158 | llama dense | clean failure)
      ↓
op executor — a sequential plan over a shared arena
      ↓
kernel dispatch {op × dtype × ISA}
      ├── f32 / f16 / bf16
      ├── GGUF quantised: Q8_0, Q4_0, Q4_K, Q6_K
      └── BitNet ternary: I2_S
```

### What is deliberately absent

No tensor IR. No graph optimiser. No backend abstraction. No dynamic kernel
loading. No plugin system.

Each of those exists in ggml for good reasons at a scale we are not operating
at. Carrying them would cost more than they return for a two-architecture
runtime, and would make the thing we actually have to get right — the ternary
kernel and the parity gate — harder to see.

A "plan" is a straight-line sequence of ops emitted by the arch adapter and
executed once. It is a graph in the trivial sense, and that is enough.

## Model format is not execution

The reason a model format can be added without touching a kernel is that the
arch adapter is the only thing that knows a model's *structure*, and the kernel
table is the only thing that knows a *dtype*. They meet at the op list.

Adding Q5_K is one row in the type table plus one dequantiser. Adding a new
architecture is one adapter that emits ops the executor already runs.

## ISA policy

x86-64-v3 is the baseline. The whole binary is compiled for it, so v3
instructions are legitimate anywhere in the code. Above v3 is opt-in per
kernel and never a substitute for the v3 path.

| Level | Adds | Used for |
| --- | --- | --- |
| `SLLM_ISA_V3` | AVX, AVX2, FMA, F16C, BMI, BMI2 | the baseline; always installed |
| `SLLM_ISA_VNNI` | `vpdpbusd` | the ternary GEMM — no int16 intermediate, no saturation |
| `SLLM_ISA_AVX512` | AVX-512F/BW/VL/DQ + VNNI | reserved; the target CPU has none |
| `SLLM_ISA_AMX` | AMX-INT8/BF16 | reserved; the target CPU has none |

AVX-512 and AMX are defined so the ladder is complete and so future hardware is
a data change rather than a redesign. **No AVX-512 or AMX kernel is written**,
because the target cannot run one and shipping untestable code would be a claim
nobody can check.

The three rules that keep this honest:

1. A kernel needing more than v3 must carry `SLLM_ISA_EXT`, placing it in
   `.sllm.isa.ext`. The section is deliberately *not* `.text.sllm_isa_ext`,
   because the default linker script merges every `.text.*` into one output
   `.text` and the guard would vanish at link time.
2. `sllm_isa_build()` is the only thing that installs a guarded function
   pointer, and it installs it only when detection found the hardware *and*
   the selected level permits.
3. `make check-isa` disassembles the binaries and fails if an above-v3
   instruction appears anywhere else. It refuses to pass when the guarded
   section is absent, so it cannot pass vacuously, and it has been verified to
   fail on a deliberately mis-built binary.

A floor above the hardware ceiling is a hard error, never a clamp. A deployment
that pinned `--isa amx` and quietly got v3 would be running different kernels
from the ones it asked for with nothing to say so.

## Error policy

`SLLM_ERR_GGUF_TRUNCATED` is not "unsupported model". Every rejection reason is
its own status, because the user's next action differs for each. The tests
assert the *distinctions*: missing-key versus wrong-type, truncated versus
bad-layout, known-but-unsupported versus not-a-type.

## Phase plan and gates

| Phase | Content | Gate |
| --- | --- | --- |
| 0 | provenance, reference build, measured baseline, golden vectors | done |
| 1 | container, ISA dispatch, first kernel, baseline proof | done |
| 2 | tensor layer, vector kernels, threading, topology-aware affinity | 28-thread collapse eliminated |
| 3 | BitNet I2_S correctness: exact ports plus a scalar reference | golden-vector match |
| 3.5 | gpt2 BPE tokenizer: load, encode, decode, golden vectors | tokeniser vectors match upstream |
| 4 | forward pass and generation | **token-identical to the reference at t=0** |
| 5 | chunked attention, KV save/load, state restore | mask correct across chunk boundaries |
| 6 | measured optimisation | full metric table on this machine |
| 7 | ordinary GGUF, `arch_llama` | correct, with a credible path to llama.cpp-class CPU perf |
| 8 | OpenAI-compatible server | streaming, prefix KV reuse |
| 9 | GPU research, optional, after the CPU engine is right | not a Saphira deliverable |

Phases 0 through 8 are the critical path and are CPU-only. Phase 9 is
deliberately last and deliberately optional; see docs/CUDA-FEASIBILITY.md.

### The parity gate

Token-identical greedy output at temperature 0 on the frozen prompt set. Logits
must agree within a tolerance **derived from measurement at Phase 4**, not
guessed. Bit-exact logit equality is the wrong criterion: saphira-llm reduces in
its own order, and demanding bit equality would force us to copy upstream's
thread partitioning instead of designing our own.

Parity is defined on the **raw prompt**. `llama-cli` applies a chat template,
so its output is a different input problem and cannot be the oracle.

## Threading

Native pthreads, no OpenMP, no orchestration language. The work distribution is
chosen from measurement.

This matters more than it sounds. Measured on the target: generation gains only
2.45x going from 1 thread to the best thread count, and **collapses to 1.42 t/s
at 28 threads** — 20x worse than the 22-thread best and 10x worse than
single-threaded. The best measured count, 22, is 8 P-cores with SMT plus 6
E-cores without their SMT siblings. Placement is therefore topology-aware, and
the policy is decided from data rather than assumed.

## The I2_S contract

Reproduced exactly, including the parts that look wrong. The output epilogue is
`(dot - act_sums) / act_scale * i2_scale`, and the `- act_sums` term
mathematically assumes `sum(w) == 0` over K. Upstream does it; parity means we
do it too. Fixing it would be a divergence, not an improvement.

The x86 kernel accumulates in int16 lanes through the *saturating*
`maddubs`, which is a precision hazard rather than a detail. The AVX-VNNI path
avoids it entirely by accumulating straight to int32, which is the concrete
performance argument for Phase 6 and the reason the dispatch pattern was built
in Phase 1.
