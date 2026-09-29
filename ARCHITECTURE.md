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
| 2 | tensor layer, vector kernels, threading, topology-aware affinity | done: no full-occupancy regression on representative workloads |
| 3 | BitNet I2_S correctness: exact ports plus a scalar reference | done: dequant over 46M real elements, and GEMV/GEMM matching the reference on real weights across two shapes |
| 3.5 | gpt2 BPE tokenizer: load, encode, decode, golden vectors | done: 80 prompts token-identical to the reference, and every one round trips |
| 4 | forward pass and generation | **token-identical to the reference at t=0** |
| 5 | chunked attention, KV save/load, state restore | mask correct across chunk boundaries |
| 6 | measured optimisation | full metric table on this machine |
| 7 | ordinary GGUF, `arch_llama` | correct, with a credible path to llama.cpp-class CPU perf |
| 8 | OpenAI-compatible server | streaming, prefix KV reuse |
| 9 | GPU research, optional, after the CPU engine is right | not a Saphira deliverable |

Phases 0 through 8 are the critical path and are CPU-only. Phase 9 is
deliberately last and deliberately optional; see docs/CUDA-FEASIBILITY.md.

Defects found in each phase, and the ones that produced a confident wrong
answer rather than a crash, are recorded in docs/EVIDENCE.md. It is written to
be read by whoever adds the next gate, because the recurring lesson is that a
gate which cannot fail is worse than no gate — one of ours did, and it cost
more time than the bug it was hiding.

### The parity gate

Token-identical greedy output at temperature 0 on the frozen prompt set. Logits
must agree within a tolerance **derived from measurement at Phase 4**, not
guessed. Bit-exact logit equality is the wrong criterion: saphira-llm reduces in
its own order, and demanding bit equality would force us to copy upstream's
thread partitioning instead of designing our own.

Parity is defined on the **raw prompt**. `llama-cli` applies a chat template,
so its output is a different input problem and cannot be the oracle.

## Threading

Native pthreads, no OpenMP, no orchestration language.

### The collapse is a barrier, not a placement problem

This was measured rather than assumed, and the assumption was wrong. The
reference implementation loses a factor of seventeen between 24 and 28 threads
on the target. Varying one variable at a time:

* 28 threads on **24** CPUs: 15.2 t/s, merely degraded
* 28 threads on **28** CPUs: 1.7 t/s, catastrophic
* **24** threads on 28 CPUs: 25.97 t/s, the best number in the table

All 14 core pairs behave identically on their own, so no core is at fault. The
failure is barrier saturation: every logical CPU ends up spinning on ggml's
atomic barrier, and the thread that must break it cannot be scheduled.

So the fix is structural, not a tuning knob. `sllm_parallel_for` has **no
inter-worker barrier at all**: workers claim chunks from a single atomic cursor,
and the only thread that ever blocks is the caller, on a condition variable.
There is nothing for a fully-occupied machine to starve on. Measured at 28
threads, saphira-llm sustains 10.8x, 9.8x and 6.9x on memory-bound,
compute-bound and many-small-item workloads respectively, where the reference
falls by 17x.

### Topology is classified by measurement, not by assumption

The target is a KVM guest that zeroes CPUID leaf 0x1A and ships no `core_type`
file; `cluster_id` merely mirrors `core_id`. The hardware will not say which
cores are fast, so `sllm_topology_calibrate` measures them: a short pinned FMA
probe per logical CPU, best of three, ranked.

Placement is on by default because it was measured to be load-bearing: without
it, threads that sleep and are woken once per region do not get spread by this
scheduler, and throughput stays at the single-thread figure. `--no-place`
reproduces the contrast.

`sllm_topology_recommended_threads` never fills every hardware thread — it
returns 27 of 28 on the target — because full occupancy is where the reference
barrier dies.

### Two bugs worth remembering

Both were found by measurement rather than by reading, and both are the kind
that produce correct output while doing nothing useful:

* **The caller starved the workers.** It claimed the whole region in a tight
  loop before any freshly-signalled worker was scheduled, so "parallel"
  regions ran on one thread. Pinning hid it, because a worker on another core
  gets a chance to run the instant it is signalled. The fix is an arrival
  rendezvous: the caller waits for the workers to arrive before it starts
  claiming. Counting elements could not have found this; counting which thread
  claimed which chunk could.
* **The worker's inner wait did not re-check shutdown.** A worker finishing its
  chunks while a region was still active parked in a wait it never woke from,
  and `join` never returned. `pool_survives_create_destroy_cycles` now covers it.

### What is not yet claimed

The end-to-end `tg128 @ 28 threads` regression is **not** demonstrated fixed,
because there is no generation path yet. It is a Phase 4/6 acceptance
measurement.

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

Note that VNNI's unsigned-first operand is *correct* here and was a bug in
Phase 1. I2_S weight codes are non-negative, so the weight operand is
genuinely unsigned; the bias-and-correction the signed int8 kernel needs would
break parity if carried over. The dispatch was built in Phase 1 to make the
Phase 3 choice obvious, and the two phases are deliberately not merged.

### The tokenizer is not a standard GPT-2 BPE

The acceptance model has no `tokenizer.ggml.pre` key, so the reference uses its
`DEFAULT` pre-type, announces that "GENERATION QUALITY WILL BE DEGRADED", and
applies **four** split passes rather than the single canonical GPT-2 pattern:
punctuation runs first, then the GPT-2 pattern, then number runs, then runs of
three ASCII digits.

The consequences are load-bearing for parity, so they are written down rather
than left in the fixture: `they're` is `they` `'` `re` because the contraction
rule is anchored at the current position; `%!` is a single token because of a
pass that exists in no GPT-2 specification; `1234567890` becomes `123` `456`
`789` `0` because of the digit-triple pass.

That is measured, not asserted. `tools/reference/make-pre-variant` produces a
second model from the acceptance model with exactly one metadata key changed --
`tokenizer.ggml.pre = "gpt-2"` -- and the same vocabulary, merges, token types
and 1.2 GB of weights otherwise byte for byte. The reference over the same 80
prompts gives different tokens for the two:

| prompt | `gpt-2` | absent (DEFAULT) |
| --- | --- | --- |
| `they're` | `they` `'re` | `they` `'` `re` |
| `1234567890` | `123` `456` `7890` | `123` `456` `789` `0` |
| `a + b = c` | `a` ` +` ` b` ` =` ` c` | `a` ` ` `+` ` b` ` ` `=` ` c` |
| `a   b` | unchanged | unchanged |

The last row is the control: whitespace handling is inside the pattern the two
pre-types share, so the two are one implementation with three extra passes
rather than unrelated ones. Both sets of ids are gated in tests.

Two consequences for Phase 7, where the ordinary GGUF models arrive. The
pre-type is per-model configuration to be *discovered*, so a name we have not
implemented is refused with that name in the message rather than approximated;
loading such a model and tokenising it with whatever is compiled in is the worst
available outcome, because the model appears to work. And whitespace is a list,
not a Unicode category: the reference sets the whitespace bit from a separate
table, and without it every double space tokenises wrongly while every single
space stays correct.

### The layout, as verified

Within a 128-element block, element `j` is field `j / 32` of byte `j % 32`, at
bit shift `6 - 2 * (j / 32)`. A 1024-byte block is 4096 weights. The code map
is `{-1, 0, +1, 0}` — code 3 is zero, not -1.

Both of these are permutations or near-misses of the naive reading, so both
produce plausible weights rather than an error. Neither is recoverable by
inspection; both were settled by comparing every element of a real tensor
against the reference. See docs/EVIDENCE.md.

`quantize_i2_s` in the pinned reference uses a *different*, sequential layout
and contradicts the dequantiser beside it. The reference is internally
inconsistent here and the model on disk arbitrates: the transposed layout
reproduces the reference's own dequantisation exactly. We follow the model.
