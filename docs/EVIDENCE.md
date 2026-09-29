# Evidence

Defects this project found, kept so the reasoning survives the commit. Every
entry is a real bug that produced a wrong number, a hang, or a check that
meant nothing — not a style note.

The rule they share: the reference is the specification. Where our code
disagreed with it, the reference won. Where the reference looked wrong, we
checked the model on disk before changing anything.

---

## Phase 0

### The reference dumper's greedy position was off by one

The reference logit dumper printed the position at which the greedy token was
selected as the *next* index rather than the current one. The `.f32` logits
and their hashes were correct, so the bug was invisible in the fixture and
only showed up as a nonsensical sampled position.

Fixed in `tools/reference/sllm-logits-ref.c`; the golden `.f32` files and
their hashes did not change, which is the confirmation that the logits were
never wrong — only the attribution of which token the argmax belonged to.

---

## Phase 1

### I2_S is not a fixed block type, and ggml's own traits table is wrong about it

`sllm_gguf_type_nbytes` exists because a plain `ne0 * type_size` calculation is
incorrect for I2_S. Its payload is `n_elements/4 + 32` bytes, with the f32
scale at `data + n_elements/4`.

ggml's traits table claims 1 byte per element and compensates with a special
case elsewhere in the codebase. Trusting the table puts the scale at the
wrong offset and produces confident garbage. This was found by calling
`ggml_blck_size`/`ggml_type_size` on the pinned reference rather than
transcribing the numbers, which is also how NVFP4 turned out to be 64/36 and
not the 32/17 one would assume.

### `.text.sllm_isa_ext` makes the ISA guard pass vacuously

Above-v3 kernels are guarded by placing them in a dedicated section and
having `make check-isa` fail on any above-v3 instruction outside it.

Naming that section `.text.sllm_isa_ext` breaks the guard silently: the
default linker script merges every `.text.*` into a single output `.text`, so
the guarded functions are merged away and the disassembler finds no guard
section to complain about. The check passes, having verified nothing.

Observed here rather than theorised — the section is named `.sllm.isa.ext`
for this reason. The check also refuses to pass when the section is absent at
all, and was verified to fail on a deliberately mis-built binary containing an
unguarded `vpdpbusd`. A safety check that cannot fail is worse than none.

### `dpbusd` reads its first operand as unsigned

The first VNNI kernel shipped with a bug its own test caught immediately.
`VPDPBUSD` multiplies an unsigned byte by a signed byte, so passing two signed
vectors read negatives as large positives and returned plausible wrong
numbers. Fixed with a bias and correction.

The lesson matters more than the fix: the real ternary GEMM will **not** need
the correction, because I2_S weight codes are non-negative and only the
activations are signed. The bias that is mandatory in Phase 1 is forbidden in
Phase 3, and adding it "for safety" would break exact parity.

### Block sizes were verified, not transcribed

Called on the pinned reference rather than written from memory, which is what
caught NVFP4. The lesson generalises to every constant in this project: a
layout constant copied from documentation is a claim, and the pinned reference
is available to check it against for free.

---

## Phase 2

### The 28-thread collapse was a barrier, not a placement problem

`tg128` at 28 threads ran at roughly 1.7 tokens per second. The first
explanation was that 28 threads were landing on 24 CPUs, so the pool had more
workers than cores and the barrier serialised them.

That was wrong, and it was wrong in the way that matters, because it looks
right from the outside. Varying one variable at a time:

| configuration | throughput |
|---|---|
| 28 threads on 24 CPUs | 15.2 t/s |
| 28 threads on 28 CPUs | 1.7 t/s |
| 24 threads on 28 CPUs | 25.97 t/s |

Filling every logical CPU is the *worst* case, and fewer threads on the same
CPUs is 15x faster. All 14 core pairs behaved identically on their own, so no
particular core is at fault. It is ggml's spinning barrier saturating: every
logical CPU spins, and the thread that must break the barrier cannot be
scheduled.

The fix is structural rather than a tuning knob. `sllm_parallel_for` has no
inter-worker barrier at all — workers claim from a single atomic cursor and
only the caller ever blocks, on a condvar. Measured at 28 threads on 28 CPUs,
where the reference falls 17x, the pool sustains 10.8x on a 32 MiB streaming
workload, 9.8x on an L1 FMA chain and 6.9x on many small work items, with no
full-occupancy regression in any of them.

### The caller starved the workers

The first version of the pool was correct and did nothing. The caller claimed
the entire region before any signalled worker had been scheduled, so parallel
regions ran on one thread while every other thread slept.

Pinning hid it. A worker on a different core gets a chance to run
immediately, so a pinned run looked healthy. Fixed with an arrival
rendezvous.

Counting elements cannot see this defect — the count is right, the work
distribution is wrong. Counting *which thread claimed which chunk* can, and
that is now a test.

### The worker's inner wait never re-checked shutdown

A worker that finished its chunks while a region was still active parked
forever, so `join` hung at every thread count above one. Covered by
create/destroy cycle tests.

### Four kernel defects, all found by the golden vectors

Every one of these produces plausible numbers in the wrong order rather than a
failure, which is why none of them would have been caught by a test that only
checked that a kernel ran:

- **SiLU** computed `x*exp(x)` instead of `x*sigmoid(x)`.
- **NeoX RoPE** paired `x[i]` with itself.
- **rms_norm** accumulated the raw values and never squared them.
- **Q4_0** grouped all low nibbles before all high ones instead of
  interleaving them per byte.

The Q4_0 one is the instructive case. It is not a rounding difference. The
deinterleaved ordering is a permutation that produces entirely valid-looking
quantities, so a magnitude check, a plausibility check, and even a spot check
of a few elements can all pass while every single value is wrong.

### Casting negative floats to `unsigned long` in the tests

ASan and UBSan found real undefined behaviour — in the tests, not the
kernels. Float-to-integer conversion of a negative or out-of-range value is
undefined. The tests now compare bit patterns.

A useful detail: the sanitiser run was what surfaced it, and it only ran at
all because the Saphira gcc has no `libasan` for this musl target, so the
sanitiser builds use clang. The release build is unaffected by the
substitution.

---

## Phase 3

### I2_S weights are stored transposed, not four consecutive per byte

The obvious reading of a 2-bit type is four elements per byte, low bits first.
The reference `dequantize_row_i2_s` does not do that. Within a 128-element
block, element `j` is field `j / 32` of byte `j % 32`, at bit shift
`6 - 2 * (j / 32)`.

The obvious reading is a *permutation* of the right answer, so it produces
entirely reasonable-looking weights. It was caught by dequantising a real
tensor and comparing all 6,553,600 elements against the reference; elements 1
through 31 differed.

### Code 3 is zero, not -1

The natural map for a 2-bit code is `{-1, 0, 1, -1}`. The reference uses
`{-1, 0, 1, 0}`. Both are defensible; only one is the reference. This changes
the weight count and therefore the output.

### `quantize_i2_s` in the reference contradicts the reference's own model

`ggml-cpu/quants.c::quantize_i2_s` writes a sequential layout: four elements
per byte, low bits first. The dequantiser in the same file, and the converter
that produced the weights, both use the transposed layout.

The reference is internally inconsistent. The model on disk is the
arbitrating evidence: dequantising real tensors with the transposed layout
reproduces the reference's dequantisation exactly, and the sequential layout
does not. We implement the layout the weights actually have.

Recorded because the next person to read `quants.c` will reach the opposite
conclusion, and they will be wrong, and it will be confidently wrong.

### `ggml_gemv_i2_i8_s` argument order is not what the names suggest

The reference signature is `gemv(n, s, bs, vx, vy, nr, nc)`. The natural
reading is `nr` rows, `nc` columns. In fact `nr` is the number of activation
columns and `nc` the number of weight rows, so one activation against 512
weight rows is `nr=1, nc=512`.

Passing them the intuitive way returns zeros rather than an error, and a
kernel returning zeros looks like a numerical bug in the kernel — a very
expensive place to look. Found by noticing the all-zero result and reading the
caller instead of the callee.

### A gate that reported the wrong reason for a correct kernel

`test_i2s.c` parses fixture fields line by line and compared computed hashes
against them. The record fields were declared *inside* the read loop, so each
was re-initialised on every line: the raw hash was read and stored correctly,
then wiped before the line that triggered the comparison. The comparison ran
against zero, and the failure message reported "reference 0" with complete
confidence.

The kernel was correct throughout. The gate was not.

This is worse than having no gate, because it points the reader at the wrong
code — a confident message naming a number that never existed sends you to
fix the kernel. Two changes: the fields now live outside the loop, and the
comparison triggers on the record's *last* field so a two-hash record cannot
be compared before both hashes are read.

### An invented epilogue API truncated every result to zero

`sllm_i2s_apply_epilogue(int32_t * dots, ...)` cast the float epilogue result
back into the int32 array it was handed, so every value became 0.

It had no callers and no counterpart in the reference. The reference applies
the epilogue in place over the GEMM's *float* output buffer, at
`ggml/src/ggml-bitnet-compute.c:164`:

    tmp[row] = (tmp[row] - act_sums[i1]) / (act_scales[i1]) * (*scale);

The signature, not the arithmetic, was the defect. A function that cannot
express the reference's behaviour is removed rather than corrected into a
different invented function. The scalar `sllm_i2s_epilogue` is what the
forward pass needs, and it is pinned by a test.

### The `- act_sum` term looks wrong and is correct

The epilogue is `(dot - act_sum) / act_scale * w_scale`. Subtracting the sum
of the quantised activation looks unjustified — the correction one would
expect depends on the sum of the *weights*, and a ternary weight sum is zero
only if the `{-1, 0, +1}` values happen to balance, which is not something to
rely on.

The reference does it anyway, unconditionally. So do we, and a test pins the
behaviour specifically, so that a later reader who spots the same apparent
bug cannot "fix" it into a different number. It is the reference's
arithmetic, and matching the reference is the gate.

---

## What these have in common

Four of the kernel defects above — SiLU, NeoX RoPE, rms_norm, Q4_0, and the
I2_S layout — were invisible on the acceptance model and visible only against
the reference, a synthesised adversarial case, or a varying shape. Two
produced confident, well-formatted messages that pointed at the wrong code.

The defences that actually caught them were the ones that compare against the
reference element by element, and the ones that vary the shape. A gate built
only from the acceptance model's own parameters would have passed every one of
these.

That is the argument for the parity rule in `ARCHITECTURE.md`, and it is worth
restating when a new gate is added: *what would this fail on?* A gate that
cannot fail is not a gate, and the one in this file that could not fail is the
one that took longest to notice.
