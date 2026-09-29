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

## Phase 3.5

### The model has no `tokenizer.ggml.pre`, so the reference uses four regexes

The obvious implementation of "gpt2 BPE" is the single canonical GPT-2
pattern. This model does not use it. `tokenizer.ggml.pre` is absent, so the
reference falls back to its `DEFAULT` pre-type and logs:

    load: missing pre-tokenizer type, using: 'default'
    load: GENERATION QUALITY WILL BE DEGRADED!

`DEFAULT` is four successive split passes, not one:

    [\p{P}\$\+<=>\^~\|]+
    's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
    \p{N}+
    [0-9][0-9][0-9]

The observable consequences, all from the fixture:

| input | reference | a textbook GPT-2 BPE |
| --- | --- | --- |
| `1234567890` | `123` `456` `789` `0` | `123` `456` `789` `0` only by luck of merge ranks |
| `they're` | `they` `'` `re` | `they` `'re` |
| `%!` | `%!` | `%` `!` |
| `a + b` | `a` ` ` `+` ` b` | `a` ` +` ` b` |

The third and fourth are pass 1, which is in no GPT-2 specification at all. The
second is pass 2's apostrophe rule being anchored at the current position rather
than searched for, so an apostrophe in the middle of a word does not begin a
contraction.

Reading the reference and reproducing its *configuration* mattered more here
than reproducing its *algorithm*. The four-regex structure and the fallback
warning are both invisible from the model file's contents alone; only running
the reference reveals them.

### Whitespace is not a general category

The first port of the Unicode tables took the category ranges and stopped. That
makes an ASCII space look like ordinary punctuation, because U+0020 is Zs and
the range table marks it `SEPARATOR` with no whitespace bit. The result is that
` ?[^\s\p{L}\p{N}]+` matches runs of spaces, `\s+(?!\S)` never fires, and every
prompt containing a double space tokenises differently from the reference.

The reference's own code shows why: it builds its per-code-point flags by
overlaying the category ranges and then, separately, setting the whitespace bit
for every entry in `unicode_set_whitespace`. Whitespace is a list, not a
category, and the two tables have to be ported together.

Worth noting what the failure looked like: it was not a crash and not an
obviously wrong token, it was a *plausible* split that differed only in where
the space tokens fell. ASCII-only prompts pass under both readings, so a fixture
made of ordinary prose would never have caught it.

### `byte_to_cpt` must return a UTF-8 encoding, not the bare byte

A passing-through byte maps to the UTF-8 encoding of the code point of the same
value. For ASCII the two are the same byte, which is exactly why the confusion
survives: `byte_to_cpt(0xCE)` is the two bytes `C3 8E`, the character `Î`, and
not the single byte `CE`.

Returning the bare byte leaves every word's byte-encoded form identical to its
original bytes, so no piece is ever found and non-ASCII text encodes to
*nothing* -- while ASCII, which is most of any fixture, passes throughout. The
symptom is a prompt that yields a bare BOS and nothing else, which looks like a
missing-vocabulary problem and is not.

### BPE without a staleness check is not BPE

The reference's merge loop pops the lowest-ranked pair and merges it, but only
after checking that the pair's text is still the text it had when the pair was
queued. A pair whose left half has since absorbed a neighbour is dropped.

Without that check, such pairs get merged anyway. The output is still plausible
tokens, still round trips through decode, and is simply not the reference's.
Seven prompts disagreed before the check was restored, all of them ones where
merges chain: a long word, a run of repeated characters, a trailing word.

The check is a length comparison rather than a string comparison, which is
equivalent here because both halves are contiguous slices of the same word.

### `kv_as_i64` could not read any array

`kv_as_i64` switched on `kv->type`, which is `SLLM_VT_ARRAY` for every array, so
it always fell through to the default case and returned false. Reading element
zero of any array failed.

No test covered it, because the only arrays in the acceptance model belong to
the tokenizer, and the tokenizer had not been built. The accessors wrapped it
looked correct: they validated the element type and then called a helper that
could only return false. The failure surfaces as a missing metadata key, which
points at the wrong file entirely.

### A string array's pointer table is not bounded by the element count

Phase 1 skipped string arrays so the vocabulary would still parse, and the
comment said so. Loading them means allocating `n` pointers where `n` comes
straight from the file: a header claiming 2^28 strings would ask for 2 GiB of
pointers out of a file with no room for them.

Each string element occupies at least 8 bytes in the file, since that is the
size of its length prefix, so `n <= remaining_bytes / 8` is a bound taken from
the file rather than from the declared count. That is the same reasoning as the
existing tensor-extent checks: a header does not get to allocate what it says.

### Two harness defects that reported the wrong thing

Both of these produced confident, well-formatted messages that were false.

A test that reported success from a count rather than an outcome printed "80
prompts, all matching the reference" while all 80 records were failing, because
the message was gated on having read 80 records and nothing else. A gate has to
report what it compared, and a success line that fires regardless of the result
is the most expensive kind of wrong: it is believed.

The same test then failed to parse the fixture at all for a while, reporting
`reference has 0` for every prompt. `"  ids"` is five characters and the parse
started at `line + 4`, so `strtol` was handed the `s`. Every comparison ran
against an empty expectation, which is why the count of 80 was the only thing
that looked healthy.

### The pre-tokeniser is model metadata, and that is now measured

The finding that `tokenizer.ggml.pre` being absent means DEFAULT rather than
"ordinary GPT-2" is easy to record and easy to forget. It was made structural
instead, by producing a second model from the first:

    tools/reference/make-pre-variant <in.gguf> gpt-2 <out.gguf>

Same vocabulary, same merges, same token types, same 1.2 GB of weights, byte
for byte. Exactly one metadata key added. Running the reference over the same
80 prompts against both:

| prompt | `pre` = gpt-2 | `pre` absent (DEFAULT) |
| --- | --- | --- |
| `they're` | `they` `'re` | `they` `'` `re` |
| `it's` | `it` `'s` | `it` `'` `s` |
| `1234567890` | `123` `456` `7890` | `123` `456` `789` `0` |
| `a + b = c` | `a` ` +` ` b` ` =` ` c` | `a` ` ` `+` ` b` ` ` `=` ` c` |
| `a   b` | unchanged | unchanged |

One key, different tokens. The last row is the useful control: whitespace
handling lives inside the pattern the two pre-types share, so prompts differing
only in whitespace are identical, which is what shows the two are one
implementation with three extra passes rather than two unrelated ones.

So the pre-type is an enum resolved from the file, with an explicit fallback,
an accessor for reporting, and a refusal for any name we have not implemented.
Refusing matters: loading a model whose `pre` names a pre-tokeniser we lack and
tokenising it with whatever happened to be compiled in is the worst available
outcome, because the model appears to work. The reference throws there too.

Phase 7's ordinary GGUF models will each carry their own `pre`, and this is the
evidence that reading it is mandatory rather than a nicety.

### Writing a scalar GGUF string as a raw blob desynchronises the file

The variant tool above emitted every non-array key as `elem_size` bytes. For a
scalar string the parser records `elem_size = 1`, because a scalar has no
meaningful element size, so every string key was written as a single byte. The
resulting file was rejected several keys later with an absurd string length
(11584560907875033284), which points at the last key rather than at the
miswriter.

A parser that reports `elem_size = 1` for a scalar string is inviting exactly
this. The writer now special-cases strings and the parser's oddity is documented
where it is set.

### A regression test that counted instead of asserting

The GGUF test for the string-array bound declared 2^28 strings and expected a
rejection. It passed vacuously: the array was written by hand, so the
key-value count was never incremented, so the parser skipped the key and the
file opened successfully. The test asserted that a file which was never
examined had been rejected.

The bound itself was correct all along, which is the uncomfortable part: a
repro outside the harness rejected the same bytes immediately. Three separate
gates in this project have now reported success without having compared
anything, which is why the completion messages now check the failure count
rather than a count of records read.

### The unescaper kept the line terminator

The fixture stores each prompt's text on one line with newlines escaped as `\n`.
The unescaper stopped only at NUL, so the line's trailing newline became part of
the prompt. Every one of the 80 prompts gained a token, and the tokenizer was
blamed for a systematic off-by-one that was in the test.

A bare newline in the fixture is always the line terminator, never prompt
content, so stopping at one is unambiguous.

---

## Phase 4

### The two RoPE layouts were swapped, and a golden vector could not see it

NEOX and NORMAL had their pairings exchanged from Phase 2 through Phase 4.

    NEOX   pairs the two halves:  (k, k + n/2)
    NORMAL pairs adjacent pairs: (2k, 2k+1)

From ggml's `rotate_pairs`, called with `(n_dims, n_dims/2)` for NEOX and with
`(n_dims, 1, scale=1)` for NORMAL. BitNet is NEOX, so every Q and K vector in
the model was rotated in the wrong pairing.

What makes this one worth writing down at length is that it **passed every test
for two phases.** The Phase 2 RoPE golden vector asserts the rotation preserves
each pair's norm. Both layouts preserve the norms of whatever pairs they
rotate, so swapping them is a norm-preserving relabelling of the entire
vector: every element is individually plausible, and the relative phase
between dimensions -- which is the entire content of a positional encoding --
is destroyed.

The property being tested was simply not a property that distinguishes the two
implementations. A golden vector is only as good as the property it asserts,
and "is a rotation" does not survive a question of *which* rotation.

The fix is in the pairing, and the test now asserts three things it did not
before: the norm of each pair **as that layout defines it**, that the
rotation actually happened at a nonzero position, and that the two layouts
produce **different** output from the same input. That last one is the
assertion that would have caught the swap immediately -- two encodings that
claim to differ and produce identical bytes means one is implemented as the
other.

It was found by the Phase 4 logit gate, which is the first thing in this
project that compares real end-to-end output against the reference. Everything
before it was a property test.

The symptom, for the record: tokenisation matched exactly, the argmax matched
at 5 of 6 positions, and the logits were wrong by 0.4 to 1.0. Close enough to
look like float noise, and not float noise at all.

### A test's success message that reported success regardless

Already recorded in Phase 3.5, and it happened again here in a new form: the
forward test's completion line was gated on the number of cases compared rather
than on the comparison passing, so it would have printed "all token-identical"
with every case failing. The failing count is captured at entry and compared
afterwards now.

### A hardcoded prompt length that included the NUL

The greedy-continuation test passed `"The capital of France is", 25` -- and the
string is 24 characters, so the tokenizer received a trailing NUL byte. That
changed the last prompt token and made all 8 continuation tokens disagree with
the reference, which presented as a forward-pass bug and was entirely a test
bug. `strlen` is used now, with a note saying why.

### `sllm_ctx_reset` did not reset

The KV cache is read up to `n_past` at every position and is never zeroed on
write, so a context reused without a reset attends to the previous
conversation's keys and values. The function existed, was called, and did
nothing; its comment described clearing the live prefix, which it did not do.

This is the same shape as the Phase 2 caller-starvation bug: correct output,
no effect, and invisible unless you happen to reuse a context, which is the
case a test is least likely to cover. The context now tracks `n_past`, the
reset clears the live prefix, the forward pass refuses to rewind a position,
and there is a test that generates twice around a reset and requires identical
ids.

### A test asserted that freq_scale divides, and it never checked

While fixing the RoPE angle the Phase 4 work turned up a second unverified
assumption. `sllm_rope_inplace` divided the angle by `freq_scale`; the
reference multiplies it:

    static void rope_yarn(..., float theta_extrap, float freq_scale, ...) {
        float theta_interp = freq_scale * theta_extrap;

and the metadata factor is `1.0f/ropescale`, so `rope.scale_linear = 2`
becomes a factor of 0.5 and the rotation slows for extrapolation. Multiplying
is what makes that sensible.

The Phase 2 test asserted the divide, framed as "a scale of 2 is the same as
doubling the position", and passed for two phases. Same shape as the NEOX
swap: an assumption that reads as reasonable and was never tested against the
thing it mirrors. A test that only re-asserts the implementation's assumption
is not a test.

### RoPE now has known-output vectors, because properties could not catch a swap

The layout swap above is the reason `ops_rope_matches_known_output_for_both_layouts`
exists. "The rotation preserves each pair's norm" is satisfied by both
pairings, so no property of that kind can distinguish NEOX from NORMAL. The
test now pins literal expected outputs for both layouts -- computed from the
reference's own algorithm, its repeated-multiplication angle cache and its
`rotate_pairs` offsets -- so an exchange moves constants that do not move on
their own.

### The float gap, and why the token gate is thinner than it looks

The first version of this entry said the deviation was "the expected
consequence of a reduction order we chose rather than copied". That was wrong,
and checking it is the most useful thing in this file.

A 2-to-4 percent deviation is not what float reordering produces; reordering
over 2560 elements gives about 1e-4. So either there was a systematic error or
a tiny one was being amplified. Running the attention products and the lm_head
in double precision moved the worst deviation from 0.917 to 0.883 -- barely.
Reordering error would have collapsed. The source was systematic, and a
systematic error can be found.

It was six of them, all listed in docs/PHASE4-GAP.md. The largest were the
RoPE angle (built by a single `powf` instead of the reference's running
multiplication) and SiLU (6912 libm exponentials per layer instead of the
reference's AVX2 polynomial, which is not the same function). Fixing them moved
the top-1 deviations from 0.34/0.16/0.40 to 0.048/0.013/0.063 at positions
2, 4 and 5.

**What remains is irreducible by choice.** `GGML_LLAMAFILE=ON`: the reference's
QK^T, P@V and lm_head all run through tinyBLAS's blocked SIMD GEMM. And this
model amplifies any float difference -- including one ulp -- to O(0.1), because
it is a chain of 30 int8 quantisers and a perturbation that crosses a rounding
boundary moves a whole quantisation step.

So the deviation cannot be driven to zero without porting tinyBLAS, which
`ARCHITECTURE.md` declines for the same reason it declines to copy the thread
partitioning. It is a design position, and it is written down so it is not
later mistaken for a shortfall.

The gate is therefore not "within some tolerance". It is:

> at every position, our top-1 deviation is smaller than the reference's
> top-1-to-top-2 margin.

That holds at 16 of 16, tightest at 70 percent of the margin. A tolerance
number would have been arbitrary; the margin is measured, and a change that
pushes the deviation past a margin fails naming the position that did it.

---

## Phase 5

Chunked prefill and KV state. 24,083 assertions, 0 failed, sanitizers clean.

### The leak that only LeakSanitizer could see

Rewriting the forward pass around a `[n * chunk]` column stride meant the
context grew ten chunk scratch buffers and dropped the nine single-token ones.
`sllm_ctx_free` was updated to the new fields. `sllm_ctx_new` was not: it kept
allocating `attn`, `proj`, `ffn_gate_buf`, `ffn_up_buf` and `ffn_h` at their old
sizes. Nothing freed them, so every context leaked 103,424 bytes in five
allocations.

This is the mirror image of the dead-code defect in the same file. Both are the
cost of changing a struct by hand, and they fail in opposite directions -- one
frees a pointer nobody allocates, the other allocates a pointer nobody frees.
Only one of them is a crash.

The second part is the reason it survived a release run. The context is 157 MB
and the leak was 103 KB, so a nine-context loop still passed while RSS climbed.
Chasing it properly meant shrinking to a 20-line standalone program that does
create-one-free-one; the 103 KB was not findable in a 2 GB test run, because
the thing that leaks is not the thing that is large.

### The double free that had two different faces

`sllm_state_load()` read the last position's logits *after* its final
`fclose`. The header is small and read early, the cache data is most of the
file, so the logits read looked like a harmless tail -- except it was a read
from a freed `FILE *`.

That single misplaced line produced:

- under the test suite, `gguf-truncated (-12)`, twice, which reads like the
  state writer was at fault
- under a standalone probe, `attempting double-free` in `fclose`
- in the release build, a plain segfault with no diagnostic at all

Three symptoms, one line, and the two builds disagreed about what had happened.
Reading from a closed handle is undefined behaviour, so the file it "read" was
whatever happened to still be mapped -- sometimes short, sometimes complete
enough to reach the second `fclose` and corrupt the heap.

The lesson worth keeping is narrower than "check your fclose": the symptom that
looks most like a *data* problem is the one that is most likely to be a *lifetime*
problem. `gguf-truncated` named the GGUF reader. The state file was never
truncated -- the reader was reading freed memory.

### The test that was a different gate in each build

```c
int32_t direct[16];
int32_t resumed[16];
/* n_new == 6, so only 6 elements are ever written */
if (memcmp(direct, resumed, sizeof(direct)) != 0) {   /* 64 bytes */
```

`sizeof(direct)` is the whole array, not the part that was filled. The compare
covered 10 elements of uninitialised stack in both operands, so it reported
whether the compiler happened to reuse the same stack in both cases.

At `-O0` (the sanitizer build) it passed. At `-O2` (release) it failed, on a
continuation that printed as identical:

```
direct : 264 6864 3363 315 9822 374
resumed: 264 6864 3363 315 9822 374
```

The state restore was correct the entire time. The fixed form compares
`n_new * sizeof(int32_t)`, and the new state test passes at both levels.

The generalisation is uncomfortable and worth stating: a gate that reads
uninitialised memory is not a weaker gate, it is a *different gate per
compiler*, and the build you trust most is the one most likely to be lying. Both
builds were reporting a real memcmp result. Neither was reporting a real model
result. Equal printed values beside a `FAIL` is the tell -- and the response to
it is to distrust the test's arithmetic, not to go looking in the kernel for a
difference that is not there.

### A fix I could not prove, and did not claim

`sllm_ctx_reset()` cleared `n_past * kv_head_stride` bytes, but
`kv_head_stride` is *per layer*, so it zeroed layer 0 and left layers 1-29
holding the previous conversation. The fix multiplies by `n_layer`.

I wrote a regression test for it -- dirty the cache with a real prompt, reset,
require the next prefill to match a fresh context -- and then reverted the fix
to check the test actually failed. **It passed with the bug in place.**

That is the honest result, and it is worth more than the fix. `sllm_ctx_reset`
sets `n_past = 0`, and every prefill overwrites the whole live prefix, so stale
data is never in range. The memset is defence in depth against a future path
that reads the cache before writing it, and it is correct now only so that it
will still be correct on the day it matters.

So the test stays, renamed to what it actually locks -- the reuse contract,
which is worth protecting and was worth proving -- and both the test comment
and the `sllm_ctx_reset` comment say plainly that it does not cover the memset's
layer count. A test named for a bug it cannot fail is the same defect as the
uninitialised `memcmp`, one level up: it reports a guarantee nobody is getting.

### Gates

- 9 chunk sizes (1, 2, 3, 4, 5, 7, 16, 64, 256) and an uneven `2|8` split, all
  bit-identical to one token per forward
- a later chunk demonstrably attends to a key written by an earlier one
- save, load, and continue 6 tokens: identical to an uninterrupted run
- a reset context is bit-identical to a fresh one
- a state file with valid magic and a 2^20-layer geometry is rejected before
  it can size a 2 GiB read
- the Phase 4 prompt argmax and margin gates unchanged and still enforced
- `make check` green, ISA guard green, ASan/UBSan/LSan green

### Still not claimed

`tg128 @ 28 threads` remains unclaimed and Phase 5 did not change that. The
Phase 4 float floor is untouched and stays closed; nothing here is evidence for
reopening it. See `PHASE5-ACCEPTANCE.md`.

---
## What these have in common

Five of the kernel defects above — SiLU, NeoX RoPE, rms_norm, Q4_0, and the
I2_S layout — were invisible on the acceptance model and visible only against
the reference, a synthesised adversarial case, or a varying shape. Two
produced confident, well-formatted messages that pointed at the wrong code.

Phase 5 adds a second family. Its three defects were not wrong arithmetic; they
were wrong *about what was true*: a freed pointer, an allocation nobody owned, a
comparison over memory that was never written. The messages pointed at the GGUF
reader, at a 2 GB test run, and at a kernel that was correct throughout. In
every case the expensive move was to distrust the message -- to shrink the
reproduction, to run the same test at two optimisation levels, to revert the fix
and check that the test could still fail.

The tokenizer added a sharper version of the same lesson. Of the six Phase 3.5
defects, every one of them passed on ASCII text. Punctuation runs, digit
triples, contractions, double spaces, multi-byte characters and corrupted
UTF-8 are all invisible to a fixture made of English prose, and the project's
own earlier conclusion — that the golden vectors had "earned their keep" — was
only ever true because the fixture was built to attack the specific thing
rather than to be representative.

The defences that actually caught them were the ones that compare against the
reference element by element, and the ones that vary the shape. A gate built
only from the acceptance model's own parameters would have passed every one of
these.

That is the argument for the parity rule in `ARCHITECTURE.md`, and it is worth
restating when a new gate is added: *what would this fail on?*

The sharpest instance in this file is one written after the others. The
`sllm_ctx_reset` regression test was added to catch a real defect, the fix was
applied, and the test passed — so the obvious next step was to revert the fix
and see whether the test could still fail. It could not. A test that cannot fail
is not a gate, and the ones that took longest to notice are not the ones that
never worked. They are the ones that look like gates: a name, a `CHECK`, a
green line, and a claim about a guarantee that was never actually tested. The
only reliable question is the one asked before the test is trusted, and it has
to be asked out loud, because "the test passes" and "the test can fail" are
different sentences and the first one is much easier to read.
