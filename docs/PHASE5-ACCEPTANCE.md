# Phase 5 acceptance

Phase 5 is **complete**. This file records what was built, what the gates
actually prove, and the two things that were deliberately left outside the
boundary.

24,083 assertions, 0 failed, sanitizers clean, ISA gate clean.

## What was built

**1. Chunked prefill over one code path.**
`sllm_forward_chunk()` evaluates any `[base, base+n)` window, and
`sllm_forward_prefill_chunked()` walks the prompt in `SLLM_MAX_CHUNK` (256)
slices. `sllm_forward()` is `n = 1` and `sllm_forward_prefill()` is one chunk,
so the single-token path that Phase 4 was gated on is not a special case
anymore -- it is the same code with `n = 1`. There is no second implementation
to keep in step.

**2. Chunk boundaries are unobservable.**
The reference masks causally on *absolute* positions and fills `-INFINITY`,
skipping keys where `p0 > p1`; the diagonal is included. A window that starts
at `base` must therefore still see every earlier key, which is what makes
chunking a question about the mask rather than about the cache. Verified at 9
chunk sizes (1, 2, 3, 4, 5, 7, 16, 64, 256) and at an uneven `2|8` split, all
**bit-identical** to one token per forward, plus an explicit check that a
later chunk attends to a key written by an earlier one.

**3. KV state saves and restores.**
`sllm_state_save()` / `sllm_state_load()` carry the live cache prefix and the
last position's logits, so a restored context continues a conversation instead
of restarting it. Geometry is checked against the model before a byte is
trusted: a file claiming 2^20 layers is rejected rather than used to size a
2 GiB read.

The logits travel with the cache deliberately. A cache alone does not say which
token was current when it was saved, and a context restored from one would have
to re-run the prompt to find out -- which would make the round-trip test pass
even with an empty file, which is the failure the test exists to catch.

## The gates

- Prompt argmax and the per-position margin gate from Phase 4 are unchanged and
  still enforced: the sealed boundary did not move.
- `make check` is green, including the ISA guard (130 instructions in
  `.sllm.isa.ext`, 0 above baseline outside it).
- `CC=clang make san` is green under ASan, UBSan and LeakSanitizer.
- Save, load, and continue 6 tokens: identical to an uninterrupted run.
- A reset context is bit-identical to a fresh one.
- 9 chunk sizes bit-identical to one token per forward.

## Deliberately out of scope

- **`tg128 @ 28 threads` is still not claimed.** Phase 4 measured
  single-threaded generation only, and nothing in Phase 5 changes that. The
  claim waits for Phase 6, where threading is actually built and measured.
- **The Phase 4 float floor is untouched.** Nothing in Phase 5 moved it, and per
  the Phase 4 reopening condition it stays closed unless a semantic failure is
  attributable to it. No new evidence argues for reopening it.
- **No tinyBLAS port.** The state format is our own (`SLKV`), not llama.cpp's.

## Three defects this phase found

Recorded in full in `EVIDENCE.md`; the shape of them is the point.

- The chunk refactor deleted the pre-chunk scratch fields from `sllm_ctx_free`
  but left their allocations in `sllm_ctx_new`, so every context leaked 103 KB.
- `sllm_state_load()` closed the file and then read the logits from the closed
  handle. Depending on the build that surfaced as a spurious "truncated state
  file" or as a double free in `fclose` and a segfault in release.
- The round-trip test compared `sizeof(direct)` -- 64 bytes of a 16-slot array
  when only 6 elements were ever written -- so it `memcmp`'d uninitialised
  stack. It passed at `-O0` and failed at `-O2`.

That last one is the reason the sanitizer build and the release build disagreed
about a passing test. A gate that reads uninitialised memory is not a weaker
gate, it is a different gate per compiler, and the one you trust most is the one
most likely to be wrong.
