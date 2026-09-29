# Phase 6 baseline and measured optimisation

Phase 6 is **not** complete. This file records what was measured, what was
changed, what the changes were worth, and what remains open. The acceptance
target in the brief is not met; the reason is stated below with the measurement
that supports it.

Three of the brief's premises did not match the repository or the machine. They
are recorded in "Premises that did not survive contact with the repo" rather
than quietly reinterpreted, because two of them change what "acceptance" means.

## Machine and build

| | |
| --- | --- |
| CPU | Intel Core i9-13900K, 14 cores / 28 threads, hybrid (8 P + 6 E) |
| ISA in use | `selected=vnni ceiling=vnni avx avx2 fma f16c avx_vnni` |
| AVX-512 | **absent** — Raptor Lake consumer silicon has it disabled |
| Baseline ISA | x86-64-v3, ISA guard green, 130 instructions in `.sllm.isa.ext` |
| Model | `ggml-model-i2_s.gguf`, bitnet-b1.58, 1,179,449,920 bytes of payload |
| Prompt | `"The capital of France is"`, 6 tokens |
| Timing | generation only, prefill excluded, warm page cache, monotonic clock |
| Estimator | best of N, minimum reported |
| Build | `-O2 -std=c11 -Wall -Wextra -Wpedantic -march=x86-64-v3` |

**The host is shared.** Three other agent sessions and a long-running `python3`
were resident throughout; measured load average ranged 4.7 to 13.6. Load is
recorded per row in `saphira-llm-tgbench` output for that reason. It is the
most likely reason the 28-thread row moved between runs here when it did not
move on the quieter machine the reference table in `BENCHMARKS.md` was taken
on. Every number below is best-of-N, because under external interference the
minimum is the closest estimate to the unloaded machine.

## Baseline, commit a7fd145

The thread curve was **flat**. Not approximately flat — 2.85 t/s at one thread
and 2.86 at fourteen, because nothing in the forward pass used the thread pool
that `main.c` had been constructing all along.

| threads | t/s | GB/s |
| ---: | ---: | ---: |
| 1 | 2.85 | 3.36 |
| 2 | 2.89 | 3.40 |
| 4 | 2.88 | 3.40 |
| 8 | 2.86 | 3.37 |
| 14 | 2.86 | 3.37 |
| 20 | 2.81 | 3.31 |
| 24 | 2.83 | 3.33 |
| 28 | 2.85 | 3.37 |

`src/main.c` had been reporting `threads 1` in its benchmark line regardless of
`-t`, with a comment explaining why. That was correct and is the reason this
was findable: the code refused to report a thread count the forward pass
ignored.

## Profiler evidence

`perf` is not installed, there is no root, and valgrind and gdb are absent
(`apk add linux-tools` fails on the log; `su` is unavailable). gprof is present
but requires `-pg` recompilation and then misattributes exactly the code this
phase is about, since inlined AVX2/VNNI intrinsics fold into their enclosing
function. So `bench/prof.c` is a sampling profiler: SIGPROF on ITIMER_PROF,
program counters captured in the handler, resolved offline.

Two details in it are worth recording because both were wrong first. The saved
program counter is found by locating the register that falls inside the
program's own text, not by a hardcoded index: index 16 is RIP under
`-std=gnu11` and is not under the project's `-std=c11`, which silently produced
kernel addresses for every sample. And the load bias is read from
`/proc/self/maps`, not `dl_iterate_phdr`, because `dlpi_addr` for a non-PIE
main program here is not the load bias.

Sampling the generation region only, 2,234 samples, 1 thread:

| share | location | what it is |
| ---: | --- | --- |
| **86.3%** | `src/forward.c:643` | tied output projection, one line |
| 12.1% | `dot_v3` | AVX2 ternary dot, 190 bytes |
| 1.4% | `src/forward.c:116-120` | `f16_to_f32`, a software converter |
| <1% | quant_act, rms_norm, silu, mul | everything else |

The mechanism is the point. Line 643 was

```c
for (int32_t d = 0; d < n_embd; ++d) { acc += f16_to_f32(row[d]) * xn[d]; }
```

and `f16_to_f32` unpacked sign, exponent and mantissa by hand and spun in a
`while` loop for subnormals. It ran 328 million times per token, over 128256
vocabulary rows by 2560 dimensions. The CPU has F16C. The project already used
`_mm256_cvtph_ps` in `quant.c`.

**The brief's stated primary kernel work was the AVX-VNNI ternary GEMM. It was
12 percent of the time, and it was not running at all** — see below.

## Changes, in commit order

Each is independently revertible and each was gated before commit.

### 1. `944c248` — F16C output projection

ggml's own kernel, not an invention: `GGML_F16_VEC` on x86 is eight floats
loaded with `_mm256_cvtph_ps`, and `ggml_vec_dot_f16_unroll` runs
`GGML_F16_STEP` (32) elements across `GGML_F16_ARR` (4) accumulators reduced as
a binary tree. Those constants and that reduction order are the reference's,
deliberately: `PHASE4-CONTRACT.md` already records that our `lm_head` reduction
order differs from upstream's and is governed by the margin gate rather than
bit equality, so matching upstream's shape is the only direction that can help.
The arithmetic is unchanged — F16 weight times F32 activation accumulated in
F32. Only the order of the additions moves.

One operand-order bug of my own: `_mm256_fmadd_ps(acc, w, a)` is `acc*w + a`,
not `acc + w*a`, which put logits at 83 instead of 21. The parity gate caught it
in one run.

### 2. `d96f361` — AVX-VNNI actually running, and not slower than AVX2

Two defects. `sllm_i2s_select_isa` was called from tests and from nothing else,
so the global stayed NULL and `sllm_i2s_dot` fell through to AVX2 while the
binary logged `selected=vnni`. A correct kernel, tested against the reference on
real tensors, dead in production for want of a call site. `sllm_model_load` now
selects, before any worker exists.

Second, `dot_vnni` chained four `dpbusd` per block into one accumulator. With a
multi-cycle latency that makes every block wait on the previous one's last add,
while the AVX2 path's four `maddubs` are independent. Isolated on real weights:

| n | v3 | vnni, 1 acc | vnni, 4 acc | 4 acc vs v3 |
| ---: | ---: | ---: | ---: | ---: |
| 2560 | 21.18 ns | 34.76 ns | **16.19 ns** | 1.31× |
| 6912 | 57.49 ns | 127.60 ns | **45.54 ns** | 1.26× |

The gap widened with `n`, which is the signature of a dependency chain rather
than a throughput limit, and that is what pointed at the accumulator count
instead of at the instruction choice. Integer addition is associative, so this
changes no result; the Phase 3 gate against the reference's own hash confirms
it.

### 3. `cfd7a4a` — threading, and the 28-thread collapse

Every parallel region is a loop over independent rows: the seven I2_S
projections per layer, and the output projection over (token, vocab) flattened
into one region. Each row's reduction stays inside one thread, so no partial
sums are ever combined across threads and **the thread count cannot change the
result**. This is a constraint, not an accident. The benchmark prints a
per-row token checksum; every row agrees at every thread count tested.

Threading immediately reproduced the 28-thread collapse — this time in our own
code. 6.25 t/s at 28 against 21.63 at 4, monotonically worse with every extra
thread, 33 percent spread. Monotonic degradation is the opposite of what a
bandwidth limit does.

The cause was a rendezvous. `sllm_parallel_for` does not wait for workers
before starting, but it does wait for all of them to *arrive* under the mutex.
Its comment claimed this "cannot starve at full occupancy" because
`pthread_cond_wait` does not burn a CPU. That is backwards: sleeping avoids
burning a CPU but means a woken worker must be scheduled before it can arrive,
and on a machine where every logical CPU is busy nothing is scheduled until
something yields. The Phase 2 microbenchmarks never saw it because their regions
are few and large; decode posts **211 regions per token**, each tens of
microseconds long, which is about 5,900 wake-and-arrive round trips per token
at 28 threads.

Fixes: workers spin for the next region before sleeping (bounded, falling back
to the condition variable), and a region is only distributed if each worker
would get at least 512 rows. The output projection, at 128256 rows, is always
distributed, so the machine is still fully used for the largest part of the
token.

## Before and after

Same harness, same model, same prompt, same token count, best of 4.

| threads | baseline t/s | now t/s | speedup | now GB/s | % of 4-thread peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.85 | 9.71 | 3.41× | 11.46 | 45% |
| 4 | 2.88 | **21.46** | **7.45×** | 25.31 | 100% |
| 8 | 2.86 | 17.88 | 6.25× | 21.09 | 83% |
| 14 | 2.86 | 14.95 | 5.23× | 17.64 | 70% |
| 20 | 2.81 | 14.39 | 5.12× | 16.97 | 67% |
| 24 | 2.83 | 14.40 | 5.09× | 16.99 | 67% |
| 28 | 2.85 | 13.39 | 4.70× | 15.80 | 62% |

An earlier run at lower load reached 23.98 t/s at 4 threads, so treat 21.5 to
24.0 as the peak band on this host.

Against the reference on the same machine class: 11.05 t/s at 1 thread and
21.44 at 14. Single-threaded we are at 88 percent of the reference. At peak we
match the reference's 14-thread figure on four threads. The reference's best
ever measured is 28.45 t/s at 22 threads.

## The 28-thread row, which was the deliverable

The reference loses about 17× between 24 and 28 threads, falling to 1.42 t/s
against 28.45 at its peak — 5 percent of peak. The mechanism is a spinning
barrier: once every logical CPU is running a spinner, the thread that has to
make progress cannot be scheduled.

This implementation now reaches **62 percent of its peak at 28 threads** on the
same CPU count, and the decline from 4 to 28 threads is gradual rather than a
cliff. The remaining decline is consistent with three things that all point the
same way and none of which is a scheduler pathology:

- 4 threads already reach 25.3 GB/s. The output projection alone is 0.66 GB of
  the 1.18 GB streamed per token, and both hot kernels are memory-bound, so
  extra threads add coordination cost without adding bandwidth.
- Past 8 threads the pool is placing work on E-cores and then on SMT siblings,
  which share execution and cache with a thread already resident.
- This host is shared, and a 28-thread pool against eleven other runnable
  threads is oversubscribing on purpose.

The correct conclusion is that the pathology is **eliminated and the residual is
a limit**, not that 28 threads is now the recommended setting. It is not: the
peak is 4 to 8 threads. Selecting 14 threads as "recommended" would be the paper-
over the brief warns against, and it is not done. `sllm_topology_recommended_threads`
still returns 27 of 28, and that number is now demonstrably wrong for this
workload — see open items.

## Per-kernel correctness

| new path | test | result |
| --- | --- | --- |
| `sllm_dot_f16_f32` F16C | 33 lengths including every tail residue mod 32 | within reduction bound |
| | 12 half-precision edge values (subnormal, ±0, ±inf, max) | agree with the portable converter |
| | unaligned vocabulary rows | agree |
| `dot_vnni` 4 accumulators | Phase 3 hash against the reference on real tensors | bit-identical |
| `sllm_i2s_select_auto` | model load installs the best kernel | VNNI on this CPU, v3 asserted where absent |
| threaded forward | per-row token checksum at every thread count | identical |
| scheduler spin | Phase 2 full-occupancy gate | still passes; sync gain 6.9× → 26.1× |

The F16C path **cannot** be bit-identical to the scalar one — reassociating a sum
changes rounding — so its gate is a reduction error bound plus an argmax-stability
check, not equality. One of those tests initially failed on `-0` versus `+0`
through `memcmp`; IEEE round-to-nearest says `+0 + -0` is `+0`, so the kernel was
right and the test was comparing floats bitwise. That is the same mistake as the
Phase 5 uninitialised `memcmp`, one level down, and it is recorded because the
second time is the one that makes it a habit.

## What is not claimed

- **The acceptance target is not met.** The brief asks for more than 26.5 t/s at
  14 threads. Measured here: 14.95 t/s at 14 threads, 21.46 t/s at 4 threads, on
  a loaded host. Even the best measurement did not clear 26.5 at 14 threads.
  26.5 t/s would require 31.3 GB/s sustained, and the reference's own best ever
  measured is 28.45 t/s (33.6 GB/s) at 22 threads on an unloaded machine, so the
  target sits above the reference's best configuration. It may be reachable here
  with a quieter machine and further work; it is not demonstrated.
- **AVX-512 is not implemented and not deferred on judgement.** This CPU has no
  AVX-512, so no AVX-512 kernel could be tested, and the brief requires every
  new SIMD path to be separately testable. Building an untestable path would
  violate that. It needs hardware that has it.
- **The `tg128 @ 28 threads` claim is now ours to make, and is not made** beyond
  the table above.
- **Topology classification is now deterministic, and on this host it reports
  "unavailable".** Classification is read from CPUID leaf 0x1A, `EAX[31:24]`,
  pinned to each logical CPU. The hypervisor zeroes it here and reports a
  non-GenuineIntel vendor string, so core classes are `UNKNOWN` and the counts
  are zero. Nothing is inferred to fill the gap. See "Timing cannot classify
  cores" below.

## Timing cannot classify cores

The Phase 2 calibration ranked cores by timing and thresholded at 0.75. On this
host it answered 14/0, 13/1, 12/2 and 11/3 performance/efficiency cores on four
consecutive runs of an idle machine, and put 27 of 28 logical CPUs in the
full-rate class at every threshold from 0.60 to 0.95.

The cause is virtualisation, and it is not a tuning problem. Guest
`sched_setaffinity` pins a vCPU to a guest logical CPU, but the host remains
free to run that vCPU's thread on any host core. A single logical CPU's samples
swing by a factor of two between rounds, at random, uncorrelated with round or
CPU — `~5.08e8` and `~1.05e9` measured. Best-of-three therefore samples the
fast mode almost every time. Guest-side timing measures host scheduling, not
the guest CPU's capability.

Threshold tuning was explicitly not attempted, because no threshold can
separate a signal that is not present.

### What replaced it

`CPUID.7.0:EBX[15]` reports whether the processor advertises a hybrid
architecture; here it returns 0. `CPUID.1A` `EAX[31:24]` is the native core-type
field, read pinned to each logical CPU; here it returns 0 on all 28. Intel
documents 0x40 for a performance core and 0x20 for an efficiency core, and those
values are stored and named — but a value we do not recognise is recorded raw
and left `UNKNOWN` rather than forced into a class.

sysfs is used to cross-check, never to derive. Two logical CPUs that sysfs says
share a physical core must report the same core type; if they disagree the two
sources contradict each other and neither is trusted. `cluster_id` is not used:
on this host it merely mirrors `2 * core_id` and carries no class information.

`sllm_topology_calibrate` is retained but narrowed to measurement: it fills
`score` and sets `calibrated`, and can no longer set any core class. A test
asserts that.

Verified: detection is byte-identical across repeated runs and under
concurrent load, and contains no timing call. `sllm_topology_describe` reports
class availability explicitly, because a bare "hybrid: no" is ambiguous between
a homogeneous machine and one that would not say.

### Egg's topology is synthetic, and does not describe homer

The physical host i9-13900K has **8 P cores x 2 threads = 16 logical, plus
16 E cores x 1 thread = 16 logical: 24 physical cores, 32 logical CPUs.**

Egg is not that machine. Its libvirt domain is declared as:

```xml
<cpu mode="host-passthrough" check="none" migratable="on">
  <topology sockets="1" dies="1" clusters="1" cores="14" threads="2"/>
</cpu>
<vcpu placement="static">28</vcpu>
```

The guest is therefore given 14 cores with 2 threads each, 28 vCPUs, SMT on
every presented core. That is a synthesised topology: it matches neither the
physical 24 cores / 32 logical CPUs nor any real hybrid arrangement, and it
puts SMT siblings on cores the silicon has no siblings on.

Two consequences follow, and they are why the CPUID route was necessary rather
than merely preferable.

First, the structural cross-check cannot confirm or deny the expected 8 P / 16 E
split, because the sibling pattern it would read is the VM's rather than the
CPU's. An inference from it would describe the hypervisor's configuration, not
the processor.

Second, timing cannot recover what the topology declines to give. Under
`host-passthrough` with no domain CPU pinning, the host scheduler places each
vCPU's thread wherever it likes, which is the ~2x random per-round swing
measured above.

The sysfs figures detection reports on Egg (14 cores, 28 logical, SMT on all)
are exactly what the domain declares, and are correct as a description of the
guest. They are not a description of homer's silicon and must not be quoted as
one.

### Reading the core-class fields

`hybrid` is false on Egg, but that does **not** mean the CPU is homogeneous: it
means core type was unavailable, and the fields were zeroed rather than left
stale. `n_perf`, `n_eff` and every `klass` are zero/UNKNOWN for the same reason.
The distinction is the whole point -- "this machine has no efficiency cores"
would justify spreading work across every core, whereas "we were not told"
justifies nothing. Call `sllm_topology_classes_known()` before consulting any of
them; it is false on any host where CPUID leaf 0x1A is hidden or zeroed.

## Open items

## Open items

1. ~~Calibration misclassification~~ — **fixed**: classification is deterministic
   from CPUID, and unavailable rather than wrong on this host. Placement
   ordering still uses measured `score`, which is a throughput observation and
   not a topology fact.
2. `sllm_topology_recommended_threads` returns 27, which is wrong for this
   workload; the peak is 4 to 8. It should be derived from measurement rather
   than from "leave one hardware thread free", which was a rule derived from the
   reference's barrier, not from this pool.
3. Single-thread kernel efficiency is at 88 percent of the reference; the gap is
   the residual float ordering, not instruction selection.
4. Peak is 4 to 8 threads, not 14. Whether the 211-regions-per-token structure
   can be coarsened — rather than skipping small regions — is unexplored.
5. Re-run every table on a quiet host before any of this is treated as a
   project record.
