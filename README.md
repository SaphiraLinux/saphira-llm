# saphira-llm

A native C11 inference runtime for BitNet 1.58-bit and conventional GGUF models,
targeting x86-64-v3 on Saphira Linux.

No Python. No launcher around someone else's binary. musl-friendly, no mandatory
BLAS, minimal dependencies.

```sh
make
./saphira-llm -m model.gguf -p "Hello"
```

> **Status.** Phases 0 to 5 are complete and sealed. Generation works: the
> binary loads a `bitnet-b1.58` GGUF, encodes a prompt, generates greedily and
> decodes the result. 23,757 assertions, sanitiser-clean, and the x86-64-v3
> baseline is proven mechanically rather than asserted. Phase 6 (decode
> throughput) is in progress. See ARCHITECTURE.md for the phase plan and gates.

## Getting a model

No model is shipped, and the runtime will not download one. You need a GGUF
whose architecture is `bitnet-b1.58`; anything else is refused with a specific
`unsupported-architecture` error rather than loaded and mis-executed.

The acceptance model is
[`microsoft/bitnet-b1.58-2B-4T-gguf`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf)
(MIT), converted from the upstream bf16 release for `bitnet.cpp`. It is about
1.2 GB. Download it wherever you like and pass it with `-m`; the path is yours
to choose and nothing in this repository assumes one.

```sh
./saphira-llm -m /path/to/ggml-model-i2_s.gguf -p "Hello"
```

The weights are I2_S ternary, about 1.58 bits per weight, and the tokenizer
(vocabulary 128,256, GPT-2 byte-level BPE) is embedded in the same GGUF file.
There is no separate vocabulary file to fetch.

## What this is and is not

It is a completion engine. There is no interactive mode, no chat template, and
no sampling beyond greedy argmax: no temperature, no top-p, no top-k, no seed.
A question is treated as text to continue rather than as an instruction to
follow, so a 2B base model given a question will often repeat it. That is the
model behaving as a base model, not a fault in the runtime; the token checksum
that `saphira-llm-tgbench` prints is what tells the two apart.

## What works today

* **GGUF v3 container**: mmap, metadata, tensor table, per-tensor extents.
  Every rejection has a specific status and a specific message.
* **ISA dispatch**: x86-64-v3 is the *baseline*, not a fallback tier. Above-v3
  kernels are placed in a guarded section and entered only when detection and
  the selected level both permit. `make check-isa` proves this mechanically.
* **First kernel and the dispatch pattern**: a signed-int8 dot product with a
  real v3 AVX2 path and a real AVX-VNNI path, required to agree exactly.
* **Threading**: a native pthread pool with no inter-worker barrier, a
  measurement-derived placement plan, and a thread count that never fills every
  hardware thread. No full-occupancy regression, where the reference loses
  about seventeen times.
* **Tensor layer**: dequantisation for F32, F16, BF16, Q8_0, Q4_0 and the
  BitNet I2_S ternary format, including the transposed tile layout and the
  scale that sits after the packed weights.
* **Vector kernels**: RMSNorm, RoPE (both layouts), softmax, SiLU, add, mul and
  get_rows, each with golden vectors.
* **23,757 assertions**, clean under `-Wall -Wextra -Wpedantic` and under
  ASan+UBSan, across all six build targets.

## Build and test

```sh
make            # release binary, -march=x86-64-v3
make test       # the test suite
make check      # tests plus the mechanical baseline proof
make san        # ASan + UBSan (clang; the Saphira gcc has no libasan)
make bench      # the scheduler benchmark
make clean
```

`make tgbench`, `make kernelbench` and `make topoprobe` build the three
measurement tools. `./testsuite.sh` does all of it in one go and then puts the
model through six prompts, printing tokens per second for each:

```sh
./testsuite.sh                          # build, test, then Q&A
./testsuite.sh --tests-only             # gates only, no model needed
MODEL=/path/to.gguf THREADS=8 ./testsuite.sh
```

`make check-isa` disassembles the binaries and fails if any instruction above
the v3 baseline appears outside `.sllm.isa.ext`. It is verified to fail on a
deliberately mis-built binary, because a safety check that cannot fail is
worse than no check.

## Design commitments

* **Model format is separate from kernels.** The layer chain is
  `GGUF -> tensor description -> arch adapter -> op executor -> kernel dispatch`.
  Adding a quantisation format means adding a row to a table, not an engine.
* **Unsupported means specific.** "Q5_K is a valid GGUF type we have no kernel
  for" and "this file is truncated" and "no adapter for architecture X" are
  three different errors with three different messages.
* **The v3 path is optimised, not generic.** Above-v3 dispatch only ever adds.
* **No benchmark estimate is ever reported as a performance claim.** Numbers go
  in BENCHMARKS.md with the method attached, or they do not go in.

## Scope: inference only

This repository performs inference. It does not train, fine-tune, LoRA-adapt or
distil a model, and it contains no optimiser, no gradient and no loss function.
Conversion is not implemented here either: `quant.c` consumes quantisation
formats (F32, F16, BF16, Q8_0, Q4_0 and the BitNet I2_S ternary format), it
never produces them. The BitNet and llama.cpp trees under `third_party/` are
pinned read-only references consulted for their algorithms and formats, as
PROVENANCE.md records; they are not build inputs, and the Makefile never
compiles them. If you want to train this architecture, use the upstream
microsoft/BitNet tooling.

## Documentation

| File | Contents |
| --- | --- |
| ARCHITECTURE.md | the design, the phase plan, and each phase's gate |
| FORMAT.md | the GGUF and I2_S layouts, verified against the reference |
| BENCHMARKS.md | measured reference baselines and the optimisation targets |
| PROVENANCE.md | every upstream revision and licence consulted |
| docs/CUDA-FEASIBILITY.md | why there is no Saphira CUDA path |
| docs/EVIDENCE.md | every defect found, and why the gate missed it |
| docs/PHASE4-CONTRACT.md | the forward graph as read from the pinned source |
| docs/PHASE4-GAP.md | the logit gap: what it was, what is fixed, and its floor |
| docs/PHASE4-ACCEPTANCE.md | the sealed Phase 4 boundary, and what would reopen it |
| tools/reference/README.md | the reference-only golden-vector tooling |
| tests/golden/README.md | the fixtures and the rules for regenerating them |

## Licence

MIT. See LICENSE and NOTICE.
