# saphira-llm

A native C11 inference runtime for BitNet 1.58-bit and conventional GGUF models,
targeting x86-64-v3 on Saphira Linux.

No Python. No launcher around someone else's binary. musl-friendly, no mandatory
BLAS, minimal dependencies.

```sh
make
./saphira-llm -m model.gguf -p "Hello"
```

> **Status.** Phases 0 and 1 are complete. The GGUF container layer and the ISA
> dispatch layer are real, tested and sanitiser-clean. Model execution does not
> exist yet — the binary parses a model and says so rather than pretending to
> generate text. See ARCHITECTURE.md for the phase plan and gates.

## What works today

* **GGUF v3 container**: mmap, metadata, tensor table, per-tensor extents.
  Every rejection has a specific status and a specific message.
* **ISA dispatch**: x86-64-v3 is the *baseline*, not a fallback tier. Above-v3
  kernels are placed in a guarded section and entered only when detection and
  the selected level both permit. `make check-isa` proves this mechanically.
* **First kernel and the dispatch pattern**: a signed-int8 dot product with a
  real v3 AVX2 path and a real AVX-VNNI path, required to agree exactly.
* **234 assertions**, clean under `-Wall -Wextra -Wpedantic` and under
  ASan+UBSan.

## Build and test

```sh
make            # release binary, -march=x86-64-v3
make test       # the test suite
make check      # tests plus the mechanical baseline proof
make san        # ASan + UBSan (clang; the Saphira gcc has no libasan)
make clean
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

## Documentation

| File | Contents |
| --- | --- |
| ARCHITECTURE.md | the design, the phase plan, and each phase's gate |
| FORMAT.md | the GGUF and I2_S layouts, verified against the reference |
| BENCHMARKS.md | measured reference baselines and the optimisation targets |
| PROVENANCE.md | every upstream revision and licence consulted |
| docs/CUDA-FEASIBILITY.md | why there is no Saphira CUDA path |
| tools/reference/README.md | the reference-only golden-vector tooling |
| tests/golden/README.md | the fixtures and the rules for regenerating them |

## Licence

MIT. See LICENSE and NOTICE.
