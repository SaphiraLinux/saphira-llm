# Provenance

Every upstream project consulted as an engineering reference, with the exact
revision inspected and its licence. Revisions were resolved with `git ls-remote`
on 2026-09-28 and are pinned in `third_party/` as submodules so that the
inspected revision is provable rather than asserted.

No third-party source has been copied into this distribution. Algorithms are
independently reimplemented in C from documented reference behaviour.

## Pinned references

| Project | Repository | Revision | Licence | Why consulted |
| --- | --- | --- | --- | --- |
| BitNet | `https://github.com/microsoft/BitNet` | `0b341e582afbf9e1011f24744b554c96a3477eb5` | MIT | Canonical specification of the b1.58 ternary representation, the BLAST ternary kernels, the T-MAC/LUT kernels, the `bitnet-b1.58` graph, and the GGUF conversion layout. |
| llama.cpp | `https://github.com/ggml-org/llama.cpp` | `680a036285273a3ff56032ec5d7f3352609eba4f` | MIT | Reference for GGUF parsing, quantisation block formats, RoPE, KV cache, flash attention, chunked prefill, and overall CPU kernel practice. |
| BitNet llama.cpp fork | `https://github.com/isHuangXin/llama.cpp` | `390c307752ab78fd8189f359d6954c9ba1be74af` (branch `release-bitnet-embedding-0.6b-270m`) | MIT | The actual BitNet-capable llama.cpp used by our `/recipes/bitnet-cpp` package. This is the fork that defines the `I2_S` tensor type and its execution path. |
| vLLM | `https://github.com/vllm-project/vllm` | `d28795f1a7af4e3ce2530d4f0bdaec4ecbede693` | Apache-2.0 | Scheduling, KV cache paging and continuous-batching ideas only. No code is taken and no Python is introduced. |
| Ollama | `https://github.com/ollama/ollama` | `05a7a91dae22d7eebd490fca9cf57a074518706b` | MIT | Model/runtime behaviour and compatibility ideas only. No product features are adopted. |

## Packaged reference used as the parity and benchmark baseline

The C++ binary that `/recipes/bitnet-cpp` installs on Egg is our parity oracle
and performance baseline. It is built from the vendored tarball, not from
upstream `ggml-org/llama.cpp`, so that "upstream BitNet" means the same thing
for our engine and for the reference.

| Item | Value |
| --- | --- |
| Tarball | `/recipes/bitnet-cpp/files/bitnet-20260830.tar.gz` |
| SHA-256 | `5b196840a75c81209c47b370387f07268bf439e4ef1b194f20c5d44b321bfde3` (verified 2026-09-28) |
| Package | `bitnet-cpp` `20260830-4`, repo `saphira` |
| Upstream root | `microsoft/BitNet` @ `0b341e5`, consistent with the revision above |
| Vendored fork | `isHuangXin/llama.cpp` @ `390c3077` at `3rdparty/llama.cpp` |
| Build flags | `-march=x86-64-v3 -O2 -DBITNET_X86_TL2=OFF`, Release, Ninja |
| Installed artefacts | `/usr/bin/llama-cli`, `/usr/bin/llama-server`, `/usr/bin/llama-quantize` |
| Runtime | musl, dynamically linked; `libstdc++`, `libgcc_s`, `libgomp` |

Note on the baseline build: `-DBITNET_X86_TL2=OFF` leaves both T-MAC paths
(`GGML_BITNET_ARM_TL1` and `GGML_BITNET_X86_TL2`) compiled out, so the live x86
execution path in that package is the AVX2-only BLAST ternary kernel. Phase 0
rebuilds the same tarball with `BITNET_X86_TL2=ON` to compare the two kernel
families on this CPU, and records the verdict in `BENCHMARKS.md`.

## Compatibility determination

All pinned references are MIT or Apache-2.0, which are compatible with the MIT
terms of this project provided attribution and licence text are retained. There
is no copyleft obligation. saphira-llm is MIT rather than the house BUSL-1.1
default so that kernel improvements can be offered back to ggml and BitNet
upstream without a relicensing conflict. All three checked-out trees carry an
MIT `LICENSE` file, confirmed 2026-09-28.

`third_party/llama.cpp` is `v0.5.0-94-g680a03628`.

## Verified reference chain

The parity oracle and benchmark baseline is not merely documented, it is
reproducible from the tree in this repository. The chain was closed and
verified on 2026-09-28:

1. `microsoft/BitNet` @ `0b341e5` pins its own `3rdparty/llama.cpp` submodule to
   `390c307752ab78fd8189f359d6954c9ba1be74af` — confirmed with
   `git -C third_party/BitNet submodule status`. This is the same revision that
   `/recipes/bitnet-cpp` records as vendored, independently confirmed.
2. The BitNet repository therefore carries the BitNet-capable llama.cpp fork as
   `third_party/BitNet/3rdparty/llama.cpp`; a separate clone of the fork is not
   needed and is not kept.
3. The files in `/recipes/bitnet-cpp/files/bitnet-20260830.tar.gz` are
   byte-identical to the checked-out tree. SHA-256 prefixes compared for
   `ggml/src/ggml-bitnet-compute.c`, `src/models/bitnet.cpp`, `src/llama-arch.h`,
   `ggml/src/ggml-quants.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/vec.cpp`,
   `src/llama-graph.cpp`, `src/llama-model.cpp`,
   `tools/llama-bench/llama-bench.cpp`, `src/ggml-bitnet-mad.cpp`,
   `src/ggml-bitnet-lut.cpp` and `include/ggml-bitnet.h` all match.
4. The two Saphira patches applied by the recipe touch only
   `3rdparty/llama.cpp/common/build-info.cpp.in`, `run_inference.py` and
   `run_inference_server.py`. They modify a build banner and the Python helper
   entry points, and no kernel, quantisation or graph source. The numerics of
   the installed reference binary are therefore pure upstream.

Consequence: "upstream BitNet" is unambiguous and rebuildable, so any numerical
disagreement between saphira-llm and the reference is attributable to saphira-llm
and not to an unrecorded local patch.

## Acceptance models

| Model | Path | Role |
| --- | --- | --- |
| `bitnet2b` (BitNet b1.58 2B-4T, GGUF v3, arch `bitnet-b1.58`, tokenizer `gpt2`) | `/var/lib/spoon/models/ggml-model-i2_s.gguf` | BitNet parity and performance acceptance model. |
| `falcon3-7b` I2_S | `/var/lib/spoon/models/falcon3-7b/ggml-model-i2_s.gguf` | `I2_S` with dense non-ternary scales, to check the same kernel generalises. |
| `falcon3-10b` I2_S | `/var/lib/spoon/models/falcon3-10b/ggml-model-i2_s.gguf` | As above, larger. |
| `Qwen3-8B-Q4_K_M` | `/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf` | Conventional-model acceptance model for the dense GGUF path. |
