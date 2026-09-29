# Reference tools

Everything in this directory is a **reference tool**. Nothing here is part of
the saphira-llm runtime, nothing here is installed, and nothing here ships.

These tools exist only to observe and characterise the upstream implementations
we are measured against, so that saphira-llm's correctness and performance are
judged against recorded evidence rather than against a claim.

## `sllm-logits-ref`

Links against the upstream BitNet-capable llama.cpp (the same pinned tree as
`third_party/BitNet/3rdparty/llama.cpp`) and dumps golden vectors for a prompt:

* every logit position as raw little-endian f32 into `<prefix>.f32`
* a `<prefix>.txt` manifest with the token ids, per-position logit statistics,
  the top-k tokens, an FNV-1a 64 hash of the raw logit bytes, timings, and a
  greedy continuation

It requires a C++ toolchain because the reference is C++. It is never compiled
into the runtime and the runtime has no equivalent.

## `sllm-tokenize-ref`

Links against the same pinned tree and dumps the reference's own tokenisation,
so `tests/golden/tokenizer.txt` records the reference rather than a belief about
it. One record per prompt: the text itself (escaped, so the fixture drives the
test), the token ids, and the reference's own rendering of each token.

The prompt set is an attack on the pre-tokeniser, not sample prose. The
acceptance model has no `tokenizer.ggml.pre` key, so the reference uses its
`DEFAULT` pre-type — four split passes, not the canonical single GPT-2 pattern
— and a fixture of ordinary English would pass against almost any
implementation. Punctuation, digits of every length, contractions, whitespace
runs and multi-byte UTF-8 are the cases that actually discriminate.

### Building

```sh
cmake -B build-base -S third_party/BitNet -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBITNET_X86_TL2=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_C_FLAGS="-march=x86-64-v3 -O2" \
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -O2"
cmake --build build-base --target llama-bench llama-cli -j
```

### Running

```sh
cc -O2 -o sllm-logits-ref sllm-logits-ref.c \
  -Ithird_party/BitNet/3rdparty/llama.cpp/include \
  -Ithird_party/BitNet/3rdparty/llama.cpp/ggml/include \
  -Lbuild-base/bin -lllama -lggml -lggml-base -lm -lpthread -ldl

cc -O2 -o sllm-tokenize-ref sllm-tokenize-ref.c \
  -Ithird_party/BitNet/3rdparty/llama.cpp/include \
  -Ithird_party/BitNet/3rdparty/llama.cpp/ggml/include \
  -Lbuild-base/bin -lllama -lggml -lggml-base -lm -lpthread -ldl

./sllm-tokenize-ref <model.gguf> tests/golden/tokenizer.txt

LD_LIBRARY_PATH=build-base/bin ./sllm-logits-ref \
  -m /var/lib/spoon/models/ggml-model-i2_s.gguf \
  -p "The capital of France is" \
  -o tests/golden/bitnet2b-capitol
```

### Reproducibility notes

* **Hold `-t` fixed.** ggml's reduction order depends on the thread count, so
  logits are only bit-identical at a fixed thread count. Fixtures record the
  thread count they were captured at.
* **Logit parity is a tolerance comparison, not bit equality.** saphira-llm uses
  its own reduction order, so a bit-exact logit match is neither expected nor
  the right gate. The gates are, in order of strength:
  1. greedy token sequence identical at temperature 0;
  2. logits agreeing within a tolerance derived from measurement;
  3. the FNV-1a hash recorded in the manifest for diagnosing *why* a run
     differs, not as a pass/fail criterion.
* `-DBITNET_X86_TL2=ON` does **not** link at the pinned revision: the
  `GGML_BITNET_X86_TL2` block of `src/ggml-bitnet-lut.cpp` omits
  `ggml_bitnet_mul_mat`. That is an upstream defect, and it is also why the
  option is not a performance question for us — see BENCHMARKS.md.
