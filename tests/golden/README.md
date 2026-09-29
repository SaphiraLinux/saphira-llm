# Golden vectors

Fixtures captured from the **upstream** BitNet-capable llama.cpp — never from
saphira-llm. They are the oracle that the saphira-llm parity gate is judged
against. Regenerating them from saphira-llm would destroy their purpose.

| File | Contents |
| --- | --- |
| `*.f32` | every logit position, raw little-endian `float`, `n_tokens × n_vocab` |
| `*.txt` | manifest: token ids, per-position logit statistics, top-k, FNV-1a 64 hash, timings, greedy continuation |
| `i2s-reference.txt` | reference dequantisation of four real tensors, and the reference's own GEMV/GEMM output on real weights against a fixed activation |
| `tokenizer.txt` | the reference's token ids for 80 prompts, with each prompt's text and the reference's own rendering of every token |

## Current fixtures

| Fixture | Prompt | Tokens |
| --- | --- | ---: |
| `bitnet2b-capitol` | `The capital of France is` | 6 |
| `bitnet2b-save` | `The name of the capital city of France is` | 10 |
| `i2s-reference` | four real I2_S tensors, dequantised and multiplied | ~46M elements |
| `tokenizer` | 80 prompts attacking the pre-tokeniser | 4 to 26 each |

Model: `/var/lib/spoon/models/ggml-model-i2_s.gguf`
(`bitnet-b1.58 2B Q1_0`, 128256-token gpt2 BPE vocabulary, BOS 128000).

## Rules for anyone adding or regenerating a fixture

1. **Capture with a reference tool**, built against
   `third_party/BitNet/3rdparty/llama.cpp`:
   `sllm-logits-ref` for logits, `sllm-tokenize-ref` for tokens,
   `capture_i2s_golden` for the I2_S records. Never capture from a saphira-llm
   binary.
2. **Hold the thread count fixed** and use `-t 1`. ggml's reduction order
   depends on the thread count, so logits are only bit-reproducible at a fixed
   thread count. The thread count is recorded in each manifest.
3. **Use the raw prompt.** Do not apply a chat template. `llama-cli` runs in
   conversation mode and templates its input, so its output is a different
   input problem and cannot be compared against these fixtures.
4. **Keep `-st --no-warmup` in mind for any CLI comparison.** The interactive
   session otherwise carries state that changes the result.
5. **Verify reproducibility** by re-capturing into a scratch path and diffing
   the `.f32` bytes. Both existing fixtures were reproduced byte-identically
   twice from scratch before being committed.
6. **Record the hash in the commit message.** It is the cheapest way to detect
   accidental fixture churn.
7. **Attack the behaviour, do not illustrate it.** Every fixture in this
   directory exists because a plausible-looking implementation got something
   wrong that ordinary input would not reveal. The I2_S layout, the
   `byte_to_cpt` encoding, whitespace classification and BPE staleness were all
   correct on ASCII prose and wrong on specific inputs. A fixture that only
   exercises the happy path is a fixture that cannot fail, and a gate that
   cannot fail is worse than no gate.
8. **Store the input alongside the output.** `tokenizer.txt` carries each
   prompt's text, so the test is driven by the fixture. A test that carries its
   own second copy of the inputs is a second thing to keep in sync, and it is
   the copy that drifts.

## Regenerating

```sh
cc -O2 -o /tmp/sllm-logits-ref tools/reference/sllm-logits-ref.c \
  -I third_party/BitNet/3rdparty/llama.cpp/include \
  -I third_party/BitNet/3rdparty/llama.cpp/ggml/include \
  -L build-base/bin -lllama -lggml -lggml-base -lm -lpthread -ldl

LD_LIBRARY_PATH=build-base/bin /tmp/sllm-logits-ref \
  -m /var/lib/spoon/models/ggml-model-i2_s.gguf \
  -p "The capital of France is" \
  -o tests/golden/bitnet2b-capitol -c 512 -t 1 -n 16 -k 5

cc -O2 -o /tmp/sllm-tokenize-ref tools/reference/sllm-tokenize-ref.c \
  -I third_party/BitNet/3rdparty/llama.cpp/include \
  -I third_party/BitNet/3rdparty/llama.cpp/ggml/include \
  -L build-base/bin -lllama -lggml -lggml-base -lm -lpthread -ldl

LD_LIBRARY_PATH=build-base/bin /tmp/sllm-tokenize-ref \
  /var/lib/spoon/models/ggml-model-i2_s.gguf tests/golden/tokenizer.txt
```

## What the gate is, and is not

The gate is **token** parity at temperature 0, plus **logit agreement within a
tolerance measured at Phase 4**. It is not bit-exact logit equality: saphira-llm
will reduce in a different order, and demanding bit equality would force us to
copy upstream's thread partitioning rather than design our own. The FNV-1a hash
in the manifest exists to diagnose disagreement, not to gate on it.

The tokenizer and I2_S gates are a different shape and are worth not
conflating. They are **exact**, with no tolerance, because they contain no
floating-point accumulation whose order is ours to choose: a tokeniser is
integer string work, and I2_S is integer arithmetic with one documented
epilogue. The tolerance exists for the float reduction in a 30-layer forward
pass. Applying it to a tokeniser would buy nothing and would hide exactly the
class of defect these fixtures were built to find.
