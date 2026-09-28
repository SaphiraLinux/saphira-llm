# Golden vectors

Fixtures captured from the **upstream** BitNet-capable llama.cpp — never from
saphira-llm. They are the oracle that the saphira-llm parity gate is judged
against. Regenerating them from saphira-llm would destroy their purpose.

| File | Contents |
| --- | --- |
| `*.f32` | every logit position, raw little-endian `float`, `n_tokens × n_vocab` |
| `*.txt` | manifest: token ids, per-position logit statistics, top-k, FNV-1a 64 hash, timings, greedy continuation |

## Current fixtures

| Fixture | Prompt | Tokens |
| --- | --- | ---: |
| `bitnet2b-capitol` | `The capital of France is` | 6 |
| `bitnet2b-save` | `The name of the capital city of France is` | 10 |

Model: `/var/lib/spoon/models/ggml-model-i2_s.gguf`
(`bitnet-b1.58 2B Q1_0`, 128256-token gpt2 BPE vocabulary, BOS 128000).

## Rules for anyone adding or regenerating a fixture

1. **Capture with the reference tool**, `tools/reference/sllm-logits-ref`, built
   against `third_party/BitNet/3rdparty/llama.cpp`. Never capture from a
   saphira-llm binary.
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
6. **Record the logit hash in the commit message.** It is the cheapest way to
   detect accidental fixture churn.

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
```

## What the gate is, and is not

The gate is **token** parity at temperature 0, plus **logit agreement within a
tolerance measured at Phase 4**. It is not bit-exact logit equality: saphira-llm
will reduce in a different order, and demanding bit equality would force us to
copy upstream's thread partitioning rather than design our own. The FNV-1a hash
in the manifest exists to diagnose disagreement, not to gate on it.
