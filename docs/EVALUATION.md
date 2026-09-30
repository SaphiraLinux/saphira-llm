# Evaluation

What this project can now measure about a model, and — more usefully — what it
still cannot.

## The gap this closes

Until now this project could prove a model was *correct* and nothing else. The
gates in `saphira-llm-test` answer "does this compute the right answer": argmax
token-identity against the reference, golden vectors, a mechanically proven ISA
baseline. All pass/fail, and none of them can tell you whether a change made
the model better or worse.

That matters because almost every planned change here is invisible without a
number. Converting BF16 masters to packed I2_S may cost a little accuracy or a
great deal, and reading the resulting text settles nothing. Nor does comparing
two checkpoints by eye, or asking whether the output "looks more fluent". A
repeatable figure over fixed text turns those into arithmetic.

The BODH's only relevant opinion on this subject is that a fine-tune nobody
measured is indistinguishable from a junior who learned your habits. This is
the receipt printer.

## Using it

```sh
make eval
./saphira-llm-eval -m /path/to/model.gguf -d corpus.txt
```

```
model              /var/lib/spoon/models/ggml-model-i2_s.gguf
corpus             tests/golden/eval-corpus.txt
corpus_tokens      386
scored_tokens      385
context            386
threads            4
nll_sum            1916.2644512626
mean_nll           4.9773102630
bits_per_token     7.180741
perplexity         145.083620
top1_accuracy      0.181818
reference_form     4.9773102633  (softmax-then-log, for comparison)
seconds            18.7341
```

`--json` emits the same figures as a machine-readable object, for scripted A/B
comparison.

## What the numbers mean

| Figure | Meaning |
| --- | --- |
| `corpus_tokens` | tokens the model's own tokenizer produced, with `add_special` and `parse_special` both set |
| `scored_tokens` | tokens actually scored; always `corpus_tokens - 1` |
| `nll_sum` | summed negative log likelihood, in nats |
| `mean_nll` | `nll_sum / scored_tokens` |
| `perplexity` | `exp(mean_nll)`. The headline figure. Lower is better. |
| `bits_per_token` | `mean_nll / ln 2`. Same information, but does not overflow and reads more naturally |
| `top1_accuracy` | fraction of scored tokens the model ranked first. Not perplexity, and a useful cross-check: a model can have good accuracy and poor calibration |

The definition is taken from the pinned upstream tree, not from memory.
`third_party/BitNet/3rdparty/llama.cpp/tools/perplexity/perplexity.cpp:60`
scores, per position `j`:

```
nll += -log( softmax(logits[j])[ tokens[j+1] ] )
perplexity = exp(nll / count)
```

## Which tokens are scored

Stated precisely so another implementation can reproduce the number:

- The corpus is tokenised with this model's tokenizer, `add_special` and
  `parse_special` both set, matching `saphira-llm`.
- It is fed in chunks of at most `SLLM_MAX_CHUNK` (256) at consecutive absolute
  positions through **one** context, so the KV cache spans the whole corpus and
  every token is scored against its real prefix rather than a sliding window.
- The logit row at index `i` of a chunk starting at position `p` is the
  distribution after consuming `tokens[p..p+i]`, so it predicts
  `tokens[p+i+1]`.
- Every token except the very first is scored exactly once. The first has no
  context, so there is nothing to predict it from; that is what a language
  model is defined to do, not a limitation of this tool.
- `scored_tokens == corpus_tokens - 1` is **asserted in the tool**, not assumed.
  The first version of this code dropped one token at every chunk boundary and
  the assertion caught it on the first run: 386 tokens in, 384 scored.

## Numerical choices, and the one that matters

The reported reduction is log-sum-exp in double:

```
nll = log( sum_i exp(L[i] - max) ) + max - L[tok]
```

This is algebraically the reference's form, and the difference is *measured*
rather than asserted — the tool prints both, and they agree to ten decimal
places (`4.9773102630` against `4.9773102633`).

Log-sum-exp is used rather than softmax-then-log for one reason that is not
stylistic: a token the model is certain it will never emit has a probability
that underflows float to zero, and `-log(0)` is infinity. One such token
poisons an entire corpus sum. The log-sum-exp form never leaves the log domain,
so the same token yields a large finite loss. `test_eval.c` asserts this
directly, and also asserts that the reference form does *not* survive it.

### The FMA finding

`mean_nll` is not portable across compilers. Measured by building this same
source six ways:

| Build | `mean_nll` |
| --- | --- |
| gcc `-O2` | 4.9773102630 |
| gcc `-O2 -ffp-contract=off` | 4.9773102630 |
| clang `-O2 -ffp-contract=off` | 4.9773102630 |
| clang `-O2` | 4.9807068980 |
| clang `-O2 -mfma -ffp-contract=fast` | 4.9807068980 |
| clang `-O1 -fsanitize=address,undefined` | 4.9807068980 |

Two things fall out of that table, and both are worth stating plainly:

1. **gcc and clang agree bit-exactly once FMA contraction is disabled.** The
   entire cross-compiler difference is the forward pass contracting
   multiply-add into fused operations.
2. **It is not the sanitizers and not the optimisation level.** clang produces
   the same value at `-O1` with ASan+UBSan as at `-O2` with FMA on.

This reaches the loss the same way it reaches the logits, and Phase 4 already
established the underlying property: our reduction order is our own, so bit
equality with the reference was never available, which is why the forward gate
is an argmax margin rather than a byte comparison. A 385-token sum of
128256-way reductions has plenty of opportunity to notice a rounding difference.

Practical consequence: **compare only against runs from the same binary.** The
gcc and clang figures differ by 0.34% (perplexity 145.0836 against 145.5773),
which is the same order as a small real improvement and would be easy to
mistake for one. `saphira-llm-eval` prints this warning for that reason.

### The arithmetic contract

Adding `-ffp-contract=off` to the **release** build is **not** done: it would
change the shipped arithmetic of a sealed release, which is a larger call than
an evaluation stage should make on its own.

For work **after** the sealed tag it is decided, and the rule is in
`DECISIONS.md`:

- `v0.0.1` at `7dbcc67` is not touched.
- The **evaluation and training validation path** builds with
  `-ffp-contract=off`, so regression arithmetic is deterministic across
  compilers.
- A fresh canonical baseline is established under that arithmetic. **The golden
  value in `test_eval.c` was captured under the default (contracted) build and
  is provisional until re-recorded.** Once the contract is in force the 5e-3
  tolerance that currently absorbs the FMA spread becomes unnecessary and
  should become exact equality.
- A contracted baseline is never compared against a non-contracted build, and
  the difference is never reported as a model improvement.
- If production inference later deliberately uses FMA for performance, that is
  benchmarked separately and does not become the quality baseline by accident.

Not yet applied; it is the first item in the next-actions list.

## Evaluation is not a correctness gate

This is worth being blunt about, because the two are easy to conflate:

- **Correctness gates** (`saphira-llm-test`, `make check`) are pass/fail and
  must never move. If token-identical output stops matching the reference, that
  is a defect regardless of whether perplexity went up or down.
- **Evaluation** (`saphira-llm-eval`) reports a number and a human decides. A
  perplexity that changes is a note, not a failure.

They can disagree, and when they do, correctness wins. A change that improves
perplexity while breaking token parity has broken the thing, and the better
number is not evidence otherwise.

## Known limitation: no rolling window

A corpus must fit the context in full. llama.cpp scores long input with a
rolling window and a stride, re-feeding an overlap so every token still sees a
full window. That is not implemented, so a corpus over `--max-tokens` (default
8192) is **refused with an explanation** rather than silently scored as a
prefix.

This is the first thing to add. It is what a wiki-scale or book-scale
measurement needs, and without it every figure here comes from text short enough
to be self-contained.

## The in-tree fixture, and why its number is not a benchmark

`tests/golden/eval-corpus.txt` is 386 tokens of prose about measurement, written
for this purpose. On the 2B acceptance model it gives:

```
corpus_tokens   386
mean_nll        4.9773102630
perplexity      145.083620
top1_accuracy   0.181818
```

Treat that as a **regression fixture, not a benchmark.** It is ~300 words of
in-house technical prose — out of distribution for a model trained on web text,
so the absolute figure is not comparable to any published BitNet number and says
nothing about the model in general. Top-1 accuracy around 18% is what a 2B base
model does on text it has not seen, and is not a defect.

It is fit for exactly one purpose: being the same 386 tokens forever, so that a
change in the number means the model changed. That is what
`test_eval.c` uses it for.

A real quality corpus — wiki, a held-out slice of the training distribution,
or domain text — is still wanted, and the honest caveat from the fixture's own
text applies: the number is worth exactly as much as the care taken choosing
what to measure.

## How to regenerate the golden value

Only when the change is deliberate and understood. The value in `test_eval.c`
is recorded from the shipping build (gcc, `-O2`).

```sh
make -s clean && make -s eval
./saphira-llm-eval -m "$SLLM_TEST_MODEL" -d tests/golden/eval-corpus.txt --log quiet --json
```

Update `EVAL_GOLDEN_TOKENS` if the fixture changed, and `EVAL_GOLDEN_MEAN_NLL`
if the model or the forward pass changed. If the token count moved, every other
number is void rather than wrong.
