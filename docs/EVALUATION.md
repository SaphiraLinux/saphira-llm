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

### The arithmetic contract — IN FORCE

`-ffp-contract=off` is now applied by the Makefile as `FPFLAGS`, to the library,
the tests, the evaluator and the training tooling alike. One arithmetic for
everything: a test suite that built the library differently from the shipped
binary would be testing something nobody runs.

`v0.0.1` at tag `7dbcc67` predates this and is unaffected — a tag names a tree,
not a build.

**Canonical baseline, recorded under the contract:**

| build | `nll_sum` | `mean_nll` |
| --- | --- | --- |
| gcc `-O2` | 1916.2644512626152 | 4.9773102630197794 |
| clang `-O2` | 1916.2644512626152 | 4.9773102630197794 |
| clang `-O1` + ASan/UBSan | 1916.2644512626152 | 4.9773102630197794 |

Identical, not close. Thread counts 1/4/8/16 are identical too.

For contrast, **with** contraction enabled clang gave `mean_nll 4.9807068980` —
a 0.34% gap in perplexity, the same order as a small real improvement, and
therefore exactly the trap this contract removes.

### The golden comparison is now exact

The tolerance in `test_eval.c` was 5e-3, and existed only to absorb the FMA
spread. The contract removed the reason for it, so the comparison is now
`==` on the bit pattern: `nll_sum` and `mean_nll` must match exactly.

That required one correction worth recording: the golden was initially captured
from a `%.10f` rendering, which is a *rounded decimal* and not the double. The
exact comparison caught it immediately, which is the right way round. The
evaluator's `--json` now emits these fields with `%.17g` so they round-trip.

A different libm or a different architecture may still move the last bits. If
so the correct response is a deliberate re-recording with the difference
explained — not a tolerance widened until the signal fits.

Overriding `FPFLAGS` is the deliberate production-FMA decision, which is a
separate performance benchmark and is **not** the quality baseline. See
`DECISIONS.md`.

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

## Rolling window

`--stride N` walks a corpus that is longer than the context in windows of
`n_ctx` tokens, scoring only the last `N` of each. The single-window path
(`--stride 0`, the default) is unchanged, and a corpus that does not fit is
still **refused**, never truncated.

### Semantics

- Window `i` begins at corpus position `start_i = i * stride`.
- Its length is `len_i = min(n_win, n_tok - start_i)`, where
  `n_win = min(n_ctx, n_tok)`. **The last window is truncated to the end of the
  corpus**, not pulled back.
- Within a window only its last `count = min(stride, len_i - 1)` tokens are
  scored. Earlier tokens were scored by the previous window and are not scored
  again.
- Each window is fed into a **cleared** context at positions `0..len-1`, so its
  predictions depend on that window's own prefix and nothing before it.
- Token 0 is never scored; it has no context in any window.

### Why truncate rather than pull back

This is the one design decision that decides whether the tiling is correct.

A window always covers `n_win - stride .. n_win - 1` of its own tokens. If the
final window is instead moved *earlier* to stay full, it **overlaps** its
predecessor and leaves a gap after it. Measured on a 1156-token corpus with a
256-token window and stride 255: 120 tokens scored twice and 135 never scored.

Truncating the last window to the corpus end makes the scored ranges a
**partition** of `[1, n_tok)` — no gap, no overlap — for any stride.

### Coverage, and why a smaller stride is refused

Windows advance by `stride` and each scores `count` tokens, so a gap opens
unless `count == stride`; and the first window must begin scoring at token 1,
which needs `count == n_win - 1`. Both hold exactly when

```
stride == n_win - 1
```

Anything else is **refused with the arithmetic shown**, rather than reporting a
perplexity over a silently-scored subset.

The upstream reference is looser than this: it advances by `stride` and scores
the last `stride` tokens of each window, so at `stride < n_ctx` it silently
leaves `1 .. n_ctx-stride-1` unscored. This tool refuses that configuration
instead.

### Equivalence, proven

With `stride == n_win - 1` and a corpus that fits in one window there is a
single window scoring corpus tokens `1 .. n_tok-1` — the same set, at the same
positions, into a cleared context, as the single-window path. Therefore:

| configuration | `nll_sum` |
| --- | --- |
| `--ctx 386 --stride 0` | 1916.2644512626152 |
| `--ctx 512 --stride 0` | 1916.2644512626152 |
| `--ctx 386 --stride 385` | 1916.2644512626152 |
| `--ctx 512 --stride 385` | 1916.2644512626152 |

Bit-identical, asserted with `==` in `test_eval.c` rather than a tolerance,
because the arithmetic contract makes exactness available and a tolerance here
would hide the class of bug this exists to catch.

### Token accounting

Every scored token increments its own counter, and afterwards:

- no token is scored more than once,
- no token is scored zero times, except token 0,
- token 0 is never scored,
- the total equals `corpus_tokens - 1`.

A stride bug would still produce a plausible-looking perplexity, so this is
checked rather than reported for comfort. It caught two real off-by-ones during
development — a `first_row` of `n_win - stride` instead of `n_win - 1 - stride`
(dropping token 1), and a `last_row` that did not exclude the row predicting a
target outside the window.

Over-context, measured: a 1156-token corpus in a 256-token window at stride 255
runs 5 windows, scores 1155 tokens, and passes accounting. The in-tree test does
the same at 771 tokens in 7 windows of 128.

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
