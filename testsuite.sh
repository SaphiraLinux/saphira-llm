#!/bin/sh
# testsuite.sh — build, run every gate, then put the model through its paces.
# Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
# https://www.akadata.co.uk
#
# Licensed under the MIT License — see LICENSE.
# Part of Saphira Linux (https://saphira.vm2.uk).
#
# Three parts, in the order you would want them if something is wrong:
#
#   1. build every binary, and say so if any warning appears
#   2. run the test suite and the mechanical ISA baseline proof
#   3. ask the model some questions and show tokens per second
#
# The model is a base language model, not a chat model, and this repository has
# no chat template and no interactive mode. So the questions below are really
# continuations: the output is the model's own text, unedited and unwrapped.
# Part 3 prints what came back whatever it looks like. A repeated or rambling
# answer here is the model behaving as a 2B base model, not a bug in the
# runtime, and the token checksum printed by tgbench is the thing that
# distinguishes the two.
#
# Usage:
#   ./testsuite.sh                 # build, test, then Q&A
#   ./testsuite.sh --no-build      # skip part 1 if you just built
#   ./testsuite.sh --tests-only    # parts 1 and 2 only, no model needed
#
#   MODEL=/path/to.gguf ./testsuite.sh
#   THREADS=8 NT=64 ./testsuite.sh

set -eu

# No model is shipped, so there is no default that works everywhere. Rather than
# hard-code a path from the machine this was developed on, look in the obvious
# places and then tell the user what to set. MODEL=... overrides all of it.
if [ -z "${MODEL:-}" ]; then
    for candidate in \
        /var/lib/spoon/models/ggml-model-i2_s.gguf \
        "$HOME/models/ggml-model-i2_s.gguf" \
        ./ggml-model-i2_s.gguf
    do
        if [ -f "$candidate" ]; then
            MODEL="$candidate"
            break
        fi
    done
fi
MODEL="${MODEL:-}"

# 4 threads, not the recommended 27. The recommendation is still the
# pre-Phase-6 "leave one hardware thread free" rule, and measurement puts the
# peak at 4 to 8, so defaulting to it would print a slow answer for a reason
# that has nothing to do with the model.
THREADS="${THREADS:-4}"

# Tokens per answer. Enough to see whether the model is going somewhere.
NT="${NT:-40}"

DO_BUILD=1
DO_TESTS=1
DO_QA=1

for arg in "$@"; do
    case "$arg" in
        --no-build)   DO_BUILD=0 ;;
        --tests-only) DO_QA=0 ;;
        -h|--help)    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "testsuite.sh: unknown option '$arg'" >&2; exit 2 ;;
    esac
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

rule() { printf '%s\n' "------------------------------------------------------------"; }
step() { rule; printf '%s\n' "$1"; rule; }
die()  { printf 'testsuite.sh: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- 1. build

if [ "$DO_BUILD" -eq 1 ]; then
    step "1. build"
    make clean >/dev/null 2>&1 || true
    # Everything is compiled here, bench and test units included, and the
    # output is kept so a warning cannot scroll past unseen.
    if make all test bench tgbench kernelbench topoprobe >"$WORK/build.log" 2>&1; then
        n_warn=$(grep -ic 'warning' "$WORK/build.log" || true)
        n_err=$(grep -ic 'error' "$WORK/build.log" || true)
        if [ "$n_warn" -ne 0 ] || [ "$n_err" -ne 0 ]; then
            printf 'FAIL: %s warning(s), %s error(s) in the build\n' "$n_warn" "$n_err"
            grep -iE 'warning|error' "$WORK/build.log" || true
            exit 1
        fi
        echo "clean: 7 binaries, 0 warnings, 0 errors"
    else
        echo "FAIL: the build did not complete"
        tail -20 "$WORK/build.log" || true
        exit 1
    fi
else
    step "1. build"
    echo "skipped (--no-build)"
fi

for bin in saphira-llm saphira-llm-test; do
    [ -x "./$bin" ] || die "$bin is not built; run without --no-build"
done

# ---------------------------------------------------------------- 2. tests

if [ "$DO_TESTS" -eq 1 ]; then
    step "2a. test suite"
    if ./saphira-llm-test >"$WORK/test.log" 2>&1; then
        grep -E '^(isa|gguf|topology|thread|ops|i2s|tokenizer|forward|dot-f16|phase5)$' \
            "$WORK/test.log" | tr '\n' ' ' | fold -sw 60 | sed 's/^/  /'
        echo
        grep -E '^[0-9]+ checks, [0-9]+ failed$' "$WORK/test.log" | sed 's/^/  /'
        echo "  PASS"
    else
        echo "  FAIL"
        grep -E 'FAIL' "$WORK/test.log" | head -20 || true
        exit 1
    fi

    step "2b. ISA baseline"
    # The mechanical proof: nothing above x86-64-v3 outside the guarded section.
    if sh scripts/check-baseline-isa.sh ./saphira-llm ./saphira-llm-test \
            >"$WORK/isa.log" 2>&1; then
        tail -1 "$WORK/isa.log" | sed 's/^/  /'
        echo "  PASS"
    else
        echo "  FAIL"
        cat "$WORK/isa.log"
        exit 1
    fi
fi

if [ "$DO_QA" -eq 0 ]; then
    printf '\n'
    echo "all tests passed; no model run requested (--tests-only)"
    exit 0
fi

# ---------------------------------------------------------------- 3. quality

if [ -n "$MODEL" ] && [ -f "$MODEL" ]; then
    step "3. model quality (perplexity)"
    CORPUS="${CORPUS:-tests/golden/eval-corpus.txt}"
    if [ -f "$CORPUS" ]; then
        ./saphira-llm-eval -m "$MODEL" -d "$CORPUS" -t "$THREADS" --log warn |
            grep -E '^(corpus_tokens|scored_tokens|mean_nll|perplexity|top1_accuracy)'
        echo
        echo "  Lower perplexity is better. This is a measurement, not a gate,"
        echo "  and it only compares against another run of the same binary on"
        echo "  the same corpus. See docs/EVALUATION.md."
    else
        echo "no corpus at $CORPUS (set CORPUS=/path/to/text.txt)"
    fi
    echo
fi

# ---------------------------------------------------------------- 4. the model

step "4. generation (Q&A)"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    die "no model found (looked in /var/lib/spoon/models, ~/models and .)
Set it explicitly, e.g.

  MODEL=/path/to/ggml-model-i2_s.gguf ./testsuite.sh

A suitable model is microsoft/bitnet-b1.58-2B-4T-gguf on Hugging Face (MIT,
about 1.2 GB). Only the bitnet-b1.58 architecture loads; see README.md.
Use --tests-only to run the gates without one."
fi

echo "model    $MODEL"
echo "bytes    $(wc -c <"$MODEL" | tr -d ' ')"
echo "sha256   $(sha256sum "$MODEL" | cut -d' ' -f1)"
echo "threads  $THREADS"
echo "tokens   $NT per answer"
echo

# What the model is, from our own GGUF reader, before we ask it anything.
./saphira-llm -m "$MODEL" --log info 2>&1 |
    grep -E 'architecture|gguf v' | sed 's/^saphira-llm: info: /  /' || true
echo

# A base model continues text; it does not answer questions. The first prompt
# is a completion because that is where the weights are informative, and the
# rest are questions so the awkward answers are on the record too.
cat >"$WORK/prompts" <<'PROMPTS'
The capital of France is
The largest planet in the Solar System is
Water freezes at
What is Saphira Linux?
The three primary colours are
In the year 1969, humans first landed on
PROMPTS

n=0
while IFS= read -r prompt; do
    n=$((n + 1))
    printf -- '--- Q%d: %s\n' "$n" "$prompt"

    # --log warn keeps the info chatter out of the answer. stdout carries the
    # generated text followed by one machine-readable tps line, which is how
    # the two get separated again below.
    if ! ./saphira-llm -m "$MODEL" -p "$prompt" -n "$NT" -t "$THREADS" \
            --log warn >"$WORK/qa.out" 2>"$WORK/qa.err"; then
        printf '    FAILED to run: %s\n' "$(head -1 "$WORK/qa.err" || true)"
        echo
        continue
    fi

    # The generated text is everything before the sllm_bench line.
    sed '/^sllm_bench tg /,$d' "$WORK/qa.out" | sed 's/^/    /'
    tps=$(sed -n 's/.*tps \([0-9.]*\).*/\1/p' "$WORK/qa.out" | tail -1)
    elapsed=$(sed -n 's/.*time \([0-9.]*\) s.*/\1/p' "$WORK/qa.out" | tail -1)
    [ -n "$tps" ] || tps="?"
    [ -n "$elapsed" ] || elapsed="?"
    printf '    [%s tok/s, %ss, %s tokens]\n\n' "$tps" "$elapsed" "$NT"
done <"$WORK/prompts"

step "done"

if [ "$DO_TESTS" -eq 1 ]; then
    echo "tests: PASS"
else
    echo "tests: skipped (--no-build given, or -h)"
fi
echo "model: answered $n prompt(s) above, unedited"
echo
echo "The runtime is deterministic, so the same prompt gives the same tokens"
echo "every time. To see throughput properly measured rather than sampled once:"
echo
echo "  ./saphira-llm-tgbench -m $MODEL -p \"The capital of France is\" \\"
echo "      -c 512 -n 64 -t 1,2,4,6,8,14,28 -r 3 -w 1"
echo
echo "It repeats each row, prints the load average, and requires every row to"
echo "produce the same token checksum. A differing checksum is a correctness"
echo "failure; a fast row is not a pass."
