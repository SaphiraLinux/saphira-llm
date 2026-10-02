#!/bin/sh
# Golden capture for the Step 2 GEMV witness. SUCCESS-GATED.
#
# A golden may be written ONLY when all of the following hold:
#     the witness process exits 0
#     every case reports BOUNDS OK
#     the exact expected number of records is present
#     every record is structurally complete, i.e. terminated by ENDCASE
#
# No partial stdout may ever become evidence merely because a file exists. The
# previous capture ran by redirecting stdout straight at the golden, so a probe that
# died halfway left a file that parsed and looked plausible. That is how a corrupt
# witness survived long enough to be mistaken for a disagreement with the kernel.
set -eu

MODEL="${1:?usage: capture_gemv_golden.sh <model.gguf> <out.golden> [xseed]}"
OUT="${2:?usage: capture_gemv_golden.sh <model.gguf> <out.golden> [xseed]}"
SEED="${3:-12345}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp "${TMPDIR:-/tmp}/gemv-golden.XXXXXX")"
trap 'rm -f "$TMP"' EXIT

# 1. run to a temp file, never to the destination
# The exit status must be captured WITHOUT a `|| true` in the pipeline: that
# pattern makes $? always zero, which would silently disable the very guard this
# script exists to provide. A success gate that cannot fail is not a gate.
set +e
QUIET=1 "$HERE/run_s0_probe.sh" "$HERE/s0_gemv_probe.cpp" "$MODEL" --seed "$SEED" \
    > "$TMP" 2>/dev/null
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ]; then
    printf 'REFUSED: witness exited %s. Golden NOT written.\n' "$STATUS" >&2
    printf '  (a partial run is not a witness regardless of what it printed)\n' >&2
    exit 1
fi

# 2. every case must report BOUNDS OK
if grep -q 'BOUNDS *= *REFUSED' "$TMP"; then
    printf 'REFUSED: a case reported BOUNDS REFUSED. Golden NOT written.\n' >&2
    exit 1
fi

# 3. the verdict line must say COMPLETE
if ! grep -q 'VERDICT: COMPLETE' "$TMP"; then
    printf 'REFUSED: no COMPLETE verdict. Golden NOT written.\n' >&2
    exit 1
fi

# 4. structural completeness: one ENDCASE per case, and every one of them OK
CASES=$(grep -c '^  \[gemv\] ' "$TMP" || true)
ENDS=$(grep -c '^    ENDCASE .* OK$' "$TMP" || true)
if [ "$CASES" -eq 0 ] || [ "$CASES" -ne "$ENDS" ]; then
    printf 'REFUSED: %s cases but %s complete ENDCASE records. Golden NOT written.\n' \
        "$CASES" "$ENDS" >&2
    exit 1
fi

# 5. every ROWS record must carry exactly n_rows values, and the same for ROWHASH
python3 - "$TMP" <<'PYEOF' || exit 1
import re, sys
txt = open(sys.argv[1]).read()
bad = 0
for blk in txt.split('  [gemv] ')[1:]:
    name = blk.split('\n', 1)[0].strip()
    m = re.search(r'n_rows\s+= (\d+)', blk)
    if not m:
        print('REFUSED: %s has no n_rows' % name, file=sys.stderr); bad += 1; continue
    n = int(m.group(1))
    for tag in ('ROWS', 'ROWHASH'):
        rm = re.search(r'^    %s(.*)$' % tag, blk, re.M)
        if not rm:
            print('REFUSED: %s missing %s' % (name, tag), file=sys.stderr); bad += 1; continue
        got = len([v for v in rm.group(1).split() if v])
        if got != n:
            print('REFUSED: %s %s has %d values, expected %d' % (name, tag, got, n),
                  file=sys.stderr); bad += 1
sys.exit(1 if bad else 0)
PYEOF

# 6. only now, and only atomically, publish
{ printf '# REFERENCE f32 GEMV results, independent of Saphira.\n#\n'
  printf '# Source: tools/s0_gemv_probe.cpp, captured through\n'
  printf '# tools/capture_gemv_golden.sh which is SUCCESS-GATED: the witness must\n'
  printf '# exit 0, every case must report BOUNDS OK and a COMPLETE verdict, and\n'
  printf '# every record must be terminated by ENDCASE with exactly n_rows values\n'
  printf '# in both ROWS and ROWHASH. A partial run is never written.\n#\n'
  printf '# Dequantisation is the vendored reference ggml type_traits.to_float;\n'
  printf '# the dot accumulates in DOUBLE, which is stricter than another f32\n'
  printf '# kernel because the reference is nearer the exact answer.\n#\n\n'
  sed -n '/^=== STEP 2 GEMV WITNESS/,$p' "$TMP"
} > "${OUT}.partial"
mv -f "${OUT}.partial" "$OUT"

printf 'OK: wrote %s (%s complete cases)\n' "$OUT" "$CASES"
