#!/bin/sh
# Build-and-run a Step 0 reference probe ATOMICALLY.
#
# Why this exists: twice, a compile failed and a PREVIOUS binary ran anyway,
# and its output was reported as new data. Detecting a stale executable is not
# good enough; this makes one impossible.
#
# The binary is compiled to a fresh unique path and only moved into place on
# success, so the run path never contains an executable that is not the product
# of the compile that just succeeded. set -e plus an explicit exit-status check
# means a failed compile aborts before any run happens.
set -eu
SRC="$1"; shift
OUT="$(dirname "$0")/.probe.bin.$$"     # unique, never a previous build
TMP="$OUT.tmp"
trap 'rm -f "$OUT" "$TMP"' EXIT INT TERM

LLAMA_DIR="${LLAMA_DIR:-third_party/llama.cpp}"
LIBDIR="${LIBDIR:-/tmp/lcpbuild/bin}"

printf 'compiling %s -> %s\n' "$SRC" "$TMP"
if ! g++ -O2 -std=c++17 -I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include" \
        "$SRC" -o "$TMP" -L"$LIBDIR" -lllama -lggml-base -Wl,-rpath,"$LIBDIR"; then
    printf 'COMPILE FAILED: refusing to run. No executable was produced.\n' >&2
    exit 70
fi
[ -x "$TMP" ] || { printf 'COMPILE PRODUCED NO BINARY: refusing to run.\n' >&2; exit 70; }

mv -f "$TMP" "$OUT"                    # publish only on success
printf 'running fresh binary\n'
set +e
"$OUT" "$@"
rc=$?
rm -f "$OUT"
exit $rc
