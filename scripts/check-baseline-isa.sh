#!/bin/sh
# check-baseline-isa.sh — prove the x86-64-v3 baseline mechanically.
# Copyright (C) 2026 Andrew Smalley for and on behalf of AKADATA LIMITED.
# Licensed under the MIT License — see LICENSE.
#
# The claim this script exists to defend:
#
#   No instruction above the x86-64-v3 baseline is reachable except inside
#   .sllm.isa.ext, and that section is entered only through the runtime
#   dispatcher.
#
# Part one is checked here, mechanically, by disassembling the binaries and
# scanning every instruction outside the guarded section for mnemonics that
# x86-64-v3 does not include. Part two is enforced by construction: the
# SLLM_ISA_EXT section attribute is what places a kernel there, and
# sllm_isa_build() is the only thing that decides whether its pointer is
# installed.
#
# The section is deliberately NOT named .text.sllm_isa_ext. The default linker
# script merges all .text.* input sections into a single output .text, so a
# .text-prefixed name is merged away at link time and the guard would pass
# vacuously forever. That was observed here before the name was changed.
#
# Usage: sh scripts/check-baseline-isa.sh [binary ...]

set -eu

GUARDED='.sllm.isa.ext'

# Mnemonics that are strictly above the x86-64-v3 baseline.
#
# v3 = AVX, AVX2, BMI, BMI2, FMA, F16C, LZCNT, MOVBE, OSXSAVE. So the things
# to look for are: AVX-512 forms, the AMX tile operations, and the VNNI
# instructions. vpdpbusd appears in both a VEX form (AVX-VNNI) and an EVEX form
# (AVX512-VNNI); both are above v3 and both are listed.
ABOVE_V3='vpdpbusd|vpdpbusds|vpdpwssd|vpdpwssds|vpternlog|vp4dpwssd|vp4dpbusd|vpdpphssd|vpdpphubsd|vcvtne2ps2bf16|vcvtneps2bf16|vcvtne2ph2ps|vcvtneph2ps|ldtilecfg|sttilecfg|tilecfgdpt|tilezero|tilerelease|tdpbssd|tdpbusd|tdpbsud|tdpbusud|tdpbsa|tdpbusa|tdphbsa|tdphbsp|kmov|kortest|kadd|kand|kor|knot|kortest|kunpck|kxnor|kxor|kunpckbw|kaddb'

# A second, deliberately broad net: anything that names a zmm/tmm register or
# an opmask. These catch AVX-512 encodings the mnemonic list would miss.
WIDE_V3='%zmm|%ymm[0-9]+,%zmm|%k[0-7]|%tmm|{k[0-7]}'

fail=0
if [ "$#" -gt 0 ]; then
    BINARIES="$*"
else
    BINARIES="./saphira-llm ./saphira-llm-test"
fi

OBJDUMP="${OBJDUMP:-objdump}"
if ! command -v "$OBJDUMP" >/dev/null 2>&1; then
    echo "check-isa: $OBJDUMP not found; cannot verify" >&2
    exit 1
fi

for bin in $BINARIES; do
    if [ ! -f "$bin" ]; then
        echo "check-isa: missing $bin (build it first)" >&2
        exit 1
    fi

    tmp="$bin.disasm"
    "$OBJDUMP" -d --no-show-raw-insn "$bin" > "$tmp" 2>/dev/null

    # 1. The guarded section must exist. If it is absent the scan below would
    #    find nothing and pass without having checked anything, which is the
    #    failure mode this step exists to prevent.
    if ! grep -q "^Disassembly of section $GUARDED:" "$tmp"; then
        echo "check-isa: FAIL $bin has no $GUARDED section." >&2
        echo "           The guard would be vacuous, so this is a failure, not a skip." >&2
        fail=1
    fi

    # 2. Single pass: attribute every instruction to its section and its
    #    function, then flag anything above the baseline outside the guard.
    #
    #    Results go to stderr as they are found, so the counts below and the
    #    violations can never disagree.
    result=$(awk -v guarded="$GUARDED" -v above="$ABOVE_V3" -v wide="$WIDE_V3" '
        /^Disassembly of section / {
            section = $4
            sub(/:$/, "", section)
            next
        }
        # A function header is "ADDRESS <name>:". Take the name from between
        # the angle brackets; the previous attempt stripped the whole line and
        # silently produced zero functions.
        match($0, /<[^>]+>:?$/) {
            fname = substr($0, RSTART + 1, RLENGTH - 3)
            if (RLENGTH == 3) fname = ""
            next
        }
        /^[[:space:]]+[0-9a-f]+:/ {
            insn = $0
            total++
            if (section == guarded) {
                guarded_n++
                next
            }
            plain_n++
            if (insn ~ above || insn ~ wide) {
                bad++
                if (bad <= 40) {
                    printf "    [%s] %s\n", section, insn > "/dev/stderr"
                }
            }
        }
        END {
            printf "%d %d %d %d\n", total+0, plain_n+0, guarded_n+0, bad+0
        }
    ' "$tmp")

    set -- $result
    total=$1; plain=$2; guarded_n=$3; bad=$4

    echo "check-isa: $bin — $total instructions scanned, $plain in baseline sections, $guarded_n in $GUARDED, $bad above-baseline outside the guard"

    if [ "$bad" -ne 0 ]; then
        echo "check-isa: FAIL $bin — $bad instruction(s) above the baseline outside $GUARDED" >&2
        fail=1
    fi

    rm -f "$tmp"
done

if [ "$fail" -ne 0 ]; then
    echo "check-isa: FAILED" >&2
    exit 1
fi

echo "check-isa: OK — nothing above ${SLLM_BASELINE_ISA:-x86-64-v3} outside $GUARDED"
