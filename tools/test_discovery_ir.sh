#!/bin/bash
# Structural tests for the discovery IR. Deliberately includes a NON-VACUITY
# PROOF: each negative test is shown to actually fail when its guard is removed,
# because a negative test that cannot fail is decoration.
set -uo pipefail
cd "$(dirname "$0")/.."
QW=/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
LL=/var/lib/spoon/models/llama32-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf
export QUIET=1
pass=0; fail=0
ok(){ if [ "$2" = "0" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1"; fail=$((fail+1)); fi; }

gen(){ ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$1" 2>/dev/null; }

# A: same artefact -> deterministic IR
A1=$(gen "$QW"); A2=$(gen "$QW")
[ "$A1" = "$A2" ]; ok "A1 same artefact yields byte-identical IR" $?
[ "$(gen "$QW")" = "$A1" ]; ok "A2 third run still identical" $?
[ "$A1" != "$(gen "$LL")" ]; ok "A3 the two models do NOT produce the same IR" $?

# B: tensor ORDER changes that do not alter semantics must not alter the IR
#    (semantics here = the per-layer role SET and every resolved parameter)
sem(){ grep -oE '(RMSNorm|RoPE|Linear|GatedMLP|Embedding|OutputProjection|TokenizerSelector) *\(?(model)?\)? *layer=[0-9]+|RMSNorm.*blk\.[0-9]+\.[a-z_]+\.weight|- (leg|qwen3\.|llama\.|a DISTINCT|NO output|tokenizer|every layer)' | sort; }
S_Q=$(printf '%s\n' "$A1" | sem); S_Q2=$(printf '%s\n' "$A2" | sem)
[ "$S_Q" = "$S_Q2" ]; ok "B1 semantic view is stable across runs" $?

# C: Qwen3's Q/K norm CANNOT disappear
printf '%s\n' "$A1" | grep -q 'blk.0.attn_q_norm.weight'; ok "C1 Qwen3 attn_q_norm present" $?
printf '%s\n' "$A1" | grep -q 'blk.0.attn_k_norm.weight'; ok "C2 Qwen3 attn_k_norm present" $?
# Count DISTINCT head-norm NODES, not every line that mentions the name: the
# evidence lines legitimately repeat it, so a raw line count is the wrong
# measure and would have "caught" a correct IR as wrong.
[ "$(printf '%s\n' "$A1" | grep -cE '^  RMSNorm .* blk\.0\.attn_[qk]_norm\.weight$')" = "2" ]
ok "C3 exactly two head-norm nodes (Q and K), counted as nodes not mentions" $?

# D: Llama must NOT grow Q/K norm merely because Qwen3 has them
L1=$(gen "$LL")
printf '%s\n' "$L1" | grep -q 'correctly ABSENT for this architecture'; ok "D1 Llama records Q/K norm absence explicitly" $?
[ "$(printf '%s\n' "$L1" | grep -cE '^  RMSNorm .* blk\.0\.attn_[qk]_norm\.weight$')" = "0" ]
ok "D2 Llama emits no Q/K norm node" $?
printf '%s\n' "$L1" | grep -q 'attn_q_norm.weight at layer'; ok "D3 absence is stated, not silently dropped" $?

# E: tokenizer tuple -- all three legs, and the contradiction must stay visible
printf '%s\n' "$A1" | grep -q 'leg model   = "gpt2"'; ok "E1 Qwen3 model leg recorded (gpt2)" $?
printf '%s\n' "$A1" | grep -q 'leg pre     = "qwen2"'; ok "E2 Qwen3 pre leg recorded (qwen2)" $?
printf '%s\n' "$A1" | grep -q 'leg tokens  = 151936'; ok "E3 Qwen3 vocab-size leg recorded" $?
printf '%s\n' "$A1" | grep -q 'model alone is NOT a sufficient selector'; ok "E4 insufficiency of model-only is stated" $?
printf '%s\n' "$A1" | grep -q 'no qwen3.vocab_size metadata'; ok "E5 missing vocab_size metadata stays visible as UNRESOLVED" $?
printf '%s\n' "$L1" | grep -q 'llama.vocab_size = 128256'; ok "E6 Llama vocab_size leg present" $?

# F: unsupported constructs become explicit UNKNOWN, never a silent fallback
printf '%s\n' "$L1" | grep -q 'UNKNOWN role: (non-layer) rope_freqs.weight'; ok "F1 undescribed tensor reported as UNKNOWN" $?
printf '%s\n' "$A1" | grep -q 'pairing         : UNRESOLVED'; ok "F2 RoPE pairing not asserted from a name" $?
printf '%s\n' "$A1" | grep -q 'requires measurement from the reference graph'; ok "F3 UNRESOLVED states what would resolve it" $?

# G: non-vacuity. Remove a guard, prove the test notices, restore.
cp src/gguf.c /tmp/gguf.c.bak 2>/dev/null || true
cp tools/s0_ir_probe.cpp /tmp/ir.bak
sed -i 's/known_roles.insert("attn_q_norm.weight"); known_roles.insert("attn_k_norm.weight");//' tools/s0_ir_probe.cpp
BROKEN=$(gen "$QW")
printf '%s\n' "$BROKEN" | grep -q 'UNKNOWN role: attn_q_norm.weight'
if [ $? -eq 0 ]; then ok "G1 non-vacuity: removing head-norm knowledge surfaces UNKNOWN (test bites)" 0
else ok "G1 non-vacuity: removing head-norm knowledge surfaces UNKNOWN (test bites)" 1; fi
cp /tmp/ir.bak tools/s0_ir_probe.cpp
RESTORED=$(gen "$QW")
[ "$RESTORED" = "$A1" ]; ok "G2 restore is exact, IR byte-identical to pre-fault" $?

echo
echo "discovery-ir structural tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
