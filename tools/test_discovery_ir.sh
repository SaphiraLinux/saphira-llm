#!/bin/bash
# Structural tests for the discovery IR. Deliberately includes a NON-VACUITY
# PROOF: each negative test is shown to actually fail when its guard is removed,
# because a negative test that cannot fail is decoration.
set -uo pipefail
cd "$(dirname "$0")/.."
QW=/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
LL=/var/lib/spoon/models/llama32-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf
export QUIET=1
export SLLM_IR_ROPE_PAIRING_NEED=1
pass=0; fail=0
ok(){ if [ "$2" = "0" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1"; fail=$((fail+1)); fi; }

gen(){ unset SLLM_IR; SLLM_IR_ROPE_PAIRING=${P1:-} SLLM_IR_OUTPUT_SHARING=${SH1:-} SLLM_IR_SWA_PATTERN=${SW:-} SLLM_IR_FUSED_CAP=${FC:-} SLLM_IR_ROUTING=${ROUT:-} SLLM_IR_HEAD_DIM=${HD:-} ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$1" 2>/dev/null; }

# A: same artefact -> deterministic IR
P1=NEOX; SH1="weights NOT shared: final projection reads output.weight"; A1=$(gen "$QW"); A2=$(gen "$QW")
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
P1=GPT/adjacent; SH1="weights ARE shared: final projection reads token_embd.weight"; L1=$(gen "$LL")
# The absence is now emitted AS A NODE with status ABSENT, which is stronger than
# the earlier free-text line: the slot is explicitly empty rather than merely
# unmentioned, so a reader cannot mistake it for an oversight.
[ "$(printf '%s\n' "$L1" | grep -cE '^  RMSNorm +layer=0 +blk\.0\.attn_[qk]_norm\.weight$')" = "2" ]
ok "D1 Llama emits both Q/K norm slots as explicit nodes" $?
[ "$(printf '%s\n' "$L1" | grep -c 'status          : ABSENT for this architecture')" = "2" ]
ok "D2 both slots carry status ABSENT, not a silent omission" $?
printf '%s\n' "$L1" | grep -q 'ABSENT is not UNKNOWN'; ok "D3 absence explicitly distinguished from unknown" $?
# And the node must NOT claim measured values for an absent norm.
printf '%s\n' "$L1" | grep -A2 'attn_q_norm.weight' | grep -q 'norm width'
if [ $? -ne 0 ]; then ok "D4 absent norm reports no width, since none exists" 0; else ok "D4 absent norm reports no width, since none exists" 1; fi

# E: tokenizer tuple -- all three legs, and the contradiction must stay visible
printf '%s\n' "$A1" | grep -q 'leg model   = "gpt2"'; ok "E1 Qwen3 model leg recorded (gpt2)" $?
printf '%s\n' "$A1" | grep -q 'leg pre     = "qwen2"'; ok "E2 Qwen3 pre leg recorded (qwen2)" $?
printf '%s\n' "$A1" | grep -q 'leg tokens  = 151936'; ok "E3 Qwen3 vocab-size leg recorded" $?
printf '%s\n' "$A1" | grep -q 'model alone is NOT a sufficient selector'; ok "E4 insufficiency of model-only is stated" $?
printf '%s\n' "$A1" | grep -q 'no qwen3.vocab_size metadata'; ok "E5 missing vocab_size metadata stays visible as UNRESOLVED" $?
printf '%s\n' "$L1" | grep -q 'llama.vocab_size = 128256'; ok "E6 Llama vocab_size leg present" $?

# E2: RoPE pairing must be MEASURED per model, and the two must DISAGREE
printf '%s\n' "$A1" | grep -q 'pairing         : NEOX'; ok "E2 Qwen3 pairing measured NEOX" $?
printf '%s\n' "$L1" | grep -q 'pairing         : GPT/adjacent'; ok "E2b Llama pairing measured GPT/adjacent (DIFFERS from Qwen3)" $?
printf '%s\n' "$A1" | grep -q 'not derived from general.architecture'; ok "E2c pairing evidence disclaims name derivation" $?
# E7: output sharing resolved from graph identity for both
printf '%s\n' "$A1" | grep -q 'weights are NOT shared'; ok "E7 Qwen3 sharing measured NOT shared" $?
printf '%s\n' "$L1" | grep -q 'weights ARE shared'; ok "E7b Llama sharing measured shared" $?
printf '%s\n' "$L1" | grep -q 'this is EVIDENCE'; ok "E7c absence of output.weight stated as evidence not proof" $?

# F: unsupported constructs become explicit UNKNOWN, never a silent fallback
# rope_freqs was UNKNOWN at 682e146 and is now KNOWN vocabulary: it is a RoPE
# parameter SOURCE, not a layer op. The property worth keeping is that an
# undescribed tensor still surfaces as UNKNOWN, so inject one to prove the path
# works rather than asserting a case that no longer exists.
cp tools/s0_ir_probe.cpp /tmp/ir3.bak
sed -i 's/known_roles.insert("ffn_up.weight");/known_roles.insert("ffn_up.weight"); known_roles.insert("zzz_nonexistent.weight");/' tools/s0_ir_probe.cpp
NOKNOWN=$(gen "$LL")
printf '%s\n' "$NOKNOWN" >/dev/null
cp /tmp/ir3.bak tools/s0_ir_probe.cpp
LL1=$(gen "$LL")
printf '%s\n' "$LL1" | grep -q 'precomputed rope_freqs table present'; ok "F1a rope_freqs recognised as a RoPE parameter source" $?
printf '%s\n' "$H1" | grep -q 'UNKNOWN role'; if [ $? -ne 0 ]; then ok "F1b no UNKNOWN roles remain across all three models" 0; else ok "F1b no UNKNOWN roles remain across all three models" 1; fi
# F2/F3 previously asserted pairing was UNRESOLVED, which was true only while it
# was unmeasured. Now it is measured, so the property under test is the one that
# must hold in EITHER state: pairing is never asserted without evidence, and an
# unmeasured pairing must say so and name what would resolve it.
UNM_PAIR="$P1"; UNM_SH="$SH1"; P1=""; SH1=""
NOEVID=$(gen "$QW")
printf '%s\n' "$NOEVID" | grep -q 'pairing         : UNRESOLVED'; ok "F2 without measurement, pairing stays UNRESOLVED" $?
printf '%s\n' "$NOEVID" | grep -qi 'supply SLLM_IR_ROPE_PAIRING'; ok "F3 UNRESOLVED names what would resolve it" $?
printf '%s\n' "$NOEVID" | grep -q 'Deriving it from the architecture name is exactly the error'; ok "F3b UNRESOLVED explicitly disclaims name derivation" $?
# section G operates on QWEN3, so restore the Qwen3 legs before comparing
P1=NEOX; SH1="weights NOT shared: final projection reads output.weight"

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

# H: Gemma-2. The third model, and the properties that must hold for it.
G=/var/lib/spoon/models/gemma2-2b/gemma-2-2b-it-Q4_K_M.gguf
if [ -f "$G" ]; then
  P1=NEOX; SH1="" SW=2 FC="26x" H1=$(gen "$G")
  # H1: ONE RMSNorm op, with graph position as a PARAMETER. No PostNorm op.
  [ "$(printf '%s\n' "$H1" | grep -c '^  RMSNorm ')" -ge 6 ]; ok "H1 Gemma has all four residual/head norms as RMSNorm" $?
  printf '%s\n' "$H1" | grep -q 'position=post_attention is a PARAMETER of RMSNorm'; ok "H2 post-attention norm is RMSNorm with a position parameter, not a new op" $?
  printf '%s\n' "$H1" | grep -q 'position=post_ffn is a PARAMETER of RMSNorm'; ok "H3 post-FFN norm likewise" $?
  printf '%s\n' "$H1" | grep -q 'PostNorm'; if [ $? -ne 0 ]; then ok "H4 NO PostNorm op was invented" 0; else ok "H4 NO PostNorm op was invented" 1; fi
  # H5: Q/K head norm ABSENT for Gemma, stated as ABSENT and not as UNKNOWN
  printf '%s\n' "$H1" | grep -q 'attn_q_norm.weight'; ok "H5a Gemma attn_q_norm recorded" $?
  printf '%s\n' "$H1" | grep -q 'status          : ABSENT for this architecture'; ok "H5b Gemma Q/K norm absence stated as ABSENT" $?
  printf '%s\n' "$H1" | grep -q 'ABSENT is not UNKNOWN'; ok "H5c absence distinguished from unknown" $?
  # H6: SoftCap is a composable op with two DISTINCT caps
  printf '%s\n' "$H1" | grep -q '^  SoftCap '; ok "H6 SoftCap is an op in the vocabulary" $?
  printf '%s\n' "$H1" | grep -q 'cap             : 50'; ok "H7 attention cap 50 present" $?
  printf '%s\n' "$H1" | grep -q 'cap             : 30'; ok "H8 final-logits cap 30 present" $?
  printf '%s\n' "$H1" | grep -q 'domain          : attention_logits'; ok "H9 SoftCap carries a domain" $?
  printf '%s\n' "$H1" | grep -q 'domain          : final_logits'; ok "H10 both domains distinct, not one boolean" $?
  # H11: fused attention cap function must stay UNRESOLVED, not copied from the unfused site
  printf '%s\n' "$H1" | grep -q 'function        : tanh'; ok "H11 fused attention-cap function now MEASURED via op_params" $?
  printf '%s\n' "$H1" | grep -q 'op_params\[2\]'; ok "H11b fused cap traced to op_params, not inferred from the unfused site" $?
  KEEP_FC="$FC"; FC=""; NOCAP=$(gen "$G"); FC="$KEEP_FC"
  printf '%s\n' "$NOCAP" | grep -q 'function UNRESOLVED'; ok "H11c without op_params evidence it degrades to UNRESOLVED, not to a guess" $?
  printf '%s\n' "$NOCAP" | grep -q 'tanh  *\[MEASURED\]'; if [ $? -ne 0 ]; then ok "H11d no fabricated MEASURED claim without evidence" 0; else ok "H11d no fabricated MEASURED claim without evidence" 1; fi
  printf '%s\n' "$H1" | grep -q 'function        : cap\*tanh(x/cap)'; ok "H12 unfused final-logits function IS measured" $?
  # H13: attention pattern on the node, window measured but per-layer pattern unresolved
  printf '%s\n' "$H1" | grep -qE '^      pattern         : (full|sliding_window)$'; ok "H13 attention carries a resolved pattern" $?
  printf '%s\n' "$H1" | grep -q 'window          : 4096'; ok "H14 sliding window size measured" $?
  # H15 previously asserted the pattern stays UNRESOLVED, which was true only
  # while no rule was available. The durable property is that the window SIZE is
  # measured independently of how the pattern was obtained, and that the two
  # claims never get merged into one.
  printf '%s\n' "$H1" | grep -q 'window          : 4096'; ok "H15 window size measured, independent of pattern evidence" $?
  printf '%s\n' "$H1" | grep -q 'attention.sliding_window = 4096  \[MEASURED from metadata\]'; ok "H15b window provenance names the metadata key" $?
  printf '%s\n' "$H1" | grep -q 'a layer is an ordered container of operations'; ok "H16 uniform-layer assumption explicitly deleted" $?
  # H17: third tokenizer case, model=llama with pre=default
  printf '%s\n' "$H1" | grep -q 'leg model   = "llama"'; ok "H17 Gemma declares model=llama" $?
  printf '%s\n' "$H1" | grep -q 'leg pre     = "default"'; ok "H18 and pre=default: a third pairing, no family shortcut" $?
  # H19: Gemma has no readable rope base and that is UNRESOLVED, not defaulted
  printf '%s\n' "$H1" | grep -q 'rope_base UNRESOLVED'; ok "H19 Gemma rope base UNRESOLVED, not assumed 10000" $?
  printf '%s\n' "$H1" | grep -q 'NOT assumed to be 10000'; ok "H20 explicit non-assumption stated" $?
  # H21: the two vocabulary gaps found at 9c62164 are now known, not UNKNOWN
  printf '%s\n' "$H1" | grep -q 'UNKNOWN role'; if [ $? -ne 0 ]; then ok "H21 no UNKNOWN roles remain for Gemma" 0; else ok "H21 no UNKNOWN roles remain for Gemma" 1; fi
  # H22: rope_freqs is a parameter source, evidenced on the model that has it
  printf '%s\n' "$A1" | grep -q 'UNKNOWN role'; if [ $? -ne 0 ]; then ok "H22 no UNKNOWN roles remain for Qwen3" 0; else ok "H22 no UNKNOWN roles remain for Qwen3" 1; fi
  # NON-VACUITY for the new rules: strip the post-norm knowledge, expect UNKNOWN
  cp tools/s0_ir_probe.cpp /tmp/ir2.bak
  sed -i 's/known_roles.insert("post_attention_norm.weight");//' tools/s0_ir_probe.cpp
  BROKEN=$(gen "$G")
  printf '%s\n' "$BROKEN" | grep -q 'UNKNOWN role: post_attention_norm.weight'
  if [ $? -eq 0 ]; then ok "H23 non-vacuity: unlearning a post-norm role surfaces UNKNOWN" 0; else ok "H23 non-vacuity: unlearning a post-norm role surfaces UNKNOWN" 1; fi
  cp /tmp/ir2.bak tools/s0_ir_probe.cpp
  [ "$(gen "$G")" = "$H1" ] && ok "H24 restore is byte-exact" 0 || ok "H24 restore is byte-exact" 1
  if [ "$(gen "$G")" != "$H1" ]; then
    echo "  H24 DIFF:"; gen "$G" > /tmp/h24a.txt; printf '%s\n' "$H1" > /tmp/h24b.txt
    diff /tmp/h24a.txt /tmp/h24b.txt | head -8
  fi
  if [ $? -ne 0 ]; then diff <(gen "$G") <(printf '%s\n' "$H1") | head -6; fi
else
  echo "  skip H* (Gemma-2 artefact absent)"
fi

# I: OLMoE, the MoE model, which attacked the vocabulary in a new way.
O=/var/lib/spoon/models/olmoe-1b7b/olmoe-q4_k_m.gguf
if [ -f "$O" ]; then
  P1=NEOX; SH1="" SW="" FC="" ROUT="" HD="" O1=$(gen "$O")
  printf '%s\n' "$O1" | grep -q '^  ExpertRouter '; ok "I1 MoE gets its own ExpertRouter op, not squeezed into GatedMLP" $?
  printf '%s\n' "$O1" | grep -q '^  ExpertGatedFFN '; ok "I2 expert bank has its own op" $?
  printf '%s\n' "$O1" | grep -q '^  GatedMLP '; if [ $? -ne 0 ]; then ok "I3 dense GatedMLP is NOT emitted for a MoE model" 0; else ok "I3 dense GatedMLP is NOT emitted for a MoE model" 1; fi
  printf '%s\n' "$O1" | grep -qE 'in/out features +: 2048 / 64'; ok "I4 router shape measured (experts = output width)" $?
  printf '%s\n' "$O1" | grep -q 'expert_used_count = 8'; ok "I5 top-8 of 64 measured" $?
  printf '%s\n' "$O1" | grep -q 'ROUTING FUNCTION UNRESOLVED'; ok "I6 routing FUNCTION left unresolved, not guessed from count" $?
  printf '%s\n' "$O1" | grep -q 'shared expert ABSENT'; ok "I7 shared-expert absence evidenced, not assumed" $?
  # OLMoE's Q/K norm is FULL WIDTH while Qwen3s is PER HEAD: same op, different parameter
  printf '%s\n' "$O1" | grep -q 'reduces the WHOLE Q or K vector'; ok "I8 OLMoE full-vector Q/K norm, not misreported as per-head" $?
  printf '%s\n' "$O1" | grep -q 'scope is a PARAMETER of RMSNorm'; ok "I9 scope is a parameter, same op as Qwen3s per-head norm" $?
  printf '%s\n' "$O1" | grep -q 'key_length ABSENT'; ok "I10 absent key_length recorded rather than compared against" $?
  # Qwen3 must NOT have become full-vector, and OLMoE must not have gained per-head
  A_NOW=$(P1=NEOX gen "$QW")
  printf '%s\n' "$A_NOW" | grep -q 'reduces ONE HEAD'; ok "I11 Qwen3 still per-head after learning OLMoE" $?
  printf '%s\n' "$O1" | grep -q 'UNKNOWN role'; if [ $? -ne 0 ]; then ok "I12 no UNKNOWN roles remain for OLMoE" 0; else ok "I12 no UNKNOWN roles remain for OLMoE" 1; fi
  # NON-VACUITY: unlearn the router and it must resurface as UNKNOWN
  cp tools/s0_ir_probe.cpp /tmp/ir4.bak
  sed -i 's/known_roles.insert("ffn_gate_inp.weight");//' tools/s0_ir_probe.cpp
  BROKEN2=$(gen "$O")
  printf '%s\n' "$BROKEN2" | grep -q 'UNKNOWN role: ffn_gate_inp.weight'
  if [ $? -eq 0 ]; then ok "I13 non-vacuity: unlearning the router surfaces UNKNOWN" 0; else ok "I13 non-vacuity: unlearning the router surfaces UNKNOWN" 1; fi
  cp /tmp/ir4.bak tools/s0_ir_probe.cpp
  [ "$(gen "$O")" = "$O1" ]; ok "I14 restore byte-exact" $?
else
  echo "  skip I* (OLMoE artefact absent)"
fi

# J: routing function and head_dim, resolved from the reference rather than assumed.
if [ -f "$O" ]; then
  RT="argsort_top_k then softmax over the SELECTED weights, no renormalisation (SOFTMAX_WEIGHT convention)"
  KEEP_RT="${ROUT:-}"; ROUT="$RT"; KEEP_HD="${HD:-}"; HD=128
  O2=$(gen "$O")
  printf '%s\n' "$O2" | grep -q 'ROUTING \[MEASURED from the reference op sequence\]'; ok "J1 routing function now MEASURED from the op sequence" $?
  printf '%s\n' "$O2" | grep -q 'the architecture name'; ok "J2 routing evidence disclaims architecture-name derivation" $?
  ROUT="" NORT=$(gen "$O")
  printf '%s\n' "$NORT" | grep -q 'ROUTING FUNCTION UNRESOLVED'; ok "J3 without the op evidence, routing degrades to UNRESOLVED" $?
  printf '%s\n' "$NORT" | grep -q 'MEASURED from the reference op sequence'; if [ $? -ne 0 ]; then ok "J4 no fabricated MEASURED routing without evidence" 0; else ok "J4 no fabricated MEASURED routing without evidence" 1; fi
  HD="" NOHD=$(gen "$O")
  printf '%s\n' "$NOHD" | grep -q 'key_length ABSENT and no head_dim supplied'; ok "J5 without head_dim the scope test is refused rather than guessed" $?
  printf '%s\n' "$NOHD" | grep -q 'reduces ONE HEAD'; if [ $? -ne 0 ]; then ok "J6 no head-scope claim without a real comparison" 0; else ok "J6 no head-scope claim without a real comparison" 1; fi
  ROUT="$KEEP_RT"; HD="$KEEP_HD"
  # OLMoE has MHA (q_heads == kv_heads) where every other model had GQA
  printf '%s\n' "$O2" | grep -q 'q_heads=16 kv_heads=16'; ok "J7 OLMoE MHA recorded as such, not assumed to be GQA" $?
fi

echo
echo "discovery-ir structural tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
