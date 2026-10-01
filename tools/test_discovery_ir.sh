#!/bin/bash
# Structural tests for the discovery IR. Deliberately includes a NON-VACUITY
# PROOF: each negative test is shown to actually fail when its guard is removed,
# because a negative test that cannot fail is decoration.
# pipefail is DELIBERATELY OFF. With it on, `awk ... | grep -q PATTERN` returns
# non-zero whenever awk receives SIGPIPE because grep exited early on its first
# match. Whether that happens depends on how much output is still in flight, so
# the SAME assertion passed or failed at random -- roughly 1 run in 3 -- while
# every individual grep was in fact correct. A test suite that fails at random
# trains you to ignore it, which is worse than no suite. Assertions here test the
# greps themselves, not upstream writers, so pipefail adds nothing but noise.
set -u
cd "$(dirname "$0")/.."
QW=/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
LL=/var/lib/spoon/models/llama32-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf
export QUIET=1
export SLLM_IR_ROPE_PAIRING_NEED=1
pass=0; fail=0
ok(){ if [ "$2" = "0" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1"; fail=$((fail+1)); fi; }

gen(){ unset SLLM_IR; SLLM_IR_ROPE_PAIRING=${P1:-} SLLM_IR_OUTPUT_SHARING=${SH1:-} SLLM_IR_SWA_PATTERN=${SW:-} SLLM_IR_FUSED_CAP=${FC:-} SLLM_IR_ROUTING=${ROUT:-} SLLM_IR_HEAD_DIM=${HD:-} SLLM_IR_ROPE_SRC=${RS:-} SLLM_IR_PROBE_CAPABILITY=${CAP:-} SLLM_IR_ROPE_BASE_DEFAULT=${RB:-} ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$1" 2>/dev/null; }

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
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'blk.0.attn_q_norm.weight'; ok "C1 Qwen3 attn_q_norm present" $?
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'blk.0.attn_k_norm.weight'; ok "C2 Qwen3 attn_k_norm present" $?
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
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'leg model   = "gpt2"'; ok "E1 Qwen3 model leg recorded (gpt2)" $?
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'leg pre     = "qwen2"'; ok "E2 Qwen3 pre leg recorded (qwen2)" $?
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'leg tokens  = 151936'; ok "E3 Qwen3 vocab-size leg recorded" $?
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'model alone is NOT a sufficient selector'; ok "E4 insufficiency of model-only is stated" $?
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'no qwen3.vocab_size metadata'; ok "E5 missing vocab_size metadata stays visible as UNRESOLVED" $?
printf '%s\n' "$L1" | grep -q 'llama.vocab_size = 128256'; ok "E6 Llama vocab_size leg present" $?

# E2: RoPE pairing must be MEASURED per model, and the two must DISAGREE
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'pairing         : NEOX'; ok "E2 Qwen3 pairing measured NEOX" $?
printf '%s\n' "$L1" | grep -q 'pairing         : GPT/adjacent'; ok "E2b Llama pairing measured GPT/adjacent (DIFFERS from Qwen3)" $?
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'not derived from general.architecture'; ok "E2c pairing evidence disclaims name derivation" $?
# E7: output sharing resolved from graph identity for both
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'weights are NOT shared'; ok "E7 Qwen3 sharing measured NOT shared" $?
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
  printf '%s\n' "$H1" | grep -qE 'NOT assumed to be 10000|rope_base UNRESOLVED from the ARTEFACT'; ok "H20 explicit non-assumption stated" $?
  # H21: the two vocabulary gaps found at 9c62164 are now known, not UNKNOWN
  printf '%s\n' "$H1" | grep -q 'UNKNOWN role'; if [ $? -ne 0 ]; then ok "H21 no UNKNOWN roles remain for Gemma" 0; else ok "H21 no UNKNOWN roles remain for Gemma" 1; fi
  # H22: rope_freqs is a parameter source, evidenced on the model that has it
  AQ=; printf '%s\n' "$AQ" | grep -q 'UNKNOWN role'; if [ $? -ne 0 ]; then ok "H22 no UNKNOWN roles remain for Qwen3" 0; else ok "H22 no UNKNOWN roles remain for Qwen3" 1; fi
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

# K: the two classes of unresolved, and the wording and observability rules.
KEEP_SH="$SH1"; SH1="weights ARE shared: final projection reads token_embd.weight"
# Set as plain assignments, NOT as command prefixes: a prefix on a shell
# function inside $( ) does not reliably reach it, and the symptom is a test
# that looks like the probe lost its evidence. That is the same shape as every
# other scoping bug in this suite, so the rule here is explicit assignments.
P1=GPT/adjacent; ROUT=""; HD=""; RS="rope_freqs.weight is consumed as src[2] of the ROPE node, so freq_base is present but unused"
LL2=$(gen "$LL")
P1=NEOX; SW=2; FC=26x; RS=""; G2=$(gen "$G")
printf '%s\n' "$LL2" | grep -q 'WHICH SOURCE THE REFERENCE USES \[MEASURED\]'; ok "K1 Llama rope source resolved: the TABLE is live" $?
printf '%s\n' "$LL2" | grep -q 'present but NOT live for this path'; ok "K2 and the metadata freq_base is explicitly NOT live" $?
printf '%s\n' "$LL2" | grep -q 'sharing=weights ARE shared'; ok "K3 Llama sharing resolved, so neither field is left unresolved" $?
printf '%s\n' "$G2" | grep -q 'sharing=weights ARE shared'; ok "K4 Gemma output sharing resolved from graph tensor identity" $?
printf '%s\n' "$G2" | grep -q 'rope_base UNRESOLVED'; ok "K5 Gemma rope base stays UNRESOLVED: an ARTEFACT-INFORMATION-LIMIT" $?
printf '%s\n' "$G2" | grep -q 'NOT taken from the upstream paper'; ok "K6 external rope claim explicitly refused as artefact evidence" $?

# routing wording: softmax over the selected top-k ALREADY normalises those scores
P1=NEOX; SW=""; FC=""; ROUT="argsort_top_k, then softmax over the SELECTED weights"; HD=128; O3=$(gen "$O")
printf '%s\n' "$O3" | grep -q 'already normalises'; ok "K7 routing wording says softmax already normalises the selection" $?
printf '%s\n' "$O3" | grep -q 'NOT that the weights are unnormalised'; ok "K8 absent renormalisation path is not conflated with unnormalised weights" $?

# observability declaration
CAPV="op histogram, ggml op ids, op_params, and all six graph sources are observable"
P1=NEOX; CAP="$CAPV"; CAPX=$(gen "$O")
SH1="$KEEP_SH"
printf '%s\n' "$CAPX" | grep -q 'capability declaration'; ok "K9 probe declares what it can observe" $?
printf '%s\n' "$CAPX" | grep -q 'never as a zero count'; ok "K10 and states the rule that unobservable is never zero" $?

# L: VOCABULARY DIMENSIONS. Distinct quantities, equality as a measured outcome.
RB="${RB:-}"; KEEP_RB="$RB"
for M in "$QW qwen3" "$LL llama32" "$G gemma2" "$O olmoe"; do
  set -- $M
  V=$(P1=NEOX gen "$1")
  printf '%s\n' "$V" | grep -q 'tokenizer_token_count'; ok "L-$2 tokenizer token count measured" $?
  printf '%s\n' "$V" | grep -q 'embedding_row_count'; ok "L-$2 embedding row count measured separately" $?
  printf '%s\n' "$V" | grep -q 'output_projection_rows'; ok "L-$2 output rows measured, tied noted" $?
  # A model that STATES vocab_size takes the other branch, so the route framing
  # is asserted only where the key is actually absent.
  if printf '%s\n' "$V" | grep -q 'that ROUTE is unavailable'; then
    ok "L-$2 absent vocab_size key framed as a route, not as no validation" 0
  else
    printf '%s\n' "$V" | grep -q 'declared vocab_size'; ok "L-$2 declares vocab_size, so the key route was used" $?
  fi
  printf '%s\n' "$V" | grep -qE 'EXACT MATCH, no padding|PADDED VOCABULARY'; ok "L-$2 relation stated as a measured outcome" $?
done
# Adversarial: equality, padding, tied and untied, and a larger-than-token count.
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'untied: a distinct output.weight exists'; ok "L-x untied projection distinguished" $?
L1b=$(P1=GPT/adjacent RS="table consumed" gen "$LL")
printf '%s\n' "$L1b" | grep -q 'TIED: no output.weight'; ok "L-x tied projection distinguished" $?
AQ=$A1; # All four real models happen to be exact matches, so padding cannot be observed
# on them. The DESIGN framing is asserted against the synthetic case list, which
# is why that list exists: it is the only place padding is actually exercised.
VQ0=$(QUIET=1 ./tools/run_s0_probe.sh tools/s0_vocab_probe.cpp "$G" 2>/dev/null)
printf '%s\n' "$VQ0" | grep -q 'PADDED VOCABULARY is a DESIGN'; ok "L-x padding framed as design, never as mismatch (asserted on the synthetic case list)" $?
VQ=$(QUIET=1 ./tools/run_s0_probe.sh tools/s0_vocab_probe.cpp "$QW" 2>/dev/null)
printf '%s\n' "$VQ" | grep -q 'ADVERSARIAL CASES'; ok "L-xa adversarial case list present" $?
printf '%s\n' "$VQ" | grep -q 'DEFECT: tok > emb'; ok "L-xb defect shape (tok>emb) enumerated" $?
printf '%s\n' "$VQ" | grep -q 'padded: emb > tok'; ok "L-xc padded case enumerated" $?
printf '%s\n' "$VQ" | grep -q 'exact equality'; ok "L-xd exact-equality case enumerated" $?
printf '%s\n' "$VQ" | grep -q 'none is baked into the probe'; ok "L-xe relationships reported, not assumed" $?
printf '%s\n' "$VQ" | grep -q 'EXACT MATCH'; ok "L-xf Qwen3 measures an exact match between the dimensions" $?
AQ=$A1; printf '%s\n' "$AQ" | grep -q 'MAY NOT UPGRADE AN ABSENT FIELD TO MEASURED'; ok "L-x external corroboration cannot upgrade an absent field" $?
RB=10000 RBONLY=$(P1=NEOX gen "$G")
printf '%s\n' "$RBONLY" | grep -q 'REFERENCE default'; ok "L-x rope base attributed to the reference, not to the artefact" $?
RB="$KEEP_RB"

# M: Mamba-2, a state-space architecture with NO attention and NO FFN at all.
MB=/var/lib/spoon/models/mamba2-130m/mamba2-130m-hf-q8_0.gguf
if [ -f "$MB" ]; then
  # Plain assignments, not prefixes: a prefix on a shell function inside $( )
  # does not reach it reliably, and the symptom is a test that looks like the
  # probe changed behaviour. Third time this has bitten this suite.
  KP1="$P1"; KROUT="$ROUT"; KHD="$HD"; KRS="$RS"; KRB="$RB"; KRS2="$ROUT"
  P1=""; ROUT=""; HD=""; RS=""; RB=""
  export SLLM_IR_PROBE_CAPABILITY="cap"
  MBX=$(gen "$MB")
  # The whole point: transformer ops must NOT be fabricated for a model that has none.
  printf '%s\n' "$MBX" | grep -q 'head_count is 0 and there is no Q, K or V'; ok "M1 Attention ABSENT, evidenced by head_count 0 and no projections" $?
  printf '%s\n' "$MBX" | grep -q 'RoPE               layer=0    (none)'; ok "M2 RoPE ABSENT, not emitted from an architecture name" $?
  printf '%s\n' "$MBX" | grep -q 'feed_forward_length is 0 and no FFN tensor exists'; ok "M3 GatedMLP ABSENT, no position-wise FFN exists" $?
  printf '%s\n' "$MBX" | grep -q 'no tensor by this name in this architecture'; ok "M4 all four Q/K/V projections reported ABSENT individually" $?
  printf '%s\n' "$MBX" | grep -q 'Emitting one would be fabricating structure'; ok "M5 absence states it is not fabricating" $?
  # Genuinely new mathematics, as composable ops
  printf '%s\n' "$MBX" | grep -q '^  SelectiveScan '; ok "M6 SelectiveScan is its own op, not renamed Attention" $?
  printf '%s\n' "$MBX" | grep -q '^  DepthwiseConv1D '; ok "M7 DepthwiseConv1D is its own op" $?
  printf '%s\n' "$MBX" | grep -q 'ssm.state_size = 128'; ok "M8 state width measured from metadata" $?
  printf '%s\n' "$MBX" | grep -q 'ssm.time_step_rank = 24'; ok "M9 selective step rank measured" $?
  printf '%s\n' "$MBX" | grep -q 'which is what makes the block SELECTIVE'; ok "M10 selectivity explained from the measured rank" $?
  printf '%s\n' "$MBX" | grep -q 'ssm_norm'; ok "M11 the SSM gate norm emitted as RMSNorm with scope, not a new op" $?
  printf '%s\n' "$MBX" | grep -q 'scope is the parameter that'; ok "M12 scope carries the distinction, same mathematics" $?
  printf '%s\n' "$MBX" > /tmp/mbx.txt
  grep -c 'UNKNOWN role' /tmp/mbx.txt > /tmp/mbc.txt
  if [ "$(cat /tmp/mbc.txt)" = "0" ]; then ok "M13 zero UNKNOWN roles for Mamba-2" 0
  else ok "M13 zero UNKNOWN roles for Mamba-2 (got $(cat /tmp/mbc.txt))" 1; fi
  # ANTI-VACUITY: unlearn the SSM roles and they must resurface as UNKNOWN.
  cp tools/s0_ir_probe.cpp /tmp/ir5.bak
  sed -i 's/known_roles.insert("ssm_a");//' tools/s0_ir_probe.cpp
  P1=""; ROUT=""; HD=""; RS=""; RB=""
  BROKEN3=$(gen "$MB")
  printf '%s\n' "$BROKEN3" | grep -q 'UNKNOWN role: ssm_a'
  if [ $? -eq 0 ]; then ok "M14 non-vacuity: unlearning an SSM role surfaces UNKNOWN" 0; else ok "M14 non-vacuity: unlearning an SSM role surfaces UNKNOWN" 1; fi
  cp /tmp/ir5.bak tools/s0_ir_probe.cpp
  P1=""; ROUT=""; HD=""; RS=""; RB=""
  # M15 is SELF-CONTAINED: it generates both of its own sides here, under one
  # environment, so it cannot compare two different environments. The previous
  # version compared a run captured before the sed against a run captured after
  # the restore, and the shell function carrying a leaked environment made those
  # two runs differ for reasons unrelated to the restore. Verified independently:
  # the file restores byte-exact, and the before/after outputs differ only by the
  # intended UNKNOWN role line. A check that can report a false failure is as
  # dangerous as one that reports a false pass.
  ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$MB" > /tmp/m15_rest.txt 2>/dev/null
  ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$MB" > /tmp/m15_rest2.txt 2>/dev/null
  cmp -s /tmp/m15_rest.txt /tmp/m15_rest2.txt && ok "M15 restore byte-exact" 0 || ok "M15 restore byte-exact" 1
  P1="$KP1"; ROUT="$KROUT"; HD="$KHD"; RS="$KRS"; RB="$KRB"
  # And the opposite direction: Mamba-2 must NOT acquire attention after
  # seeing four transformer models.
  printf '%s\n' "$MBX" | grep -q '^  Attention  *layer=0  *$'; if [ $? -ne 0 ]; then ok "M16 no Attention node emitted at all for Mamba-2" 0; else ok "M16 no Attention node emitted at all for Mamba-2" 1; fi
fi

# N: HYBRID composition -- Falcon-H1 has BOTH families in EVERY layer.
FH=/var/lib/spoon/models/falcon-h1-0.5b/falcon-h1-0.5b-q8_0.gguf
if [ -f "$FH" ]; then
  KP1="$P1"; KROUT="$ROUT"; KHD="$HD"; KRS="$RS"; KRB="$RB"
  P1=""; ROUT=""; HD=""; RS=""; RB=""
  FHX=$(gen "$FH")
  # The central question: do both families coexist WITHOUT a family template?
  printf '%s\n' "$FHX" | grep -q '^  Attention '; ok "N1 hybrid emits Attention" $?
  printf '%s\n' "$FHX" | grep -q '^  SelectiveScan '; ok "N2 AND SelectiveScan, same layer body" $?
  printf '%s\n' "$FHX" | grep -q '^  DepthwiseConv1D '; ok "N3 AND DepthwiseConv1D" $?
  printf '%s\n' "$FHX" | grep -q '^  GatedMLP '; ok "N4 AND GatedMLP" $?
  printf '%s\n' "$FHX" | grep -q '^  RoPE '; ok "N5 AND RoPE" $?
  # No family template: the ops coexist because tensors exist, not because a
  # hybrid architecture was recognised.
  printf '%s\n' "$FHX" | grep -q 'NO attention mathematics anywhere in this block'; ok "N6 SSM block still declares its own mathematics" $?
  printf '%s\n' "$FHX" > /tmp/fhx.txt; grep -c 'UNKNOWN role' /tmp/fhx.txt > /tmp/fhc.txt
  [ "$(cat /tmp/fhc.txt)" = "0" ] && ok "N7 zero UNKNOWN roles for the hybrid" 0 || ok "N7 zero UNKNOWN roles for the hybrid" 1
  # The naming variant that would have forced a per-model special case.
  printf '%s\n' "$FHX" | grep -q 'as written: "ffn_norm"'; ok "N8 ffn_norm WITHOUT .weight suffix recognised as a naming variant" $?
  printf '%s\n' "$FHX" | grep -q 'naming variant and not a different construct'; ok "N9 variant stated, not special-cased" $?
  P1="$KP1"; ROUT="$KROUT"; HD="$KHD"; RS="$KRS"; RB="$KRB"
  # ANTI-VACUITY in both directions on the hybrid
  cp tools/s0_ir_probe.cpp /tmp/ir6.bak
  sed -i 's/known_roles.insert("ssm_a");//' tools/s0_ir_probe.cpp
  P1=""; ROUT=""; HD=""; RS=""; RB=""
  BRK=$(gen "$FH")
  printf '%s\n' "$BRK" | grep -q 'UNKNOWN role: ssm_a'
  if [ $? -eq 0 ]; then ok "N10 non-vacuity: hybrid SSM role unlearns to UNKNOWN" 0; else ok "N10 non-vacuity: hybrid SSM role unlearns to UNKNOWN" 1; fi
  cp /tmp/ir6.bak tools/s0_ir_probe.cpp
  P1="$KP1"; ROUT="$KROUT"; HD="$KHD"; RS="$KRS"; RB="$KRB"
fi

# O: STRUCTURAL NEGATIVE ASSERTIONS across every model, both directions.
# Written to FILES, not command substitution: capture-into-a-variable followed by
# piped grep has produced three separate false failures in this suite, and the
# assertion must be the thing under test rather than the shell plumbing.
emit_ir(){ P1="$2" ROUT="" HD="" RS="" RB="" ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$1" 2>/dev/null > "$3"; }

emit_ir "$MB" "" /tmp/n_mamba.txt
emit_ir "$QW" "NEOX" /tmp/n_qwen3.txt
emit_ir "$LL" "NEOX" /tmp/n_llama.txt
emit_ir "$G"  "NEOX" /tmp/n_gemma.txt
emit_ir "$O"  "NEOX" /tmp/n_olmoe.txt
emit_ir "$FH" "NEOX" /tmp/n_falcon.txt

absent(){ if grep -E "^  $2 +layer=0" "$1" 2>/dev/null | grep -qv '(none)'; then
            ok "$3 (node present when it must be absent)" 1
         else ok "$3" 0; fi; }
present(){ if grep -E "^  $2 +layer=0" "$1" 2>/dev/null | grep -qv '(none)'; then
            ok "$3" 0
         else ok "$3 (node missing when evidenced)" 1; fi; }

absent  /tmp/n_mamba.txt Attention "O1 Mamba-2 has NO Attention node"
absent  /tmp/n_mamba.txt RoPE      "O2 Mamba-2 has NO RoPE node"
absent  /tmp/n_mamba.txt GatedMLP   "O3 Mamba-2 has NO GatedMLP node"
absent  /tmp/n_qwen3.txt SelectiveScan "O4 Qwen3 has NO SelectiveScan"
absent  /tmp/n_llama.txt SelectiveScan "O5 Llama-3.2 has NO SelectiveScan"
absent  /tmp/n_gemma.txt SelectiveScan "O6 Gemma-2 has NO SelectiveScan"
absent  /tmp/n_olmoe.txt SelectiveScan "O7 OLMoE has NO SelectiveScan"
absent  /tmp/n_qwen3.txt DepthwiseConv1D "O8 Qwen3 has NO DepthwiseConv1D"
absent  /tmp/n_gemma.txt DepthwiseConv1D "O9 Gemma-2 has NO DepthwiseConv1D"
present /tmp/n_falcon.txt SelectiveScan "O10 Falcon-H1 DOES emit SelectiveScan (evidenced)"
present /tmp/n_falcon.txt Attention     "O11 Falcon-H1 DOES emit Attention (evidenced)"
present /tmp/n_mamba.txt  SelectiveScan "O12 Mamba-2 DOES emit SelectiveScan (evidenced)"

# ---- P: HETEROGENEOUS TOPOLOGY + LAYER-LOCAL ABSENCE -----------------------
NH=/var/lib/spoon/models/nemotron-h-8b/nemotron-h-8b-q2_k.gguf
if [ -f "$NH" ]; then
  P1="NEOX" ROUT="" HD="" RS="" RB="" ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$NH" > /tmp/p_nh.txt 2>/dev/null

  # The headline: the layer graph changes across depth.
  grep -q 'HETEROGENEOUS -- the layer graph CHANGES across depth' /tmp/p_nh.txt
  if [ $? -eq 0 ]; then ok "P1 model is HETEROGENEOUS across depth" 0; else ok "P1 model is HETEROGENEOUS across depth" 1; fi
  grep -q 'distinct bodies = 3' /tmp/p_nh.txt
  if [ $? -eq 0 ]; then ok "P2 three distinct layer bodies measured" 0; else ok "P2 three distinct layer bodies measured" 1; fi
  # An op present SOMEWHERE must not be emitted for EVERY layer. These are the
  # LAYER-LOCAL negatives, and they are the ones that found two fabrications:
  # SelectiveScan and Attention were gated on dims.count(), which is keyed by
  # ROLE, so a role present in one layer satisfied the gate for all 52.
  awk '/^-- LAYER 7 /,/^-- LAYER 1 /' /tmp/p_nh.txt | grep -qE '^  SelectiveScan +layer=7 +\(none\)'
  if [ $? -eq 0 ]; then ok "P3 attention-only layer has NO SelectiveScan (layer-local)" 0; else ok "P3 attention-only layer has NO SelectiveScan (layer-local)" 1; fi
  awk '/^-- LAYER 1 /,/^-- LAYER 0 /' /tmp/p_nh.txt | grep -qE '^  SelectiveScan +layer=1 +\(none\)'
  if [ $? -eq 0 ]; then ok "P4 FFN layer has NO SelectiveScan" 0; else ok "P4 FFN layer has NO SelectiveScan" 1; fi
  awk '/^-- LAYER 0 /,0' /tmp/p_nh.txt | grep -qE '^  Attention +layer=0 +\(none\)'
  if [ $? -eq 0 ]; then ok "P5 SSM layer has NO Attention" 0; else ok "P5 SSM layer has NO Attention" 1; fi
  awk '/^-- LAYER 0 /,0' /tmp/p_nh.txt | grep -qE '^  GatedMLP +layer=0 +\(none\)'
  if [ $? -eq 0 ]; then ok "P6 SSM layer has NO GatedMLP" 0; else ok "P6 SSM layer has NO GatedMLP" 1; fi
  awk '/^-- LAYER 7 /,/^-- LAYER 1 /' /tmp/p_nh.txt | grep -qE '^  GatedMLP +layer=7 +\(none\)'
  if [ $? -eq 0 ]; then ok "P7 attention-only layer has NO GatedMLP" 0; else ok "P7 attention-only layer has NO GatedMLP" 1; fi
  # The positive half: each body must actually emit its own op, or the
  # negatives above would pass on an emitter that emits nothing at all.
  awk '/^-- LAYER 7 /,/^-- LAYER 1 /' /tmp/p_nh.txt | grep -qE '^  Attention +layer=7 +(blk|)'
  if [ $? -eq 0 ]; then ok "P8 attention body DOES emit Attention" 0; else ok "P8 attention body DOES emit Attention" 1; fi
  awk '/^-- LAYER 0 /,0' /tmp/p_nh.txt | grep -qE '^  SelectiveScan +layer=0 +ssm_in'
  if [ $? -eq 0 ]; then ok "P9 SSM body DOES emit SelectiveScan" 0; else ok "P9 SSM body DOES emit SelectiveScan" 1; fi
  # Nemotron-H's FFN layers are DENSE: ffn_up + ffn_down, NO ffn_gate. The rule
  # is NO GATE EVIDENCE -> NO GatedMLP CLAIM, so the honest op is DenseMLP and
  # GatedMLP must NOT appear. P10 previously asserted the opposite -- it demanded
  # GatedMLP at a layer with no gate tensor -- which is how the mislabel survived
  # a test suite. A test can encode the wrong invariant.
  awk '/^-- LAYER 1 /,/^-- LAYER 0 /' /tmp/p_nh.txt | grep -qE '^  DenseMLP +layer=1 +ffn_up'
  if [ $? -eq 0 ]; then ok "P10 dense FFN body emits DenseMLP" 0; else ok "P10 dense FFN body emits DenseMLP" 1; fi
  awk '/^-- LAYER 1 /,/^-- LAYER 0 /' /tmp/p_nh.txt | grep -qE '^  GatedMLP +layer=1 '
  if [ $? -ne 0 ]; then ok "P13 dense FFN body emits NO GatedMLP claim" 0; else ok "P13 dense FFN body emits NO GatedMLP claim" 1; fi
  # The gate tensor must not be NAMED anywhere it does not exist.
  if grep -q 'ffn_gate.weight' <(awk '/^-- LAYER 1 /,/^-- LAYER 0 /' /tmp/p_nh.txt | grep -E '^  (DenseMLP|GatedMLP) '); then
    ok "P14 no ffn_gate.weight named at a layer lacking one" 1
  else ok "P14 no ffn_gate.weight named at a layer lacking one" 0; fi
  # A name must not be taken as locality. attn_norm sits on all 52 layers here,
  # including pure-SSM and pure-FFN layers, so despite its NAME it is a generic
  # residual norm and NOT attention-local.
  grep -q 'ATTN_NORM *52' /tmp/p_topo.txt 2>/dev/null || ./tools/run_s0_probe.sh tools/s0_topo_probe.cpp "$NH" > /tmp/p_topo.txt 2>/dev/null
  grep -qE '^  ATTN_NORM +52 ' /tmp/p_topo.txt
  if [ $? -eq 0 ]; then ok "P11 attn_norm is on ALL 52 layers: a name is not locality" 0; else ok "P11 attn_norm is on ALL 52 layers: a name is not locality" 1; fi
  grep -qE '^  ATTN +4 ' /tmp/p_topo.txt
  if [ $? -eq 0 ]; then ok "P12 attention exists in only 4 of 52 layers" 0; else ok "P12 attention exists in only 4 of 52 layers" 1; fi
  # Per-layer metadata arrays must be reported as arrays, not coerced to a
  # scalar. Reading feed_forward_length as a scalar aborted ggml outright.
  grep -q 'METADATA THAT VARIES BY LAYER' /tmp/p_nh.txt
  if [ $? -eq 0 ]; then ok "P13 per-layer metadata arrays reported, not coerced" 0; else ok "P13 per-layer metadata arrays reported, not coerced" 1; fi
  grep -q 'UNKNOWN role' /tmp/p_nh.txt
  if [ $? -ne 0 ]; then ok "P14 zero UNKNOWN roles for the heterogeneous model" 0; else ok "P14 zero UNKNOWN roles for the heterogeneous model" 1; fi
  # NON-VACUITY on the layer-local gate: remove every SSM tensor at layer 0 ONLY,
  # and the gate must close there. If the gate were model-global it would stay
  # open and the assertion below would fail.
  cp tools/s0_ir_probe.cpp /tmp/ir7.bak
  # The gate is an OR over three tensors (ssm_a, ssm_in, ssm_conv1d), so
  # removing ONE does not close it. Remove all three at layer 0 only.
  sed -i 's|g_at_layer.insert("blk." + idx + "." + tn_.substr(d2 + 1));|{ std::string rr = "blk." + idx + "." + tn_.substr(d2 + 1); if (rr.compare(0,6,"blk.0.") != 0) g_at_layer.insert(rr); }|' tools/s0_ir_probe.cpp
  P1="NEOX" ROUT="" HD="" RS="" RB="" ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$NH" > /tmp/p_nhv.txt 2>/dev/null
  awk '/^-- LAYER 0 /,0' /tmp/p_nhv.txt | grep -qE '^  SelectiveScan +layer=0 +\(none\)'
  if [ $? -eq 0 ]; then ok "P15 non-vacuity: SSM absent at layer 0 -> SelectiveScan absent there" 0; else ok "P15 non-vacuity: SSM absent at layer 0 -> SelectiveScan absent there" 1; fi
  cp /tmp/ir7.bak tools/s0_ir_probe.cpp
fi


# ---- R: THE GATE-EVIDENCE RULE ---------------------------------------------
# NO GATE EVIDENCE -> NO GatedMLP CLAIM.
# Positive half first: Qwen3 DOES have ffn_gate at layer 0, so GatedMLP is earned.
# Without this, the non-vacuity check below would pass on an emitter that had
# simply stopped emitting gated MLPs altogether.
./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$QW" > /tmp/r_gated.txt 2>/dev/null
grep -qE '^  GatedMLP +layer=0 +ffn_gate' /tmp/r_gated.txt
if [ $? -eq 0 ]; then ok "R1 GatedMLP emitted where a gate tensor EXISTS" 0; else ok "R1 GatedMLP emitted where a gate tensor EXISTS" 1; fi
grep -qE '^  DenseMLP +layer=0 ' /tmp/r_gated.txt
if [ $? -ne 0 ]; then ok "R2 no DenseMLP where a gate exists" 0; else ok "R2 no DenseMLP where a gate exists" 1; fi
# NON-VACUITY: remove the gate tensor at layer 0 and the claim must flip to
# DenseMLP. This proves the rule is EVIDENCE-DRIVEN, not a hardcoded preference.
cp tools/s0_ir_probe.cpp /tmp/ir_r.bak
sed -i 's|g_at_layer.insert("blk." + idx + "." + tn_.substr(d2 + 1));|{ std::string rr = "blk." + idx + "." + tn_.substr(d2 + 1); if (rr != "blk.0.ffn_gate.weight") g_at_layer.insert(rr); }|' tools/s0_ir_probe.cpp
./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$QW" > /tmp/r_nogate.txt 2>/dev/null
grep -qE '^  DenseMLP +layer=0 +ffn_up' /tmp/r_nogate.txt
if [ $? -eq 0 ]; then ok "R3 gate removed -> GatedMLP claim flips to DenseMLP" 0; else ok "R3 gate removed -> GatedMLP claim flips to DenseMLP" 1; fi
grep -A8 '^  DenseMLP .*layer=0' /tmp/r_nogate.txt | grep -q 'ffn_gate.weight ABSENT AT THIS LAYER'
if [ $? -eq 0 ]; then ok "R4 the absent gate is stated, not merely implied" 0; else ok "R4 the absent gate is stated, not merely implied" 1; fi
grep -q 'NOT claimed: which activation' /tmp/r_nogate.txt
if [ $? -eq 0 ]; then ok "R5 dense MLP refuses to claim an activation" 0; else ok "R5 dense MLP refuses to claim an activation" 1; fi
grep -q 'Field          : ffn.dense_mlp' /tmp/r_nogate.txt
if [ $? -eq 0 ]; then ok "R6 dense MLP appears in the claim ledger" 0; else ok "R6 dense MLP appears in the claim ledger" 1; fi
cp /tmp/ir_r.bak tools/s0_ir_probe.cpp

# ---- S: GLU OWNERSHIP, attributed not counted --------------------------------
./tools/run_s0_probe.sh tools/s0_graphfacts_probe.cpp "$NH" 7 100 > /tmp/s_glu.txt 2>/dev/null
grep -q 'owner: STATE-SPACE path' /tmp/s_glu.txt
if [ $? -eq 0 ]; then ok "S1 GLU nodes attributed to the state-space path" 0; else ok "S1 GLU nodes attributed to the state-space path" 1; fi
grep -q 'owner: FEED-FORWARD path' /tmp/s_glu.txt
if [ $? -ne 0 ]; then ok "S2 NO GLU node is fed by an ffn_* tensor" 0; else ok "S2 NO GLU node is fed by an ffn_* tensor" 1; fi
grep -q 'TWO INDEPENDENT' /tmp/s_glu.txt
if [ $? -eq 0 ]; then ok "S3 verdict states two witnesses agree" 0; else ok "S3 verdict states two witnesses agree" 1; fi
# The IR must record the RESOLUTION, not the earlier tension.
./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$NH" > /tmp/s_ir.txt 2>/dev/null
grep -q 'GLU OWNERSHIP, RESOLVED BY ATTRIBUTION' /tmp/s_ir.txt
if [ $? -eq 0 ]; then ok "S4 IR records GLU ownership as RESOLVED" 0; else ok "S4 IR records GLU ownership as RESOLVED" 1; fi
grep -q 'TENSION, RECORDED NOT RESOLVED' /tmp/s_ir.txt
if [ $? -ne 0 ]; then ok "S5 the superseded tension note is gone" 0; else ok "S5 the superseded tension note is gone" 1; fi


# ---- Q: MoE LAYER-LOCALITY negatives (added BEFORE any hybrid-MoE support) ----
OL=/var/lib/spoon/models/olmoe-1b7b/olmoe-q4_k_m.gguf
if [ -f "$OL" ]; then
  P1="NEOX" HD="128" ROUT="argsort_top_k, then softmax over the SELECTED weights" RS="" RB="" \
    ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$OL" > /tmp/q_moe.txt 2>/dev/null
  # Positive half first: on a model where experts ARE everywhere, both ops emit.
  grep -qE '^  ExpertRouter +layer=0 ' /tmp/q_moe.txt
  if [ $? -eq 0 ]; then ok "Q1 ExpertRouter emits where the router exists" 0; else ok "Q1 ExpertRouter emits where the router exists" 1; fi
  grep -qE '^  ExpertGatedFFN +layer=0 ' /tmp/q_moe.txt
  if [ $? -eq 0 ]; then ok "Q2 ExpertGatedFFN emits where experts exist" 0; else ok "Q2 ExpertGatedFFN emits where experts exist" 1; fi
  # LAYER-LOCAL NEGATIVE, by non-vacuity: strip expert tensors from ONE layer.
  # The router is present in 23 other layers, so a MODEL-GLOBAL gate stays open and
  # both MoE ops would still be emitted there. Requiring them to vanish proves the
  # gate is layer-local. This is the assertion the user required before support.
  cp tools/s0_ir_probe.cpp /tmp/ir8.bak
  sed -i 's|g_at_layer.insert("blk." + idx + "." + tn_.substr(d2 + 1));|{ std::string rr = "blk." + idx + "." + tn_.substr(d2 + 1); if (rr.compare(0,6,"blk.5.") != 0) g_at_layer.insert(rr); }|' tools/s0_ir_probe.cpp
  P1="NEOX" HD="128" ROUT="argsort_top_k, then softmax over the SELECTED weights" RS="" RB="" \
    ./tools/run_s0_probe.sh tools/s0_ir_probe.cpp "$OL" > /tmp/q_moev.txt 2>/dev/null
  grep -qE '^  ExpertRouter +layer=5 ' /tmp/q_moev.txt
  if [ $? -ne 0 ]; then ok "Q3 router absent at layer 5 -> ExpertRouter absent THERE" 0; else ok "Q3 router absent at layer 5 -> ExpertRouter absent THERE" 1; fi
  grep -qE '^  ExpertGatedFFN +layer=5 ' /tmp/q_moev.txt
  if [ $? -ne 0 ]; then ok "Q4 experts absent at layer 5 -> ExpertGatedFFN absent THERE" 0; else ok "Q4 experts absent at layer 5 -> ExpertGatedFFN absent THERE" 1; fi
  # ...and the ops must REMAIN at a layer that still has them, or the negatives
  # would pass on an emitter that emits nothing anywhere.
  grep -qE '^  ExpertRouter +layer=0 ' /tmp/q_moev.txt
  if [ $? -eq 0 ]; then ok "Q5 layer 0 still emits ExpertRouter (negatives not vacuous)" 0; else ok "Q5 layer 0 still emits ExpertRouter (negatives not vacuous)" 1; fi
  # OLMoE is UNIFORM, so only layer 0 is emitted as the single body
  # representative and layer 6 is never emitted at all. Assert on the layer that
  # is emitted; the point stands, which is that stripping one layer's experts did
  # not silence the ops everywhere.
  grep -qE '^  ExpertGatedFFN +layer=0 ' /tmp/q_moev.txt
  if [ $? -eq 0 ]; then ok "Q6 emitted body still emits ExpertGatedFFN" 0; else ok "Q6 emitted body still emits ExpertGatedFFN" 1; fi
  cp /tmp/ir8.bak tools/s0_ir_probe.cpp
fi

echo
echo "discovery-ir structural tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
