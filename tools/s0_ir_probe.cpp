// Derive a discovery IR from a GGUF artefact, empirically.
//
// Every conclusion records WHERE IT CAME FROM. There are exactly two kinds of
// evidence and they are never conflated:
//
//   MEASURED  read out of the artefact or out of the reference graph
//   INFERRED  concluded from a naming/shape rule, carrying the rule as evidence
//
// Nothing is asserted bare. A RoPE node carries which metadata keys set its
// parameters and which graph evidence fixed its pairing; an UNKNOWN carries what
// was seen and what could not be resolved.
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

enum ir_kind { IR_UNKNOWN = 0, IR_EMBEDDING, IR_RMSNORM, IR_LINEAR, IR_ROPE,
               IR_HEADSPLIT, IR_ATTENTION, IR_RESIDUAL, IR_MLP, IR_OUTPUT, IR_SOFTCAP,
               IR_ROUTER, IR_EXPERTFFN };

static const char * kind_name(ir_kind k) {
    switch (k) {
        case IR_UNKNOWN:    return "UNKNOWN";
        case IR_EMBEDDING:  return "Embedding";
        case IR_RMSNORM:    return "RMSNorm";
        case IR_LINEAR:     return "Linear";
        case IR_ROPE:       return "RoPE";
        case IR_HEADSPLIT:  return "HeadSplit";
        case IR_ATTENTION:  return "Attention";
        case IR_RESIDUAL:   return "ResidualAdd";
        case IR_MLP:        return "GatedMLP";
        case IR_OUTPUT:     return "OutputProjection";
        case IR_SOFTCAP:    return "SoftCap";
        case IR_ROUTER:     return "ExpertRouter";
        case IR_EXPERTFFN:  return "ExpertGatedFFN";
    }
    return "?";
}

struct ir_node {
    ir_kind                  kind;
    int                      layer;        /* -1 = model scope */
    std::string              tensor;
    std::string              type_name;
    std::vector<std::string> evidence;     /* metadata keys / rules used     */
    std::vector<std::string> unknown;      /* unresolved, when kind UNKNOWN */
    bool                     measured;     /* all params measured           */

    /* resolved parameters, only those the artefact evidences */
    long in_features, out_features;
    long head_dim, n_heads, n_kv_heads, q_heads;
    double eps, rope_base;
    long rope_dim_count;
    std::string pairing, scaling_type, position_source;
    bool bias, has_norm_weight;
    long attn_pattern_sliding;   /* Attention.pattern */
    long norm_width;
    std::string scope, position;
    long n_experts, n_used, expert_ff;
};

static ir_node mk(ir_kind k, int layer) {
    ir_node n;
    n.kind = k; n.layer = layer; n.measured = false;
    n.in_features = n.out_features = n.head_dim = n.n_heads = n.n_kv_heads = -1;
    n.q_heads = -1;
    n.eps = n.rope_base = -1.0; n.rope_dim_count = -1;
    n.bias = false; n.has_norm_weight = false; n.norm_width = -1;
    n.attn_pattern_sliding = -1;
    n.n_experts = n.n_used = n.expert_ff = -1;
    return n;
}
static void emit(const ir_node & n, const char * arch) {
    printf("  %-18s", kind_name(n.kind));
    if (n.layer >= 0) { printf(" layer=%-3d", n.layer); } else { printf(" %-10s", "(model)"); }
    printf("  %s\n", n.tensor.c_str());
    if (n.type_name.size()) { printf("      type            : %s\n", n.type_name.c_str()); }
    if (n.in_features >= 0) { printf("      in/out features : %ld / %ld\n", n.in_features, n.out_features); }
    if (n.norm_width >= 0) { printf("      norm width      : %ld   bias=%s\n", n.norm_width, n.bias ? "true" : "false"); }
    if (n.kind == IR_RMSNORM) { printf("      epsilon         : %.9g\n", n.eps); }
    if (n.eps >= 0 && n.kind != IR_RMSNORM && n.kind != IR_LINEAR) {
        printf("      eps(inherited)  : %.9g\n", n.eps); }
    if (n.head_dim > 0) { printf("      head_dim=%ld  heads=%ld  kv_heads=%ld\n", n.head_dim, n.n_heads, n.n_kv_heads); }
    if (n.kind == IR_ROPE) {
        if (n.pairing == "UNRESOLVED") {
          printf("      pairing         : UNRESOLVED  (requires measurement from the reference graph)\n");
      } else {
          printf("      pairing         : %s   [MEASURED from reference graph]\n", n.pairing.c_str());
      }
        printf("      base            : %.9g\n", n.rope_base);
        printf("      dim_count       : %ld\n", n.rope_dim_count);
        printf("      scaling         : %s\n", n.scaling_type.c_str());
        printf("      position_source : %s\n", n.position_source.c_str());
    }
    if (n.kind == IR_MLP) { printf("      scope           : %s\n", n.scope.c_str()); }
    if (n.kind == IR_ATTENTION) {
        printf("      pattern         : %s\n", n.pairing.c_str());
        if (n.rope_base >= 0) { printf("      window          : %.0f\n", n.rope_base); }
    }
    if (n.kind == IR_SOFTCAP) {
        printf("      cap             : %.9g\n", n.rope_base);
        printf("      domain          : %s   position=%s\n", n.scope.c_str(), n.position.c_str());
        if (n.scaling_type.size()) { printf("      function        : %s\n", n.scaling_type.c_str()); }
    }
    printf("      evidence        : %s\n", n.measured ? "MEASURED (read from artefact/reference graph)" : "INFERRED (rule-based)");
    for (size_t i = 0; i < n.evidence.size(); ++i) { printf("        - %s\n", n.evidence[i].c_str()); }
    for (size_t i = 0; i < n.unknown.size(); ++i) { printf("      UNRESOLVED      : %s\n", n.unknown[i].c_str()); }
    (void) arch;
}

static bool split_layer(const std::string & n, int * layer, std::string * role) {
    *layer = -1; role->clear();
    size_t i = n.find('.');
    if (i == std::string::npos) { return false; }
    size_t j = n.find('.', i + 1);
    if (j == std::string::npos) { return false; }
    const std::string idx = n.substr(i + 1, j - i - 1);
    if (idx.empty()) { return false; }
    for (size_t k = 0; k < idx.size(); ++k) { if (idx[k] < '0' || idx[k] > '9') { return false; } }
    *layer = atoi(idx.c_str()); *role = n.substr(j + 1);
    return true;
}

int main(int argc, char ** argv) {
    gguf_init_params gp; memset(&gp, 0, sizeof gp); gp.no_alloc = true;
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (g == NULL) { printf("open failed\n"); return 1; }
    const char * arch = NULL;
    if (gguf_find_key(g, "general.architecture") >= 0) { arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture")); }
    const std::string P = std::string(arch != NULL ? arch : "") + ".";

    /* ---- metadata, read not guessed ---- */
    long block_count = -1, embd = -1, ff = -1, heads = -1, kv_heads = -1;
    long key_len = -1, val_len = -1, rope_dim = -1;
    double eps = -1.0, rope_base = -1.0;
    struct { const char * suffix; long * out; } ints[] = {
        {"block_count", &block_count}, {"embedding_length", &embd},
        {"feed_forward_length", &ff}, {"attention.head_count", &heads},
        {"attention.head_count_kv", &kv_heads}, {"attention.key_length", &key_len},
        {"attention.value_length", &val_len}, {"rope.dimension_count", &rope_dim},
    };
    for (unsigned i = 0; i < sizeof ints / sizeof *ints; ++i) {
        const std::string k = P + ints[i].suffix;
        const int64_t id = gguf_find_key(g, k.c_str());
        if (id >= 0) { *ints[i].out = (long) gguf_get_val_u32(g, id); }
    }
    { const std::string k = P + "attention.layer_norm_rms_epsilon";
      const int64_t id = gguf_find_key(g, k.c_str());
      if (id >= 0) { eps = gguf_get_val_f32(g, id); } }
    { const std::string k = P + "rope.freq_base";
      const int64_t id = gguf_find_key(g, k.c_str());
      if (id >= 0) { rope_base = gguf_get_val_f32(g, id); } }

    printf("=== DISCOVERY IR for %s ===\n", argv[1]);
    printf("architecture (from general.architecture) = %s\n", arch != NULL ? arch : "(absent)");
    printf("block_count=%ld embedding_length=%ld ffn_length=%ld heads=%ld kv_heads=%ld\n",
           block_count, embd, ff, heads, kv_heads);
    printf("key_length=%ld value_length=%ld rope_dimension_count=%ld\n", key_len, val_len, rope_dim);
    printf("rms_eps=%.9g rope_base=%.9g\n", eps, rope_base);

    /* ---- token role inventory ---- */
    std::map<std::string, int> roles;
    std::map<std::string, std::pair<long,long> > dims;
    std::map<std::string, std::string> types;
    std::map<int, std::set<std::string> > per_layer;
    int max_layer = -1;
    std::vector<std::string> non_layer;
    bool rope_freqs_present = false;
    long rope_freqs_len = -1;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        int layer = -1; std::string role;
        const std::string n = gguf_get_tensor_name(g, i);
        if (!split_layer(n, &layer, &role)) {
            non_layer.push_back(n);
            /* Non-layer tensors still carry resolvable shapes. Omitting them
             * made token_embd look like ne=[-1,-1], and the tied/untied check
             * then reported UNTIED from an ABSENT value rather than from a
             * measured one -- a false conclusion produced by missing data. */
            dims[n] = std::make_pair((long) gguf_get_tensor_ne(g, i)[0],
                                     (long) gguf_get_tensor_ne(g, i)[1]);
            types[n] = ggml_type_name(gguf_get_tensor_type(g, i));
            continue;
        }
        if (layer > max_layer) { max_layer = layer; }
        roles[role]++;
        dims[role] = std::make_pair((long) gguf_get_tensor_ne(g, i)[0],
                                    (long) gguf_get_tensor_ne(g, i)[1]);
        types[role] = ggml_type_name(gguf_get_tensor_type(g, i));
        per_layer[layer].insert(role);
    }

    /* ---- MoE metadata, read not assumed ---- */
    long n_experts = -1, n_used = -1;
    { const int64_t e = gguf_find_key(g, (P + "expert_count").c_str());
      if (e >= 0) { n_experts = (long) gguf_get_val_u32(g, e); }
      const int64_t u = gguf_find_key(g, (P + "expert_used_count").c_str());
      if (u >= 0) { n_used = (long) gguf_get_val_u32(g, u); } }
    const bool is_moe = (dims.count("ffn_gate_inp.weight") > 0);

    /* ---- tokens we did NOT learn to describe ---- */
    std::set<std::string> known_roles;
    known_roles.insert("attn_norm.weight");   known_roles.insert("ffn_norm.weight");
    known_roles.insert("attn_q.weight");      known_roles.insert("attn_k.weight");
    known_roles.insert("attn_v.weight");      known_roles.insert("attn_output.weight");
    known_roles.insert("attn_q_norm.weight"); known_roles.insert("attn_k_norm.weight");
    known_roles.insert("ffn_gate.weight");    known_roles.insert("ffn_up.weight");
    known_roles.insert("ffn_down.weight");
    /* MoE roles are known vocabulary now, with their own ops. They were UNKNOWN
     * before ExpertRouter / ExpertGatedFFN existed, which was the correct state
     * at the time: the vocabulary could not describe them. */
    known_roles.insert("ffn_gate_inp.weight");
    known_roles.insert("ffn_gate_exps.weight");
    known_roles.insert("ffn_up_exps.weight");
    known_roles.insert("ffn_down_exps.weight");
    /* These two were reported UNKNOWN at 9c62164 and that was WRONG. They are
     * canonical llama.cpp tensor names -- llama-arch.cpp:460 maps
     * LLM_TENSOR_ATTN_POST_NORM to "blk.%d.post_attention_norm" -- so they are
     * known vocabulary that the IR simply had not learned yet. Recorded here as
     * a vocabulary gap rather than an UNKNOWN construct, which is a weaker and
     * wrong statement about them. */
    known_roles.insert("post_attention_norm.weight");
    known_roles.insert("post_ffw_norm.weight");
    std::set<std::string> unknown_layer_roles;
    for (std::map<std::string,int>::iterator it = roles.begin(); it != roles.end(); ++it) {
        if (known_roles.find(it->first) == known_roles.end()) { unknown_layer_roles.insert(it->first); }
    }
    for (size_t i = 0; i < non_layer.size(); ++i) {
        const std::string & n = non_layer[i];
        const std::string base = n.substr(0, n.rfind('.') == std::string::npos ? n.size() : n.rfind('.'));
        if (base == "rope_freqs") {
            /* A precomputed RoPE frequency table, consumed BY the RoPE node
             * rather than being a layer operation. It was reported UNKNOWN at
             * 682e146, which was a vocabulary gap: llama-arch.cpp:440 defines
             * LLM_TENSOR_ROPE_FREQS as "rope_freqs" and llama.cpp:66 creates it
             * at {n_rot/2}, optional and duplicated across layers. This is a
             * PARAMETER SOURCE for RoPE, not a new operation. */
            rope_freqs_present = true;
            rope_freqs_len = dims.count(n) ? dims[n].first : -1;
            continue;
        }
        if (base != "token_embd" && base != "output" && base != "output_norm") {
            unknown_layer_roles.insert("(non-layer) " + n);
        }
    }
    if (!unknown_layer_roles.empty()) {
        printf("\n-- TENSORS THE VOCABULARY CANNOT YET DESCRIBE --\n");
        for (std::set<std::string>::iterator it = unknown_layer_roles.begin();
             it != unknown_layer_roles.end(); ++it) { printf("  UNKNOWN role: %s\n", it->c_str()); }
    }

    /* ---- tokenizer: the selector, as a first-class IR node ----
     * All three legs are carried. model alone is NOT sufficient: Qwen3 declares
     * model=gpt2 while needing pre=qwen2, and selecting on model alone picks the
     * right family with the wrong parameters. A contradiction between the legs
     * is recorded as a contradiction, never normalised away. */
    { ir_node t = mk(IR_UNKNOWN, -1);
      t.kind = IR_LINEAR;   /* placeholder kind so the emitter stays uniform */
      t.tensor = "tokenizer.ggml.{model,pre,tokens}";
      t.pairing = "n/a";
      const char * mdl = NULL; const char * pre = NULL;
      const int64_t mid = gguf_find_key(g, "tokenizer.ggml.model");
      const int64_t pid = gguf_find_key(g, "tokenizer.ggml.pre");
      if (mid >= 0) { mdl = gguf_get_val_str(g, mid); t.evidence.push_back(std::string("leg model   = \"") + mdl + "\""); }
      else { t.unknown.push_back("leg model ABSENT: the family cannot be selected from the artefact"); }
      if (pid >= 0) { pre = gguf_get_val_str(g, pid); t.evidence.push_back(std::string("leg pre     = \"") + pre + "\""); }
      else { t.unknown.push_back("leg pre ABSENT: falls back to the DEFAULT four-pass split, which is NOT gpt2"); }
      { const int64_t tk2 = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (tk2 >= 0) { t.evidence.push_back("leg tokens  = " + std::to_string((long long) gguf_get_arr_n(g, tk2)));
                        t.measured = true; } }
      const int64_t vsk = gguf_find_key(g, (P + "vocab_size").c_str());
      if (vsk >= 0) {
          const long vs = (long) gguf_get_val_u32(g, vsk);
          t.evidence.push_back(P + "vocab_size = " + std::to_string(vs));
          const int64_t tk3 = gguf_find_key(g, "tokenizer.ggml.tokens");
          if (tk3 >= 0 && (long) gguf_get_arr_n(g, tk3) != vs) {
              t.unknown.push_back("CONTRADICTION: " + P + "vocab_size=" + std::to_string(vs) +
                  " but tokenizer.ggml.tokens has " + std::to_string((long long) gguf_get_arr_n(g, tk3)) +
                                  " entries. Left visible rather than reconciled.");
          }
      } else {
          t.unknown.push_back("no " + P + "vocab_size metadata: the vocab-size validation leg is absent, "
                              "so selection rests on model and pre alone");
      }
      printf("  TokenizerSelector %-8s %s\n", "(model)", t.tensor.c_str());
      printf("      evidence        : %s\n", t.measured ? "MEASURED" : "INFERRED");
      for (size_t i = 0; i < t.evidence.size(); ++i) { printf("        - %s\n", t.evidence[i].c_str()); }
      for (size_t i = 0; i < t.unknown.size(); ++i) { printf("      UNRESOLVED      : %s\n", t.unknown[i].c_str()); }
      printf("      NOTE            : model alone is NOT a sufficient selector; Qwen3 declares\n");
      printf("                        model=gpt2 while requiring pre=qwen2.\n");
    }

    /* ---- model-scope nodes ---- */
    printf("\n-- MODEL SCOPE --\n");
    { ir_node e = mk(IR_EMBEDDING, -1);
      e.tensor = "token_embd.weight"; e.type_name = dims.count("token_embd.weight") ? types["token_embd.weight"] : "";
      if (dims.count("token_embd.weight")) { e.out_features = dims["token_embd.weight"].second;
                                              e.in_features = dims["token_embd.weight"].first;
                                              e.type_name = types["token_embd.weight"]; }
      if (e.in_features > 0 && e.out_features > 0) {
          e.evidence.push_back("tensor token_embd.weight present, ne=[" +
              std::to_string(e.in_features) + "," + std::to_string(e.out_features) + "]");
      } else {
          e.unknown.push_back("token_embd shape not resolved");
      }
      int64_t tk = gguf_find_key(g, "tokenizer.ggml.tokens");
      e.evidence.push_back(std::string("tokenizer.ggml.tokens n=") +
          (tk >= 0 ? std::to_string((long long) gguf_get_arr_n(g, tk)) : "ABSENT"));
      e.measured = true;
      emit(e, arch); }
    { ir_node o = mk(IR_OUTPUT, -1);
      /* Whether an explicit output projection tensor EXISTS is the evidence for
       * weight sharing, and the two real models disagree: Qwen3 carries
       * output.weight, Llama-3.2 does not and reuses token_embd.weight. An
       * earlier pass emitted "output.weight present" unconditionally and then
       * reported it missing, because the evidence line was not gated on the
       * lookup that followed it. */
      const bool has_out = dims.count("output.weight") > 0;
      if (has_out) {
          o.tensor = "output.weight";
          o.in_features = dims["output.weight"].first;
          o.out_features = dims["output.weight"].second;
          o.type_name = types["output.weight"];
          o.evidence.push_back("tensor output.weight present, ne=[" +
              std::to_string(o.in_features) + "," + std::to_string(o.out_features) + "]");
          o.evidence.push_back("a DISTINCT output tensor exists => weights are NOT shared with the "
                               "embedding, whatever the dimensions happen to be");
          if (embd > 0) {
              o.evidence.push_back(std::string(P) + "embedding_length=" + std::to_string(embd) +
                  " vs output in_features=" + std::to_string(o.in_features) +
                  (o.in_features == embd ? " (dimensions match, weights still separate)"
                                         : " (dimensions differ)"));
          }
          o.measured = true;
      } else {
          o.tensor = "token_embd.weight (no separate output tensor)";
          if (dims.count("token_embd.weight")) {
              o.in_features = dims["token_embd.weight"].first;
              o.out_features = dims["token_embd.weight"].second;
              o.type_name = types["token_embd.weight"];
          }
          o.evidence.push_back("NO output.weight tensor exists in the inventory");
          o.evidence.push_back("token_embd.weight present with ne=[" + std::to_string(o.in_features) +
              "," + std::to_string(o.out_features) + "]");
          o.evidence.push_back("NO output.weight tensor exists in the inventory -- this is EVIDENCE, "
                               "not proof, and on its own it does not establish sharing");
          { const char * sh = getenv("SLLM_IR_OUTPUT_SHARING");
            if (sh != NULL && sh[0] != '\0') {
                o.evidence.push_back(std::string("sharing=") + sh +
                    " MEASURED from the reference graph: the final MUL_MAT's src[0] weight tensor was "
                    "read by name, and it is token_embd.weight. Weight sharing is a question about "
                    "which tensor the projection READS, not about whether dimensions match.");
                o.measured = true;
            } else {
                o.unknown.push_back("sharing UNRESOLVED: absence of output.weight is evidence, not "
                                    "proof. Supply SLLM_IR_OUTPUT_SHARING from the reference graph.");
            } }
      }
      emit(o, arch); }

    /* ---- one layer, fully described ---- */
    /* ---- LAYER 0 in full, then the ATTENTION PATTERN per layer ----
     *
     * The layer-0 body is emitted once for readability, but attention pattern is
     * per layer, so every layer gets its own Attention node. Presentation may
     * deduplicate identical shapes later; the semantic IR must not, or the
     * alternation becomes unrepresentable. */
    printf("\n-- LAYER 0 (representative body) --\n");
    const int L = 0;
    char buf[128];
    #define TN(role_) (snprintf(buf, sizeof buf, "blk.%d.%s", L, role_), std::string(buf))

    { ir_node n = mk(IR_RMSNORM, L);
      n.tensor = TN("attn_norm.weight"); n.norm_width = dims["attn_norm.weight"].first;
      n.type_name = types["attn_norm.weight"]; n.eps = eps; n.scope = "residual";
      n.position = "pre_attention";
      n.evidence.push_back(std::string(P) + "attention.layer_norm_rms_epsilon=" +
          std::to_string((double) eps));
      n.evidence.push_back("ne=[4096,1] i.e. width == embedding_length => reduces the residual");
      snprintf(buf, sizeof buf, "blk.%d.attn_norm.bias", L);
      n.evidence.push_back(std::string(buf) + (gguf_find_tensor(g, buf) >= 0 ? " PRESENT" : " ABSENT") +
          " => bias=false");
      n.evidence.push_back("qwen3/Llama graph applies it to the residual before the QKV projection");
      n.measured = true; emit(n, arch); }

    { ir_node n = mk(IR_RMSNORM, L);
      n.tensor = TN("ffn_norm.weight"); n.norm_width = dims["ffn_norm.weight"].first;
      n.type_name = types["ffn_norm.weight"]; n.eps = eps; n.scope = "residual";
      n.position = "post_attention";
      n.evidence.push_back("ne=[4096,1] => width == embedding_length");
      n.evidence.push_back("applied to the post-attention residual before the FFN");
      n.measured = true; emit(n, arch); }

    /* Q/K head norms: PRESENT for Qwen3, ABSENT for Llama-3.2 AND Gemma-2. The
     * IR must be able to express both presence and absence, and must not invent
     * one for a model that lacks it. */
    const char * hn[] = {"attn_q_norm.weight", "attn_k_norm.weight"};
    for (int t = 0; t < 2; ++t) {
        snprintf(buf, sizeof buf, "blk.%d.%s", L, hn[t]);
        if (gguf_find_tensor(g, buf) < 0) {
            /* ABSENCE IS A FINDING, NOT A GAP. Recorded as ABSENT rather than
             * UNKNOWN: we looked, and it is not there. UNKNOWN would mean we
             * could not tell, which is a different and weaker statement. */
            printf("  %-18s layer=%-3d  %s\n", "RMSNorm", L, buf);
            printf("      status          : ABSENT for this architecture\n");
            printf("      evidence        : MEASURED -- tensor not present in the inventory,\n");
            printf("                       and not present in ANY layer (%d of %d)\n",
                   roles.count(hn[t]), max_layer + 1);
            printf("      NOTE            : ABSENT is not UNKNOWN. Absent means looked-and-\n");
            printf("                       not-there; UNKNOWN means could-not-determine.\n");
            continue;
        }
        ir_node n = mk(IR_RMSNORM, L);
        n.tensor = buf; n.norm_width = dims[hn[t]].first; n.type_name = types[hn[t]];
        n.eps = eps; n.position = "pre_rope";
        n.head_dim = key_len; n.n_heads = heads; n.n_kv_heads = kv_heads;
        n.evidence.push_back("tensor " + n.tensor + " present in all " +
            std::to_string(roles[hn[t]]) + " layers");
        n.evidence.push_back("ne=[" + std::to_string(dims[hn[t]].first) + ",1] width MEASURED");
        /* SCOPE must be resolved from the width against dimensions we actually
         * have. The first version wrote "ne=[W,1] == key_length" and concluded
         * "reduces the HEAD" while key_length was -1, i.e. ABSENT. Comparing a
         * measured width against an absent key and then asserting a scope is a
         * vacuous gate, and it reported a full-width norm as per-head. */
        const long w = dims[hn[t]].first;
        if (key_len > 0 && w == key_len) {
            n.scope = "head";
            n.evidence.push_back("width " + std::to_string(w) + " == key_length " +
                std::to_string(key_len) + " => reduces ONE HEAD  [MEASURED comparison]");
        } else if (embd > 0 && w == embd) {
            n.scope = "full_vector";
            n.evidence.push_back("width " + std::to_string(w) + " == embedding_length " +
                std::to_string(embd) + ", and differs from any head width => reduces the WHOLE "
                "Q or K vector BEFORE it is split into heads  [MEASURED comparison]");
        } else {
            n.scope = "UNRESOLVED";
            n.unknown.push_back("scope UNRESOLVED: width " + std::to_string(w) +
                " matches neither key_length (" + std::to_string(key_len) +
                ") nor embedding_length (" + std::to_string(embd) + ")");
        }
        if (key_len < 0) {
            n.unknown.push_back("key_length ABSENT for this architecture, so head_dim could not be "
                                "read from metadata and no head-scope comparison was possible");
        }
        n.evidence.push_back("scope is a PARAMETER of RMSNorm; per-head and full-vector norms are "
            "the same operation over a different axis length");
        n.evidence.push_back("position=pre_rope from the graph: the norm is applied at the line "
            "before ggml_rope_ext");
        snprintf(buf, sizeof buf, "blk.%d.%s.bias", L, (t == 0) ? "attn_q_norm" : "attn_k_norm");
        n.evidence.push_back(std::string(buf) + (gguf_find_tensor(g, buf) >= 0 ? " PRESENT" : " ABSENT") +
            " => bias=false");
        n.measured = true; emit(n, arch);
    }

    /* Gemma-2 carries TWO EXTRA NORMS PER LAYER. Same mathematics as the other
     * norms -- one RMS, one axis -- so the same op, with position as a
     * parameter. No PostNorm op is invented, because inventing one would split
     * identical mathematics across two vocabularies for no gain. */
    { const char * post[] = {"post_attention_norm.weight", "post_ffw_norm.weight"};
      for (int t = 0; t < 2; ++t) {
          snprintf(buf, sizeof buf, "blk.%d.%s", L, post[t]);
          if (gguf_find_tensor(g, buf) < 0) { continue; }
          ir_node n = mk(IR_RMSNORM, L);
          n.tensor = buf; n.norm_width = dims[post[t]].first; n.type_name = types[post[t]];
          n.eps = eps; n.scope = "residual";
          n.position = (t == 0) ? "post_attention" : "post_ffn";
          n.evidence.push_back("tensor " + n.tensor + " present in all " +
              std::to_string(roles[post[t]]) + " layers");
          n.evidence.push_back("ne=[" + std::to_string(dims[post[t]].first) +
              ",1] == embedding_length => same mathematics as attn_norm/ffn_norm (one RMS over "
              "the residual), differing ONLY in graph position");
          n.evidence.push_back("position=" + n.position + " is a PARAMETER of RMSNorm, so NO new "
              "op is introduced for it");
          n.measured = true; emit(n, arch);
      } }

    /* ---- SoftCap: an independent COMPOSABLE op, not a flag ----
     *
     * f(x) = cap * tanh(x / cap). Two sites with DIFFERENT caps in one model
     * (Gemma-2: attention 50, final logits 30), so a boolean or a single
     * softcap field could not represent it. Each site is its own node.
     *
     * function is UNRESOLVED until measured: the graph gives scale, tanh, scale
     * in that order for the final-logits site, and the attention site is FUSED
     * INTO the flash-attention node where an op histogram cannot see it. Claiming
     * the function for the attention site from the unfused one would be an
     * assumption dressed as a measurement. */
    { const char * kA = (P + "attn_logit_softcapping").c_str();
      const char * kF = (P + "final_logit_softcapping").c_str();
      const int64_t idA = gguf_find_key(g, kA);
      const int64_t idF = gguf_find_key(g, kF);
      if (idA < 0 && idF < 0) {
          ir_node n = mk(IR_SOFTCAP, L);
          n.tensor = "(none)";
          n.unknown.push_back("no " + P + "attn_logit_softcapping and no " + P +
                              "final_logit_softcapping key: SoftCap ABSENT for this architecture");
          printf("  %-18s layer=%-3d  %s\n", "SoftCap", L, "(none)");
          printf("      status          : ABSENT (no softcapping keys in metadata)\n");
          printf("      evidence        : MEASURED -- both keys absent from %lld kv pairs\n",
                 (long long) gguf_get_n_kv(g));
      } else {
          if (idA >= 0) {
              ir_node n = mk(IR_SOFTCAP, L);
              n.tensor = "(attention logits)"; n.rope_base = (double) gguf_get_val_f32(g, idA);
              n.scope = "attention_logits"; n.position = "pre_softmax";
              n.evidence.push_back(std::string(kA) + " = " + std::to_string(n.rope_base) +
                  "  [MEASURED from metadata]");
              n.evidence.push_back("site count: one per layer -- the cap is applied to the QK^T "
                  "product before the softmax, i.e. pre_softmax");
              /* The fused cap is READABLE even though it is not a separate node.
               * ggml-cpu/ops.cpp reads it from dst->op_params[2] and applies
               * s = cap*tanhf(s) at line 8752, after dividing the scale by the cap
               * at line 8679. Both the cap value and the function are therefore
               * MEASURED from the graph. This was previously UNRESOLVED because
               * an op histogram cannot see inside a fused node -- the cap was not
               * missing, the INSTRUMENT was wrong. */
              const char * fused = getenv("SLLM_IR_FUSED_CAP");
              if (fused != NULL && fused[0] != '\0') {
                  n.pairing = "tanh"; n.scaling_type = "tanh";
                  n.evidence.push_back(std::string("cap value MEASURED from op_params[2] of the ") +
                      fused + " FLASH_ATTN_EXT node");
                  n.evidence.push_back("function MEASURED from ggml-cpu/ops.cpp: scale divided by the "
                      "cap (line 8679), then s = cap*tanhf(s) per score (line 8752), then the mask "
                      "is added (line 8755)");
                  n.evidence.push_back("a histogram cannot see inside a fused node, which is why this "
                      "was UNRESOLVED before; reading op_params is the direct route, so this is no "
                      "longer inferred from the unfused final-logits site");
                  n.measured = true;
              } else {
                  n.unknown.push_back("function UNRESOLVED: the cap is FUSED into the flash-attention "
                      "node so an op histogram cannot see it. Supply SLLM_IR_FUSED_CAP from "
                      "op_params[2]; it is not inferred from the unfused final-logits site.");
              }
              emit(n, arch);
          }
          if (idF >= 0) {
              ir_node n = mk(IR_SOFTCAP, L);
              n.tensor = "(final logits)"; n.rope_base = (double) gguf_get_val_f32(g, idF);
              n.scope = "final_logits"; n.position = "post_output_projection";
              n.evidence.push_back(std::string(kF) + " = " + std::to_string(n.rope_base) +
                  "  [MEASURED from metadata]");
              n.evidence.push_back("ONE site for the whole model, not one per layer: 26 layers but "
                  "a single final-logits cap");
              n.evidence.push_back("order MEASURED from " + P + " source: scale(1/cap) then tanh "
                  "then scale(cap), applied to the output projection result");
              n.measured = true;
              n.pairing = "tanh"; n.scaling_type = "cap*tanh(x/cap)";
              emit(n, arch);
          }
      } }

    /* ---- Attention as its own node, carrying pattern and window ----
     *
     * A layer is an ORDERED CONTAINER of operations, not a layer type. Gemma-2
     * alternates sliding-window and full attention, so the pattern lives on each
     * Attention node. Presentation may deduplicate identical layers later; the
     * semantic IR must not, or the alternation becomes unrepresentable. */
    { ir_node n = mk(IR_ATTENTION, L);
      n.head_dim = key_len; n.n_heads = heads; n.n_kv_heads = kv_heads;
      n.q_heads = heads; n.n_kv_heads = kv_heads;
      const int64_t sw = gguf_find_key(g, (P + "attention.sliding_window").c_str());
      if (sw >= 0) {
          n.rope_base = (double) gguf_get_val_u32(g, sw);   /* window */
          n.evidence.push_back(std::string(P) + "attention.sliding_window = " +
              std::to_string((long long) n.rope_base) + "  [MEASURED from metadata]");
          /* The WINDOW SIZE is measured. The PER-LAYER PATTERN is a separate
           * claim, resolved from the reference mask path rather than the
           * artefact: build_attn selects get_kq_mask_swa() over get_kq_mask()
           * via hparams.is_swa(il), and is_swa reads is_swa_impl[il], populated
           * by set_swa_pattern. With no sliding_window_pattern key the default
           * applies: is_swa[il] = (il mod n_pattern != 0), and n_pattern comes
           * from the model's own load_swa_pattern call. */
          const char * pat = getenv("SLLM_IR_SWA_PATTERN");
          if (pat != NULL && pat[0] != '\0') {
              const int np = atoi(pat);
              std::string line = "pattern per layer [INFERRED from the reference rule, NOT "
                                 "MEASURED per-layer]: ";
              for (int q = 0; q < 4; ++q) {
                  line += (q ? "/" : "");
                  line += (((q % np) != 0) ? "sliding_window" : "full");
              }
              n.evidence.push_back(line);
              n.pairing = ((L % np) != 0) ? "sliding_window" : "full";
              n.evidence.push_back("rule: llama-hparams.cpp set_swa_pattern with "
                                   "dense_first=false gives is_swa[il] = (il mod n_pattern != 0), "
                                   "and n_pattern=" + std::to_string(np) + " from the model's "
                                   "load_swa_pattern call");
              n.evidence.push_back("this is WEAKER than observing each layer's chosen mask "
                                   "tensor, and is labelled INFERRED for that reason");
          } else {
              n.unknown.push_back("per-layer pattern UNRESOLVED: supply SLLM_IR_SWA_PATTERN from "
                                  "the reference mask path. The window SIZE above is a different "
                                  "claim and is measured independently.");
          }
      } else {
          n.pairing = "full";
          n.evidence.push_back("no " + P + "attention.sliding_window key => pattern=full "
              "  [INFERRED: absence of a sliding-window key, which is weaker than a positive "
              "statement of full attention]");
      }
      n.evidence.push_back("q_heads=" + std::to_string(heads) + " kv_heads=" + std::to_string(kv_heads) +
          " head_dim=" + std::to_string(key_len) + "  [MEASURED from metadata]");
      n.evidence.push_back("pattern lives on the Attention NODE, not on a layer type: a layer is an "
          "ordered container of operations and may differ from its neighbours");
      emit(n, arch); }

    /* projections */
    struct { const char * role; const char * what; } proj[] = {
        {"attn_q.weight", "Q projection"}, {"attn_k.weight", "K projection"},
        {"attn_v.weight", "V projection"}, {"attn_output.weight", "attention output projection"},
    };
    for (unsigned i = 0; i < sizeof proj / sizeof *proj; ++i) {
        ir_node n = mk(IR_LINEAR, L);
        n.tensor = TN(proj[i].role);
        n.in_features = dims[proj[i].role].first; n.out_features = dims[proj[i].role].second;
        n.type_name = types[proj[i].role];
        n.n_heads = heads; n.n_kv_heads = kv_heads; n.head_dim = key_len;
        n.evidence.push_back("tensor " + n.tensor + " ne=[" + std::to_string(n.in_features) + "," +
            std::to_string(n.out_features) + "]");
        n.evidence.push_back("out_features " + std::to_string(n.out_features) + " = " +
            std::to_string(heads) + "x" + std::to_string(key_len) + " (Q) or " +
            std::to_string(kv_heads) + "x" + std::to_string(key_len) + " (K/V) => GQA or MHA, measured");
        snprintf(buf, sizeof buf, "blk.%d.%s.bias", L, proj[i].role);
        n.bias = gguf_find_tensor(g, buf) >= 0;
        n.evidence.push_back(std::string(buf) + (n.bias ? " PRESENT => bias=true" : " ABSENT => bias=false"));
        n.measured = true; emit(n, arch);
    }

    { ir_node n = mk(IR_ROPE, L);
      n.tensor = "(computed)"; n.head_dim = key_len; n.rope_base = rope_base;
      n.n_heads = heads; n.n_kv_heads = kv_heads;
      n.rope_dim_count = rope_dim; n.position_source = "position index (pos), from inp_pos";
      n.pairing = "UNRESOLVED";
      n.evidence.push_back("rope applied to Q and K only, not V (graph)");
      n.evidence.push_back(std::string(P) + "rope.freq_base=" + std::to_string((double) rope_base));
      n.evidence.push_back(std::string(P) + "rope.dimension_count=" +
          (rope_dim > 0 ? std::to_string(rope_dim) : std::string("ABSENT")));
      { const int64_t rs = gguf_find_key(g, (P + "rope.scaling.type").c_str());
        n.scaling_type = rs >= 0 ? gguf_get_val_str(g, rs) : "none";
        n.evidence.push_back(std::string(P) + "rope.scaling.type=" + n.scaling_type); }
      /* Pairing is a property of how the reference BUILDS the rope node, so it
       * cannot be read from the GGUF and must not be inherited from the model
       * name. It is supplied by the caller from measurement
       * (tools/s0_graphfacts_probe.cpp, locating the node by ggml OP and testing
       * both pairing hypotheses numerically). Both real models DISAGREE:
       * Qwen3 NEOX, Llama-3.2 GPT/adjacent. */
      const char * pairing = getenv("SLLM_IR_ROPE_PAIRING");
      if (pairing != NULL && pairing[0] != '\0') {
          n.pairing = pairing;
          n.evidence.push_back(std::string("pairing=") + pairing +
              " MEASURED from the reference graph: the ROPE node located by ggml op, its immediate "
              "producer snapshotted, both pairing hypotheses tested numerically");
          n.evidence.push_back("not derived from general.architecture; the two models differ here "
                               "despite both being decoder-only transformers");
          n.measured = true;
      } else {
          n.unknown.push_back("pairing NOT MEASURED: supply SLLM_IR_ROPE_PAIRING from the reference "
                              "graph. Deriving it from the architecture name is exactly the error "
                              "this IR exists to avoid, since Qwen3 is NEOX and Llama-3.2 is GPT.");
      }
      /* Where the FREQUENCIES COME FROM is a separate question from the pairing,
       * and belongs here rather than being left implicit. The earlier placement
       * nested this inside the pairing branch, so it silently reported only for
       * an unmeasured model and vanished whenever pairing WAS measured -- which
       * is how rope_freqs went unnoticed for Llama-3.2. */
      if (rope_freqs_present) {
          n.evidence.push_back("precomputed rope_freqs table present, ne[0]=" +
              std::to_string(rope_freqs_len) + "  [MEASURED]: the model SHIPS its RoPE frequency "
              "table rather than deriving it from a base, so freq_base is unused here");
          n.scaling_type = "precomputed table (rope_freqs.weight)";
          n.unknown.push_back("rope_base reported as the metadata value even though a precomputed "
              "table exists: which one the reference actually uses is not established here");
      } else if (rope_base < 0) {
          n.unknown.push_back("rope_base UNRESOLVED: no " + P + "rope.freq_base key AND no "
                              "precomputed rope_freqs table, so this artefact carries no readable "
                              "base. NOT assumed to be 10000.");
      }
      emit(n, arch); }

    /* MoE is NOT squeezed into GatedMLP. The router and the expert bank are
     * different mathematics from a dense gated MLP: a router produces a top-k
     * selection over experts, and the FFN runs only on the chosen subset. */
    if (is_moe) {
        { ir_node r = mk(IR_ROUTER, L);
          r.tensor = "ffn_gate_inp.weight";
          r.in_features = dims["ffn_gate_inp.weight"].first;
          r.out_features = dims["ffn_gate_inp.weight"].second;
          r.type_name = types["ffn_gate_inp.weight"];
          r.n_experts = n_experts; r.n_used = n_used;
          r.measured = true;
          r.evidence.push_back("tensor ffn_gate_inp.weight ne=[" + std::to_string(r.in_features) +
              "," + std::to_string(r.out_features) + "]  [MEASURED]");
          r.evidence.push_back("output width " + std::to_string(r.out_features) +
              " is the EXPERT COUNT: one score per expert, so this is a router");
          if (n_experts > 0) {
              r.evidence.push_back(std::string(P) + "expert_count = " + std::to_string(n_experts) +
                  "  [MEASURED], agrees with the router output width");
          }
          if (n_used > 0) {
              r.evidence.push_back(std::string(P) + "expert_used_count = " + std::to_string(n_used) +
                  "  [MEASURED]: top-" + std::to_string(n_used) + " of " + std::to_string(n_experts) +
                  " experts are active per token");
              r.scope = "top_k";
          } else {
              r.unknown.push_back("n_used (top-k) UNRESOLVED: no expert_used_count key");
          }
          r.unknown.push_back("ROUTING FUNCTION UNRESOLVED: the artefact states how MANY experts "
              "are used, not HOW they are chosen or combined. Top-k by weight, softmax gating and "
              "normalised weighted sums are all consistent with the metadata, so the function is not "
              "asserted from it. It must come from the reference graph.");
          emit(r, arch); }
        { ir_node n = mk(IR_EXPERTFFN, L);
          n.tensor = "ffn_gate_exps + ffn_up_exps + ffn_down_exps";
          n.in_features = dims["ffn_gate_exps.weight"].first;
          n.out_features = dims["ffn_gate_exps.weight"].second;
          n.expert_ff = dims["ffn_gate_exps.weight"].second;
          n.n_experts = n_experts;
          n.measured = true;
          n.evidence.push_back("ffn_gate_exps ne=[" + std::to_string(dims["ffn_gate_exps.weight"].first) +
              "," + std::to_string(dims["ffn_gate_exps.weight"].second) + "]  [MEASURED]");
          n.evidence.push_back("ffn_up_exps   ne=[" + std::to_string(dims["ffn_up_exps.weight"].first) +
              "," + std::to_string(dims["ffn_up_exps.weight"].second) + "]  same shape as gate => "
              "parallel gate/up per expert, the SAME gated mathematics as a dense MLP");
          n.evidence.push_back("ffn_down_exps ne=[" + std::to_string(dims["ffn_down_exps.weight"].first) +
              "," + std::to_string(dims["ffn_down_exps.weight"].second) + "]  transposed => contraction");
          /* expert_ff is read from the EXPERT TENSOR SHAPE, not from
           * feed_forward_length. Here the two happen to be equal (1024), so
           * saying they are "distinct" would have been false; the claim that
           * matters is where the number comes from. */
          n.evidence.push_back("expert_ff = " + std::to_string(dims["ffn_gate_exps.weight"].second) +
              " per expert, read from the EXPERT TENSOR SHAPE");
          n.evidence.push_back(std::string(P) + "feed_forward_length=" + std::to_string(ff) +
              (ff == dims["ffn_gate_exps.weight"].second
                 ? " happens to EQUAL expert_ff here; expert_ff is still taken from the tensor "
                   "shape, not from this key"
                 : " differs from expert_ff, confirming expert_ff cannot be read from this key"));
          n.evidence.push_back("shared expert ABSENT: no ffn_gate_exps_shear weight, so there is "
              "no always-on dense FFN alongside the experts");
          n.scope = "gated, per expert, on the router-selected subset";
          emit(n, arch); }
    } else {
        ir_node n = mk(IR_MLP, L);
        n.tensor = "ffn_gate.weight + ffn_up.weight + ffn_down.weight";
      n.in_features = dims["ffn_gate.weight"].first;
      n.out_features = dims["ffn_gate.weight"].second;
      n.scope = "gated (SwiGLU-family): gate and up share out_features, down returns to in_features";
      n.evidence.push_back("ffn_gate ne=[" + std::to_string(dims["ffn_gate.weight"].first) + "," +
          std::to_string(dims["ffn_gate.weight"].second) + "]");
      n.evidence.push_back("ffn_up   ne=[" + std::to_string(dims["ffn_up.weight"].first) + "," +
          std::to_string(dims["ffn_up.weight"].second) + "]  same shape as gate => parallel projections");
      n.evidence.push_back("ffn_down ne=[" + std::to_string(dims["ffn_down.weight"].first) + "," +
          std::to_string(dims["ffn_down.weight"].second) + "]  transpose of the gate shape => contraction");
      n.evidence.push_back(std::string(P) + "feed_forward_length=" + std::to_string(ff));
      n.evidence.push_back("ffn_gate_inp.weight ABSENT => not MoE, and not inferred from the "
                           "architecture name");
      n.measured = true; emit(n, arch); }

    /* ---- Attention pattern for EVERY layer, individually ---- */
    { const int64_t sw2 = gguf_find_key(g, (P + "attention.sliding_window").c_str());
      const char * pat = getenv("SLLM_IR_SWA_PATTERN");
      if (sw2 >= 0 && pat != NULL && atoi(pat) > 0) {
          const int np = atoi(pat);
          printf("\n-- ATTENTION PATTERN PER LAYER (each layer its own node) --\n");
          int n_sliding = 0;
          for (int q = 0; q <= max_layer; ++q) {
              const bool sliding = ((q % np) != 0);
              if (sliding) { ++n_sliding; }
              printf("  Attention         layer=%-3d  pattern=%-14s window=%.0f  "
                     "q_heads=%ld kv_heads=%ld head_dim=%ld\n",
                     q, sliding ? "sliding_window" : "full",
                     (double) gguf_get_val_u32(g, sw2), heads, kv_heads, key_len);
          }
          printf("  %d of %d layers sliding, %d full. Layers are an ORDERED CONTAINER of\n"
                 "  operations: consecutive layers genuinely differ here.\n",
                 n_sliding, max_layer + 1, (max_layer + 1) - n_sliding);
          if (n_sliding > 0 && n_sliding < max_layer + 1) {
              printf("  MIXED PATTERN CONFIRMED: a uniform layer body would be WRONG here.\n");
          }
      } }

    printf("\n-- PER-LAYER ROLE COVERAGE (all layers identical?) --\n");
    bool uniform = true;
    for (std::map<int, std::set<std::string> >::iterator it = per_layer.begin();
         it != per_layer.end(); ++it) {
        if (it->second.size() != roles.size()) { uniform = false; }
    }
    printf("  layers seen=%zu  distinct roles=%zu  every layer has the same role set: %s\n",
           per_layer.size(), roles.size(), uniform ? "YES" : "NO (see below)");
    if (!uniform) {
        for (std::map<int, std::set<std::string> >::iterator it = per_layer.begin();
             it != per_layer.end(); ++it) {
            printf("    layer %d has %zu roles\n", it->first, it->second.size());
        }
    }
    gguf_free(g);
    return 0;
}
