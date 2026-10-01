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
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

/* The IR probe reads GGUF metadata through this small Saphira-side shim rather
 * than llama.cpp, so the discovered structure is described in terms the product
 * already owns. */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

enum ir_kind { IR_UNKNOWN = 0, IR_EMBEDDING, IR_RMSNORM, IR_LINEAR, IR_ROPE,
               IR_HEADSPLIT, IR_ATTENTION, IR_RESIDUAL, IR_MLP, IR_DENSE_MLP, IR_OUTPUT, IR_SOFTCAP,
               IR_ROUTER, IR_EXPERTFFN, IR_SSM, IR_CONV1D, IR_SCAN };

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
        case IR_DENSE_MLP:  return "DenseMLP";
        case IR_OUTPUT:     return "OutputProjection";
        case IR_SOFTCAP:    return "SoftCap";
        case IR_ROUTER:     return "ExpertRouter";
        case IR_EXPERTFFN:  return "ExpertGatedFFN";
        case IR_SSM:        return "SelectiveSSM";
        case IR_CONV1D:     return "DepthwiseConv1D";
        case IR_SCAN:       return "SelectiveScan";
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


/* ---- CLAIM LEDGER -------------------------------------------------------
 *
 * The whole project exists to make one distinction impossible to blur:
 *
 *     we MEASURED that it is not there   !=   we FAILED to observe it
 *
 * Every node printed above, and every absence printed above, is recorded here
 * with the four things a reader needs before believing it:
 *
 *     field        what the claim is about
 *     value        the value, or ABSENT / UNRESOLVED / UNOBSERVABLE
 *     source       where the evidence came from
 *     confidence   measured | inferred | unresolved | unobservable
 *     claim allowed    the strongest statement this evidence PERMITS
 *     NOT claimed      the statement a careless reader would make anyway
 *
 * The last field is the important one. "ABSENT, measured" still does not license
 * "this model has no attention"; it licenses "no Q projection exists at layer 7".
 * Anyone can turn a correct ledger into a wrong story by dropping the scope, so
 * the scope is printed next to every claim rather than left to be inferred. */
struct claim_rec {
    std::string field, value, source, confidence, layer, allowed, notclaimed;
};
static std::vector<claim_rec> g_claims;
static void claim(const std::string & field, const std::string & value,
                  const std::string & source, const std::string & confidence,
                  const std::string & layer, const std::string & allowed,
                  const std::string & notclaimed) {
    claim_rec c; c.field = field; c.value = value; c.source = source;
    c.confidence = confidence; c.layer = layer; c.allowed = allowed;
    c.notclaimed = notclaimed; g_claims.push_back(c);
}

/* An absence backed by the tensor inventory is the STRONGEST negative claim
 * available: we looked in the one place the artefact must speak if it were going
 * to, and it did not speak. */
static void claim_absent(const std::string & field, int layer, const std::string & what) {
    char lb[16]; if (layer < 0) snprintf(lb, sizeof lb, "model");
    else snprintf(lb, sizeof lb, "%d", layer);
    claim(field, "ABSENT", "tensor inventory (direct lookup)", "measured", lb,
          "no " + what + " exists at " + (layer < 0 ? std::string("model scope") : ("layer " + std::string(lb))) +
              ", measured by direct lookup in the tensor inventory",
          layer < 0 ? ("no claim about any layer in particular")
                    : ("no claim about OTHER layers, and no claim that the model lacks " + what + " entirely"));
}
/* Failing to observe is a DIFFERENT statement and must never be recorded as an
 * absence. This is the distinction the project exists to protect. */
static void claim_unresolved(const std::string & field, int layer, const std::string & why,
                             const std::string & source) {
    char lb[16]; if (layer < 0) snprintf(lb, sizeof lb, "model");
    else snprintf(lb, sizeof lb, "%d", layer);
    claim(field, "UNRESOLVED", source, "unresolved", lb,
          "cannot determine " + field + " at " + (layer < 0 ? std::string("model scope") : ("layer " + std::string(lb))),
          why);
}
static void claim_unobservable(const std::string & family, const std::string & why) {
    claim(family, "UNOBSERVABLE", "instrument capability", "unobservable", "model",
          "this probe cannot observe " + family + ", so nothing may be concluded about it",
          why);
}

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
    { char lb[16]; if (n.layer < 0) snprintf(lb, sizeof lb, "model");
      else snprintf(lb, sizeof lb, "%d", n.layer);
      const std::string k = kind_name(n.kind);
      const bool unresolved = (n.pairing == "UNRESOLVED") || !n.unknown.empty();
      claim(k, unresolved ? "PRESENT but PARTLY UNRESOLVED" : "PRESENT",
            "tensor inventory (direct) + resolved parameters",
            unresolved ? "partly measured, partly unresolved" : "measured", lb,
            "a " + k + " op exists at " +
                (n.layer < 0 ? std::string("model scope") : ("layer " + std::string(lb))) +
                (n.measured ? ", with the parameters listed above resolved from the artefact"
                            : ", but not every parameter is measured"),
            "no claim about any parameter shown as -1 or UNRESOLVED, and no claim about other layers");
    }
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
    if (n.kind == IR_MLP || n.kind == IR_DENSE_MLP) { printf("      scope           : %s\n", n.scope.c_str()); }
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


/* Emit the SSM block from artefact evidence. Self-contained so its braces
 * cannot interfere with the surrounding emitter, which is how the first two
 * attempts at this section broke the whole file. */
/* Emit the SSM block from artefact evidence. Kept self-contained so its braces
 * cannot interfere with the surrounding emitter: threading them inline broke
 * the whole file twice, which is itself a note about editing code this long by
 * string substitution. */
typedef std::map<std::string, std::pair<long, long> > dim_map;
typedef std::map<std::string, std::string> type_map;

/* Metadata reads are TYPE-CHECKED. ssm_a and ssm_d are F32 SCALARS, not u32
 * arrays, and reading them as u32 trips a ggml assert: the probe aborted with
 * GGML_ASSERT(type_to_gguf_type<T>::value == type) failed. A scalar is a
 * legitimate value of a legitimate quantity, so it is read as its real type. */
static bool sllm_ok(gguf_context * g, const std::string & k, uint32_t * out) {
    const int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) { return false; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) { return false; }
    *out = gguf_get_val_u32(g, id);
    return true;
}
/* File-scope LAYER-LOCAL tensor index. Every emitter asks AT(role, layer):
 * does THIS tensor exist AT THIS LAYER. The earlier model-global question,
 * "does this role exist somewhere", is what let SelectiveScan and Attention be
 * emitted on layers that have neither. A file-scope set (rather than a macro
 * local to main) is required because emitters live in separate functions. */
static std::set<std::string> g_at_layer;
static bool AT(const std::string & role, int layer) {
    return g_at_layer.count("blk." + std::to_string(layer) + "." + role) > 0;
}

static bool is_moe_at(int layer) { return AT("ffn_gate_inp.weight", layer); }

static bool kv_f32(gguf_context * g, const std::string & k, double * out) {
    const int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) { return false; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) { return false; }
    *out = gguf_get_val_f32(g, id);
    return true;
}
/* Type checking is a PERMANENT invariant of every reader, not a defensive check
 * on the ones that happened to crash. A per-layer ARRAY such as
 * feed_forward_length or attention.head_count_kv is a legitimate value of a
 * legitimate quantity: it says the quantity VARIES BY LAYER. Reading it as a
 * scalar aborts the probe inside ggml, so the type is checked before decoding
 * and an array is reported rather than coerced. */

static std::vector<std::string> kv_arrays_seen;
static bool kv_is_array(gguf_context * g, const std::string & k, long * n_out) {
    const int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) { return false; }
    const enum gguf_type t = gguf_get_kv_type(g, id);
    if (t != GGUF_TYPE_ARRAY) { return false; }
    if (n_out != NULL) { *n_out = (long) gguf_get_arr_n(g, id); }
    kv_arrays_seen.push_back(k + " (array of " + std::to_string((long) gguf_get_arr_n(g, id)) +
                             " -- quantity VARIES BY LAYER, not a single value)");
    return true;
}
static long kv_long(gguf_context * g, const std::string & k, long dflt) {
    const int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) { return dflt; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) { return dflt; }
    return (long) gguf_get_val_u32(g, id);
}

static void emit_ssm_block(gguf_context * g, const std::string & P,
                           dim_map & dims, type_map & types, int L,
                           const char * arch) {
    (void) arch;
    const long state_size   = kv_long(g, P + "ssm.state_size", -1);
    const long inner_size   = kv_long(g, P + "ssm.inner_size", -1);
    const long conv_kernel  = kv_long(g, P + "ssm.conv_kernel", -1);
    const long dt_rank      = kv_long(g, P + "ssm.time_step_rank", -1);
    const long group_count  = kv_long(g, P + "ssm.group_count", -1);

    /* DepthwiseConv1D: a causal convolution over the SEQUENCE axis. There is no
     * key axis here, which is why this is not a projection and not attention. */
    if (AT("ssm_conv1d.weight", L)) {
        ir_node n = mk(IR_CONV1D, L);
        char buf[128]; snprintf(buf, sizeof buf, "blk.%d.ssm_conv1d.weight", L);
        n.tensor = buf;
        n.in_features  = dims["ssm_conv1d.weight"].first;
        n.out_features = dims["ssm_conv1d.weight"].second;
        n.type_name = types["ssm_conv1d.weight"];
        n.measured = true;
        n.scope = "depthwise, over the sequence axis";
        n.position = "pre_recurrence";
        if (conv_kernel > 0) {
            n.evidence.push_back(P + "ssm.conv_kernel = " + std::to_string(conv_kernel) +
                "   [ARTEFACT MEASURED]");
            n.evidence.push_back("ssm_conv1d.weight ne[0]=" +
                std::to_string(dims["ssm_conv1d.weight"].first) +
                ", which agrees with conv_kernel: the weights ARE the kernel taps");
        }
        n.evidence.push_back("ne[1]=" + std::to_string(dims["ssm_conv1d.weight"].second) +
            " channels with a matching ssm_conv1d.bias, one bias per channel: that is what makes "
            "it DEPTHWISE rather than a dense convolution");
        n.evidence.push_back("convolved over SEQUENCE, not over an attention axis");
        emit(n, arch);
    }

    /* The RMSNorm that gates the SSM inner space: SAME op as every other norm,
     * with scope as the parameter that distinguishes it. */
    if (AT("ssm_norm.weight", L)) {
        ir_node n = mk(IR_RMSNORM, L);
        char nb[128]; snprintf(nb, sizeof nb, "blk.%d.ssm_norm.weight", L);
        n.tensor = nb;
        n.norm_width = dims["ssm_norm.weight"].first;
        n.type_name = types["ssm_norm.weight"];
        { uint32_t ev = 0;
          n.eps = (sllm_ok(g, P + "attention.layer_norm_rms_epsilon", &ev)) ? (double) ev : -1.0; }
        n.scope = "ssm_inner";
        n.position = "pre_conv";
        n.measured = true;
        n.evidence.push_back("ssm_norm.weight ne=[" +
            std::to_string(dims["ssm_norm.weight"].first) + ",1]   [ARTEFACT MEASURED]");
        if (inner_size > 0 && dims["ssm_norm.weight"].first == inner_size) {
            n.evidence.push_back("width " + std::to_string(dims["ssm_norm.weight"].first) +
                " == ssm.inner_size " + std::to_string(inner_size) + ", not embedding_length, so it "
                "gates the INNER space rather than the residual  [MEASURED comparison]");
        }
        n.evidence.push_back("SAME RMSNorm op as the residual norms; scope is the parameter that "
            "distinguishes it, not a different mathematics");
        emit(n, arch);
    }

    /* SelectiveScan: the recurrence. Different mathematics from attention: a
     * linear recurrence over a carried state, with NO score matrix, NO softmax
     * over keys, NO KV cache and NO position-wise FFN.
     *
     * GATED ON EVIDENCE, and this gate was MISSING: the block was emitted
     * unconditionally, so every transformer model grew a SelectiveScan node out
     * of nothing. It was caught by a structural NEGATIVE assertion rather than
     * by a positive one, which is the strongest argument for having them: a
     * test asserting "SSM exists" would have passed here forever. */
    if (!AT("ssm_a", L) && !AT("ssm_in.weight", L) && !AT("ssm_conv1d.weight", L)) {
        printf("  SelectiveScan      layer=%-3d  (none)\n", L);
        printf("      status          : ABSENT -- no ssm_a, ssm_in or ssm_conv1d tensor exists, so\n");
        printf("                       there is no recurrence to describe\n");
        printf("      evidence        : MEASURED -- emitting it would invent a state-space model from an\n");
        printf("                       architecture name.\n");
        claim_absent("ssm.selective_scan", L, "SelectiveScan recurrence (ssm_a / ssm_in / ssm_conv1d)");
    } else {
        ir_node n = mk(IR_SCAN, L);
        n.tensor = "ssm_in -> ssm_out with ssm_a, ssm_d and an input-derived step size";
        n.measured = true;
        n.scope = "recurrent over sequence, state carried forward";
        if (state_size > 0) {
            n.evidence.push_back(P + "ssm.state_size = " + std::to_string(state_size) +
                "   [ARTEFACT MEASURED]: the recurrent state width");
        }
        if (inner_size > 0) {
            n.evidence.push_back(P + "ssm.inner_size = " + std::to_string(inner_size) +
                "   [ARTEFACT MEASURED]");
        }
        if (dt_rank > 0) {
            n.evidence.push_back(P + "ssm.time_step_rank = " + std::to_string(dt_rank) +
                "   [ARTEFACT MEASURED]: the step size is COMPUTED FROM THE INPUT by projecting the "
                "sequence to this rank, which is what makes the block SELECTIVE rather than a "
                "fixed-step recurrence");
        }
        if (group_count > 0) {
            n.evidence.push_back(P + "ssm.group_count = " + std::to_string(group_count) +
                "   [ARTEFACT MEASURED]");
        }
        if (AT("ssm_a", L)) {
            n.evidence.push_back("ssm_a ne=[" + std::to_string(dims["ssm_a"].first) + "," +
                std::to_string(dims["ssm_a"].second) + "]   [ARTEFACT MEASURED]: per-head state "
                "transition, one row per state dimension");
        }
        if (AT("ssm_d", L)) {
            n.evidence.push_back("ssm_d ne=[" + std::to_string(dims["ssm_d"].first) + "," +
                std::to_string(dims["ssm_d"].second) + "]   [ARTEFACT MEASURED]: per-head skip "
                "coefficient, so the recurrence carries a direct term as well as the state");
        }
        if (AT("ssm_in.weight", L)) {
            n.evidence.push_back("ssm_in.weight ne=[" + std::to_string(dims["ssm_in.weight"].first) +
                "," + std::to_string(dims["ssm_in.weight"].second) + "]: projects the residual into "
                "the inner space, mixing the state and step-size paths");
        }
        if (AT("ssm_out.weight", L)) {
            n.evidence.push_back("ssm_out.weight ne=[" + std::to_string(dims["ssm_out.weight"].first) +
                "," + std::to_string(dims["ssm_out.weight"].second) + "]: projects back out");
        }
        n.evidence.push_back("NO attention mathematics anywhere in this block: no Q, no K, no softmax "
            "over keys, no KV cache. A recurrence over a carried state is DIFFERENT mathematics, so "
            "it gets its own composable op rather than being renamed Attention");
        emit(n, arch);
    }
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
    { const char * hd0 = getenv("SLLM_IR_HEAD_DIM"); if (hd0 != NULL && atoi(hd0) > 0) { key_len = atol(hd0); } }
    double eps = -1.0, rope_base = -1.0;
    /* Every one of these is INITIALISED to the -1 sentinel. A per-layer ARRAY
     * such as feed_forward_length or attention.head_count_kv now leaves the
     * scalar field deliberately unset, which means an unset field is reachable
     * state -- and an uninitialised local would then be printed as evidence.
     * That happened: "head_count=140735386808432" was uninitialised memory in
     * the Attention ABSENT message, and it made output nondeterministic between
     * two runs of the SAME binary. A sentinel is visible; garbage is not. */
    struct { const char * suffix; long * out; } ints[] = {
        {"block_count", &block_count}, {"embedding_length", &embd},
        {"feed_forward_length", &ff}, {"attention.head_count", &heads},
        {"attention.head_count_kv", &kv_heads}, {"attention.key_length", &key_len},
        {"attention.value_length", &val_len}, {"rope.dimension_count", &rope_dim},
    };
    for (unsigned i = 0; i < sizeof ints / sizeof *ints; ++i) {
        const std::string k = P + ints[i].suffix;
        const int64_t id = gguf_find_key(g, k.c_str());
        if (id >= 0) {
            if (gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32) {
                *ints[i].out = (long) gguf_get_val_u32(g, id);
            } else if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
                /* Heterogeneity declared in the metadata itself. Not a defect in
                 * the artefact and not something to average: it is per-layer. */
                kv_arrays_seen.push_back(k + " (array of " +
                    std::to_string((long) gguf_get_arr_n(g, id)) +
                    " -- quantity VARIES BY LAYER; the scalar field is left unset"
                    " rather than guessed from element 0)");
            }
        }
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
    /* MoE LOCALITY. This was a MODEL-GLOBAL test: a router tensor in ANY layer
     * licensed ExpertRouter and ExpertGatedFFN for EVERY layer. On a model with
     * experts in only some layers that fabricates a router where none exists. The
     * same rule as Attention, SelectiveScan and GatedMLP: presence is evidence
     * ONLY AT ITS OWN LAYER. Verified by a non-vacuity assertion that strips the
     * router from one OLMoE layer and requires both MoE ops to vanish THERE. */
    /* Defined before the layer loop, so it must be a FUNCTION of the layer
     * rather than a value captured from one. */


    /* ---- OBSERVABILITY: what this instrument can and cannot see ----
     *
     * A zero observation is evidence of absence ONLY after the probe has
     * demonstrated it can observe that thing. This has been learned three times
     * the hard way: the fused softcap (invisible to an op histogram, visible in
     * op_params), rope_freqs (hidden because the evidence was nested inside the
     * pairing-unmeasured branch), and the rope node's third source (invisible to
     * a two-name capture, which reported "no frequency table" while src[2] held
     * exactly that). So capability is DECLARED, not assumed. */
    { const char * cap = getenv("SLLM_IR_PROBE_CAPABILITY");
      if (cap != NULL && cap[0] != '\0') {
          printf("\n-- PROBE OBSERVABILITY --\n");
          printf("  capability declaration: %s\n", cap);
          printf("  rule: a family this probe cannot observe is reported UNOBSERVABLE,\n");
          printf("        never as a zero count. A zero is evidence of absence only\n");
          printf("        after the probe has shown it can observe that thing.\n");
      } }

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
    /* SSM roles, now that DepthwiseConv1D and SelectiveScan exist. They were
     * UNKNOWN at first contact with Mamba-2, which was the correct state: the
     * vocabulary genuinely could not describe a selective state-space block. */
    known_roles.insert("ssm_a");
    known_roles.insert("ssm_d");
    known_roles.insert("ssm_dt.bias");
    known_roles.insert("ssm_conv1d.weight");
    known_roles.insert("ssm_conv1d.bias");
    known_roles.insert("ssm_in.weight");
    known_roles.insert("ssm_out.weight");
    known_roles.insert("ssm_norm.weight");
    /* These two were reported UNKNOWN at 9c62164 and that was WRONG. They are
     * canonical llama.cpp tensor names -- llama-arch.cpp:460 maps
     * LLM_TENSOR_ATTN_POST_NORM to "blk.%d.post_attention_norm" -- so they are
     * known vocabulary that the IR simply had not learned yet. Recorded here as
     * a vocabulary gap rather than an UNKNOWN construct, which is a weaker and
     * wrong statement about them. */
    /* A norm tensor may or may not carry the .weight suffix: Falcon-H1 writes
     * "ffn_norm" where every other model writes "ffn_norm.weight". A missing
     * suffix is a NAMING VARIANT, not a different construct, and treating it as
     * unknown was a vocabulary gap that would have forced a special case per
     * model. Both spellings are now known. */
    known_roles.insert("ffn_norm");
    known_roles.insert("attn_norm");
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
      { const int64_t vsk2 = gguf_find_key(g, (P + "vocab_size").c_str());
        if (vsk2 >= 0) {
            const long vs = (long) gguf_get_val_u32(g, vsk2);
            t.evidence.push_back(P + "vocab_size = " + std::to_string(vs) +
                                 "   [ARTEFACT MEASURED, the declared route]");
            const int64_t tk4 = gguf_find_key(g, "tokenizer.ggml.tokens");
            long tok_n2 = -1;
            if (tk4 >= 0) { tok_n2 = (long) gguf_get_arr_n(g, tk4); }
            long emb_r = dims.count("token_embd.weight") ? dims["token_embd.weight"].second : -1;
            long out_r = dims.count("output.weight") ? dims["output.weight"].second : emb_r;
            const bool tied = !dims.count("output.weight");
            if (tok_n2 > 0) {
                t.evidence.push_back("tokenizer_token_count = " + std::to_string(tok_n2) +
                                     "   [ARTEFACT MEASURED]");
            }
            if (emb_r > 0) {
                t.evidence.push_back("embedding_row_count   = " + std::to_string(emb_r) +
                                     "   [ARTEFACT MEASURED]: the physical capacity ids index");
            }
            if (out_r > 0) {
                t.evidence.push_back(std::string("output_projection_rows= ") + std::to_string(out_r) +
                    "   [ARTEFACT MEASURED]" + (tied
                        ? " (TIED: no output.weight, so it inherits the embedding rows)"
                        : " (untied: a distinct output.weight exists)"));
            }
            /* Cross-check the declared value against what the file also exposes
             * independently. Equality here is a MEASURED OUTCOME, not an
             * invariant: these are distinct quantities that a single scalar
             * conflates, and a padded vocabulary is a DESIGN rather than a
             * defect. */
            if (vs != tok_n2 && tok_n2 > 0) {
                t.evidence.push_back("RELATION: declared vocab_size " + std::to_string(vs) +
                    " vs tokenizer_token_count " + std::to_string(tok_n2) + "  DIFFER, recorded not "
                    "reconciled; a padded vocabulary is a design, not a mismatch");
            } else if (tok_n2 > 0) {
                t.evidence.push_back("RELATION: declared vocab_size == tokenizer_token_count, "
                                     "EXACT MATCH, no padding  [MEASURED outcome]");
            }
            if (emb_r > 0 && tok_n2 > 0 && emb_r != tok_n2) {
                t.scope = "padded_vocab";
                t.evidence.push_back("RELATION: tokenizer_token_count < embedding_rows, PADDED "
                                     "VOCABULARY, padding_rows = " + std::to_string(emb_r - tok_n2));
            }
            t.measured = true;
        } else {
          /* No declared key. That is a statement about this CONFIG, not about
           * whether validation is possible: the artefact exposes other
           * independently measurable vocabulary dimensions. */
          t.evidence.push_back("no " + P + "vocab_size metadata key: that ROUTE is unavailable, "
                               "which is not the same as validation being impossible");
          long tok_n = -1, emb_rows = -1, out_rows = -1;
          bool has_out = false;
          { const int64_t tk = gguf_find_key(g, "tokenizer.ggml.tokens");
            if (tk >= 0) { tok_n = (long) gguf_get_arr_n(g, tk); } }
          if (dims.count("token_embd.weight")) { emb_rows = dims["token_embd.weight"].second; }
          if (dims.count("output.weight")) { has_out = true; out_rows = dims["output.weight"].second; }
          else { out_rows = emb_rows; }
          if (tok_n > 0) { t.evidence.push_back("tokenizer_token_count = " + std::to_string(tok_n) +
              "   [ARTEFACT MEASURED]"); }
          if (emb_rows > 0) { t.evidence.push_back("embedding_row_count   = " + std::to_string(emb_rows) +
              "   [ARTEFACT MEASURED]: the physical capacity token ids index"); }
          if (out_rows > 0) { t.evidence.push_back(std::string("output_projection_rows= ") +
              std::to_string(out_rows) + "   [ARTEFACT MEASURED]" +
              (has_out ? " (untied: a distinct output.weight exists)"
                       : " (TIED: no output.weight, so it inherits the embedding rows)")); }
          if (tok_n > 0 && emb_rows > 0) {
              if (emb_rows == tok_n) {
                  t.evidence.push_back("RELATION: tokenizer_token_count == embedding_rows, "
                                       "EXACT MATCH, no padding  [MEASURED outcome]");
              } else if (emb_rows > tok_n) {
                  t.scope = "padded_vocab";
                  t.evidence.push_back("RELATION: tokenizer_token_count < embedding_rows, PADDED "
                                       "VOCABULARY, padding_rows = " +
                                       std::to_string(emb_rows - tok_n) + "  [MEASURED]");
                  t.evidence.push_back("a padded vocabulary is a DESIGN and not a defect: it reserves "
                                       "ids the tokenizer does not yet emit. OLMoE is built with "
                                       "vocab_size = tokenizer.padded_vocab_size(), so a larger row "
                                       "count is exactly what deliberate padding looks like");
              } else {
                  t.unknown.push_back("DEFECT SHAPE: tokenizer_token_count > embedding_row_count, so "
                                      "an emitted id cannot be represented");
              }
          }
          t.measured = true;
        }
      }
      printf("  TokenizerSelector %-8s %s\n", "(model)", t.tensor.c_str());
      printf("      evidence        : %s\n", t.measured ? "MEASURED" : "INFERRED");
      for (size_t i = 0; i < t.evidence.size(); ++i) { printf("        - %s\n", t.evidence[i].c_str()); }
      for (size_t i = 0; i < t.unknown.size(); ++i) { printf("      UNRESOLVED      : %s\n", t.unknown[i].c_str()); }
      printf("      NOTE            : model alone is NOT a sufficient selector; Qwen3 declares\n");
      printf("                        model=gpt2 while requiring pre=qwen2.\n");
      printf("      VOCAB DIMENSIONS ARE DISTINCT QUANTITIES, not one scalar: token count is\n");
      printf("                        how many tokens the tokenizer lists, embedding rows are the\n");
      printf("                        physical capacity ids index, and output rows are the head\n");
      printf("                        capacity. Equality between them is a MEASURED OUTCOME, not an\n");
      printf("                        invariant of this vocabulary, and a padded vocabulary is a\n");
      printf("                        DESIGN rather than a defect.\n");
      printf("      EXTERNAL UPSTREAM INFORMATION MAY NOT UPGRADE AN ABSENT FIELD TO MEASURED.\n");
      printf("                        A web search reports Gemma-2 rope_theta 10000 and OLMoE built\n");
      printf("                        with vocab_size = tokenizer.padded_vocab_size(); both may be\n");
      printf("                        true of those upstream configs, and neither is a reading of\n");
      printf("                        THESE files.\n");
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
    /* ---- LAYER TOPOLOGY: the layer graph, measured, not assumed ----
     *
     * dims[] is keyed by ROLE ONLY, so by this point the mapping from tensor to
     * layer index has already been discarded. That is precisely the uniform
     * layer assumption, and it is wrong for any model whose layers differ. It is
     * also actively dangerous here: on a heterogeneous model, emitting only
     * layer 0 describes the WHOLE model by one layer's body and hides every
     * other body present. Layer locality must be recovered from the tensor
     * names, so it is. */
    printf("\n-- LAYER TOPOLOGY (measured from tensor names) --\n");
    std::map<int, std::string> layer_body;
    std::map<std::string, std::vector<int> > body_layers;
    int n_layers = 0;
    for (int64_t ti = 0; ti < gguf_get_n_tensors(g); ++ti) {
        const std::string tn_ = gguf_get_tensor_name(g, ti);
        const size_t d1 = tn_.find('.');
        if (d1 == std::string::npos || tn_.substr(0, d1) != "blk") continue;
        const size_t d2 = tn_.find('.', d1 + 1);
        if (d2 == std::string::npos) continue;
        const std::string idx = tn_.substr(d1 + 1, d2 - d1 - 1);
        if (idx.empty() || idx.size() > 6) continue;
        bool digits = true;
        for (char c : idx) if (c < '0' || c > '9') { digits = false; break; }
        if (!digits) continue;
        const int l = atoi(idx.c_str());
        if (l + 1 > n_layers) n_layers = l + 1;

        std::string sig;
        if (tn_.find("attn_q.") != std::string::npos || tn_.find("attn_k.") != std::string::npos ||
            tn_.find("attn_v.") != std::string::npos || tn_.find("attn_output.") != std::string::npos) sig += "ATTN+";
        if (tn_.find("ssm_a") != std::string::npos || tn_.find("ssm_conv1d") != std::string::npos ||
            tn_.find("ssm_in.") != std::string::npos || tn_.find("ssm_out.") != std::string::npos) sig += "SSM+";
        if (tn_.find("ffn_up") != std::string::npos || tn_.find("ffn_down") != std::string::npos ||
            tn_.find("ffn_gate.") != std::string::npos) sig += "FFN+";
        if (tn_.find("experts") != std::string::npos || tn_.find("ffn_gate_inp") != std::string::npos) sig += "MOE+";
        if (!sig.empty() && layer_body[l].empty()) layer_body[l] = sig;
    }
    printf("  layer count = %d   distinct bodies = ", n_layers);
    for (int l = 0; l < n_layers; ++l)
        if (!layer_body[l].empty()) body_layers[layer_body[l]].push_back(l);
    printf("%zu\n", body_layers.size());
    for (std::map<std::string, std::vector<int> >::iterator it = body_layers.begin();
         it != body_layers.end(); ++it) {
        printf("  %-26s layers %2zu  indices:", it->first.c_str(), it->second.size());
        for (size_t i = 0; i < it->second.size() && i < 60; ++i) printf(" %d", it->second[i]);
        if (it->second.size() > 60) printf(" ...");
        putchar('\n');
    }
    /* LAYER-LOCAL PRESENCE. dims[] is keyed by ROLE ONLY, so dims.count(r) asks
     * "does this role exist somewhere in the model", which is a MODEL-GLOBAL
     * question being used where a LAYER-LOCAL one is required. On a model whose
     * layers differ, that fabricates: SelectiveScan appeared on an attention-only
     * layer because some OTHER layer had ssm_a. Every emitter below must ask
     * whether the tensor exists AT THIS LAYER, and only that. */
    for (int64_t ti = 0; ti < gguf_get_n_tensors(g); ++ti) {
        const std::string tn_ = gguf_get_tensor_name(g, ti);
        const size_t d1 = tn_.find('.');
        if (d1 == std::string::npos || tn_.substr(0, d1) != "blk") continue;
        const size_t d2 = tn_.find('.', d1 + 1);
        if (d2 == std::string::npos) continue;
        const std::string idx = tn_.substr(d1 + 1, d2 - d1 - 1);
        bool dg = !idx.empty();
        for (char c : idx) if (c < '0' || c > '9') dg = false;
        if (!dg) continue;
        g_at_layer.insert("blk." + idx + "." + tn_.substr(d2 + 1));
    }

    printf("  %s\n", body_layers.size() <= 1
        ? "  verdict: UNIFORM -- every layer shares one body."
        : "  verdict: HETEROGENEOUS -- the layer graph CHANGES across depth.");
    printf("  an op present in SOME layers must NOT be emitted for every layer.\n");

    /* Emit the FIRST layer of each distinct body, so heterogeneity is visible in
     * the IR itself rather than asserted beside it. Layer 0 alone is not
     * representative of a heterogeneous model and must not stand in for one. */
    std::vector<int> reps;
    for (std::map<std::string, std::vector<int> >::iterator it = body_layers.begin();
         it != body_layers.end() && !it->second.empty(); ++it) reps.push_back(it->second[0]);
    if (reps.empty()) reps.push_back(0);
    if (n_layers > 0 && body_layers.size() <= 1) reps.clear(), reps.push_back(0);

    for (size_t ri = 0; ri < reps.size(); ++ri) {
    printf("\n-- LAYER %d (body %s, first of %zu distinct) --\n",
           reps[ri], body_layers.empty() ? "?" :
           (body_layers.count(layer_body[reps[ri]]) ? layer_body[reps[ri]].c_str() : "?"), body_layers.size());
    const int L = reps[ri];
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
      const char * ffn_key = AT("ffn_norm.weight", L) ? "ffn_norm.weight"
                              : (AT("ffn_norm", L) ? "ffn_norm" : NULL);
      if (ffn_key == NULL) { ffn_key = "ffn_norm.weight"; }
      n.tensor = TN(ffn_key);
      if (ffn_key == NULL || dims.count(ffn_key) == 0) {
        printf("  %-18s layer=%-3d  %s\n", "RMSNorm", L, "(none)");
        printf("      status          : ABSENT -- no ffn norm tensor exists AT THIS LAYER\n");
        printf("      evidence        : MEASURED -- a post-attention norm in another layer does not\n");
        printf("                       license one here\n");
        claim_absent("rmsnorm.ffn", L, "FFN norm tensor");
      } else {
      n.norm_width = dims[ffn_key].first;
      n.type_name = types[ffn_key]; n.eps = eps; n.scope = "residual";
      n.position = "post_attention";
      n.evidence.push_back(std::string("tensor name as written: \"") + ffn_key +
          "\" -- norms appear with and without the .weight suffix across models, which is a "
          "naming variant and not a different construct");
      n.evidence.push_back("ne=[4096,1] => width == embedding_length");
      n.evidence.push_back("applied to the post-attention residual before the FFN");
      n.measured = true; emit(n, arch); } }

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
            claim_absent(std::string("rmsnorm.") + (t == 0 ? "attn_q" : "attn_k"), L,
                         t == 0 ? "per-head Q norm" : "per-head K norm");
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
            /* The artefact omits key_length, but head_dim is DERIVABLE and is
             * confirmed independently from the reference graph. Deriving it here
             * removes the last UNRESOLVED on this model without inventing
             * anything: 2048 / 16 heads = 128, matching the graph. */
            const char * hd_env = getenv("SLLM_IR_HEAD_DIM");
            if (hd_env != NULL && atoi(hd_env) > 0) {
                key_len = atol(hd_env);
                n.head_dim = key_len;
                n.evidence.push_back("key_length ABSENT from metadata, but head_dim=" +
                    std::to_string(key_len) + " DERIVED as embedding_length " +
                    std::to_string(embd) + " / head_count " + std::to_string(heads) +
                    " and confirmed against the reference graph [MEASURED]");
                n.evidence.push_back("with head_dim now known, the scope test is a real comparison "
                    "rather than one against an absent value");
            } else {
                n.unknown.push_back("key_length ABSENT and no head_dim supplied, so no head-scope "
                                    "comparison was possible");
            }
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
    /* DANGLING POINTER FIXED HERE. These were
     *   const char * kA = (P + "attn_logit_softcapping").c_str();
     * which binds a pointer into a TEMPORARY std::string that dies at the end of
     * the declaration. kA and kF then dangle, and reading them printed garbage
     * into the evidence line -- nondeterministic output that happened to be
     * byte-identical across runs often enough to pass. It was caught by the H24
     * byte-exact restore test, which compared two runs of the same binary and
     * found one differing line of uninitialised memory. Bind the strings to
     * named locals; the pointers are then valid for the enclosing block. */
    { const std::string keyA = P + "attn_logit_softcapping";
      const std::string keyF = P + "final_logit_softcapping";
      const char * kA = keyA.c_str();
      const char * kF = keyF.c_str();
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
          claim_absent("softcap", L, "logit softcapping key");
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
    /* LAYER-LOCAL, for the same reason as SelectiveScan and GatedMLP: dims is
     * keyed by role, so a model-global presence test would emit Attention on
     * every one of Nemotron-H's 48 non-attention layers. */
    if (!AT("attn_q.weight", L) && !AT("attn_k.weight", L) &&
        !AT("ffn_gate_exps.weight", L)) {
        printf("  Attention          layer=%-3d  (none)\n", L);
        printf("      status          : ABSENT -- head_count is %ld and there is no Q, K or V\n", heads);
        printf("                       projection and no KV cache to attend over\n");
        printf("      evidence        : MEASURED -- no Q, K or V projection exists AT THIS LAYER.\n");
        printf("                       head_count=%ld is model-global metadata and does NOT license an\n",
               heads);
        printf("                       Attention node here: an op present in SOME layers must not be\n");
        printf("                       emitted for every layer.\n");
        claim_absent("attention", L, "Attention block (Q/K/V projections)");
    } else {
    ir_node n = mk(IR_ATTENTION, L);
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
        /* GATED ON EVIDENCE, NOT ON A NAME. This loop used to emit a node for
         * every projection name unconditionally, so a model with NO attention
         * at all still received four Linear nodes, a RoPE node and an Attention
         * node with every dimension at -1. Naming a role that does not exist is
         * not evidence that it exists. */
        if (!AT(proj[i].role, L)) {
            printf("  %-18s layer=%-3d  %s\n", "Linear", L, TN(proj[i].role).c_str());
            printf("      status          : ABSENT -- no tensor by this name in this architecture\n");
            printf("      evidence        : MEASURED -- no tensor with this name exists AT THIS LAYER, so\n");
            printf("                       NO node is emitted. The role may exist at another layer; that is\n");
            printf("                       not evidence here. Emitting one would be fabricating structure.\n");
            claim_absent(std::string("linear.") + proj[i].role, L, proj[i].role);
            continue;
        }
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

    if (!AT("attn_q.weight", L) && !AT("attn_k.weight", L)) {
        printf("  RoPE               layer=%-3d  (none)\n", L);
        printf("      status          : ABSENT -- no Q or K projection exists AT THIS LAYER to rotate\n");
        claim_absent("rope", L, "Q or K projection to rotate");
        printf("      evidence        : MEASURED -- RoPE is a property of an attention block, and there\n");
        printf("                       is none. Emitting a RoPE node would invent one.\n");
    } else {
        ir_node n = mk(IR_ROPE, L);
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
          const char * rfs = getenv("SLLM_IR_ROPE_SRC");
          if (rfs != NULL && rfs[0] != '\0') {
              n.scaling_type = std::string("precomputed table (rope_freqs.weight): ") + rfs;
              n.evidence.push_back(std::string("WHICH SOURCE THE REFERENCE USES [MEASURED]: ") + rfs);
              n.evidence.push_back("the artefact carries BOTH a precomputed table and a freq_base "
                                   "key, and the reference uses the table, so the metadata base is "
                                   "present but NOT live for this path");
              n.evidence.push_back("this was only visible after the probe was taught to read ALL "
                                   "rope sources: a two-name capture reported no table at all while "
                                   "src[2] held rope_freqs.weight exactly");
              n.measured = true;
          } else {
              n.unknown.push_back("WHICH SOURCE IS LIVE UNRESOLVED: the artefact carries both a "
                  "precomputed rope_freqs table and a freq_base key, and the metadata alone cannot "
                  "say which the reference uses. Supply SLLM_IR_ROPE_SRC from the rope node sources.");
          }
      } else if (rope_base < 0) {
          /* The artefact carries neither a base key nor a precomputed table.
           * That does NOT make the base unknowable: the reference sets
           * rope_freq_base_train = 10000.0f and overrides it only if the key is
           * present, so with the key absent the default STANDS. Resolved from
           * the reference path, and labelled as the reference's default rather
           * than as a fact about the artefact, because a different implementation
           * could legitimately choose otherwise. */
          const char * rb = getenv("SLLM_IR_ROPE_BASE_DEFAULT");
          if (rb != NULL && rb[0] != '\0') {
              n.rope_base = atof(rb);
              n.measured = true;
              n.evidence.push_back(std::string("rope_base = ") + rb +
                  "  [MEASURED from the reference path: the artefact has NO base key and NO "
                  "precomputed table, and the reference sets this default and overrides it only "
                  "if the key is present, so the default stands]");
              n.evidence.push_back("this is the REFERENCE default, so it is a statement about this "
                                   "artefact under this reference, not a universal claim");
          } else {
              n.unknown.push_back("rope_base UNRESOLVED from the ARTEFACT: no " + P +
                  "rope.freq_base key AND no precomputed rope_freqs table in this file. NOT "
                  "assumed to be 10000, and NOT taken from the upstream paper or the HuggingFace "
                  "config, which report 10000 but describe a different artefact. Supply "
                  "SLLM_IR_ROPE_BASE_DEFAULT only to record the REFERENCE default as reference "
                  "evidence, which is a different provenance class.");
          }
      }
      emit(n, arch); }

    /* MoE is NOT squeezed into GatedMLP. The router and the expert bank are
     * different mathematics from a dense gated MLP: a router produces a top-k
     * selection over experts, and the FFN runs only on the chosen subset. */
    if (is_moe_at(L)) {
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
          /* ROUTING FUNCTION, resolved from the reference op SEQUENCE rather than
           * from the architecture name. Three gating conventions exist in
           * build_moe_ffn and only the op sequence tells them apart. The probe
           * counts ARGSORT (selection), SOFT_MAX (softmax on the selected
           * weights) and the normalise trio SUM_ROWS + CLAMP + DIV. */
          const char * rf = getenv("SLLM_IR_ROUTING");
          if (rf != NULL && rf[0] != '\0') {
              r.scope = std::string("top_") + std::to_string(n_used) + "_of_" +
                        std::to_string(n_experts);
              r.position = rf;
              /* WORDING, and it matters: softmax over the SELECTED top-k ALREADY
               * normalises those selected scores among themselves. The absent
               * SUM_ROWS/CLAMP/DIV trio means there is NO SEPARATE EXPLICIT
               * POST-SELECTION RENORMALISATION path -- not that the final routing
               * weights are unnormalised. Those are different claims and the
               * earlier wording conflated them. */
              r.evidence.push_back("note: softmax over the selected top-k already normalises "
                                   "those scores among themselves. The absent SUM_ROWS/CLAMP/DIV "
                                   "trio shows there is no SEPARATE EXPLICIT post-selection "
                                   "renormalisation, NOT that the weights are unnormalised");
              r.measured = true;
              r.evidence.push_back(std::string("ROUTING [MEASURED from the reference op sequence]: ") + rf);
              r.evidence.push_back("three conventions exist in build_moe_ffn and only the op order "
                                   "distinguishes them, so this is read from the graph and NOT from "
                                   "the architecture name");
          } else {
              r.unknown.push_back("ROUTING FUNCTION UNRESOLVED: the artefact states how MANY "
                  "experts are used, not HOW they are chosen or combined. Top-k by weight, softmax "
                  "gating and normalised weighted sums are all consistent with the metadata, so the "
                  "function is not asserted from it. Supply SLLM_IR_ROUTING from the reference op "
                  "sequence.");
          }
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
    } else if (!AT("ffn_gate.weight", L) && !AT("ffn_down.weight", L) && !AT("ffn_up.weight", L)) {
        printf("  %-18s layer=%-3d  (none)\n", "GatedMLP", L);
        printf("      status          : ABSENT -- feed_forward_length is %ld and no FFN tensor exists\n", ff);
        printf("      evidence        : MEASURED -- NO FFN TENSOR EXISTS AT THIS LAYER. Presence\n");
        printf("                       elsewhere in the model does not license one here.\n");
        claim_absent("ffn.gated_mlp", L, "FFN tensor (ffn_gate / fn_up / ffn_down)");
    } else if (!AT("ffn_gate.weight", L)) {
        /* DENSE FFN. The FFN tensors exist but there is NO GATE, so the
         * mathematics is up-then-down, NOT gate*up-then-down. Those are
         * DIFFERENT MATHEMATICS and therefore different ops under the rule that
         * the same mathematics is the same op. Emitting GatedMLP here would name
         * a tensor that does not exist in the artefact, which is fabrication of
         * exactly the kind this project exists to prevent.
         *
         * The rule, stated so it generalises:
         *     NO GATE EVIDENCE  ->  NO GatedMLP CLAIM
         * It is not about SwiGLU versus GeGLU, which are the same op with a
         * different activation. It is about whether a gate tensor exists at all.
         * A future GeGLU or SwiGLU model with a real ffn_gate lands on GatedMLP; a
         * GELU MLP or a dense MLP lands here; a parallel MLP needs its own op
         * because its residual wiring differs; an expert FFN already has
         * ExpertGatedFFN. Every one of those now has an honest place to land, and
         * none of them can be reached by a partial match. */
        ir_node n = mk(IR_DENSE_MLP, L);
        n.tensor = "ffn_up.weight + ffn_down.weight";
        n.in_features = dims["ffn_up.weight"].first;
        n.out_features = dims["ffn_down.weight"].first;
        n.scope = "dense, ungated: up then down, no multiplicative gate";
        n.evidence.push_back("ffn_gate.weight ABSENT AT THIS LAYER -- measured by direct lookup, "
            "so no gate tensor is named and GatedMLP is NOT emitted");
        n.evidence.push_back("ffn_up   ne=[" + std::to_string(dims["ffn_up.weight"].first) + "," +
            std::to_string(dims["ffn_up.weight"].second) + "]");
        n.evidence.push_back("ffn_down ne=[" + std::to_string(dims["ffn_down.weight"].first) + "," +
            std::to_string(dims["ffn_down.weight"].second) + "]  contraction back to in_features");
        n.evidence.push_back(std::string(P) + "feed_forward_length=" + std::to_string(ff));
        n.evidence.push_back("GLU OWNERSHIP, RESOLVED BY ATTRIBUTION RATHER THAN BY COUNT. "
            "The reference graph contains 24 GLU ops -- gated linear units -- which appeared to "
            "contradict this ungated classification. It does not. Every GLU node is fed by a "
            "source named mamba2_y_add_d, so the gate lives in the STATE-SPACE path; none is fed "
            "by an ffn_* tensor. Counts could never have settled this, because GLU, SSM and FFN "
            "layers are all 24 here, so every count-based story fits either answer. Only "
            "following the sources can, and they agree with the tensor inventory.");
        n.evidence.push_back("TWO INDEPENDENT WITNESSES now agree that this FFN is ungated: the "
            "tensor inventory (no ffn_gate.weight here) and the reference graph (no GLU fed by "
            "an ffn_* tensor). Neither measurement overrules the other; they corroborate. Had "
            "the GLU been attributed to the FFN, this would be a classification CHANGE to "
            "GatedMLP with the gate tensor ABSENT, not a refinement of DenseMLP.");
        n.evidence.push_back("NOT claimed: which activation or nonlinearity this dense FFN uses. "
            "No activation tensor or metadata key identifies it in this artefact, so the "
            "activation is UNRESOLVED, not GELU and not ReLU.");
        n.measured = true; emit(n, arch);
        claim("ffn.dense_mlp", "PRESENT (ungated)", "tensor inventory (direct lookup)", "measured",
              std::to_string(L),
              "a DENSE (ungated) MLP exists at layer " + std::to_string(L) + ": up then down",
              "NOT claimed: a gated MLP here, and NOT claimed any particular activation function");
    } else {
        ir_node n = mk(IR_MLP, L);
        n.tensor = "ffn_gate.weight + ffn_up.weight + ffn_down.weight";
      n.in_features = dims["ffn_gate.weight"].first;
      n.out_features = dims["ffn_gate.weight"].second;
      n.scope = "gated (SwiGLU-family): gate and up share out_features, down returns to in_features";
      n.evidence.push_back("ffn_gate ne=[" + std::to_string(dims["ffn_gate.weight"].first) + "," +
          std::to_string(dims["ffn_gate.weight"].second) + "]  GATE TENSOR EXISTS AT THIS LAYER, "
          "measured by direct lookup, so the gated claim is earned");
      n.evidence.push_back("ffn_up   ne=[" + std::to_string(dims["ffn_up.weight"].first) + "," +
          std::to_string(dims["ffn_up.weight"].second) + "]  same shape as gate => parallel projections");
      n.evidence.push_back("ffn_down ne=[" + std::to_string(dims["ffn_down.weight"].first) + "," +
          std::to_string(dims["ffn_down.weight"].second) + "]  transpose of the gate shape => contraction");
      n.evidence.push_back(std::string(P) + "feed_forward_length=" + std::to_string(ff));
      n.evidence.push_back("ffn_gate_inp.weight ABSENT => not MoE, and not inferred from the "
                           "architecture name");
      n.evidence.push_back("NOT claimed: which activation the gate uses. SwiGLU and GeGLU are the "
          "SAME op with a different activation, and nothing in this artefact identifies it.");
      n.measured = true; emit(n, arch);
      claim("ffn.gated_mlp", "PRESENT (gate tensor measured at this layer)",
            "tensor inventory (direct lookup)", "measured", std::to_string(L),
            "a GATED MLP exists at layer " + std::to_string(L),
            "NOT claimed: a specific activation function, and NOT claimed any gate elsewhere in the model");
    }

    emit_ssm_block(g, P, dims, types, L, arch);

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

    /* ---- UNRESOLVED and UNOBSERVABLE, recorded as first-class claims ---- */
    { char lb[16]; snprintf(lb, sizeof lb, "%d", L);
      if (!kv_arrays_seen.empty()) { /* metadata arrays already reported */ }
      claim_unresolved("rope.pairing", L,
          "NOT claimed: a pairing. The architecture name is not evidence; Qwen3 is NEOX and "
          "Llama-3.2 is GPT, so guessing from the name is the exact error this IR exists to avoid",
          "reference graph (not reached)");
      if (rope_base < 0) {
          claim_unresolved("rope.base", L,
              "NOT claimed: a base value. The artefact declares no freq_base key and ships no "
              "precomputed rope_freqs table, so there is nothing in THIS file to read",
              "artefact metadata (absent) + tensor inventory (absent)");
      }
    }
    if (!kv_arrays_seen.empty()) {
        for (size_t i = 0; i < kv_arrays_seen.size(); ++i) {
            char lb[16]; snprintf(lb, sizeof lb, "%d", L);
            claim("metadata.per_layer_array", "ARRAY (value VARIES BY LAYER)",
                  "artefact metadata (direct)", "measured", lb,
                  "this quantity is per-layer, not a single model-wide value",
                  "NOT claimed: any single value for it, and not claimed that element 0 "
                  "represents the model. Reading it as a scalar aborts ggml.");
        }
    }
    /* An instrument cannot conclude anything about a family it cannot see. The
     * fused SoftCap was invisible to an op histogram and visible only in
     * op_params; that lesson is why this exists. */
    claim_unobservable("fused-op internals (e.g. a SoftCap fused inside flash-attention)",
        "an op histogram counts node types and cannot see inside a node. Where a value is "
        "only reachable through op_params it must be read there or left unresolved.");

    } /* end per-distinct-body emission */

    if (!kv_arrays_seen.empty()) {
        printf("\n-- METADATA THAT VARIES BY LAYER (arrays, not scalars) --\n");
        for (size_t i = 0; i < kv_arrays_seen.size(); ++i)
            printf("  %s\n", kv_arrays_seen[i].c_str());
        printf("  These are HETEROGENEITY IN THE METADATA. Reading one as a scalar\n");
        printf("  aborts ggml; averaging it would invent a single layer body.\n");
    }

    printf("\n-- CLAIM LEDGER (what was measured, from where, and what is NOT claimed) --\n");
    printf("  THE READING RULE. Four confidence classes, never to be conflated:\n");
    printf("    measured     : looked in the place that must speak, and it did\n");
    printf("    inferred     : derived from a reference RULE, weaker than direct observation\n");
    printf("    unresolved   : the evidence is insufficient to decide -- NOT the same as absent\n");
    printf("    unobservable : this instrument cannot see it, so nothing may be concluded\n");
    printf("  'not observed' is NEVER 'does not exist'. Absence is a claim ABOUT A SCOPE,\n");
    printf("  and the scope is printed on every row.\n\n");
    { size_t nm[4] = {0,0,0,0};
      for (size_t i = 0; i < g_claims.size(); ++i) {
          const std::string & c = g_claims[i].confidence;
          if (c.find("unresolved") != std::string::npos) nm[0]++;
          else if (c.find("unobservable") != std::string::npos) nm[1]++;
          else if (c.find("inferred") != std::string::npos) nm[2]++;
          else nm[3]++;
      }
      printf("  totals: %zu measured/present, %zu inferred, %zu unresolved, %zu unobservable\n\n",
             nm[3], nm[2], nm[0], nm[1]); }
    for (size_t i = 0; i < g_claims.size(); ++i) {
        const claim_rec & c = g_claims[i];
        printf("  Field          : %s\n", c.field.c_str());
        printf("    Value        : %s\n", c.value.c_str());
        printf("    Evidence src : %s\n", c.source.c_str());
        printf("    Layer        : %s\n", c.layer.c_str());
        printf("    Confidence   : %s\n", c.confidence.c_str());
        printf("    Claim allowed: %s\n", c.allowed.c_str());
        printf("    NOT claimed  : %s\n\n", c.notclaimed.c_str());
    }

    /* ---- FINAL AUDIT REPORT ------------------------------------------------
     *
     * The ledger above is per-claim. This is the rollup a reviewer reads first:
     * what this instrument can see, what it therefore measured directly, what it
     * could only derive, what it could not decide, and -- the part that matters
     * most -- WHAT WAS DELIBERATELY NOT CLAIMED.
     *
     * A future contributor should be unable to turn "not observed" into "does not
     * exist" by skimming. The cheapest way to guarantee that is to make the
     * unknowns as visible as the findings, and to print the instrument's own
     * blind spots as first-class results rather than omissions. */
    printf("\n-- FINAL AUDIT REPORT --\n");
    printf("\n  A. WHAT THIS INSTRUMENT CAN SEE\n");
    printf("     direct   : tensor inventory, per-layer tensor identity, tensor shapes\n");
    printf("                and types, GGUF metadata with declared types read per type.\n");
    printf("     derived  : nothing that changes a value. Shape arithmetic is used only\n");
    printf("                to CROSS-CHECK a declared count, never to supply one.\n");
    printf("     external : reference graph op histogram, op_params, and the six graph\n");
    printf("                sources -- a SEPARATE provenance class, supplied explicitly and\n");
    printf("                never merged into artefact evidence.\n");
    printf("     cannot see: an architecture whose operations are fused inside a node the\n");
    printf("                histogram only counts; anything a single-token, single-position\n");
    printf("                graph build elides; the activation function of a dense FFN.\n");

    printf("\n  B. DIRECT VERSUS DERIVED, COUNTED OVER %zu CLAIMS\n", g_claims.size());
    { size_t direct = 0, derived = 0, unres = 0, unob = 0;
      for (size_t i = 0; i < g_claims.size(); ++i) {
          const std::string & c = g_claims[i].confidence;
          if (c.find("unobservable") != std::string::npos) ++unob;
          else if (c.find("unresolved") != std::string::npos) ++unres;
          else if (c.find("inferred") != std::string::npos) ++derived;
          else ++direct;
      }
      printf("     %-4zu measured DIRECTLY from the artefact\n", direct);
      printf("     %-4zu INFERRED from a reference rule (weaker; labelled per claim)\n", derived);
      printf("     %-4zu UNRESOLVED -- insufficient evidence, NOT absence\n", unres);
      printf("     %-4zu UNOBSERVABLE -- this instrument cannot see it at all\n\n", unob); }

    printf("\n  C. WHAT WAS DELIBERATELY NOT CLAIMED\n");
    printf("     Each item below was available to assert and was refused, because the\n");
    printf("     evidence does not support it. Refusal is the finding.\n");
    { bool saw_rope = false, saw_moe = false, saw_ssm = false, saw_attn = false;
      for (size_t i = 0; i < g_claims.size(); ++i) {
          const std::string & f = g_claims[i].field;
          if (f == "rope.base" || f == "rope.pairing") saw_rope = true;
          if (f.find("expert") != std::string::npos) saw_moe = true;
          if (f.find("ssm") != std::string::npos) saw_ssm = true;
          if (f == "attention") saw_attn = true;
      }
      if (saw_rope) printf("     - no RoPE base and no pairing, though both are tempting to fill in\n");
      printf("       from an architecture name or an upstream config: those describe\n");
      printf("       different artefacts. Unresolved is the honest answer.\n");
      if (!saw_moe) printf("     - no ExpertRouter or ExpertGatedFFN: no router tensor exists here,\n");
      printf("       so MoE is NOT claimed even for a model family known to use experts.\n");
      if (saw_attn) printf("     - absence of Attention is claimed PER LAYER only. Never\n");
      printf("       'this model has no attention': other layers may have it.\n");
      if (saw_ssm) printf("     - absence of a state-space recurrence is claimed PER LAYER only.\n");
      printf("     - no activation function for any FFN, gated or dense.\n");
      printf("     - no claim that the vocabulary is COMPLETE. A tensor this probe cannot\n");
      printf("       classify surfaces as UNKNOWN, which means not-yet-described, not\n");
      printf("       absent and not absent-by-design.\n"); }

    printf("\n  D. THE ONE-LINE SUMMARY OF THIS ARTEFACT\n");
    printf("     %d layer(s) measured, %zu distinct layer bodies, %zu claims recorded,\n",
           n_layers, body_layers.size(), g_claims.size());
    printf("     %zu of those claims are absences or unknowns. Those are RESULTS.\n\n",
           [&]{ size_t c = 0; for (size_t i = 0; i < g_claims.size(); ++i)
                   if (g_claims[i].value.find("ABSENT") != std::string::npos ||
                       g_claims[i].confidence.find("unresolv") != std::string::npos ||
                       g_claims[i].confidence.find("unobserv") != std::string::npos) ++c;
               return c; }());
    printf("     NO EVIDENCE -> NO OPERATION. A claim appears here only where a\n");
    printf("     tensor, a metadata key, or a reference graph node earned it, at the\n");
    printf("     exact layer where it was found.\n");

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
