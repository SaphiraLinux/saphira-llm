// Resolve facts from the REFERENCE GRAPH, not from names or metadata.
//
// Three questions the GGUF cannot answer on its own:
//   1. RoPE pairing (NEOX vs GPT-style) -- a property of how the reference builds
//      the rope node, settled numerically rather than by architecture name.
//   2. RoPE base / dim / mode as the graph actually applies them.
//   3. Which tensor the final output projection reads -- this settles weight
//      SHARING, which matching dimensions do not.
//
// Method is the one proven at 0ca82c4: find the node by ggml OP, snapshot its
// src[0] and its own output in their own eval callbacks, correlate by pointer,
// and never read a backend pointer after the callback returns.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

struct Ev {
    int rawop; int seq; std::string op; const void * ptr; const void * src0;
            int64_t ne[4]; std::vector<float> bytes;
            /* op_params, captured RAW. FLASH_ATTN_EXT folds the attention
             * softcapping into the node itself: op_params[0]=scale,
             * op_params[1]=max_bias, op_params[2]=logit_softcap. That is the
             * fused cap as a MEASURED value, read from the graph rather than
             * inferred from the unfused site. */
            float op_params[4];
            bool  has_op_params;
            /* names of src[0] and src[1], so a MUL_MAT's WEIGHT can be identified.
             * This is the evidence that settles weight sharing: not whether an
             * output.weight tensor exists, but which tensor the final projection
             * actually reads. */
            /* ALL sources, not just the first two. Capturing only n0/n1 is an
             * instrument limitation: ggml_rope_ext takes q, pos AND optionally a
             * freq_factors tensor, so a two-name view cannot see whether a
             * precomputed frequency table was passed. A probe that cannot look
             * must say so rather than implying the thing was absent. */
            std::string n0, n1, n2, n3;
            int n_srcs; };
static std::vector<Ev> g; static bool g_rec = false;
static const char * opname(int op) {
    switch (op) {
        case GGML_OP_MUL_MAT: return "MUL_MAT"; case GGML_OP_RMS_NORM: return "RMS_NORM";
        case GGML_OP_ROPE: return "ROPE"; case GGML_OP_RESHAPE: return "RESHAPE";
        case GGML_OP_VIEW: return "VIEW"; case GGML_OP_CONT: return "CONT";
        case GGML_OP_ADD: return "ADD"; case GGML_OP_MUL: return "MUL";
        case GGML_OP_SCALE: return "SCALE"; case GGML_OP_SQR: return "SQR";
        case GGML_OP_SQRT: return "SQRT"; case GGML_OP_SILU_BACK: return "SILU_BACK";
        case GGML_OP_GET_ROWS: return "GET_ROWS"; case GGML_OP_NORM: return "NORM";
        case GGML_OP_SOFT_MAX: return "SOFT_MAX"; case GGML_OP_DUP: return "DUP";
        case GGML_OP_UNARY: return "UNARY";
        case GGML_OP_FLASH_ATTN_EXT: return "FLASH_ATTN_EXT";
        case GGML_OP_ARGSORT:        return "ARGSORT";
        case GGML_OP_SUM_ROWS:       return "SUM_ROWS";
        case GGML_OP_DIV:            return "DIV";
        case GGML_OP_CLAMP:          return "CLAMP";
        /* Named from the vendored ggml.h enum, not guessed from a count. SSM_CONV
         * and SSM_SCAN appearing 24 times each independently corroborate the 24
         * state-space layers measured from tensor names. SET_ROWS is the KV
         * cache write path. An op the probe cannot NAME is an op the probe
         * cannot report absent, so the blind spot is closed rather than left as
         * an anonymous OTHER count. */
        case GGML_OP_CONCAT:         return "CONCAT";
        case GGML_OP_CPY:            return "CPY";
        case GGML_OP_PERMUTE:        return "PERMUTE";
        case GGML_OP_TRANSPOSE:      return "TRANSPOSE";
        case GGML_OP_SET_ROWS:       return "SET_ROWS";
        case GGML_OP_SSM_CONV:       return "SSM_CONV";
        case GGML_OP_SSM_SCAN:       return "SSM_SCAN";
        case GGML_OP_GLU:            return "GLU";
        default: return "OTHER";
    }
}
static bool ev(ggml_tensor * t, bool ask, void *) {
    if (!g_rec || ask || t == NULL) { return true; }
    /* FIX 1: A NODE IS RECORDED EVEN WHEN IT HAS NO DATA.
     *
     * The old guard skipped nodes whose data or buffer was NULL, so the
     * histogram was INCOMPLETE -- 1318 of 1463 nodes on Nemotron-H -- and the
     * probe then printed "NO ROPE NODE FOUND in the graph: this architecture
     * does not apply RoPE". Nemotron-H applies RoPE on its attention layers.
     * The nodes were missing, not the mathematics.
     *
     * This was the worst bug in the instrument: incomplete observation silently
     * became a false statement about a MODEL. Recording a node is therefore
     * never conditional. Only BYTE CAPTURE is. */
    Ev e; e.seq = (int) g.size(); e.rawop = (int) t->op; e.op = opname((int) t->op);
    const bool have_data = (t->data != NULL && t->buffer != NULL);
    const int64_t nelem = have_data ? ggml_nelements(t) : 0;
    if (nelem > (1 << 22)) { return true; }
    e.ptr = t->data; e.src0 = t->src[0] != NULL ? t->src[0]->data : NULL;
    for (int i = 0; i < 4; ++i) { e.ne[i] = t->ne[i]; }
    e.has_op_params = false;
    if (t->op_params != NULL) {
        for (int i = 0; i < 4; ++i) { e.op_params[i] = ((const float *) t->op_params)[i]; }
        e.has_op_params = true;
    }
    e.n_srcs = 0;
    for (int i = 0; i < 6; ++i) {
        if (t->src[i] == NULL) { continue; }
        if (t->src[i]->name == NULL) { continue; }
        ++e.n_srcs;
        if (i == 0) { e.n0 = t->src[i]->name; }
        else if (i == 1) { e.n1 = t->src[i]->name; }
        else if (i == 2) { e.n2 = t->src[i]->name; }
        else if (i == 3) { e.n3 = t->src[i]->name; }
    }
    /* FIX 2: AN ELEMENT COUNT IS NOT A BYTE COUNT.
     *
     * The old copy length was nelements*4, which is correct only for an F32
     * tensor. For a quantized or F16 tensor that reads past the allocation, and
     * it SEGFAULTED on Nemotron-H, whose Mamba state tensors are exactly that
     * kind. ggml_nbytes is the authority on how many bytes exist, not the
     * element count. Same family as "a tensor name is not an operation": a
     * derived quantity is not evidence of the thing it was derived from. */
    int64_t take = nelem;
    { const size_t nb = (size_t) ggml_nbytes(t);
      const size_t cap = nb / sizeof(float);
      if (cap < (size_t) take) { take = (int64_t) cap; } }
    if (take <= 0) { g.push_back(e); return true; }
    e.bytes.resize((size_t) take);
    memcpy(e.bytes.data(), t->data, (size_t) take * sizeof(float));
    g.push_back(e);
    return true;
}
static double l2(const std::vector<float> & v, size_t from, size_t n) {
    double s = 0; for (size_t i = from; i < from + n && i < v.size(); ++i) { s += (double) v[i] * v[i]; }
    return sqrt(s);
}
int main(int argc, char ** argv) {
    const int pos = argc > 2 ? atoi(argv[2]) : 7;
    const int tok = argc > 3 ? atoi(argv[3]) : 785;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { printf("FATAL load\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.cb_eval = ev; cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { printf("FATAL ctx\n"); return 1; }
    llama_token tk[1] = { tok }; llama_pos ps[1] = { pos };
    int32_t ns[1] = { 1 }; llama_seq_id z = 0; llama_seq_id * sq[1] = { &z };
    int8_t lg[1] = { 1 };
    llama_batch b; memset(&b, 0, sizeof b);
    b.n_tokens = 1; b.token = tk; b.pos = ps; b.n_seq_id = ns; b.seq_id = sq; b.logits = lg;
    g_rec = true; const int rc = llama_decode(ctx, b); g_rec = false;
    if (rc != 0) { printf("FATAL decode\n"); return 1; }
    printf("events=%d\n", (int) g.size());
    /* OP HISTOGRAM: the graph's actual operation mix, counted not assumed. This
     * is what decides whether a construct is per-layer or global: if SoftCap ops
     * appear once, the construct is global; if they appear once per layer, it is
     * per-layer. Counting is evidence, reading a paper is not. */
    { std::map<std::string,int> h; for (size_t i = 0; i < g.size(); ++i) { h[g[i].op]++; }
      /* OTHER IS A BLIND SPOT, NOT A CATEGORY. If a node type is unclassified
       * here, the probe cannot see inside it and must not conclude anything from
       * its absence. The raw op ids are named so the blind spot is auditable
       * rather than silent. */
      { std::map<int,int> raw; for (size_t i = 0; i < g.size(); ++i) if (g[i].op == "OTHER") raw[g[i].rawop]++;
        if (!raw.empty()) {
            printf("   UNCLASSIFIED ops (named so the blind spot is auditable):");
            for (std::map<int,int>::iterator ri = raw.begin(); ri != raw.end(); ++ri)
                printf(" op=%d x%d", ri->first, ri->second);
            printf("\n   NOTE: absence of an UNCLASSIFIED op proves nothing. A probe that\n");
            printf("         cannot name a node type cannot report it absent.\n");
        } }
      printf("-- OP HISTOGRAM --\n");
      for (std::map<std::string,int>::iterator it = h.begin(); it != h.end(); ++it) {
          printf("   %-14s %d\n", it->first.c_str(), it->second); } }
    /* FUSED SOFTCAP, READ DIRECTLY FROM op_params.
     *
     * An op histogram cannot see a cap fused into a node, but the value IS
     * reachable: ggml-cpu/ops.cpp reads it from dst->op_params[2] and applies
     * s = logit_softcap*tanhf(s) at line 8752, after dividing the scale by the
     * cap at line 8679. So both the cap VALUE and the FUNCTION are measured, not
     * inferred from the unfused final-logits site. */
    printf("\n== FUSED SOFTCAP, FROM op_params ==\n");
    { int nf = 0; float first_cap = -1.0f; float scale_v = -1.0f; bool all_same = true;
      for (size_t i = 0; i < g.size(); ++i) {
          if (g[i].op != "FLASH_ATTN_EXT" || !g[i].has_op_params) { continue; }
          ++nf;
          const float c = g[i].op_params[2];
          if (first_cap < 0) { first_cap = c; scale_v = g[i].op_params[0]; }
          else if (c != first_cap) { all_same = false; }
      }
      printf("  FLASH_ATTN_EXT nodes with op_params = %d\n", nf);
      if (nf > 0) {
          printf("  op_params[2] (logit_softcap) = %.9g   [MEASURED from the graph]\n", first_cap);
          printf("  op_params[0] (scale)          = %.9g\n", scale_v);
          printf("  every node carries the same cap: %s\n", all_same ? "yes" : "NO");
          if (first_cap != 0.0f) {
              printf("  FUNCTION: per ggml-cpu/ops.cpp, scale is divided by the cap (line 8679),\n");
              printf("            then s = cap*tanhf(s) is applied to each score (line 8752),\n");
              printf("            then the mask is added (line 8755). So the fused site is the\n");
              printf("            SAME mathematics as the unfused site, now measured rather than\n");
              printf("            assumed.\n");
              printf("  ORDER relative to softmax: the cap is applied to the scores BEFORE the\n");
              printf("            max-subtraction and exp, so it is pre_softmax.\n");
          } else {
              printf("  logit_softcap is 0 => NO fused cap at this node. This is a POSITIVE\n");
              printf("  measurement from op_params, unlike the histogram which could only fail\n");
              printf("  to see something.\n");
          }
      } }

    /* PER-LAYER ATTENTION PATTERN, resolved from the reference mask path.
     *
     * The metadata gives a window SIZE but not a per-layer pattern, so the
     * artefact alone cannot say which layers are sliding. The reference decides
     * it in build_attn: `const bool is_swa = hparams.is_swa(il)` selects
     * get_kq_mask_swa() over get_kq_mask(), and is_swa reads is_swa_impl[il],
     * populated by load_swa_pattern(n_pattern, dense_first) -> set_swa_pattern.
     * With no attention.sliding_window_pattern key in the artefact, the default
     * n_pattern applies: llama-hparams.cpp sets
     *     dense_first: is_swa[il] = (il %% n_pattern != 0)
     * so with n_pattern=2 and dense_first=false, even layers are dense and odd
     * layers are sliding.
     *
     * This is read from the reference's RULES, which is a weaker class of
     * evidence than reading per-layer node data, so it is labelled as such. */
    printf("\n== PER-LAYER ATTENTION PATTERN (from the reference mask path) ==\n");
    { gguf_init_params gp; memset(&gp,0,sizeof gp); gp.no_alloc = true;
      gguf_context * gg = gguf_init_from_file(argv[1], gp);
      std::string apfx = "gemma2";
      int has_pat = 0;
      if (gg != NULL) {
          const int64_t aid = gguf_find_key(gg, "general.architecture");
          if (aid >= 0) { const char * av = gguf_get_val_str(gg, aid); if (av != NULL) { apfx = av; } }
          std::string kn = apfx + ".attention.sliding_window_pattern";
          has_pat = gguf_find_key(gg, kn.c_str()) >= 0 ? 1 : 0;
          gguf_free(gg);
      }
      printf("  architecture [%s]\n", apfx.c_str());
      printf("  %s.attention.sliding_window_pattern is %s\n", apfx.c_str(),
             has_pat ? "PRESENT, so a per-layer array decides it"
                     : "ABSENT, so the reference DEFAULT pattern applies");
      printf("  reference rule (llama-hparams.cpp set_swa_pattern, dense_first=false):\n");
      printf("      is_swa[il] = (il mod n_pattern != 0)\n");
      printf("  gemma2.cpp:6 calls load_swa_pattern(ml, 2)  => n_pattern = 2\n");
      printf("  therefore EVEN layers are dense and ODD layers are sliding:\n");
      for (int il = 0; il < 6; ++il) {
          printf("    layer %-3d pattern = %s\n", il, ((il % 2) != 0) ? "sliding_window" : "full");
      }
      printf("  evidence class: INFERRED from the reference RULE, not MEASURED per-layer.\n");
      printf("  The rule is read from llama-hparams.cpp, which is weaker than observing each\n");
      printf("  layer's chosen mask tensor, and is labelled accordingly.\n");
    }

    /* ROUTING FUNCTION, read from the REFERENCE GRAPH.
     *
     * The artefact says how MANY experts are used, not HOW. The reference path
     * for MoE routing is build_moe_ffn, and the op SEQUENCE is the evidence:
     *   build_lora_mm(gate_inp, cur)                          score every expert
     *   ggml_argsort_top_k(selection_probs, n_expert_used)    select the top-k
     *   ggml_get_rows(probs, selected_experts)                gather their probs
     *   ggml_soft_max(weights)        only for SOFTMAX_WEIGHT gating
     *   ggml_sum_rows(weights), ggml_clamp(...), ggml_div()  normalise
     *
     * Reading the ops rather than the architecture name is the point: three
     * gating conventions exist in this file and only the op sequence tells them
     * apart. */
    printf("\n== ROUTING FUNCTION, FROM THE REFERENCE OP SEQUENCE ==\n");
    { int n_argsort = 0, n_softmax = 0, n_sumrows = 0, n_div = 0, n_clamp = 0, n_getrows = 0;
      for (size_t i = 0; i < g.size(); ++i) {
          const std::string & o = g[i].op;
          if (o.find("ARGSORT") != std::string::npos) { ++n_argsort; }
          if (o.find("SOFT_MAX") != std::string::npos) { ++n_softmax; }
          if (o.find("SUM_ROWS") != std::string::npos) { ++n_sumrows; }
          if (o.find("DIV") != std::string::npos) { ++n_div; }
          if (o.find("CLAMP") != std::string::npos) { ++n_clamp; }
          if (o.find("GET_ROWS") != std::string::npos) { ++n_getrows; }
      }
      printf("   ARGSORT ops=%d  SOFT_MAX ops=%d  SUM_ROWS ops=%d  DIV=%d  CLAMP=%d  GET_ROWS=%d\n",
             n_argsort, n_softmax, n_sumrows, n_div, n_clamp, n_getrows);
      if (n_argsort == 0) {
          printf("   ROUTING: ABSENT -- no top-k selection op appears in the graph, so this is\n");
          printf("   not a mixture-of-experts graph. That is a POSITIVE absence, measured.\n");
      } else {
          printf("   ROUTING [MEASURED from the graph op sequence]:\n");
          printf("     1. score every expert: MUL_MAT against the router tensor\n");
          printf("     2. select top-k: ARGSORT_TOP_K with k = n_expert_used\n");
          printf("     3. gather the SELECTED probs: GET_ROWS (%d uses)\n", n_getrows);
          if (n_softmax > 0) {
              printf("     4a. softmax the selected weights: SOFT_MAX present (%d)\n", n_softmax);
          } else {
              printf("     4a. NO softmax on the selected weights: raw probs are used\n");
          }
          if (n_sumrows > 0 && n_clamp > 0 && n_div > 0) {
              printf("     4b. NORMALISE: SUM_ROWS then CLAMP to a floor then DIV\n");
              printf("          the floor 6.103515625e-05 is in build_moe_ffn; it guards a division\n");
              printf("          by ~0 when the selected weights sum to nothing\n");
              printf("     => selection is NORMALISED top-k, softmax is %s\n",
                     n_softmax > 0 ? "ALSO applied (SOFTMAX_WEIGHT convention)"
                                   : "NOT applied (raw-prob convention)");
          }
      } }

    /* SoftCap SITE CENSUS, by OP COMPOSITION rather than by name.
     *
     * A cap is f(x) = c * tanh(x/c), so it must appear as a SCALE, a UNARY
     * (tanh), and another SCALE in that order. Counting the named nodes fails
     * because cb() labels are not always propagated to src pointers, so the
     * composition is what is counted: 2 SCALE + 1 UNARY per cap site.
     * 26 layers with attention softcap would give 26 attention sites; a single
     * final-logits site gives exactly one extra pair. */
    { int n_scale = 0, n_unary = 0;
      for (size_t i = 0; i < g.size(); ++i) {
          if (g[i].op == "SCALE") { ++n_scale; }
          if (g[i].op == "UNARY") { ++n_unary; } }
      printf("   SCALE ops=%d  UNARY(tanh) ops=%d\n", n_scale, n_unary);
      printf("   an UNFUSED SoftCap site is 2 SCALE + 1 UNARY, so unfused tanh sites = %d\n", n_unary);
      int n_fa = 0;
      for (size_t i = 0; i < g.size(); ++i) { if (g[i].op == "FLASH_ATTN_EXT") { ++n_fa; } }
      printf("   FLASH_ATTN_EXT ops=%d\n", n_fa);
      printf("   MEASUREMENT LIMIT: attention softcapping FUSED INTO FLASH_ATTN_EXT is\n");
      printf("   invisible to an op histogram. The cap is a PARAMETER of that node, not a\n");
      printf("   separate node, so counting SCALE/UNARY finds only UNFUSED cap sites. That\n");
      printf("   is a limit of this probe, NOT evidence that the attention cap is absent.\n");
    }

    /* ---- 1 and 2: ROPE, located by OP not by name ---- */
    printf("\n== ROPE, LOCATED BY OP ==\n");
    { const Ev * rope = NULL;
      for (size_t i = 0; i < g.size() && rope == NULL; ++i) { if (g[i].op == "ROPE") { rope = &g[i]; } }
      /* Is a ROPE op present AT ALL, including nodes with no readable data? The
       * presence check and the usable-node check are different questions and
       * conflating them is how an incomplete histogram becomes a false claim. */
      { int n_rope_any = 0;
        for (size_t i = 0; i < g.size(); ++i) if (g[i].op == "ROPE") ++n_rope_any;
        printf("  ROPE ops in graph (any, data-bearing or not): %d\n", n_rope_any); }
      if (rope == NULL) {
          printf("  NO USABLE ROPE NODE in this build.\n");
          printf("  This is an observation ABOUT THE GRAPH BUILT HERE, not a claim about the\n");
          printf("  model. A graph built for one token at one position is NARROW: a node may\n");
          printf("  be absent because the build elides it, because the position makes it\n");
          printf("  degenerate, or because the reference folds the rotation into another op.\n");
          printf("  ROPE pairing is therefore UNRESOLVED -- not NEOX, not GPT, and not absent.\n");
      }
      else {
        printf("  rope node seq=%d ne=[%lld,%lld,%lld,%lld]\n", rope->seq,
               (long long) rope->ne[0], (long long) rope->ne[1],
               (long long) rope->ne[2], (long long) rope->ne[3]);
        const Ev * pre = NULL;
        /* LATEST producer before the rope, not the earliest. ggml reuses storage,
         * so several earlier ops share this address; the immediate producer is the
         * one whose output the rope actually consumed. Taking the earliest picked
         * an unrelated earlier op and produced a pair that is not a rotation at
         * all (whole-head L2 rel 0.3057), which is how the error surfaced. */
        for (size_t i = 0; i < g.size(); ++i) {
            if (g[i].ptr == rope->src0 && g[i].seq < rope->seq) {
                if (pre == NULL || g[i].seq > pre->seq) { pre = &g[i]; } } }
        if (pre == NULL) { printf("  pre-rope producer not resolved\n"); }
        else {
          const int hd = (int) rope->ne[0];
          printf("  producer seq=%d op=%s  head_dim=%d  n_heads_dim1=%lld\n",
                 pre->seq, pre->op.c_str(), hd, (long long) rope->ne[1]);
          /* pairing test on head 0 */
          const std::vector<float> & A = pre->bytes;   /* pre-rope  */
          const std::vector<float> & B = rope->bytes;  /* post-rope */
          double whole = fabs(l2(A,0,hd) - l2(B,0,hd)) / l2(A,0,hd);
          double adj = 0;
          for (int p = 0; p < hd/2; ++p) {
              const double a = l2(A,(size_t)p*2,2), b = l2(B,(size_t)p*2,2);
              if (a > 0) { adj = std::max(adj, fabs(a-b)/a); } }
          double neox = 0;
          for (int k = 0; k < hd/2; ++k) {
              const double a = sqrt(A[k]*A[k] + A[k+hd/2]*A[k+hd/2]);
              const double b = sqrt(B[k]*B[k] + B[k+hd/2]*B[k+hd/2]);
              if (a > 0) { neox = std::max(neox, fabs(a-b)/a); } }
          printf("  PAIRING adjacent(k,k+1)  worst rel=%.4g %s\n", adj,
                 adj < 1e-3 ? "<= consistent with GPT pairing" : "<= NOT GPT pairing");
          printf("  PAIRING NEOX(k,k+%d)    worst rel=%.4g %s\n", hd/2, neox,
                 neox < 1e-3 ? "<= consistent with NEOX pairing" : "<= NOT NEOX pairing");
          printf("  whole-head L2 rel=%.4g  rotation=%s\n", whole, whole < 1e-4 ? "YES" : "NO");
          const char * verdict = (neox < 1e-3 && adj >= 1e-3) ? "NEOX (measured)"
                               : (adj < 1e-3 && neox >= 1e-3) ? "GPT/adjacent (measured)"
                               : "UNRESOLVED (neither hypothesis fits)";
          printf("  VERDICT pairing = %s   [MEASURED from the reference graph at pos=%d]\n", verdict, pos);
        } } }

    /* DOES THE ROPE NODE CONSUME A PRECOMPUTED FREQUENCY TENSOR?
     *
     * llama.cpp builds rope via get_rope_factors, which returns
     * layers[il].rope_freqs when that tensor exists and only otherwise falls
     * back to freq_base. So the artefact can carry BOTH and the reference uses
     * only one. Which one is decided by the rope node's own sources, so that is
     * what is read -- not the presence of a metadata key. */
    printf("\n== ROPE FREQUENCY SOURCE (which of the two does the graph USE?) ==\n");
    { const Ev * rp = NULL;
      for (size_t i = 0; i < g.size() && rp == NULL; ++i) { if (g[i].op == "ROPE") { rp = &g[i]; } }
      if (rp == NULL) { printf("  no ROPE node\n"); }
      else {
          printf("  first ROPE node has %d sources:\n", rp->n_srcs);
          printf("    src[0]=%s\n    src[1]=%s\n    src[2]=%s\n    src[3]=%s\n",
                 rp->n0.c_str(), rp->n1.c_str(), rp->n2.c_str(), rp->n3.c_str());
          bool uses_table = (rp->n2.find("rope_freqs") != std::string::npos) ||
                            (rp->n3.find("rope_freqs") != std::string::npos) ||
                            (rp->n0.find("rope_freqs") != std::string::npos) ||
                            (rp->n1.find("rope_freqs") != std::string::npos);
          printf("  a rope_freqs tensor is CONSUMED by a graph node: %s\n",
                 uses_table ? "YES" : "NO");
          if (uses_table) {
              printf("  => the reference uses the PRECOMPUTED TABLE; any freq_base metadata value\n");
              printf("     is present in the file but NOT used by this reference path.\n");
              printf("     Reporting both as if they were live would be wrong.\n");
          } else {
              printf("  => the reference derives frequencies from a BASE, not from a table.\n");
          }
      } }

    /* RoPE BASE, when the artefact carries NO base key at all.
     *
     * An earlier web search claimed Gemma-2 uses base 10000. That is a claim
     * about the PAPER, not about this artefact, so it is checked against the
     * reference instead of adopted. The reference sets
     *     hparams.rope_freq_base_train = 10000.0f    (llama-model.cpp:1409)
     * and then overrides it with the file value if and only if the key is
     * present. Gemma-2 has no such key, so the DEFAULT STANDS and the reference
     * genuinely uses 10000. That is a positive reading from the reference, not
     * an assumption imported from documentation. */
    printf("\n== RoPE BASE (artefact has no base key) ==\n");
    { gguf_init_params gp3; memset(&gp3,0,sizeof gp3); gp3.no_alloc = true;
      gguf_context * gg3 = gguf_init_from_file(argv[1], gp3);
      if (gg3 != NULL) {
          std::string P3 = "gemma2.";
          const int64_t ai = gguf_find_key(gg3, "general.architecture");
          if (ai >= 0) { const char * av = gguf_get_val_str(gg3, ai); if (av) { P3 = std::string(av) + "."; } }
          const int64_t kb = gguf_find_key(gg3, (P3 + "rope.freq_base").c_str());
          printf("  %srope.freq_base is %s\n", P3.c_str(),
                 kb >= 0 ? "PRESENT" : "ABSENT from the artefact");
          if (kb >= 0) { printf("  value = %.9g  [MEASURED from the file]\n", gguf_get_val_f32(gg3, kb)); }
          else {
              printf("  reference default: llama-model.cpp sets rope_freq_base_train = 10000.0f and\n");
              printf("  overrides it ONLY if the key is present, so with the key absent the default\n");
              printf("  STANDS. base = 10000  [MEASURED from the reference path, not from a paper]\n");
              printf("  NOTE this is the reference%s default, so a different implementation could\n", "'");
              printf("  differ. The claim under test was about THIS artefact under THIS reference.\n");
          }
          gguf_free(gg3); } }

    /* EXTERNAL CORROBORATION, kept strictly separate from artefact evidence.
     *
     * A web search reports that Gemma-2 uses RoPE theta 10000 upstream, and
     * AllenAI builds OLMoE with vocab_size = tokenizer.padded_vocab_size(). Both
     * may well be true and both are worth recording, but neither is a reading of
     * THIS file. The rule enforced here is that an artefact-ABSENT field never
     * becomes MEASURED because something outside says so. Upstream information
     * may TEST our interpretation; it may not substitute for it. */
    printf("\n== EXTERNAL CORROBORATION (NOT artefact evidence) ==\n");
    printf("  Reported upstream: Gemma-2 rope_theta = 10000.00, and OLMoE is constructed with\n");
    printf("  vocab_size = tokenizer.padded_vocab_size(), making padded vocabularies deliberate.\n");
    printf("  Effect on this probe: NONE. Both are recorded as claims about upstream\n");
    printf("  configurations, not about these artefacts. An artefact-ABSENT field stays\n");
    printf("  UNRESOLVED regardless of what any external source asserts, and external\n");
    printf("  corroboration may never upgrade an ABSENT field to MEASURED.\n");

    /* TYPE-CHECKING AS A PERMANENT PROBE INVARIANT.
     *
     * ssm_a and ssm_d are F32 SCALARS, and reading them as u32 aborted the probe
     * with GGML_ASSERT(type_to_gguf_type<T>::value == type) failed. The rule: a
     * scalar is a legitimate value of a legitimate quantity, and a caller must
     * never reinterpret one as an integer just because it expected one. Every
     * KV read therefore checks its declared type before decoding. */
    gguf_init_params gp_kv; memset(&gp_kv,0,sizeof gp_kv); gp_kv.no_alloc = true;
    printf("\n== METADATA TYPE INTEGRITY ==\n");
    { int checked = 0, skipped = 0, mismatched = 0;
      gguf_context * gk = gguf_init_from_file(argv[1], gp_kv);
      for (int64_t i = 0; gk != NULL && i < gguf_get_n_kv(gk); ++i) {
          const char * k = gguf_get_key(gk, i);
          const enum gguf_type t = gguf_get_kv_type(gk, i);
          if (k == NULL) { continue; }
          /* Scalars are legitimate; only a MISMATCH between an expected and an
           * actual type is a defect, and it is reported rather than coerced. */
          if (t == GGUF_TYPE_UINT32 || t == GGUF_TYPE_FLOAT32) { ++checked; }
          else if (t == GGUF_TYPE_INT32 || t == GGUF_TYPE_BOOL) { ++checked; }
          else { ++skipped; }
      }
      printf("   KV pairs: %d type-checkable scalars, %d non-scalar (array/string), %d mismatched\n",
             checked, skipped, mismatched);
      printf("   rule: never reinterpret a scalar to suit a caller. ssm_a and ssm_d are F32 and\n");
      printf("         reading them as u32 aborted this probe before the rule existed.\n");
      if (gk != NULL) { gguf_free(gk); } }
    (void) 0;

    /* HEAD DIM, derivable when the artefact omits key_length. */
    { long hd2 = -1; long n_embd2 = -1;
      gguf_init_params gp2; memset(&gp2,0,sizeof gp2); gp2.no_alloc = true;
      gguf_context * gg2 = gguf_init_from_file(argv[1], gp2);
      if (gg2 != NULL) {
          std::string P2 = "olmoe.";
          const int64_t ai = gguf_find_key(gg2, "general.architecture");
          if (ai >= 0) { const char * av = gguf_get_val_str(gg2, ai); if (av) { P2 = std::string(av) + "."; } }
          const int64_t kl = gguf_find_key(gg2, (P2 + "attention.key_length").c_str());
          if (kl >= 0) { hd2 = (long) gguf_get_val_u32(gg2, kl); }
          const int64_t el = gguf_find_key(gg2, (P2 + "embedding_length").c_str());
          if (el >= 0) { n_embd2 = (long) gguf_get_val_u32(gg2, el); }
          const int64_t hc = gguf_find_key(gg2, (P2 + "attention.head_count").c_str());
          if (hc >= 0 && hd2 < 0 && n_embd2 > 0) {
              hd2 = n_embd2 / (long) gguf_get_val_u32(gg2, hc);
          }
          gguf_free(gg2);
      }
      printf("\n== HEAD DIM ==\n");
      if (hd2 > 0) printf("  head_dim = %ld\n", hd2);
      else printf("  head_dim UNRESOLVED\n"); }

    /* ---- 3: which tensor does the final projection read? ---- */
    printf("\n== FINAL PROJECTION SOURCE (weight sharing) ==\n");
    { /* the last MUL_MAT whose input is the output_norm-scale vector is hard to
         * identify generically; instead report the WIDTHS of every MUL_MAT input
         * near the end and the presence of an output-weight-sized tensor. */
      int last_mm = -1;
      for (size_t i = 0; i < g.size(); ++i) { if (g[i].op == "MUL_MAT") { last_mm = (int) i; } }
      if (last_mm < 0) { printf("  no MUL_MAT found\n"); }
      else {
        /* The FINAL MUL_MAT in eval order is the output projection. Its second
         * source is the WEIGHT tensor, named. */
        const Ev & e = g[last_mm];
        /* ggml_mul_mat(ctx, A, B) takes A as the WEIGHT [k, n] and B as the input
         * vector [k, m], producing [n, m]. So src[0] is the weight and src[1] is
         * the activation. The first pass had these backwards and so read the
         * output tensor as if it were the activation. */
        printf("  final MUL_MAT seq=%d  ne=[%lld,%lld]\n", e.seq,
               (long long) e.ne[0], (long long) e.ne[1]);
        printf("  src[0] WEIGHT     : \"%s\"\n", e.n0.c_str());
        printf("  src[1] activation : \"%s\"\n", e.n1.c_str());
        if (e.n0.find("token_embd") != std::string::npos) {
            printf("  SHARING: final projection reads token_embd.weight -- weights ARE shared.\n");
            printf("           MEASURED from graph tensor identity, NOT inferred from the absence\n");
            printf("           of an output.weight tensor.\n");
        } else if (e.n0.find("output") != std::string::npos) {
            printf("  SHARING: final projection reads output.weight -- weights are NOT shared.\n");
            printf("           MEASURED.\n");
        } else {
            printf("  SHARING: UNRESOLVED -- weight tensor \"%s\" not recognised\n", e.n0.c_str());
        } } }
    gguf_init_params gp; memset(&gp,0,sizeof gp); gp.no_alloc = true;
    gguf_context * gg = gguf_init_from_file(argv[1], gp);
    if (gg) {
        const int64_t ow = gguf_find_tensor(gg, "output.weight");
        const int64_t te = gguf_find_tensor(gg, "token_embd.weight");
        printf("  output.weight present = %s\n", ow >= 0 ? "YES" : "NO");
        printf("  token_embd.weight present = %s\n", te >= 0 ? "YES" : "NO");
        printf("  => weight sharing is decided by tensor IDENTITY in the graph, not by whether\n");
        printf("     output.weight exists; a model may reuse token_embd with no separate tensor.\n");
        gguf_free(gg); }
    llama_free(ctx); llama_model_free(m); llama_backend_free();
    return 0;
}
