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

struct Ev { int seq; std::string op; const void * ptr; const void * src0;
            int64_t ne[4]; std::vector<float> bytes;
            /* names of src[0] and src[1], so a MUL_MAT's WEIGHT can be identified.
             * This is the evidence that settles weight sharing: not whether an
             * output.weight tensor exists, but which tensor the final projection
             * actually reads. */
            std::string n0, n1; };
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
        case GGML_OP_UNARY: return "UNARY"; case GGML_OP_GLU: return "GLU";
        default: return "OTHER";
    }
}
static bool ev(ggml_tensor * t, bool ask, void *) {
    if (!g_rec || ask || t == NULL || t->data == NULL || t->buffer == NULL) { return true; }
    const int64_t n = ggml_nelements(t);
    if (n <= 0 || n > (1 << 22)) { return true; }
    Ev e; e.seq = (int) g.size(); e.op = opname((int) t->op);
    e.ptr = t->data; e.src0 = t->src[0] != NULL ? t->src[0]->data : NULL;
    for (int i = 0; i < 4; ++i) { e.ne[i] = t->ne[i]; }
    if (t->src[0] != NULL && t->src[0]->name) { e.n0 = t->src[0]->name; }
    if (t->src[1] != NULL && t->src[1]->name) { e.n1 = t->src[1]->name; }
    e.bytes.resize((size_t) n);
    memcpy(e.bytes.data(), t->data, (size_t) n * sizeof(float));
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

    /* ---- 1 and 2: ROPE, located by OP not by name ---- */
    printf("\n== ROPE, LOCATED BY OP ==\n");
    { const Ev * rope = NULL;
      for (size_t i = 0; i < g.size() && rope == NULL; ++i) { if (g[i].op == "ROPE") { rope = &g[i]; } }
      if (rope == NULL) { printf("  NO ROPE NODE FOUND in the graph: this architecture does not apply RoPE\n"); }
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
