// Q/K head-norm reference capture, correlated by GRAPH IDENTITY rather than by
// tensor name.
//
// Why: "Qcur", "Qcur_normed", "Kcur" and "Kcur_normed" each occur once per
// layer -- 36 times each in Qwen3-8B -- so matching on name and taking the
// first hit can pair one layer's normed tensor with another layer's roped one.
// The visible symptom was a post-norm vector spanning +-12.2 beside a
// post-RoPE vector spanning +-0.27, which cannot be the same vector because
// RoPE is a rotation and preserves magnitude.
//
// The correlation rule is not assumed. Every callback event records its data
// pointer, and RoPE's src[0] IS the norm output, so a normed/roped pair is
// accepted only when roped.src0 == normed.data exactly. That makes the pairing
// a graph fact rather than a naming convention, and it also yields the
// pre-norm input, because the norm's own src[0] is the pre-norm tensor.
//
// Nothing here reimplements attention: these are tensors llama.cpp produced,
// read through its own ggml_backend_sched_eval_callback.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>

static uint64_t fnv(const void * d, size_t n) {
    const unsigned char * p = (const unsigned char *) d;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
/* llama.cpp pads op names to a fixed column and appends a node index, so a node
 * reads like "          Qcur_normed-89". Strip both, then compare EXACTLY.
 * Substring matching is unsafe here: "Qcur" is a substring of "Qcur_normed",
 * which made both captures take the same tensor in an earlier pass. */
static std::string norm_name(const char * raw) {
    if (raw == NULL) { return ""; }
    const char * q = raw;
    while (*q == ' ') { ++q; }
    std::string s(q);
    const size_t d = s.rfind('-');
    if (d != std::string::npos && d + 1 < s.size() &&
        s[d + 1] >= '0' && s[d + 1] <= '9') { s.erase(d); }
    return s;
}

struct Ev {
    int         seq;
    std::string name;
    const void * data;        /* identity of this tensor's storage */
    int64_t     nel;
    int64_t     ne[4];
    int         op;
    const void * src0;        /* identity of this node's primary input */
    std::string src0name;
    int         src0_op;
    ggml_tensor * t;          /* retained so the data can be read after decode */
};
static std::vector<Ev> g_evs;
static int g_seq = 0;
static bool g_record = false;
/* Pre-norm values, captured on the ASK pass, keyed by the input pointer.
 *
 * ggml_rms_norm runs IN PLACE here: the normed tensor's src[0] pointer equals
 * its own data pointer, so by the time the eval pass fires the pre-norm values
 * have already been overwritten and are unrecoverable. The ask pass runs BEFORE
 * the node executes, so reading there captures the pre-norm values while they
 * still exist. Keyed by pointer so the lookup is by graph identity, and because
 * the norm is in place the key is the same pointer the normed tensor reports. */
static std::map<const void *, std::vector<float> > g_prenorm;

static bool ev(ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (!g_record) { return true; }
    if (t == NULL) { return true; }
    if (ask) {
        /* Pre-execution: snapshot the input of an in-place norm while intact. */
        if (norm_name(t->name) == "norm" && t->src[0] != NULL &&
            t->src[0]->data != NULL && ggml_is_contiguous(t->src[0])) {
            const int64_t n = ggml_nelements(t->src[0]);
            if ((n == 4096 || n == 1024) && g_prenorm.find(t->src[0]->data) == g_prenorm.end()) {
                std::vector<float> v((size_t) n);
                memcpy(v.data(), t->src[0]->data, (size_t) n * sizeof(float));
                g_prenorm[t->src[0]->data] = v;
            }
        }
        return true;
    }
    if (t->data == NULL || t->buffer == NULL) { return true; }
    const int64_t n = ggml_nelements(t);
    if (n <= 0 || n > (1 << 22)) { return true; }
    Ev e;
    e.seq = g_seq++;
    e.name = norm_name(t->name);
    e.data = t->data;
    e.nel  = n;
    e.op   = (int) t->op;
    for (int i = 0; i < 4; ++i) { e.ne[i] = t->ne[i]; }
    e.src0 = NULL; e.src0name = ""; e.src0_op = -1;
    if (ggml_is_view(t) == false && t->src[0] != NULL) {
        e.src0     = t->src[0]->data;
        e.src0name = norm_name(t->src[0]->name);
        e.src0_op  = (int) t->src[0]->op;
    } else if (t->src[0] != NULL) {
        e.src0     = t->src[0]->data;
        e.src0name = norm_name(t->src[0]->name);
        e.src0_op  = (int) t->src[0]->op;
    }
    e.t = t;
    g_evs.push_back(e);
    return true;
}

static void stats(const char * tag, const float * v, int n) {
    double s = 0, mn = v[0], mx = v[0]; int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) { ++bad; continue; }
        s += v[i]; if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i];
    }
    printf("  %-14s n=%3d hash=%016llx min=%.9g max=%.9g mean=%.9g nonfinite=%d\n",
           tag, n, (unsigned long long) fnv(v, (size_t) n * sizeof(float)), mn, mx, s / n, bad);
    printf("  %-14s first6 =", "");
    for (int i = 0; i < 6 && i < n; ++i) { printf(" %.9g", v[i]); }
    printf("\n  %-14s last4  =", "");
    for (int i = (n > 4 ? n - 4 : 0); i < n; ++i) { printf(" %.9g", v[i]); }
    printf("\n");
}

static std::vector<float> read_t(const Ev & e, int count) {
    std::vector<float> o((size_t) count);
    ggml_backend_tensor_get(e.t, o.data(), 0, (size_t) count * sizeof(float));
    return o;
}
static double l2(const std::vector<float> & v, size_t from, size_t n) {
    double s = 0; for (size_t i = from; i < from + n; ++i) { s += (double) v[i] * v[i]; }
    return sqrt(s);
}

/* Normed/roped pairing is ACCEPTED ONLY when the roped node's src[0] pointer
 * equals the normed node's data pointer. That is a graph fact. */
static bool pair_up(const char * nname, const char * rname, int64_t nel,
                    const Ev ** normed, const Ev ** roped, const Ev ** prenorm) {
    for (size_t i = 0; i < g_evs.size(); ++i) {
        const Ev & n = g_evs[i];
        if (n.name != nname || n.nel != nel) { continue; }
        for (size_t j = 0; j < g_evs.size(); ++j) {
            const Ev & r = g_evs[j];
            if (r.name != rname || r.nel != nel) { continue; }
            if (r.src0 == n.data) {
                *normed = &n; *roped = &r; *prenorm = NULL;
                for (size_t k = 0; k < g_evs.size(); ++k) {
                    if (g_evs[k].data == n.src0) { *prenorm = &g_evs[k]; break; }
                }
                return true;
            }
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    const int tok_id = argc > 2 ? atoi(argv[2]) : 785;
    const int pos    = argc > 3 ? atoi(argv[3]) : 0;
    const int q_head = argc > 4 ? atoi(argv[4]) : 3;
    const int k_head = argc > 5 ? atoi(argv[5]) : 2;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("FATAL load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.cb_eval = ev; cp.cb_eval_user_data = NULL;
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("FATAL ctx failed\n"); return 1; }

    llama_token tk[1] = { tok_id }; llama_pos ps[1] = { pos };
    int32_t ns[1] = { 1 }; llama_seq_id sq0 = 0; llama_seq_id * sq[1] = { &sq0 };
    int8_t lg[1] = { 1 };
    llama_batch b; memset(&b, 0, sizeof b);
    b.n_tokens = 1; b.token = tk; b.pos = ps; b.n_seq_id = ns; b.seq_id = sq; b.logits = lg;

    g_record = true;
    const int rc = llama_decode(ctx, b);
    g_record = false;
    if (rc != 0) { printf("FATAL decode failed\n"); return 1; }

    printf("=== STRUCTURAL INVENTORY OF CALLBACK EVENTS ===\n");
    printf("events recorded = %d\n", (int) g_evs.size());
    { std::map<std::string, int> hist;
      for (size_t i = 0; i < g_evs.size(); ++i) { hist[g_evs[i].name]++; }
      printf("distinct normalised names, with counts (36+ means per-layer):\n");
      for (std::map<std::string, int>::iterator it = hist.begin(); it != hist.end(); ++it) {
        if (it->first.find("cur") != std::string::npos) {
            printf("   %-16s x%d\n", it->first.c_str(), it->second);
        }
      }
    }

    const Ev * qn = NULL, * qr = NULL, * qp = NULL;
    const Ev * kn = NULL, * kr = NULL, * kp = NULL;
    printf("\n=== PAIRING BY GRAPH IDENTITY (roped.src0 == normed.data) ===\n");
    const bool qok = pair_up("Qcur_normed", "Qcur", 4096, &qn, &qr, &qp);
    const bool kok = pair_up("Kcur_normed", "Kcur", 1024, &kn, &kr, &kp);
    printf("Q pair found = %s   K pair found = %s\n", qok ? "yes" : "NO", kok ? "yes" : "NO");
    if (!qok || !kok) { printf("NO TRUSTWORTHY PAIR: refusing to publish numbers\n"); return 2; }
    printf("Q normed seq=%d data=%p | roped seq=%d src0=%p  -> pointer match %s\n",
           qn->seq, qn->data, qr->seq, qr->src0, qr->src0 == qn->data ? "YES" : "NO");
    printf("K normed seq=%d data=%p | roped seq=%d src0=%p  -> pointer match %s\n",
           kn->seq, kn->data, kr->seq, kr->src0, kr->src0 == kn->data ? "YES" : "NO");
    printf("pre-norm located for Q = %s, for K = %s\n",
           qp ? qp->name.c_str() : "NONE", kp ? kp->name.c_str() : "NONE");
    /* If the norm wrote into its own input's storage then src0 == data, and the
     * pre-norm VALUES ARE NOT RECOVERABLE from a post-eval callback: the tensor
     * we would read has already been overwritten. That must be reported, not
     * silently presented as a pre-norm capture. */
    printf("Q norm in-place (normed.src0 == normed.data) = %s\n",
           (qn->src0 == qn->data) ? "YES -- pre-norm NOT recoverable here" : "no");
    printf("K norm in-place (normed.src0 == normed.data) = %s\n",
           (kn->src0 == kn->data) ? "YES -- pre-norm NOT recoverable here" : "no");
    printf("NOTE pos=%d: RoPE angle is pos*freq, so at pos 0 RoPE is the IDENTITY\n", pos);

    const int hd = 128;
    std::vector<float> q_n = read_t(*qn, 4096), q_r = read_t(*qr, 4096);
    std::vector<float> k_n = read_t(*kn, 1024), k_r = read_t(*kr, 1024);
    /* pre-norm comes from the ask-pass snapshot, keyed by the normed pointer,
     * which is the same pointer because the norm is in place. */
    std::vector<float> q_p, k_p;
    { std::map<const void *, std::vector<float> >::iterator it = g_prenorm.find(qn->data);
      if (it != g_prenorm.end()) { q_p = it->second; } }
    { std::map<const void *, std::vector<float> >::iterator it = g_prenorm.find(kn->data);
      if (it != g_prenorm.end()) { k_p = it->second; } }
    printf("pre-norm snapshot found: Q=%zu elems K=%zu elems\n", q_p.size(), k_p.size());
    auto head = [](const std::vector<float> & v, int h) {
        std::vector<float> o((size_t) hd);
        for (int i = 0; i < hd; ++i) { o[i] = v[(size_t) h * hd + i]; }
        return o; };

    std::vector<float> qn_h = head(q_n, q_head), qr_h = head(q_r, q_head);
        std::vector<float> kn_h = head(k_n, k_head), kr_h = head(k_r, k_head);
    
    printf("\n=== SANITY GATES ===\n");
    int bad = 0;
    /* RoPE is a rotation: whole-head and pairwise L2 must be preserved. */
    { double a = l2(qn_h, 0, hd), b = l2(qr_h, 0, hd);
      const double rel = fabs(a - b) / a;
      printf("Q whole-head L2 normed=%.9g roped=%.9g rel_diff=%.3g %s\n",
             a, b, rel, rel < 1e-4 ? "PASS" : "FAIL");
      if (!(rel < 1e-4)) { ++bad; } }
    { int worst = 0; double worstrel = 0;
      for (int pr = 0; pr < hd / 2; ++pr) {
          double a = l2(qn_h, (size_t) pr * 2, 2), b = l2(qr_h, (size_t) pr * 2, 2);
          if (a <= 0) { continue; }
          const double rel = fabs(a - b) / a;
          if (rel > worstrel) { worstrel = rel; worst = pr; }
      }
      printf("Q worst pairwise L2 rel_diff=%.3g at pair %d %s\n", worstrel, worst,
             worstrel < 1e-3 ? "PASS" : "FAIL");
      if (!(worstrel < 1e-3)) { ++bad; } }
    { double a = l2(kn_h, 0, hd), b = l2(kr_h, 0, hd);
      const double rel = fabs(a - b) / a;
      printf("K whole-head L2 normed=%.9g roped=%.9g rel_diff=%.3g %s\n",
             a, b, rel, rel < 1e-4 ? "PASS" : "FAIL");
      if (!(rel < 1e-4)) { ++bad; } }
    if (qn->nel != 4096 || kn->nel != 1024) { printf("dimension FAIL\n"); ++bad; }
    printf("dims: Q=%d K=%d head_dim=%d  %s\n", (int) qn->nel, (int) kn->nel, hd,
           (qn->nel == 4096 && kn->nel == 1024) ? "PASS" : "FAIL");

    printf("\n=== SELECTED HEADS (q_head=%d k_head=%d) ===\n", q_head, k_head);
    if (!q_p.empty()) { stats("Q pre-norm",  head(q_p, q_head).data(), hd); }
    stats("Q post-norm",  qn_h.data(), hd);
    stats("Q post-RoPE",  qr_h.data(), hd);
    if (!k_p.empty()) { stats("K pre-norm",  head(k_p, k_head).data(), hd); }
    stats("K post-norm",  kn_h.data(), hd);
    stats("K post-RoPE",  kr_h.data(), hd);

    printf("\nRESULT bad=%d\n", bad);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return bad == 0 ? 0 : 3;
}
