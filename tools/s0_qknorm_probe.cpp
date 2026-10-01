// Q/K head-norm reference capture.
//
// TWO SEPARATE PIECES OF EVIDENCE, deliberately not conflated:
//
//   1. POINTER IDENTITY proves graph structure: that a node's src[0] is the
//      same storage another node produced, i.e. the edge exists.
//   2. COPIED BYTES prove what existed at that instant. Pointer equality alone
//      is not a reading, because ggml runs several of these ops IN PLACE: if
//      the tensor is retained and inspected after the callback returns, an
//      in-place RoPE has since overwritten the very bytes that were there.
//
// So every tensor is snapshotted INTO PROBE-OWNED STORAGE at the moment its
// eval callback fires, and the pointer is kept alongside as provenance only.
//
// PRE-NORM, OBTAINED BY WALKING BACKWARDS. ggml_rms_norm runs in place here, so
// src[0] at eval time already holds POST-norm values, and peeking at storage on
// the ask pass is unreliable because the producer may not be materialised yet.
// Instead: snapshot every event, then find the EARLIEST event sharing the
// normed tensor's storage. That is the producer's output as it stood BEFORE the
// norm overwrote it, obtained from the producer's own eval callback.
//
// Nothing here reimplements attention. These are tensors llama.cpp produced,
// read through its own ggml_backend_sched_eval_callback.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

static uint64_t fnv(const void * d, size_t n) {
    const unsigned char * p = (const unsigned char *) d;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
/* llama.cpp pads op names to a fixed column and appends a node index, so a node
 * reads "          Qcur_normed-89". Substring matching is UNSAFE: "Qcur" is a
 * substring of "Qcur_normed", which once made both captures take the same
 * tensor. Strip padding and the -NN suffix, then compare EXACTLY. */
static std::string op_name_local(int op) {
    switch (op) {
        case GGML_OP_MUL_MAT:    return "MUL_MAT";
        case GGML_OP_RMS_NORM:   return "RMS_NORM";
        case GGML_OP_ROPE:       return "ROPE";
        case GGML_OP_RESHAPE:    return "RESHAPE";
        case GGML_OP_VIEW:       return "VIEW";
        case GGML_OP_CONT:       return "CONT";
        case GGML_OP_ADD:        return "ADD";
        case GGML_OP_SCALE:      return "SCALE";
        case GGML_OP_SQR:        return "SQR";
        case GGML_OP_SQRT:       return "SQRT";
        case GGML_OP_SILU_BACK:  return "SILU_BACK";
        case GGML_OP_GET_ROWS:   return "GET_ROWS";
        case GGML_OP_NORM:       return "NORM";
        default: { char b[32]; snprintf(b, sizeof b, "OP%d", op); return b; }
    }
}
static std::string norm_name(const char * raw) {
    if (raw == NULL) { return ""; }
    const char * q = raw; while (*q == ' ') { ++q; }
    std::string s(q);
    const size_t d = s.rfind('-');
    if (d != std::string::npos && d + 1 < s.size() &&
        s[d + 1] >= '0' && s[d + 1] <= '9') { s.erase(d); }
    return s;
}

struct Ev {
    int         seq;
    std::string name;
    int         opid;          /* ggml op id; named via ggml_op_name_local */
    const void * ptr;           /* backend storage address: provenance only   */
    const void * src0;         /* producer storage address: provenance only  */
    int64_t     ne[4];
    std::vector<float> bytes;  /* OWNED SNAPSHOT, taken at eval time         */
};
static std::vector<Ev> g_evs;
static bool g_record = false;

static bool ev(ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (!g_record || ask || t == NULL) { return true; }
    if (t->data == NULL || t->buffer == NULL) { return true; }
    const int64_t n = ggml_nelements(t);
    if (n <= 0 || n > (1 << 22)) { return true; }
    Ev e;
    e.seq  = (int) g_evs.size();
    e.name = norm_name(t->name);
    e.opid = (int) t->op;
    e.ptr  = t->data;
    e.src0 = t->src[0] != NULL ? t->src[0]->data : NULL;
    for (int i = 0; i < 4; ++i) { e.ne[i] = t->ne[i]; }
    /* Snapshot NOW. Reading this storage after the callback returns can read an
     * in-place successor's values instead. */
    e.bytes.resize((size_t) n);
    memcpy(e.bytes.data(), t->data, (size_t) n * sizeof(float));
    g_evs.push_back(e);
    return true;
}

struct Seal {
    std::string label;
    int         layer;
    int         head;
    int         pos;
    int         n_heads;
    std::vector<float> pre, w, post_norm, post_rope;
    uint64_t h_pre, h_w, h_post_norm, h_post_rope;
    const void * normed_ptr;
    const void * roped_ptr;
    const void * producer_ptr;
    std::string pre_op, norm_op, rope_op;
};

static void stats(const char * tag, const std::vector<float> & v) {
    const int n = (int) v.size();
    if (n == 0) { printf("  %-16s EMPTY -- not captured\n", tag); return; }
    double s = 0, mn = v[0], mx = v[0]; int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) { ++bad; continue; }
        s += v[i]; if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i];
    }
    printf("  %-16s n=%3d hash=%016llx min=%.9g max=%.9g mean=%.9g nonfinite=%d\n",
           tag, n, (unsigned long long) fnv(v.data(), (size_t) n * sizeof(float)), mn, mx, s / n, bad);
    printf("  %-16s first6 =", "");
    for (int i = 0; i < 6 && i < n; ++i) { printf(" %.9g", v[i]); }
    printf("\n  %-16s last4  =", "");
    for (int i = (n > 4 ? n - 4 : 0); i < n; ++i) { printf(" %.9g", v[i]); }
    printf("\n");
}
static double l2(const std::vector<float> & v, size_t from, size_t n) {
    double s = 0; for (size_t i = from; i < from + n; ++i) { s += (double) v[i] * v[i]; }
    return sqrt(s);
}
static std::vector<float> head_of(const std::vector<float> & v, int hd, int h) {
    std::vector<float> o((size_t) hd);
    for (int i = 0; i < hd; ++i) { o[i] = v[(size_t) h * hd + i]; }
    return o;
}
/* Read a norm weight tensor straight out of the model file, bound by name to
 * the layer under measurement. */
static bool read_norm_weight(const char * path, const char * name, int nel, std::vector<float> & out) {
    gguf_init_params gp; memset(&gp, 0, sizeof gp); gp.no_alloc = true;
    gguf_context * g = gguf_init_from_file(path, gp);
    if (g == NULL) { return false; }
    const int64_t ti = gguf_find_tensor(g, name);
    if (ti < 0 || (int64_t) gguf_get_tensor_ne(g, ti)[0] != nel) { gguf_free(g); return false; }
    if (gguf_get_tensor_type(g, ti) != GGML_TYPE_F32) { gguf_free(g); return false; }
    FILE * f = fopen(path, "rb");
    if (f == NULL) { gguf_free(g); return false; }
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
    out.assign((size_t) nel, 0.0f);
    const bool ok = fseek(f, (long) off, SEEK_SET) == 0 &&
                    fread(out.data(), sizeof(float), (size_t) nel, f) == (size_t) nel;
    fclose(f); gguf_free(g);
    return ok;
}

int main(int argc, char ** argv) {
    const int tok_id = argc > 2 ? atoi(argv[2]) : 785;
    const int pos    = argc > 3 ? atoi(argv[3]) : 7;   /* NON-ZERO: RoPE must act */
    const int q_head = argc > 4 ? atoi(argv[4]) : 3;   /* NON-ZERO               */
    const int k_head = argc > 5 ? atoi(argv[5]) : 2;   /* NON-ZERO               */
    const char * path = argv[1];
    const int hd = 128;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == NULL) { printf("FATAL load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.cb_eval = ev; cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == NULL) { printf("FATAL ctx failed\n"); return 1; }

    llama_token tk[1] = { tok_id }; llama_pos ps[1] = { pos };
    int32_t ns[1] = { 1 }; llama_seq_id sq0 = 0; llama_seq_id * sq[1] = { &sq0 };
    int8_t lg[1] = { 1 };
    llama_batch b; memset(&b, 0, sizeof b);
    b.n_tokens = 1; b.token = tk; b.pos = ps; b.n_seq_id = ns; b.seq_id = sq; b.logits = lg;

    g_record = true;
    const int rc = llama_decode(ctx, b);
    g_record = false;
    if (rc != 0) { printf("FATAL decode failed\n"); return 1; }
    printf("events=%d token=%d pos=%d q_head=%d k_head=%d\n",
           (int) g_evs.size(), tok_id, pos, q_head, k_head);

    /* Events are stored in EVAL order, so the first matching norm is layer 0. */
    struct Side { const char * nname; const char * rname; int nel; int heads; int h; };
    Side sides[2] = { {"Qcur_normed", "Qcur", 4096, 32, q_head},
                      {"Kcur_normed", "Kcur", 1024,  8, k_head} };
    Seal seals[2];
    int bad = 0;

    for (int si = 0; si < 2; ++si) {
        const Side & S = sides[si];
        const Ev * normed = NULL; const Ev * roped = NULL;
        for (size_t i = 0; i < g_evs.size() && normed == NULL; ++i) {
            if (g_evs[i].name != S.nname || (int) g_evs[i].bytes.size() != S.nel) { continue; }
            for (size_t j = 0; j < g_evs.size(); ++j) {
                if (g_evs[j].name != S.rname || (int) g_evs[j].bytes.size() != S.nel) { continue; }
                if (g_evs[j].src0 == g_evs[i].ptr) { normed = &g_evs[i]; roped = &g_evs[j]; break; }
            }
        }
        printf("\n--- %s ---\n", S.nname);
        if (normed == NULL || roped == NULL) {
            printf("  no graph-identity pair; REFUSING\n"); ++bad; continue;
        }
        /* PRODUCER SELECTED ARITHMETICALLY, NOT BY HEURISTIC.
         *
         * ggml reuses storage, so one address carries several ops in sequence:
         * attn_norm, then the Q/K projection, then the head norm, then RoPE.
         * "Earliest on this address" and "latest before the head norm" both pick
         * something, and neither is guaranteed to be the projection output --
         * for K it was not, and the arithmetic caught it. So every earlier event
         * on this address is offered as a candidate and RMSNorm's own identity
         * out = in * weight * inv_rms selects the correct one. The rule is
         * verified by the reconstruction rather than assumed. */
        const Ev * producer = NULL; double prod_err = 1e30;
        for (size_t k = 0; k < g_evs.size(); ++k) {
            const Ev & c = g_evs[k];
            if (c.ptr != normed->ptr || c.seq >= normed->seq) { continue; }
            if ((int) c.bytes.size() != S.nel) { continue; }
            const std::vector<float> h = head_of(c.bytes, hd, S.h);
            double ss = 0;
            for (int i = 0; i < hd; ++i) { ss += (double) h[i] * h[i]; }
            const float inv = 1.0f / sqrtf((float)(ss / hd) + 1e-6f);
            const std::vector<float> pn = head_of(normed->bytes, hd, S.h);
            std::vector<float> wt;
            { std::vector<float> full;
              char nm[128];
              snprintf(nm, sizeof nm, "blk.0.attn_%c_norm.weight",
                       (S.nname[0] >= 'A' && S.nname[0] <= 'Z') ? S.nname[0] + 32 : S.nname[0]);
              if (!read_norm_weight(path, nm, hd, full)) { continue; } wt = full; }
            double worst = 0;
            for (int i = 0; i < hd; ++i) {
                const double rec = (double) h[i] * wt[i] * inv;
                worst = std::max(worst, fabs(rec - pn[i]));
            }
            if (worst < prod_err) { prod_err = worst; producer = &c; }
        }
        printf("  candidate producer: seq=%d op=%s reconstruction err=%.6g\n",
               producer ? producer->seq : -1,
               producer ? op_name_local(producer->opid).c_str() : "NONE", prod_err);
        printf("  normed : seq=%d op=%-8s ptr=%p ne=[%lld,%lld]\n", normed->seq,
               op_name_local(normed->opid).c_str(), normed->ptr, (long long) normed->ne[0], (long long) normed->ne[1]);
        printf("  roped  : seq=%d op=%-8s src0=%p  edge(src0==normed.ptr)=%s\n", roped->seq,
               op_name_local(roped->opid).c_str(), roped->src0, roped->src0 == normed->ptr ? "YES" : "NO");
        printf("  producer: seq=%d op=%-8s ptr=%p  <- pre-norm snapshot\n",
               producer ? producer->seq : -1, producer ? op_name_local(producer->opid).c_str() : "NONE",
               producer ? producer->ptr : NULL);
        if (producer == NULL) { printf("  no producer; pre-norm UNRECOVERED\n"); ++bad; continue; }

        Seal & Z = seals[si];
        Z.label = S.nname; Z.layer = 0; Z.head = S.h; Z.pos = pos; Z.n_heads = S.heads;
        Z.pre = head_of(producer->bytes, hd, S.h);
        Z.post_norm = head_of(normed->bytes, hd, S.h);
        Z.post_rope = head_of(roped->bytes, hd, S.h);
        Z.normed_ptr = normed->ptr; Z.roped_ptr = roped->ptr; Z.producer_ptr = producer->ptr;
        Z.pre_op = op_name_local(producer->opid); Z.norm_op = op_name_local(normed->opid); Z.rope_op = op_name_local(roped->opid);
        char nm[128];
        snprintf(nm, sizeof nm, "blk.0.attn_%c_norm.weight",
                 (S.nname[0] >= 'A' && S.nname[0] <= 'Z') ? S.nname[0] + 32 : S.nname[0]);
        if (!read_norm_weight(path, nm, hd, Z.w)) { printf("  weights MISSING for %s\n", nm); ++bad; }

        /* IS THE "PRE-NORM" VECTOR ACTUALLY THE Q-PROJECTION OUTPUT?
         *
         * Storage is reused by ggml, so the earliest snapshot on an address can
         * be an EARLIER op entirely -- here attn_norm, which normalises the
         * residual before the Q projection, not the projection's own output.
         * That is indistinguishable by name, position or pointer, so it is
         * settled ARITHMETICALLY: RMSNorm is out = in * weight * inv_rms, with
         * inv_rms from in. If this candidate is the true pre-norm vector, the
         * reconstruction must reproduce post_norm to float tolerance. If it does
         * not, the candidate is the wrong tensor and the capture is void. */
        { double ss = 0;
          for (int i = 0; i < hd; ++i) { ss += (double) Z.pre[i] * Z.pre[i]; }
          const float inv = 1.0f / sqrtf((float)(ss / hd) + 1e-6f);
          double worst = 0;
          for (int i = 0; i < hd; ++i) {
              const double rec = (double) Z.pre[i] * Z.w[i] * inv;
              const double d2 = fabs(rec - Z.post_norm[i]);
              worst = std::max(worst, d2);
          }
          printf("  PRE-NORM CHECK max|reconstructed-post_norm| = %.6g %s\n", worst,
                 worst < 1e-2 ? "PASS (this really is the norm input)"
                              : "FAIL (WRONG TENSOR: not the projection output)");
          if (!(worst < 1e-2)) { ++bad; } }

        stats("pre-RMSNorm",  Z.pre);
        stats("norm weights",  Z.w);
        stats("post-RMSNorm", Z.post_norm);
        stats("post-RoPE",    Z.post_rope);

        /* GATES */
        const uint64_t hp = fnv(Z.pre.data(), Z.pre.size() * 4);
        const uint64_t hn = fnv(Z.post_norm.data(), Z.post_norm.size() * 4);
        const uint64_t hr = fnv(Z.post_rope.data(), Z.post_rope.size() * 4);
        Z.h_pre = hp; Z.h_post_norm = hn; Z.h_post_rope = hr;
        Z.h_w = fnv(Z.w.data(), Z.w.size() * 4);
        if (hp == hn) { printf("  GATE FAIL pre==post-norm (inconclusive)\n"); ++bad; }
        else          { printf("  GATE ok  pre!=post-norm\n"); }
        if (hn == hr) { printf("  GATE FAIL post-norm==post-RoPE at pos=%d: RoPE unobserved\n", pos); ++bad; }
        else          { printf("  GATE ok  post-norm != post-RoPE at pos=%d\n", pos); }
        const double a = l2(Z.post_norm, 0, hd), b2 = l2(Z.post_rope, 0, hd);
        printf("  GATE L2 whole head normed=%.9g roped=%.9g rel=%.3g\n", a, b2, fabs(a-b2)/a);
        if (!(fabs(a - b2) / a < 1e-4)) { printf("  GATE FAIL magnitude not preserved\n"); ++bad; }
        /* PAIRING DETERMINED EMPIRICALLY, NOT ASSUMED. A rotation preserves the
         * norm of every rotated PAIR, but WHICH pairs are rotated is a property
         * of the model, not of RoPE. GPT-style pairs are adjacent (2k, 2k+1);
         * NEOX pairs each element with the one half a vector away (k, k+hd/2).
         * Test both and report which holds. Guessing here would have produced a
         * FAIL for a correct model and a PASS for a wrong one. */
        double worst_adj = 0, worst_neox = 0;
        for (int pr = 0; pr < hd/2; ++pr) {
            const double na = l2(Z.post_norm, (size_t)pr*2, 2), nb = l2(Z.post_rope, (size_t)pr*2, 2);
            if (na > 0) { worst_adj = std::max(worst_adj, fabs(na - nb) / na); }
        }
        for (int k = 0; k < hd/2; ++k) {
            double na = 0, nb = 0;
            na = (double) Z.post_norm[k] * Z.post_norm[k] +
                 (double) Z.post_norm[k + hd/2] * Z.post_norm[k + hd/2];
            nb = (double) Z.post_rope[k] * Z.post_rope[k] +
                 (double) Z.post_rope[k + hd/2] * Z.post_rope[k + hd/2];
            na = sqrt(na); nb = sqrt(nb);
            if (na > 0) { worst_neox = std::max(worst_neox, fabs(na - nb) / na); }
        }
        printf("  PAIRING worst adjacent(k, k+1)  rel=%.3g %s\n", worst_adj,
               worst_adj < 1e-3 ? "<- consistent with GPT pairing" : "<- NOT GPT pairing");
        printf("  PAIRING worst NEOX(k, k+64)    rel=%.3g %s\n", worst_neox,
               worst_neox < 1e-3 ? "<- consistent with NEOX pairing" : "<- NOT NEOX pairing");
        if (!(worst_neox < 1e-3)) {
            printf("  GATE FAIL neither pairing hypothesis explains the rotation\n"); ++bad;
        } else {
            printf("  GATE ok  rotation pair structure identified as NEOX\n");
        }
    }
    printf("\nBAD=%d\n", bad);
    if (bad == 0) {
        printf("\n=== SEALED SUMMARY ===\n");
        for (int i = 0; i < 2; ++i) {
            printf("%s layer=%d head=%d of %d pos=%d\n", seals[i].label.c_str(),
                   seals[i].layer, seals[i].head, seals[i].n_heads, seals[i].pos);
            printf("   pre-RMSNorm  %016llx\n", (unsigned long long) seals[i].h_pre);
            printf("   weights      %016llx\n", (unsigned long long) seals[i].h_w);
            printf("   post-RMSNorm %016llx\n", (unsigned long long) seals[i].h_post_norm);
            printf("   post-RoPE    %016llx\n", (unsigned long long) seals[i].h_post_rope);
            printf("   provenance: producer=%p(%s) normed=%p(%s) roped=%p(%s)\n",
                   seals[i].producer_ptr, seals[i].pre_op.c_str(),
                   seals[i].normed_ptr, seals[i].norm_op.c_str(),
                   seals[i].roped_ptr, seals[i].rope_op.c_str());
        }
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return bad == 0 ? 0 : 3;
}
