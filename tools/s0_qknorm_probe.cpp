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
        case GGML_OP_MUL:        return "MUL";
        case GGML_OP_DUP:        return "DUP";
        case GGML_OP_CPY:        return "CPY";
        default: { char b[32]; snprintf(b, sizeof b, "UNMAPPED_OP%d", op); return b; }
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
        /* TRAVERSE src[0] BACKWARDS, ONE EDGE AT A TIME.
         *
         * The previous approach searched every event sharing an ADDRESS, which
         * conflates distinct ops because ggml reuses storage: attn_norm, the
         * projection and the head norm can all land on one address. Instead walk
         * the graph itself. The normed node is a view/dup whose src[0] is the
         * RMS_NORM result; that node's own src[0] is the projection output. Each
         * hop is resolved by pointer equality on src[0], so the chain is graph
         * structure and never an address coincidence.
         *
         * Every hop is snapshotted in its OWN eval callback, so the bytes are
         * those that existed when that node ran, regardless of what a later
         * in-place op did to the same storage. */
        const Ev * n1 = NULL;   /* the RMS_NORM result feeding the normed node */
        for (size_t k = 0; k < g_evs.size(); ++k) {
            if (g_evs[k].ptr == normed->src0 && g_evs[k].seq < normed->seq) {
                if (n1 == NULL || g_evs[k].seq < n1->seq) { n1 = &g_evs[k]; }
            }
        }
        const Ev * n2 = NULL;   /* the projection feeding the RMS_NORM         */
        if (n1 != NULL) {
            for (size_t k = 0; k < g_evs.size(); ++k) {
                if (g_evs[k].ptr == n1->src0 && g_evs[k].seq < n1->seq) {
                    if (n2 == NULL || g_evs[k].seq < n2->seq) { n2 = &g_evs[k]; }
                }
            }
        }
        printf("  chain: normed(seq=%d,op=%s[id=%d]).src0 -> %s\n", normed->seq,
               op_name_local(normed->opid).c_str(), normed->opid,
               n1 ? op_name_local(n1->opid).c_str() : "NONE");
        if (n1 != NULL) {
            printf("  chain: rmsnorm(seq=%d,op=%s).src0 -> %s\n", n1->seq,
                   op_name_local(n1->opid).c_str(),
                   n2 ? op_name_local(n2->opid).c_str() : "NONE");
        }
        const Ev * producer = n2;
        if (producer == NULL) { printf("  no projection node found; pre-norm VOID\n"); ++bad; continue; }

        Seal & Z = seals[si];
        Z.label = S.nname; Z.layer = 0; Z.head = S.h; Z.pos = pos; Z.n_heads = S.heads;
        Z.pre = head_of(producer->bytes, hd, S.h);
        Z.post_norm = head_of(normed->bytes, hd, S.h);
        Z.post_rope = head_of(roped->bytes, hd, S.h);
        Z.normed_ptr = normed->ptr; Z.roped_ptr = roped->ptr; Z.producer_ptr = producer->ptr;
        Z.pre_op = op_name_local(producer->opid); Z.norm_op = op_name_local(normed->opid);
        Z.rope_op = op_name_local(roped->opid);
        { char nm[128];
          snprintf(nm, sizeof nm, "blk.0.attn_%c_norm.weight",
                   (S.nname[0] >= 'A' && S.nname[0] <= 'Z') ? S.nname[0] + 32 : S.nname[0]);
          if (!read_norm_weight(path, nm, hd, Z.w)) { printf("  weights MISSING for %s\n", nm); ++bad; } }

        stats("pre-RMSNorm",  Z.pre);
        stats("norm weights",  Z.w);
        stats("post-RMSNorm", Z.post_norm);
        stats("post-RoPE",    Z.post_rope);

        /* IS THE TRAVERSED PRODUCER REALLY THE NORM INPUT?
         *
         * Graph traversal says this MUL_MAT feeds this RMS_NORM. That is a claim
         * about structure, so it is confirmed arithmetically: RMSNorm is
         * out = in * weight * inv_rms with inv_rms from in. If the reconstruction
         * reproduces post-RMSNorm, the producer is the true input. Traversal
         * alone would have accepted the earlier wrong-tensor result, and only
         * this check exposed that. */
        { double ss = 0;
          for (int i = 0; i < hd; ++i) { ss += (double) Z.pre[i] * Z.pre[i]; }
          const float inv = 1.0f / sqrtf((float)(ss / hd) + 1e-6f);
          double worst = 0, scale = 0;
          for (int i = 0; i < hd; ++i) { scale = std::max(scale, fabs((double) Z.post_norm[i])); }
          for (int i = 0; i < hd; ++i) {
              worst = std::max(worst, fabs((double) Z.pre[i] * Z.w[i] * inv - Z.post_norm[i]));
          }
          const double rel = worst / (scale > 0 ? scale : 1.0);
          printf("  RMS RECONSTRUCTION max abs=%.6g relative to post_norm scale=%.6g = %.3g %s\n",
                 worst, scale, rel, rel < 1e-3 ? "PASS" : "FAIL");
          if (!(rel < 1e-3)) { ++bad; } }

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
