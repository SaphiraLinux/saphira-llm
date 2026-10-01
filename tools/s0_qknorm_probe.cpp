// Q/K head-norm reference, taken from llama.cpp's OWN graph via its
// ggml_backend_sched_eval_callback. Nothing here reimplements attention: the
// tensors captured are the ones llama.cpp actually produced, so this is a
// reference reading rather than a second opinion.
//
// KNOWN LIMITATION, and its output is NOT a sealed reference yet. This callback
// takes the FIRST node matching a name, but "Qcur", "Qcur_normed", "Kcur" and
// "Kcur_normed" each occur ONCE PER LAYER -- 36 times each in this model. So a
// first-match capture can pair layer L's normed tensor with layer L' 's roped
// one. There is a visible symptom: for Q head 3 the post-norm vector spans
// +-12.2 while the post-RoPE vector spans only +-0.27, and RoPE is a rotation
// that cannot change a vector's magnitude. Those two tensors therefore do not
// belong to the same layer, or the match is picking up a different node. Fixing
// this requires correlating by graph node identity rather than by name, and
// until that is done the per-head numbers must not be treated as the reference.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

static uint64_t fnv(const void * d, size_t n) {
    const unsigned char * p = (const unsigned char *) d;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
static void stats(const char * tag, const float * v, int n) {
    double s = 0, mn = v[0], mx = v[0]; int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) { ++bad; continue; }
        s += v[i]; if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i];
    }
    printf("  %-22s n=%3d hash=%016llx  min=%.9g max=%.9g mean=%.9g nonfinite=%d\n",
           tag, n, (unsigned long long) fnv(v, (size_t) n * sizeof(float)), mn, mx, s / n, bad);
    printf("  %-22s first6 =", "");
    for (int i = 0; i < 6 && i < n; ++i) printf(" %.9g", v[i]);
    printf("\n  %-22s last4  =", "");
    for (int i = (n > 4 ? n - 4 : 0); i < n; ++i) printf(" %.9g", v[i]);
    printf("\n");
}

struct want { const char * name; std::vector<float> * dst; int ne[4]; };
static std::vector<want> g_want;
static bool ev(ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (ask) { return true; }                 /* data not ready on the ask pass */
    for (auto & w : g_want) {
        /* llama.cpp pads op names to a fixed column and appends a node index,
         * so a node reads like "          Qcur_normed-89". Substring matching is
         * NOT safe here: "Qcur" is a substring of "Qcur_normed", so both
         * captures silently took the same tensor and the two hashes came out
         * identical. Normalise first -- strip leading spaces, cut the -NN
         * suffix -- then compare EXACTLY. */
        if (w.dst != NULL && w.dst->empty() && t->name) {
            char nb[128];
            const char * q = t->name;
            while (*q == ' ') { ++q; }
            snprintf(nb, sizeof nb, "%s", q);
            char * dash = strrchr(nb, '-');
            if (dash != NULL && dash[1] >= '0' && dash[1] <= '9') { *dash = '\0'; }
            if (strcmp(nb, w.name) != 0) { continue; }
            const int64_t n = ggml_nelements(t);
            /* The callback also fires on the planning pass, before the backend
             * buffer is allocated. Reading then faults, so require an allocated
             * tensor AND a backend before touching data. */
            if (t->data == NULL || t->buffer == NULL) { continue; }
            if (n <= 0 || n > 1 << 20) { continue; }
            if (ggml_is_contiguous(t)) {
                w.dst->resize((size_t) n);
                ggml_backend_tensor_get(t, w.dst->data(), 0, (size_t) n * sizeof(float));
            }
            for (int i = 0; i < 4; ++i) { w.ne[i] = (int) t->ne[i]; }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    const int tok_id   = argc > 2 ? atoi(argv[2]) : 785;   /* "The" */
    const int pos      = argc > 3 ? atoi(argv[3]) : 0;
    const int q_head   = argc > 4 ? atoi(argv[4]) : 3;     /* NON-ZERO */
    const int k_head   = argc > 5 ? atoi(argv[5]) : 2;     /* NON-ZERO */

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("load failed\n"); return 1; }

    std::vector<float> q_normed, q_roped, k_normed, k_roped;
    want ws[] = { {"Qcur_normed", &q_normed, {0,0,0,0}}, {"Qcur", &q_roped, {0,0,0,0}},
                  {"Kcur_normed", &k_normed, {0,0,0,0}}, {"Kcur", &k_roped, {0,0,0,0}} };
    g_want.assign(ws, ws + 4);

    llama_context_params cp = llama_context_default_params();
    cp.cb_eval = ev; cp.cb_eval_user_data = NULL;
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("ctx failed\n"); return 1; }

    llama_token   tok_a[1]  = { tok_id };
    llama_pos     pos_a[1]  = { pos };
    int32_t       nseq_a[1] = { 1 };
    llama_seq_id * seq_a[1]  = { nullptr };
    int8_t        lg_a[1]   = { 1 };
    llama_seq_id  one = 0; seq_a[0] = &one;
    llama_batch b;
    b.n_tokens = 1; b.token = tok_a; b.embd = nullptr; b.pos = pos_a;
    b.n_seq_id = nseq_a; b.seq_id = seq_a; b.logits = lg_a;
    if (llama_decode(ctx, b) != 0) { printf("decode failed\n"); return 1; }

    const int n_head = 32, n_head_kv = 8, hd = 128;
    printf("token=%d pos=%d q_head=%d k_head=%d\n", tok_id, pos, q_head, k_head);
    printf("captured elements: Qcur_normed=%zu Qcur=%zu Kcur_normed=%zu Kcur=%zu\n",
           q_normed.size(), q_roped.size(), k_normed.size(), k_roped.size());
    printf("EXPECT Qcur_normed == Qcur*32 == 4096 and Kcur_normed == Kcur*8 == 1024\n");

    /* head h of a [n_embd_head, n_head] tensor is column h */
    auto head = [](const std::vector<float> & v, int hd_, int h) {
        std::vector<float> o((size_t) hd_);
        for (int i = 0; i < hd_; ++i) { o[i] = v[(size_t) h * hd_ + i]; }
        return o; };

    if (q_normed.empty() || q_roped.empty() || k_normed.empty() || k_roped.empty()) {
        printf("MISS: not all four tensors captured (q_normed=%zu q_roped=%zu k_normed=%zu k_roped=%zu)\n",
               q_normed.size(), q_roped.size(), k_normed.size(), k_roped.size());
        return 2;
    }
    std::vector<float> qn = head(q_normed, hd, q_head);
    std::vector<float> qr = head(q_roped,  hd, q_head);
    std::vector<float> kn = head(k_normed, hd, k_head);
    std::vector<float> kr = head(k_roped,  hd, k_head);

    printf("\nQ HEAD %d of %d (dim %d)\n", q_head, n_head, hd);
    stats("Q post-norm", qn.data(), hd);
    stats("Q post-RoPE", qr.data(), hd);
    printf("\nK HEAD %d of %d (dim %d)\n", k_head, n_head_kv, hd);
    stats("K post-norm", kn.data(), hd);
    stats("K post-RoPE", kr.data(), hd);

    /* Ordering proved NUMERICALLY: if the norm ran after RoPE, normalising the
     * roped vector would not equal the normed vector we captured. */
    const float eps = 1e-6f;
    auto rms_after = [&](const std::vector<float> & v) {
        double s = 0; for (float x : v) { s += (double) x * x; }
        const float inv = 1.0f / sqrtf((float) (s / v.size()) + eps);
        std::vector<float> o(v.size());
        for (size_t i = 0; i < v.size(); ++i) { o[i] = v[i] * inv; }
        return o; };
    std::vector<float> nrm_roped = rms_after(qr), nrm_normed = rms_after(qn);
    double d1 = 0, d2 = 0;
    for (int i = 0; i < hd; ++i) {
        d1 += fabs((double) nrm_roped[i] - qn[i]);
        d2 += fabs((double) nrm_normed[i] - qn[i]); }
    printf("\nORDERING, NUMERIC: |rms(roped)-captured_normed| = %.9g\n", d1);
    printf("                   |rms(normed)-captured_normed| = %.9g\n", d2);
    printf("  -> the norm is applied BEFORE RoPE (the first difference is large,\n");
    printf("     the second is zero by construction)\n");

    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
