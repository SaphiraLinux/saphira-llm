/* Layer topology: which computational bodies exist at WHICH depth.
 *
 * Count-based surveys answer "how many layers", which is not the question. The
 * question is whether the layer graph CHANGES across depth, and that needs the
 * exact index set per body. A count of 4 attention layers among 52 is a very
 * different structure from 4 layers of 4 spread evenly.
 *
 * Written to be per-layer and index-exact by construction. */
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <set>
#include <map>
#include <string>
#include <vector>

/* "blk.7.attn_q.weight" -> 7. The convention is a PREFIX field "blk" then an
 * INTEGER index field then the role. Splitting on the last dot, or assuming a
 * fixed substring, silently yields zero layers, which then reports every
 * artefact as UNIFORM. A topology probe that cannot find its own layers is
 * worse than no probe, so the prefix is matched rather than assumed and the
 * index field is validated as digits. */
static int split_layer(const std::string & n) {
    const size_t i = n.find('.');
    if (i == std::string::npos) return -1;
    const std::string head = n.substr(0, i);
    if (head != "blk") return -1;
    const size_t j = n.find('.', i + 1);
    if (j == std::string::npos) return -1;
    const std::string idx = n.substr(i + 1, j - i - 1);
    if (idx.empty() || idx.size() > 6) return -1;
    for (char c : idx) if (c < '0' || c > '9') return -1;
    return atoi(idx.c_str());
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    gguf_init_params p; memset(&p, 0, sizeof p); p.no_alloc = true;
    gguf_context * g = gguf_init_from_file(argv[1], p);
    if (!g) { fprintf(stderr, "cannot open\n"); return 1; }
    const int64_t nt = gguf_get_n_tensors(g);

    /* Classify every tensor into a body: what computation does it belong to. */
    struct Body { std::set<int> layers; std::set<int> nonlayer; };
    std::map<std::string, Body> body;

    auto add = [&](const std::string & role, int l) {
        Body & b = body[role];
        if (l < 0) b.nonlayer.insert(-2); else b.layers.insert(l);
    };

    int maxl = -1;
    for (int64_t i = 0; i < nt; ++i) {
        const std::string n = gguf_get_tensor_name(g, i);
        const int l = split_layer(n);
        if (l > maxl) maxl = l;

        /* SSM body. */
        if (n.find("ssm_a") != std::string::npos || n.find("ssm_d") != std::string::npos ||
            n.find("ssm_conv1d") != std::string::npos || n.find("ssm_in.") != std::string::npos ||
            n.find("ssm_out.") != std::string::npos || n.find("ssm_dt") != std::string::npos ||
            n.find("ssm_norm") != std::string::npos) { add("SSM", l); continue; }
        /* Attention body. Q/K norm is attention-local: it cannot exist without
         * a Q or K projection to norm, and it is not a generic residual norm. */
        if (n.find("attn_q") != std::string::npos || n.find("attn_k") != std::string::npos ||
            n.find("attn_v") != std::string::npos || n.find("attn_output") != std::string::npos) {
            add("ATTN", l); continue;
        }
        if (n.find("attn_q_norm") != std::string::npos || n.find("attn_k_norm") != std::string::npos) {
            add("ATTN", l); continue;
        }
        /* MoE router / experts. */
        if (n.find("experts") != std::string::npos || n.find("ffn_gate_inp") != std::string::npos) {
            add("MOE", l); continue; }
        /* Feed-forward body, gated or dense. */
        if (n.find("ffn_up") != std::string::npos || n.find("ffn_down") != std::string::npos ||
            n.find("ffn_gate.") != std::string::npos) { add("FFN", l); continue; }
        /* Generic residual norm. attn_norm is named like a residual norm but is
         * attention-local in practice; report it separately rather than
         * assuming, and let the layer map show where it actually lands. */
        if (n.find("attn_norm") != std::string::npos) { add("ATTN_NORM", l); continue; }
        if (n.find("ffn_norm") != std::string::npos) { add("FFN_NORM", l); continue; }
        add("OTHER", l);
    }

    const int nl = maxl + 1;
    printf("\n=== LAYER TOPOLOGY (index-exact) ===\n");
    printf("  layer count = %d\n", nl);
    printf("  %-10s %6s  %s\n", "BODY", "COUNT", "LAYER INDICES");
    for (const auto & kv : body) {
        if (kv.second.layers.empty() && kv.second.nonlayer.empty()) continue;
        printf("  %-10s %6zu  ", kv.first.c_str(), kv.second.layers.size());
        int n = 0;
        for (int l : kv.second.layers) { if (n) putchar(' '); printf("%d", l); if (++n >= 64) { printf(" ..."); break; } }
        if (kv.second.nonlayer.size()) printf("  (+%zu non-layer)", kv.second.nonlayer.size());
        putchar('\n');
    }

    /* The decisive table: every distinct layer body, and which layers share it. */
    printf("\n=== DISTINCT LAYER BODIES ===\n");
    std::map<std::string, std::vector<int>> sigs;
    for (int l = 0; l < nl; ++l) {
        std::string s;
        for (const char * b : {"ATTN","SSM","MOE","FFN"})
            if (body.count(b) && body[b].layers.count(l)) { s += b; s += "+"; }
        if (s.empty()) s = "NONE+";
        sigs[s].push_back(l);
    }
    for (const auto & kv : sigs)
        printf("  %-28s layers %zu  first=%d last=%d\n", kv.first.c_str(), kv.second.size(),
               kv.second.front(), kv.second.back());

    const size_t distinct = sigs.size();
    printf("\n  distinct layer bodies = %zu over %d layers\n", distinct, nl);
    printf("  verdict: %s\n", distinct <= 1
        ? "UNIFORM -- every layer has the same body. A per-layer IR cannot be"
          " distinguished from a uniform-layer one on this artefact."
        : "HETEROGENEOUS -- the layer graph CHANGES across depth. Any IR that"
          " emits one body per model, or one body per layer by assumption, is"
          " already wrong here.");
    printf("  NOTE: heterogeneity is a property of the ARTEFACT's tensor layout,\n");
    printf("        measured above, not a property of the architecture name.\n");

    gguf_free(g);
    return 0;
}
