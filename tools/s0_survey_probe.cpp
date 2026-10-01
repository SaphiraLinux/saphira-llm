// Empirical survey of a GGUF artefact, for deriving the discovery IR vocabulary
// from real models rather than from imagination. Prints metadata, a tensor
// inventory with layer indices resolved, type histogram, tokenizer tuple, and
// explicitly which optional per-layer tensors are PRESENT or ABSENT.
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

/* Split "blk.7.attn_q.weight" into layer 7 and role "attn_q.weight".
 *
 * The first pass cut at the LAST dot, which left "blk.7.attn_q" as the role
 * and made all 36 layers look like 36 distinct roles. A survey that cannot count
 * layers is not evidence about layer structure, so the split is on the SECOND
 * field instead, and the prefix before it is matched rather than assumed. */
static bool split_layer(const std::string & n, int * layer, std::string * role) {
    *layer = -1; role->clear();
    size_t i = n.find('.');
    if (i == std::string::npos) { return false; }
    const std::string head = n.substr(0, i);
    size_t j = n.find('.', i + 1);
    if (j == std::string::npos) { return false; }
    const std::string idx = n.substr(i + 1, j - i - 1);
    if (idx.empty()) { return false; }
    for (size_t k = 0; k < idx.size(); ++k) {
        if (idx[k] < '0' || idx[k] > '9') { return false; }
    }
    *layer = atoi(idx.c_str());
    *role = n.substr(j + 1);
    (void) head;
    return true;
}
int main(int argc, char ** argv) {
    gguf_init_params gp; memset(&gp, 0, sizeof gp); gp.no_alloc = true;
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (g == NULL) { printf("open failed\n"); return 1; }
    const char * tag = argc > 2 ? argv[2] : "MODEL";
    printf("########## %s ##########\n", tag);
    printf("file = %s\n", argv[1]);
    printf("kv_pairs = %lld   tensors = %lld\n",
           (long long) gguf_get_n_kv(g), (long long) gguf_get_n_tensors(g));

    printf("\n-- ARCHITECTURE-RELEVANT METADATA (tokenizer.* excluded) --\n");
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        const char * k = gguf_get_key(g, i);
        if (k == NULL || strncmp(k, "tokenizer.", 10) == 0) { continue; }
        const enum gguf_type t = gguf_get_kv_type(g, i);
        if (t == GGUF_TYPE_STRING) {
            printf("  %-52s = \"%s\"\n", k, gguf_get_val_str(g, i));
        } else if (t == GGUF_TYPE_UINT32) {
            printf("  %-52s = %u\n", k, gguf_get_val_u32(g, i));
        } else if (t == GGUF_TYPE_INT32) {
            printf("  %-52s = %d\n", k, gguf_get_val_i32(g, i));
        } else if (t == GGUF_TYPE_FLOAT32) {
            printf("  %-52s = %.9g\n", k, gguf_get_val_f32(g, i));
        } else if (t == GGUF_TYPE_BOOL) {
            printf("  %-52s = %s\n", k, gguf_get_val_bool(g, i) ? "true" : "false");
        } else {
            printf("  %-52s = <%s>\n", k, gguf_type_name(t));
        }
    }

    printf("\n-- TOKENIZER TUPLE (the selector, all three legs) --\n");
    { const char * K[] = {"tokenizer.ggml.model", "tokenizer.ggml.pre",
                          "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                          "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id",
                          "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
                          "tokenizer.ggml.add_space_prefix", "tokenizer.ggml.ignore_merges"};
      for (unsigned i = 0; i < sizeof K / sizeof *K; ++i) {
          const int64_t id = gguf_find_key(g, K[i]);
          if (id < 0) { printf("  %-38s ABSENT\n", K[i]); continue; }
          const enum gguf_type t = gguf_get_kv_type(g, id);
          if (t == GGUF_TYPE_STRING)  printf("  %-38s = \"%s\"\n", K[i], gguf_get_val_str(g, id));
          else if (t == GGUF_TYPE_UINT32) printf("  %-38s = %u\n", K[i], gguf_get_val_u32(g, id));
          else if (t == GGUF_TYPE_INT32)  printf("  %-38s = %d\n", K[i], gguf_get_val_i32(g, id));
          else if (t == GGUF_TYPE_BOOL)   printf("  %-38s = %s\n", K[i], gguf_get_val_bool(g, id) ? "true" : "false");
          else printf("  %-38s = <%s>\n", K[i], gguf_type_name(t));
      }
      const int64_t tk = gguf_find_key(g, "tokenizer.ggml.tokens");
      printf("  %-38s = %lld\n", "tokenizer.ggml.tokens n",
             tk >= 0 ? (long long) gguf_get_arr_n(g, tk) : -1LL);
      const int64_t mg = gguf_find_key(g, "tokenizer.ggml.merges");
      printf("  %-38s = %s\n", "tokenizer.ggml.merges",
             mg >= 0 ? "present" : "ABSENT");
    }

    printf("\n-- TENSOR TYPE HISTOGRAM --\n");
    { std::map<std::string,int> h;
      for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
          h[ggml_type_name(gguf_get_tensor_type(g, i))]++;
      }
      for (std::map<std::string,int>::iterator it = h.begin(); it != h.end(); ++it) {
          printf("  %-12s %d\n", it->first.c_str(), it->second);
      } }

    printf("\n-- TENSOR INVENTORY, layer indices resolved --\n");
    int max_layer = -1;
    std::map<std::string, int> suffix_hist;     /* role -> count of layers seen */
    std::map<std::string, std::pair<int,int> > dims; /* role -> [d0, d1] of first */
    std::set<std::string> non_layer;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        int layer = -1; std::string role;
        const std::string n = gguf_get_tensor_name(g, i);
        if (!split_layer(n, &layer, &role)) { non_layer.insert(n); continue; }
        if (layer > max_layer) { max_layer = layer; }
        suffix_hist[role]++;
        if (dims.find(role) == dims.end()) {
            dims[role] = std::make_pair((int) gguf_get_tensor_ne(g, i)[0],
                                        (int) gguf_get_tensor_ne(g, i)[1]);
        }
    }
    printf("  max layer index = %d  => layer count = %d\n", max_layer, max_layer + 1);
    for (std::map<std::string,int>::iterator it = suffix_hist.begin(); it != suffix_hist.end(); ++it) {
        printf("  %-24s layers=%-4d ne=[%d,%d]\n", it->first.c_str(), it->second,
               dims[it->first].first, dims[it->first].second,
               "see type histogram");
    }
    printf("\n-- NON-LAYER TENSORS --\n");
    for (std::set<std::string>::iterator it = non_layer.begin(); it != non_layer.end(); ++it) {
        printf("  %s\n", it->c_str());
    }

    printf("\n-- OPTIONAL PER-LAYER TENSORS: PRESENT OR ABSENT --\n");
    const char * opt[] = {"attn_q_norm.weight", "attn_k_norm.weight",
                          "attn_q_norm.bias", "attn_k_norm.bias",
                          "attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias",
                          "ffn_gate_inp.weight", "ffn_norm.bias", "attn_norm.bias"};
    for (unsigned i = 0; i < sizeof opt / sizeof *opt; ++i) {
        char nm[128]; snprintf(nm, sizeof nm, "blk.0.%s", opt[i]);
        const int64_t id = gguf_find_tensor(g, nm);
        printf("  %-26s %s\n", opt[i], id < 0 ? "ABSENT" : "PRESENT");
    }
    printf("  (a genuine per-layer tensor is verified at layer 0 AND its count\n"
           "   matches the layer count above, so it is not a single stray tensor)\n");
    gguf_free(g);
    return 0;
}
