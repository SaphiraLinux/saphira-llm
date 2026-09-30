// s0_probe.cpp -- Step 0 reference probe. Copyright (C) 2026 Andrew Smalley for
// and on behalf of AKADATA Limited. https://www.akadata.co.uk
// Licensed under the MIT License -- see LICENSE. Part of Saphira Linux.
//
// Step 0 of docs/PLAN-MAINSTREAM-GGUF.md, and it is MEASUREMENT ONLY: there is
// no compute here and none is planned in this file. It links the vendored
// llama.cpp and asks that library what a mainstream GGUF contains, so the
// values it prints are independent of Saphira. Comparing Saphira against
// numbers Saphira produced would be the same circularity that let a finite
// logit computed from all-zero projections pass as a working model.
//
// Build (CPU only, no network):
//   cd third_party/llama.cpp
//   cmake -B /tmp/lcpbuild -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
//         -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF \
//         -DGGML_OPENCL=OFF -DGGML_SYCL=OFF -DGGML_RPC=OFF \
//         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
//   make -C /tmp/lcpbuild -j llama
//   g++ -O2 -std=c++17 -Iinclude -Iggml/include ../../tools/s0_probe.cpp \
//       -o /tmp/s0_probe -L/tmp/lcpbuild/bin -lllama -lggml-base \
//       -Wl,-rpath,/tmp/lcpbuild/bin
//
// Output is captured in tests/golden/mainstream-qwen3-ref.txt.
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
static void hdr(const char*s){ printf("\n===== %s =====\n", s); }
int main(int argc, char** argv){
  const char* path = argv[1];
  const char* text = argc>2 ? argv[2] : "The quick brown fox";
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;

  llama_model* model = llama_model_load_from_file(path, mp);
  if(!model){ printf("LOAD FAILED\n"); return 1; }
  hdr("model");
  printf("n_vocab        = %d\n", llama_vocab_n_tokens(llama_model_get_vocab(model)));
  printf("n_ctx_train    = %u\n", llama_model_n_ctx_train(model));
  printf("n_embd         = %u\n", llama_model_n_embd(model));
  printf("n_layer        = %u\n", llama_model_n_layer(model));
  printf("n_head         = %u\n", llama_model_n_head(model));
  printf("n_head_kv      = %u\n", llama_model_n_head_kv(model));
  printf("n_embd_out     = %u\n", llama_model_n_embd_out(model));
  printf("n_swa         = %u\n", llama_model_n_swa(model));

  hdr("tokenizer: input ids for a known string");
  const llama_vocab* vocab = llama_model_get_vocab(model);
  llama_token toks[256];
  int n = llama_tokenize(vocab, text, (int32_t)strlen(text), toks, 256, true, true);
  printf("text = \"%s\"\n", text);
  printf("n_pieces = %d\n", n);
  for(int i=0;i<n;i++){
    char buf[64]={0};
    llama_token_to_piece(vocab, toks[i], buf, sizeof buf, 0, true);
    printf("  [%2d] id=%6d  piece=\"%s\"\n", i, toks[i], buf);
  }
  hdr("raw GGUF via llama.cpp's own reader");
  gguf_init_params gp = { true, NULL };
  gguf_context* gctx = gguf_init_from_file(path, gp);
  if(gctx){
    int64_t nkv = gguf_get_n_kv(gctx);
    printf("n_kv = %lld\n", (long long)nkv);
    for(int64_t i=0;i<nkv;i++){
      const char* k = gguf_get_key(gctx, i);
      if(strncmp(k,"tokenizer.ggml.tokens",20) && strncmp(k,"general.",8)) {
        printf("  %-46s type=%d\n", k, (int)gguf_get_kv_type(gctx,i));
      }
    }
    // vocab arrays are large; counted separately
    
    gguf_free(gctx);
  }
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
