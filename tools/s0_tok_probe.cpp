// Expanded tokenizer reference. llama.cpp's own vocab API decides the ids; this
// only prints them. Nothing is compared and nothing is normalised.
#include "llama.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
struct T { const char* label; const char* text; };
int main(int argc,char**argv){
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  llama_model* m = llama_model_load_from_file(argv[1], mp);
  if(!m){ printf("LOAD FAILED\n"); return 1; }
  const llama_vocab* v = llama_model_get_vocab(m);

  printf("=== tokenizer metadata, from GGUF via llama.cpp ===\n");
  printf("n_vocab                 = %d\n", llama_vocab_n_tokens(v));
  printf("vocab_type (llama.cpp)  = %s\n",
    llama_vocab_type(v)==LLAMA_VOCAB_TYPE_SPM ? "SPM" :
    llama_vocab_type(v)==LLAMA_VOCAB_TYPE_BPE ? "BPE" :
    llama_vocab_type(v)==LLAMA_VOCAB_TYPE_WPM ? "WPM" : "?");
  printf("bos=%d eos=%d eot=%d sep=%d pad=%d\n",
    llama_vocab_bos(v), llama_vocab_eos(v), llama_vocab_eot(v),
    llama_vocab_sep(v), llama_vocab_pad(v));
  printf("add_bos=%d add_eos=%d add_sep=%d\n",
    llama_vocab_get_add_bos(v), llama_vocab_get_add_eos(v), llama_vocab_get_add_sep(v));
  // The selector must be read from the GGUF, because that is all a Saphira
  // runtime has. Read the raw keys directly rather than trusting a helper.
  { gguf_init_params gp={true,NULL};
    gguf_context* g = gguf_init_from_file(argv[1], gp);
    const char* K[]={"tokenizer.ggml.pre","tokenizer.ggml.model",
                      "tokenizer.ggml.bos_token_id","tokenizer.ggml.eos_token_id",
                      "tokenizer.ggml.unknown_token_id","tokenizer.ggml.padding_token_id",
                      "tokenizer.ggml.add_bos_token","tokenizer.ggml.add_eos_token",
                      "tokenizer.ggml.add_space_prefix","tokenizer.ggml.ignore_merges",
                      "tokenizer.chat_template","general.name"};
    for(unsigned i=0;i<sizeof K/sizeof*K;++i){
      int64_t k=gguf_find_key(g,K[i]); if(k<0){ printf("GGUF[absent] %s\n",K[i]); continue; }
      enum gguf_type t=gguf_get_kv_type(g,k);
            if(t==GGUF_TYPE_STRING){ const char*sv=gguf_get_val_str(g,k);
        printf("GGUF %-34s = [%s]\n",K[i],sv?sv:""); }
      else if(t==GGUF_TYPE_UINT32){ printf("GGUF %-34s = %u\n",K[i],gguf_get_val_u32(g,k)); }
      else if(t==GGUF_TYPE_BOOL){ printf("GGUF %-34s = %s\n",K[i],gguf_get_val_bool(g,k)?"true":"false"); }
      else printf("GGUF %-34s = (type %d)\n",K[i],(int)t);
    }
    // token_type array, so "special/control tokens actually declared" is evidenced
    int64_t tk=gguf_find_key(g,"tokenizer.ggml.token_type");
    if(tk>=0 && gguf_get_arr_n(g,tk)>0){
      const int32_t* tt=(const int32_t*)gguf_get_arr_data(g,tk);
      int h[16]={0}; for(int64_t i=0;i<gguf_get_arr_n(g,tk);++i) h[tt[i]&15]++;
      printf("GGUF token_type array n=%lld  counts:",(long long)gguf_get_arr_n(g,tk));
      for(int i=0;i<16;i++) if(h[i]) printf(" type%d=%d",i,h[i]); printf("\n");
    }
    int64_t at=gguf_find_key(g,"tokenizer.ggml.added_tokens");
    if(at>=0) printf("GGUF added_tokens (serialised block) present, n=%lld\n",(long long)gguf_get_arr_n(g,at));
    gguf_free(g); }
  printf("token_type histogram (llama_token_attr):\n");
  { int hist[16]={0};
    for(int i=0;i<llama_vocab_n_tokens(v);++i) hist[(int)llama_vocab_get_attr(v,i) & 15]++;
    for(int i=0;i<16;i++) if(hist[i]) printf("   attr %2d : %d\n", i, hist[i]); }

  T cases[] = {
    {"baseline",            "The quick brown fox"},
    {"leading-space",       " The"},
    {"trailing-space",      "The "},
    {"both-spaces",         " The "},
    {"repeated-spaces",     "a  b   c    d"},
    {"only-spaces",         "   "},
    {"punctuation",         "Hello, world! (yes) [no] {maybe} <tag>"},
    {"punct-dense",         "a,b;c:d.e/f\\g|h~i`j\"k"},
    {"digits",              "0123456789"},
    {"newline",             "line one\nline two\n\nline four"},
    {"tab-cr",              "a\tb\rc"},
    {"utf8-accented",       "café naïve résumé"},
    {"utf8-cjk",            "日本語のテキスト"},
    {"utf8-emoji",          "hello 👋🏽 world 🌍 done"},
    {"utf8-mixed",          "naïve café 日本語 👋 ok"},
    {"hyphen-underscore",   "snake_case kebab-case"},
    {"url-like",            "https://example.com/path?q=1&r=2"},
    {"json-like",           "{\"k\": [1, 2, null], \"s\": \"v\"}"},
    {"long-word",           "supercalifragilisticexpialidocious"},
    {"single-char",         "x"},
    {"empty",               ""},
  };
  printf("\n=== encode cases (add_special=true, parse_special=true) ===\n");
  for(unsigned ci=0; ci<sizeof cases/sizeof*cases; ++ci){
    std::string t = cases[ci].text;
    /* The exact INPUT BYTES, hex encoded, so the Saphira test can reconstruct
     * them byte-exactly. Derived here from the same literal the llama.cpp call
     * receives, so the expectation cannot drift from what was actually run --
     * and it is bytes, not a UTF-8 string, because these pieces are not
     * individually valid UTF-8. */
    { std::string h; char hb[4];
      for(unsigned char c : t){ snprintf(hb,sizeof hb,"%02X",c); h+=hb; }
      printf("   text_hex = %s\n", h.c_str()); }
    std::vector<llama_token> toks(512);
    int n = llama_tokenize(v, t.c_str(), (int32_t)t.size(), toks.data(), 512, true, true);
    printf("[%s] %zu bytes -> n=%d\n", cases[ci].label, t.size(), n);
    if(n<0){ printf("   (tokenizer returned %d)\n", n); continue; }
    printf("   ids =");
    for(int i=0;i<n;i++) printf(" %d", (int)toks[i]);
    printf("\n   pieces =");
    for(int i=0;i<n;i++){ char b[128]={0};
      llama_token_to_piece(v, toks[i], b, sizeof b, 0, true);
      printf(" \"%s\"", b); }
    printf("\n");
    // encode -> decode round trip, text only (no specials)
    if(n>0){
      std::vector<char> buf(4096);
      int dn = llama_detokenize(v, toks.data(), n, buf.data(), (int)buf.size(), false, true);
      if(dn>0){
        std::string back(buf.data(), dn);
        std::string hb; char cb[4];
        for(unsigned char c : back){ snprintf(cb,sizeof cb,"%02X",c); hb+=cb; }
        printf("   decode_hex = %s   byte_identical=%s\n",
               hb.c_str(), (back==t)?"yes":"NO");
      } else printf("   decode_rt = (failed %d)\n", dn);
    }
  }
  llama_model_free(m); llama_backend_free();
  return 0; }
