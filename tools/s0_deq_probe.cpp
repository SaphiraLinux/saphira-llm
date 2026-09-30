// Reference dequantisation capture. llama.cpp's own type traits do the decoding,
// via to_float for the type. No Saphira dequantisation exists or is used here.
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
static uint64_t fnv(const void* p, size_t n, uint64_t h=1469598103934665603ULL){
  const uint8_t* b=(const uint8_t*)p; for(size_t i=0;i<n;i++){ h^=b[i]; h*=1099511628211ULL; } return h; }
struct Case { const char* name; int row; };
int main(int argc,char**argv){
  // Read the file ourselves so the RAW ENCODED BYTES are in hand to hash, and
  // so the byte offset is computed rather than assumed.
  FILE* f=fopen(argv[1],"rb"); if(!f){printf("open fail\n");return 1;}
  fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* file=(uint8_t*)malloc((size_t)fsz);
  if(fread(file,1,(size_t)fsz,f)!=(size_t)fsz){printf("short read\n");return 1;}
  fclose(f);
  gguf_init_params gp={true,NULL};
  gguf_context* g=gguf_init_from_file(argv[1],gp);
  if(!g){printf("gguf fail\n");return 1;}
  const size_t data_base = gguf_get_data_offset(g);
  // Q4_K, Q6_K, and an F32 control. F16 is absent from this model -- recorded
  // as a finding rather than silently substituted.
  Case cases[] = {
    {"blk.0.attn_q.weight",      0},   // Q4_K, 4096 in  x 4096 out
    {"blk.0.ffn_gate.weight",    7},   // Q4_K, a non-zero row
    {"blk.0.attn_v.weight",      0},   // Q6_K, 4096 in  x 1024 out
    {"blk.0.ffn_down.weight",    3},   // Q6_K, a non-zero row
    {"token_embd.weight",        0},   // Q4_K embedding row
    {"blk.0.attn_norm.weight",   0},   // F32 CONTROL, 1-D
  };
  for(unsigned ci=0; ci<sizeof cases/sizeof*cases; ++ci){
    int64_t ti = gguf_find_tensor(g, cases[ci].name);
    if(ti<0){ printf("MISSING %s\n", cases[ci].name); continue; }
    ggml_type ty = (ggml_type) gguf_get_tensor_type(g, ti);
    const int64_t* ne = gguf_get_tensor_ne(g, ti);
    const int64_t rowlen = ne[0];
    const int64_t r = cases[ci].row;
    const void* base = file + data_base + gguf_get_tensor_offset(g, ti);
    const int64_t blck = ggml_blck_size(ty);
    const size_t  tsz  = ggml_type_size(ty);
    if(rowlen % blck){ printf("SKIP %s: ne0=%lld not a multiple of blck=%lld\n",
                              cases[ci].name,(long long)rowlen,(long long)blck); continue; }
    const size_t row_bytes = (size_t)(rowlen/blck)*tsz;
    const void* src = (const void*)((const uint8_t*)base + r*row_bytes);
    printf("\n[case] %s\n", cases[ci].name);
    printf("  type            = %s (ggml %d)\n", ggml_type_name(ty), (int)ty);
    printf("  ne              = [%lld, %lld]\n", (long long)ne[0], (long long)ne[1]);
    printf("  row             = %lld\n", (long long)r);
    printf("  block_elems     = %lld   type_bytes = %zu   row_encoded_bytes = %zu\n",
           (long long)blck, tsz, row_bytes);
    printf("  decoded_elems   = %lld\n", (long long)rowlen);
    printf("  RAW_HASH        = %016llx\n", (unsigned long long)fnv(src,row_bytes));
    float* dst=(float*)malloc(sizeof(float)*(size_t)rowlen);
    const ggml_type_traits* tr = ggml_get_type_traits(ty);
    if (ty == GGML_TYPE_F32) {
      /* ggml's f32 traits intentionally carry no to_float: f32 IS the decoded
       * form, so the decode is a copy. Treating "no to_float" as a failure here
       * would have silently dropped the control case. */
      memcpy(dst, src, sizeof(float)*(size_t)rowlen);
    } else if(!tr || !tr->to_float){
      printf("  NO KERNEL for %s in the reference -- cannot capture a control\n",
             ggml_type_name(ty));
      free(dst); continue;
    } else {
      tr->to_float(src, dst, (int)rowlen);
    }
    printf("  DECODED_HASH    = %016llx\n", (unsigned long long)fnv(dst,sizeof(float)*(size_t)rowlen));
    printf("  first8          =");
    for(int i=0;i<8 && i<rowlen;i++) printf(" %.9g", dst[i]);
    printf("\n  last4           =");
    for(int64_t i=(rowlen>4?rowlen-4:0);i<rowlen;i++) printf(" %.9g", dst[i]);
    double mn=dst[0],mx=dst[0],sum=0; int nonfinite=0;
    for(int64_t i=0;i<rowlen;i++){ float v=dst[i];
      if(!std::isfinite(v)) nonfinite++;
      if(v<mn)mn=v; if(v>mx)mx=v; sum+=v; }
    printf("\n  min=%.9g max=%.9g mean=%.9g nonfinite=%d\n", mn,mx,sum/(double)rowlen,nonfinite);
    free(dst);
  }
  gguf_free(g); return 0; }
