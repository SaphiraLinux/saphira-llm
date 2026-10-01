#include "gguf.h"
#include <cstdio>
#include <cstring>
int main(int argc,char**argv){ gguf_init_params gp; memset(&gp,0,sizeof gp); gp.no_alloc=true;
 gguf_context*g=gguf_init_from_file(argv[1],gp);
 printf("keys containing 'token' that are NOT the token array:\n");
 for(int64_t i=0;i<gguf_get_n_kv(g);++i){const char*k=gguf_get_key(g,i);
   if(!k||!strstr(k,"tokenizer."))continue; if(strstr(k,".tokens")||strstr(k,".token_type")||strstr(k,".scores")||strstr(k,".merges"))continue;
   printf("  %-46s type=%s",k,gguf_type_name(gguf_get_kv_type(g,i)));
   enum gguf_type t=gguf_get_kv_type(g,i);
   if(t==GGUF_TYPE_UINT32)printf(" = %u",gguf_get_val_u32(g,i));
   else if(t==GGUF_TYPE_INT32)printf(" = %d",gguf_get_val_i32(g,i));
   else if(t==GGUF_TYPE_BOOL)printf(" = %s",gguf_get_val_bool(g,i)?"true":"false");
   else if(t==GGUF_TYPE_FLOAT32)printf(" = %.9g",gguf_get_val_f32(g,i));
   printf("\n");}
 gguf_free(g); return 0;}
