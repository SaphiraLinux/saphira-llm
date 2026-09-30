// Prove or disprove: does this GGUF store pieces in the GPT-2 bytes_to_unicode
// alphabet? Read RAW BYTES of tokenizer.ggml.tokens. Do not ask llama.cpp to
// render pieces, because llama.cpp byte-DECODES them and would hide the answer.
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
static std::string hex(const std::string& s){
  std::string o; char b[8];
  for(unsigned char c : s){ snprintf(b,sizeof b,"%02X ",c); o+=b; }
  if(!o.empty()) o.pop_back(); return o; }
int main(int argc,char**argv){
  gguf_init_params gp={true,NULL};
  gguf_context* g=gguf_init_from_file(argv[1],gp);
  if(!g){ printf("open fail\n"); return 1; }
  int64_t k=gguf_find_key(g,"tokenizer.ggml.tokens");
  if(k<0){ printf("no tokens key\n"); return 1; }
  int64_t n=gguf_get_arr_n(g,k);
  printf("tokens n = %lld\n",(long long)n);

  // Build the GPT-2 bytes_to_unicode map, exactly as specified.
  std::map<int,unsigned> b2u;
  { /* GPT-2 bytes_to_unicode, transcribed exactly: bytes in bs map to
       themselves, every other byte maps to 256+n in increasing byte order. */
    std::vector<int> bs;
    for(int i=33;i<=126;++i) bs.push_back(i);
    for(int i=161;i<=172;++i) bs.push_back(i);
    for(int i=174;i<=255;++i) bs.push_back(i);
    std::vector<bool> in(256,false);
    for(int b : bs) in[b]=true;
    for(int b=0;b<256;++b) if(in[b]) b2u[b]=(unsigned)b;
    int m=0;
    for(int b=0;b<256;++b) if(!in[b]) b2u[b]=(unsigned)(256+m++);
  }
  // index vocab by exact byte string
  std::map<std::string,int32_t> bybytes;
  for(int64_t i=0;i<n;++i){
    const char* s=gguf_get_arr_str(g,k,(size_t)i);
    if(s) bybytes[std::string(s)]=(int32_t)i;
  }
  printf("distinct byte strings in vocab = %zu\n", bybytes.size());

  // TEST 1: is the FULL 256-entry byte alphabet present, one token per byte?
  int present=0, absent_first=-1; std::string absent_desc;
  for(int b=0;b<256;++b){
    // encode the unicode codepoint as UTF-8
    unsigned cp=b2u[b]; std::string e;
    if(cp<0x80) e.push_back((char)cp);
    else if(cp<0x800){ e.push_back((char)(0xC0|(cp>>6))); e.push_back((char)(0x80|(cp&0x3F))); }
    else { e.push_back((char)(0xE0|(cp>>12))); e.push_back((char)(0x80|((cp>>6)&0x3F))); e.push_back((char)(0x80|(cp&0x3F))); }
    auto it=bybytes.find(e);
    if(it!=bybytes.end()) ++present;
    else if(absent_first<0){ absent_first=b; absent_desc=hex(e); }
  }
  printf("\nTEST 1  GPT-2 byte alphabet completeness\n");
  printf("  byte values with a matching token : %d / 256\n", present);
  if(present==256) printf("  VERDICT: FULL 256-ENTRY ALPHABET PRESENT\n");
  else printf("  VERDICT: INCOMPLETE, first missing byte %d (expected utf8 %s)\n", absent_first, absent_desc.c_str());

  // TEST 2: the decisive pair. Raw space 0x20 vs U+0120.
  printf("\nTEST 2  space: raw 0x20 or U+0120 (0xC4 0xA0)?\n");
  { std::string raw(" "); std::string u0120="\xC4\xA0";
    auto a=bybytes.find(raw); auto b=bybytes.find(u0120);
    printf("  token whose bytes are 20            : %s\n", a==bybytes.end()?"ABSENT":std::to_string(a->second).c_str());
    printf("  token whose bytes are C4 A0         : %s\n", b==bybytes.end()?"ABSENT":std::to_string(b->second).c_str());
    printf("  fixture says id 220 is \" \" and id 64 is \"a\"\n"); }

  // TEST 3: what are the actual raw bytes of known fixture ids?
  int64_t ids[]={220,198,262,785,576,64,15,1879,314,29,101059,102819,15767,14990,61804,233,145375};
  printf("\nTEST 3  raw bytes of representative fixture ids\n");
  for(unsigned i=0;i<sizeof ids/sizeof*ids;++i){
    int64_t id=ids[i]; if(id>=n) continue;
    std::string s=gguf_get_arr_str(g,k,(size_t)id);
    printf("  %6lld : %-18s utf8=[%s]\n",(long long)id,hex(s).c_str(),s.c_str());
  }

  // TEST 4: byte-decode a few pieces with the INVERSE map and compare to fixture
  printf("\nTEST 4  inverse-map byte-decode vs the llama.cpp fixture pieces\n");
  std::map<unsigned,int> u2b; for(auto&kv:b2u) u2b[kv.second]=kv.first;
  auto decode=[&](const std::string& in)->std::string{
    std::string out; size_t i=0;
    while(i<in.size()){
      unsigned char c=(unsigned char)in[i]; unsigned cp; size_t adv;
      if(c<0x80){ cp=c; adv=1; } else if((c&0xE0)==0xC0){ cp=((c&0x1Fu)<<6)|((unsigned char)in[i+1]&0x3Fu); adv=2; }
      else { cp=((c&0x0Fu)<<12)|(((unsigned char)in[i+1]&0x3Fu)<<6)|((unsigned char)in[i+2]&0x3Fu); adv=3; }
      i+=adv; auto it=u2b.find(cp);
      if(it!=u2b.end()) out.push_back((char)it->second);
      else out.push_back('?');
    } return out; };
  const char* want[]={" ","\n","   ","The"," The","a","0"," world"," {",">"};
  int64_t chk[]={220,198,262,785,576,64,15,1879,314,29};
  for(unsigned i=0;i<10;++i){
    if(chk[i]>=n) continue;
    std::string s=gguf_get_arr_str(g,k,(size_t)chk[i]);
    printf("  id %6lld  stored=[%-14s] byte-decoded=[%s]  fixture expects=[%s]\n",
      (long long)chk[i], hex(s).c_str(), decode(s).c_str(), want[i]);
  }
  // TEST 5: CJK - a 3-byte char must be 3 single-byte tokens
  printf("\nTEST 5  CJK 日本語 -> 6 fixture tokens; stored bytes of each\n");
  int64_t cjk[]={101059,102819,15767,56833,61803,70534};
  for(unsigned i=0;i<6;++i){ if(cjk[i]>=n) continue;
    std::string s=gguf_get_arr_str(g,k,(size_t)cjk[i]);
    printf("  id %6lld stored=[%s] decodes to U+%04X  bytes=%zu\n",(long long)cjk[i],hex(s).c_str(),0u,s.size());
    std::string d=decode(s);
    for(unsigned char ch : d) printf("        byte %02X\n", ch); }
  gguf_free(g); return 0; }
