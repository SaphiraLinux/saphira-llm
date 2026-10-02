/* ------------------------------------------------------------------ *
 * test_dispatch.c -- Step 3 gate.
 *
 * A dispatch layer earns its keep by REFUSING, so most of these tests are about
 * what must NOT execute. The contract under test:
 *
 *     measured architecture evidence
 *             |
 *     are the required execution capabilities present?
 *             |
 *           yes -> exact operation dispatch
 *            no -> explicit refusal
 *
 * There is no fallback of the form "this looks like Qwen". The tests that matter
 * most are therefore the ADVERSARIAL ones: evidence hand-built to look like a
 * family we do support while the layout says otherwise. If any of those reach a
 * dispatchable verdict, the layer has become a name classifier and every claim
 * made with it is void.
 * ------------------------------------------------------------------ */

#include "saphira_llm/dispatch.h"
#include "saphira_llm/gguf.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int g_pass = 0, g_fail = 0;
static void ck(int cond, const char * what, const char * detail) {
    if (cond) {
        printf("    ok   %s\n", what);
        g_pass++;
    } else {
        printf("    FAIL %s%s%s\n", what, detail ? " -- " : "", detail ? detail : "");
        g_fail++;
    }
}

/* Evidence as it would be measured for a uniform dense-attention model. */
static sllm_arch_evidence dense_attention(uint32_t layers) {
    sllm_arch_evidence e;
    memset(&e, 0, sizeof e);
    e.state = SLLM_EVIDENCE_MEASURED;
    e.n_layers = layers;
    e.n_attn_layers = layers;
    e.n_ffn_layers = layers;
    e.n_gated_ffn = layers;
    e.n_dense_ffn = 0;
    e.n_heads = 32;
    e.n_kv_heads = 8;
    e.has_untied_output = true;
    return e;
}

int main_k_dispatch_gate(void) {
    sllm_profile p;
    printf("  Step 3 dispatch gate (execution by measured evidence, never by name)\n");

    /* --- the positive case, so the negatives mean something --- */
    {
        sllm_arch_evidence e = dense_attention(36);
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_EXECUTABLE, "uniform dense attention is EXECUTABLE", e.reason);
        ck(p == SLLM_PROFILE_DENSE_ATTENTION, "and resolves to the dense-attention profile", NULL);
    }

    /* --- state-space only, measured by absence of attention --- */
    {
        sllm_arch_evidence e;
        memset(&e, 0, sizeof e);
        e.state = SLLM_EVIDENCE_MEASURED;
        e.n_layers = 24;
        e.n_ssm_layers = 24;
        e.n_heads = 0;              /* MEASURED zero: this model has no heads */
        e.n_kv_heads = 0;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_EXECUTABLE, "uniform state-space with head_count 0 is EXECUTABLE", e.reason);
        ck(p == SLLM_PROFILE_STATE_SPACE_ONLY, "and resolves to the state-space profile", NULL);
    }

    /* --- MoE is refused, not approximated by a dense graph --- */
    {
        sllm_arch_evidence e = dense_attention(24);
        e.n_moe_layers = 24;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "MoE is REFUSED", e.reason);
        ck(p == SLLM_PROFILE_NONE, "a refusal sets no profile at all", NULL);
        ck(strstr(e.reason, "dense graph in its place") != NULL,
           "the refusal says what it declined to substitute", e.reason);
    }

    /* --- heterogeneous bodies: the Nemotron-H shape --- */
    {
        sllm_arch_evidence e = dense_attention(52);
        e.n_attn_layers = 4;       /* attention in only 4 of 52 */
        e.n_ssm_layers  = 24;      /* state space in 24 of 52 */
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "heterogeneous bodies are REFUSED", e.reason);
        ck(p == SLLM_PROFILE_HETEROGENEOUS, "and are identified as heterogeneous", NULL);
        ck(strstr(e.reason, "whichever body happens to be first") != NULL,
           "the refusal declines to pick a body arbitrarily", e.reason);
    }

    /* --- PARTIAL attention: the Granite-4.0-H shape. Every layer has a body,
     *     but not the same body, so it is NOT a dense profile. --- */
    {
        sllm_arch_evidence e = dense_attention(40);
        e.n_attn_layers = 4;       /* 4 of 40 */
        e.n_ssm_layers  = 36;      /* 36 of 40 */
        e.n_gated_ffn = 40;
        e.n_dense_ffn = 0;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "attention in only some layers is REFUSED", e.reason);
    }

    /* --- THE ADVERSARIAL CASES: evidence shaped to trick a name classifier --- */

    /* 1. Partial attention dressed up as a dense model. A layer-global dispatcher
     *    that only asked "is there any attention?" would say yes and run. */
    {
        sllm_arch_evidence e = dense_attention(40);
        e.n_attn_layers = 39;      /* one layer short */
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "attention missing from ONE layer still REFUSED", e.reason);
    }

    /* 2. Attention everywhere but a feed-forward in only some layers. Dispatching
     *    an FFN onto layers that lack one is the uniform-layer assumption. */
    {
        sllm_arch_evidence e = dense_attention(32);
        e.n_ffn_layers = 30;
        e.n_gated_ffn = 30;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "FFN missing from some layers is REFUSED", e.reason);
    }

    /* 3. Mixed gating. Two different graphs in one model; picking either by
     *    majority would be a guess dressed as a decision. */
    {
        sllm_arch_evidence e = dense_attention(36);
        e.n_gated_ffn = 20;
        e.n_dense_ffn = 16;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "mixed gated/ungated FFN is REFUSED", e.reason);
        ck(strstr(e.reason, "two graphs, not one") != NULL,
           "the refusal explains the mixture rather than picking one", e.reason);
    }

    /* 4. Unresolved evidence is NOT executable. This is the state that must never
     *    be allowed to fall through to a nearby profile. */
    {
        sllm_arch_evidence e;
        memset(&e, 0, sizeof e);
        e.state = SLLM_EVIDENCE_ABSENT;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_UNRESOLVED, "absent evidence is UNRESOLVED, not executable", e.reason);
        ck(p == SLLM_PROFILE_NONE, "unresolved evidence sets no profile", NULL);
    }

    /* 5. Zero layers measured is UNRESOLVED, not an empty-but-valid model. */
    {
        sllm_arch_evidence e = dense_attention(0);
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_UNRESOLVED, "zero measured layers is UNRESOLVED", e.reason);
    }

    /* 6. Neither attention nor state space anywhere: matches nothing. */
    {
        sllm_arch_evidence e;
        memset(&e, 0, sizeof e);
        e.state = SLLM_EVIDENCE_MEASURED;
        e.n_layers = 16;
        e.n_ffn_layers = 16;
        e.n_gated_ffn = 16;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "no attention and no SSM is REFUSED", e.reason);
    }

    /* 7. Partial state-space coverage is not the state-space profile. */
    {
        sllm_arch_evidence e;
        memset(&e, 0, sizeof e);
        e.state = SLLM_EVIDENCE_MEASURED;
        e.n_layers = 24;
        e.n_ssm_layers = 20;
        const sllm_dispatch_result r = sllm_dispatch_resolve(&e, &p);
        ck(r == SLLM_DISPATCH_REFUSED, "partial state-space coverage is REFUSED", e.reason);
    }

    /* 8. EVERY refusal must EXPLAIN itself. A refusal that cannot say why is
     *    indistinguishable from a bug, and an unexplained refusal is one people
     *    route around. */
    {
        const sllm_dispatch_result rs[] = {
            SLLM_DISPATCH_REFUSED, SLLM_DISPATCH_UNRESOLVED
        };
        int explained = 1;
        for (size_t i = 0; i < sizeof rs / sizeof *rs; ++i) {
            if (sllm_dispatch_result_name(rs[i])[0] == '\0') { explained = 0; }
        }
        sllm_arch_evidence e = dense_attention(24);
        e.n_moe_layers = 24;
        sllm_dispatch_resolve(&e, &p);
        if (e.reason[0] == '\0') { explained = 0; }
        ck(explained, "refusals and unresolved verdicts both carry a reason", e.reason);
    }

    /* --- against the REAL artefact: measure, do not assume --- */
    {
        const char * path = "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf";
        if (access(path, R_OK) == 0) {
            sllm_gguf g;
            char err[256];
            if (sllm_gguf_open(path, &g, err, sizeof err) == SLLM_OK) {
                sllm_arch_evidence e;
                const sllm_dispatch_result r =
                    sllm_dispatch_measure(&g, "qwen3", &e);
                printf("    qwen3-8b: %u layers, attn %u, ssm %u, ffn %u (gated %u, dense %u), "
                       "moe %u, heads %u/%u -> %s\n",
                       e.n_layers, e.n_attn_layers, e.n_ssm_layers, e.n_ffn_layers,
                       e.n_gated_ffn, e.n_dense_ffn, e.n_moe_layers,
                       e.n_heads, e.n_kv_heads, sllm_dispatch_result_name(r));
                ck(e.n_layers == 36, "qwen3-8b: 36 layers measured from metadata", e.reason);
                ck(e.n_attn_layers == 36, "qwen3-8b: attention in all 36 layers", e.reason);
                ck(e.n_ssm_layers == 0, "qwen3-8b: no state-space recurrence", e.reason);
                ck(e.n_moe_layers == 0, "qwen3-8b: no router", e.reason);
                ck(r == SLLM_DISPATCH_EXECUTABLE, "qwen3-8b is EXECUTABLE", e.reason);
                sllm_gguf_close(&g);
            } else {
                printf("    skipped: cannot open %s (%s)\n", path, err);
            }
        } else {
            printf("    skipped: %s not available\n", path);
        }
    }

    /* The label is recorded but must not matter. Measuring the same model twice
     * with different labels must give identical verdicts, because the label is not
     * consulted. This is the single assertion that distinguishes a dispatcher from
     * a name classifier. */
    {
        const char * path = "/var/lib/spoon/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf";
        if (access(path, R_OK) == 0) {
            sllm_gguf g;
            char err[256];
            if (sllm_gguf_open(path, &g, err, sizeof err) == SLLM_OK) {
                sllm_arch_evidence a, b;
                const sllm_dispatch_result ra =
                    sllm_dispatch_measure(&g, "qwen3", &a);
                const sllm_dispatch_result rb =
                    sllm_dispatch_measure(&g, "llama", &b);   /* deliberately wrong */
                ck(ra == rb, "a deliberately WRONG label changes nothing", b.reason);
                ck(a.n_layers == b.n_layers && a.n_attn_layers == b.n_attn_layers &&
                   a.n_ssm_layers == b.n_ssm_layers && a.n_ffn_layers == b.n_ffn_layers &&
                   a.n_gated_ffn == b.n_gated_ffn && a.n_moe_layers == b.n_moe_layers &&
                   a.n_heads == b.n_heads && a.n_kv_heads == b.n_kv_heads,
                   "every measured quantity is identical under both labels", NULL);
                ck(strcmp(a.reason, b.reason) == 0, "and so is the verdict text", NULL);
                sllm_gguf_close(&g);
            }
        }
    }

    printf("  Step 3 dispatch gate: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}