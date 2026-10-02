/* ------------------------------------------------------------------ *
 * dispatch.c -- architecture dispatch by MEASURED evidence.
 *
 * The whole design is one refusal: there is no code path here that reaches a
 * profile by comparing a model name. If the tensor layout does not match a
 * profile we have, the answer is a refusal that says which measured quantity
 * failed to match, because a refusal that cannot explain itself is
 * indistinguishable from a crash.
 * ------------------------------------------------------------------ */

#include <saphira_llm/dispatch.h>
#include <saphira_llm/gguf.h>

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void say(char * dst, size_t cap, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dst, cap, fmt, ap);
    va_end(ap);
}

const char * sllm_dispatch_result_name(sllm_dispatch_result r) {
    switch (r) {
        case SLLM_DISPATCH_EXECUTABLE: return "EXECUTABLE";
        case SLLM_DISPATCH_REFUSED:     return "REFUSED";
        case SLLM_DISPATCH_UNRESOLVED:  return "UNRESOLVED";
        default:                        return "INVALID";
    }
}

/* Count layers carrying a given tensor role, by probing EXACT tensor names.
 *
 * The roles this runtime knows about are a short, explicit list, so they are named
 * rather than pattern-matched. An explicit list has a second virtue: a role we do
 * not name cannot be silently half-matched, and adding one is a visible edit rather
 * than a change in how an existing string is interpreted.
 *
 * Each layer is probed by its OWN name, so a role present in one layer never
 * licenses anything in another. */
static uint32_t count_layers_with(const sllm_gguf * g, uint32_t n_layers,
                                  const char * suffix) {
    uint32_t n = 0;
    char buf[192];
    for (uint32_t l = 0; l < n_layers; ++l) {
        snprintf(buf, sizeof buf, "blk.%u.%s", l, suffix);
        if (sllm_gguf_find_tensor(g, buf) != NULL) { ++n; }
    }
    return n;
}

static uint32_t n_layers_from_metadata(const sllm_gguf * g, const char ** prefix_out) {
    const char * arch = NULL;
    char key[192];
    uint32_t bc = 0;

    if (prefix_out != NULL) { *prefix_out = NULL; }
    if (sllm_gguf_kv_str(g, "general.architecture", &arch) != SLLM_OK || arch == NULL) {
        return 0;
    }
    if (prefix_out != NULL) { *prefix_out = arch; }
    snprintf(key, sizeof key, "%s.block_count", arch);
    if (sllm_gguf_kv_u32(g, key, &bc) != SLLM_OK) { return 0; }
    /* A layer count this large is not a real model; refusing to enumerate it stops
     * a corrupt header from turning into an enormous allocation and a long hang,
     * and reports a fact instead of a symptom. */
    if (bc == 0 || bc > 4096) { return 0; }
    return bc;
}

sllm_dispatch_result sllm_dispatch_measure(const void * gguf_v,
                                           const char * architecture_label,
                                           sllm_arch_evidence * out) {
    if (out == NULL) { return SLLM_DISPATCH_UNRESOLVED; }
    memset(out, 0, sizeof *out);

    const sllm_gguf * g = (const sllm_gguf *) gguf_v;
    if (g == NULL) {
        say(out->reason, sizeof out->reason,
            "no model to measure; evidence is absent, not zero");
        out->state = SLLM_EVIDENCE_ABSENT;
        return SLLM_DISPATCH_UNRESOLVED;
    }

    (void) architecture_label;   /* recorded by callers; NEVER selects a graph */

    const char * arch = NULL;
    out->n_layers = n_layers_from_metadata(g, &arch);
    if (out->n_layers == 0) {
        out->state = SLLM_EVIDENCE_ABSENT;
        say(out->reason, sizeof out->reason,
            "block_count could not be read from metadata, so the layer set is unknown; "
            "an unknown layer count is not a zero layer count");
        return SLLM_DISPATCH_UNRESOLVED;
    }

    out->n_attn_layers = count_layers_with(g, out->n_layers, "attn_q.weight");
    out->n_ssm_layers  = count_layers_with(g, out->n_layers, "ssm_in.weight");
    out->n_ffn_layers  = count_layers_with(g, out->n_layers, "ffn_down.weight");
    out->n_gated_ffn   = count_layers_with(g, out->n_layers, "ffn_gate.weight");
    out->n_moe_layers  = count_layers_with(g, out->n_layers, "ffn_gate_inp.weight");
    out->n_dense_ffn   = (out->n_ffn_layers > out->n_gated_ffn)
                       ? (out->n_ffn_layers - out->n_gated_ffn) : 0;

    out->has_untied_output  = (sllm_gguf_find_tensor(g, "output.weight") != NULL);
    out->has_rope_freqs_key =
        (sllm_gguf_find_tensor(g, "rope_freqs.weight") != NULL);

    /* Heads are read from METADATA with a type check. A missing key is not a zero
     * count: head_count == 0 means "measured to have no heads", which is a very
     * different statement from "we could not read it". Dividing by, or branching
     * on, an unread key is how a missing value becomes a fabricated one. */
    {
        char key[192];
        uint32_t h = 0, kh = 0, w = 0;
        snprintf(key, sizeof key, "%s.attention.head_count", arch);
        if (sllm_gguf_kv_u32(g, key, &h) == SLLM_OK) { out->n_heads = h; }
        snprintf(key, sizeof key, "%s.attention.head_count_kv", arch);
        if (sllm_gguf_kv_u32(g, key, &kh) == SLLM_OK) { out->n_kv_heads = kh; }
        snprintf(key, sizeof key, "%s.embedding_length", arch);
        if (sllm_gguf_kv_u32(g, key, &w) == SLLM_OK) { out->width = w; }
        /* A missing head_count is NORMAL for a model with no attention, and is
         * recorded as zero without being called a fault. What would be a fault is
         * treating that zero as a head count and dividing by it. */
    }

    if (out->state != SLLM_EVIDENCE_PARTIAL) { out->state = SLLM_EVIDENCE_MEASURED; }
    return sllm_dispatch_resolve(out, NULL);
}

sllm_dispatch_result sllm_dispatch_resolve(sllm_arch_evidence * ev,
                                           sllm_profile * profile_out) {
    if (profile_out != NULL) { *profile_out = SLLM_PROFILE_NONE; }
    if (ev == NULL) { return SLLM_DISPATCH_UNRESOLVED; }

    /* The gate comes FIRST and it is a capability gate, not a name gate. If we
     * could not measure enough to decide, the answer is UNRESOLVED, which is not
     * executable. Falling through to a "close enough" profile here would be the
     * exact failure this contract exists to prevent. */
    if (ev->state == SLLM_EVIDENCE_ABSENT) {
        say(ev->reason, sizeof ev->reason, "%s", ev->reason[0] ? ev->reason
            : "evidence absent: cannot decide, so not executable");
        return SLLM_DISPATCH_UNRESOLVED;
    }
    if (ev->n_layers == 0) {
        say(ev->reason, sizeof ev->reason,
            "no layers measured: the tensor layout could not be read at all");
        ev->state = SLLM_EVIDENCE_ABSENT;
        return SLLM_DISPATCH_UNRESOLVED;
    }

    const bool has_attn = ev->n_attn_layers > 0;
    const bool has_ssm  = ev->n_ssm_layers > 0;
    const bool has_moe  = ev->n_moe_layers > 0;

    /* MoE is NOT implemented. Saying so is the whole point of a dispatch layer
     * that can refuse: an expert router is a different computation, and running a
     * dense graph in its place would produce a plausible model that is wrong. */
    if (has_moe) {
        say(ev->reason, sizeof ev->reason,
            "MoE routing measured in %u of %u layers, and no expert-routing execution "
            "path exists; refusing rather than running a dense graph in its place",
            ev->n_moe_layers, ev->n_layers);
        return SLLM_DISPATCH_REFUSED;
    }

    if (has_attn && has_ssm) {
        say(ev->reason, sizeof ev->reason,
            "layer bodies differ: %u of %u layers carry attention and %u carry a "
            "state-space recurrence. No heterogeneous execution path exists yet; "
            "refusing rather than dispatching whichever body happens to be first",
            ev->n_attn_layers, ev->n_layers, ev->n_ssm_layers);
        if (profile_out != NULL) { *profile_out = SLLM_PROFILE_HETEROGENEOUS; }
        return SLLM_DISPATCH_REFUSED;
    }

    if (has_ssm) {
        if (ev->n_ssm_layers != ev->n_layers) {
            say(ev->reason, sizeof ev->reason,
                "state-space recurrence measured in %u of %u layers: partial coverage "
                "is not a profile",
                ev->n_ssm_layers, ev->n_layers);
            return SLLM_DISPATCH_REFUSED;
        }
        if (profile_out != NULL) { *profile_out = SLLM_PROFILE_STATE_SPACE_ONLY; }
        return SLLM_DISPATCH_EXECUTABLE;
    }

    if (has_attn) {
        /* A dense-attention profile requires attention on EVERY layer. Attention on
         * some layers only is a different topology, and admitting it here would be
         * the model-global assumption this project deleted at Nemotron-H. */
        if (ev->n_attn_layers != ev->n_layers) {
            say(ev->reason, sizeof ev->reason,
                "attention measured in %u of %u layers: partial attention coverage is "
                "not a dense-attention profile, and dispatching it as one would assume "
                "every layer has a body only some of them have",
                ev->n_attn_layers, ev->n_layers);
            if (profile_out != NULL) { *profile_out = SLLM_PROFILE_HETEROGENEOUS; }
            return SLLM_DISPATCH_REFUSED;
        }
        /* Every layer has a feed-forward, and its gating is uniform. Mixed gating
         * across layers is a different graph and is refused rather than guessed. */
        if (ev->n_ffn_layers != ev->n_layers) {
            say(ev->reason, sizeof ev->reason,
                "attention on all %u layers but a feed-forward in only %u: refusing "
                "rather than dispatching an FFN onto layers that lack one",
                ev->n_layers, ev->n_ffn_layers);
            return SLLM_DISPATCH_REFUSED;
        }
        if (ev->n_gated_ffn != 0 && ev->n_dense_ffn != 0) {
            say(ev->reason, sizeof ev->reason,
                "feed-forward gating is mixed across layers (%u gated, %u ungated): "
                "that is two graphs, not one, and no rule for choosing between them "
                "exists in evidence",
                ev->n_gated_ffn, ev->n_dense_ffn);
            return SLLM_DISPATCH_REFUSED;
        }
        if (profile_out != NULL) { *profile_out = SLLM_PROFILE_DENSE_ATTENTION; }
        return SLLM_DISPATCH_EXECUTABLE;
    }

    say(ev->reason, sizeof ev->reason,
        "neither attention nor a state-space recurrence was measured in any of %u "
        "layers: the layout matches nothing this runtime implements",
        ev->n_layers);
    return SLLM_DISPATCH_REFUSED;
}