/* ------------------------------------------------------------------ *
 * dispatch.h -- architecture dispatch by MEASURED evidence.
 *
 * THE CONTRACT, in order:
 *
 *     measured architecture evidence
 *             |
 *     are the required execution capabilities present?
 *             |
 *           yes -> exact operation dispatch
 *            no -> explicit refusal
 *
 * There is deliberately no fallback. In particular there is no path of the form
 * "this looks like Qwen, so run the Qwen graph". The label general.architecture
 * is read for reporting and is NEVER consulted to choose a graph, because a
 * label is metadata and this project has already established, repeatedly and at
 * cost, that a name locates evidence rather than creating it. A model whose
 * tensors do not match a profile is refused, even if its name matches perfectly.
 *
 * Evidence is measured from the TENSOR LAYOUT at the exact layer where it appears.
 * A role present in one layer never licenses an operation in another. Where the
 * evidence cannot decide, the result is SLLM_DISPATCH_UNRESOLVED, which is
 * non-executable: unresolved evidence is not permission to try the nearest thing
 * we do know.
 *
 * Three distinct questions, kept distinct because collapsing them is how a tool
 * becomes confidently wrong:
 *
 *   - the LAYOUT is known and matches a profile        -> executable
 *   - the layout is known but matches nothing we have  -> refused (not executable)
 *   - the layout cannot be determined                   -> unresolved (not executable)
 * ------------------------------------------------------------------ */

#ifndef SAPHIRA_LLM_DISPATCH_H
#define SAPHIRA_LLM_DISPATCH_H

#include <saphira_llm/status.h>

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Topology profiles, named by WHAT IS IN THE LAYOUT rather than by who ships it.
 * The name is documentation for a human; selection is by the measured struct
 * below, never by comparing these names. */
typedef enum {
    SLLM_PROFILE_NONE = 0,
    /* every layer carries Q/K/V projections and an ungated-or-gated FFN */
    SLLM_PROFILE_DENSE_ATTENTION,
    /* every layer carries a state-space recurrence and NO attention */
    SLLM_PROFILE_STATE_SPACE_ONLY,
    /* layer bodies differ: some layers attention, some state-space, some neither */
    SLLM_PROFILE_HETEROGENEOUS
} sllm_profile;

/* How the evidence was obtained. Kept alongside the verdict so a report can say
 * how it knows, rather than only what it concluded. */
typedef enum {
    SLLM_EVIDENCE_MEASURED = 0,
    SLLM_EVIDENCE_PARTIAL,     /* some questions could not be decided */
    SLLM_EVIDENCE_ABSENT       /* the evidence needed to decide was not present */
} sllm_evidence_state;

typedef struct {
    /* MEASURED, from tensor names and shapes. */
    uint32_t n_layers;
    uint32_t width;
    uint32_t n_attn_layers;   /* layers carrying a Q projection */
    uint32_t n_ssm_layers;    /* layers carrying a state-space recurrence */
    uint32_t n_ffn_layers;    /* layers carrying a feed-forward */
    uint32_t n_gated_ffn;     /* of those, how many carry a gate tensor */
    uint32_t n_dense_ffn;     /* of those, how many do not */
    uint32_t n_moe_layers;    /* layers carrying a router tensor */
    uint32_t n_heads;
    uint32_t n_kv_heads;
    bool     has_untied_output;   /* a separate output projection exists */
    bool     has_rope_freqs_key;  /* a precomputed rope frequency key exists */

    sllm_evidence_state state;
    /* Populated when the verdict is a refusal, so the refusal EXPLAINED ITSELF.
     * A refusal that cannot say why is indistinguishable from a bug. */
    char     reason[320];   /* large enough that a refusal is never truncated mid-sentence */
} sllm_arch_evidence;

typedef enum {
    SLLM_DISPATCH_EXECUTABLE = 0,
    SLLM_DISPATCH_REFUSED,       /* evidence decided, and matches nothing we have */
    SLLM_DISPATCH_UNRESOLVED    /* evidence could not decide: NOT executable */
} sllm_dispatch_result;

/* Measure architecture evidence from an OPENED, MAPPED model. Reads tensor names
 * and shapes only. `architecture_label` is recorded for reporting and is NOT used
 * to choose anything; pass NULL if absent. */
sllm_dispatch_result sllm_dispatch_measure(const void * gguf,
                                           const char * architecture_label,
                                           sllm_arch_evidence * out);

/* Decide whether the measured evidence may be executed. Separated from measuring
 * so a caller (and a test) can supply evidence by hand and prove that a
 * mislabelled or incomplete model is refused rather than guessed at. */
sllm_dispatch_result sllm_dispatch_resolve(sllm_arch_evidence * ev,
                                           sllm_profile * profile_out);

/* Human-readable, for reports and test failure messages. Never NULL. */
const char * sllm_dispatch_result_name(sllm_dispatch_result r);

#ifdef __cplusplus
}
#endif

#endif /* SAPHIRA_LLM_DISPATCH_H */