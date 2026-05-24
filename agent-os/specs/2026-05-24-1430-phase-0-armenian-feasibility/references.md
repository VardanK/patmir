# References for Phase 0 — Armenian Feasibility Gate

## In-repo references

The repository is greenfield — no existing Python code. The only references that shaped this plan are the product docs.

### `agent-os/product/roadmap.md` — Phase 0 section

- **Relevance:** Defines Phase 0 verbatim. Sets the eval set composition (cartoon dialogue, simple narration, idioms, proper nouns), the two language pairs (EN→HY, RU→HY), the scoring approach (fluent Armenian speaker + target child), and the exit criteria (one-pager + audio samples + explicit go/no-go).
- **Key patterns to follow:** ~20 utterances, 2–3 candidates per stage, a cloud baseline for translation ceiling, human listening eval as the source of truth for TTS.

### `agent-os/product/tech-stack.md` — Stages 3 & 4, plus orchestration

- **Relevance:** Names specific candidate models per stage and explicitly calls translation and Armenian TTS the highest-risk stages. Specifies Python 3.11+, `uv` as preferred dep manager, and the principle that *"per-stage config loaded from a single `patmir.toml` so model swaps are config changes, not code changes"* — which the eval harness honors from day one even though the pipeline doesn't exist yet.
- **Key patterns to borrow:** TOML-driven candidate selection; protocol-style stage interfaces (here applied at a smaller scale to `Translator` / `TTS` protocols inside the harness); preference for `transformers`-based loading for HuggingFace models.

### `agent-os/product/mission.md`

- **Relevance:** Confirms the "would a child enjoy this?" framing that the TTS rubric encodes. Also the source of the **local-first-strict** stance applied throughout Phase 0: cloud is opt-in per stage, never default-enabled, even for research/ceiling-reference. This stance is stricter than the original roadmap wording; the roadmap was updated in lockstep to match.

## External references

None consulted yet. Candidate model selection is taken directly from `tech-stack.md`. When implementing Task 4/5, double-check that the named HuggingFace model IDs (`facebook/nllb-200-distilled-600M`, `facebook/mms-tts-hye`) still resolve.
