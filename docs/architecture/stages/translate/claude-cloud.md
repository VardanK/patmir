# Claude cloud (translate)

- **Repo / version:** Anthropic Claude (Opus 4.x / Sonnet) via API
- **Apple Silicon / CUDA:** n/a (cloud)
- **Languages:** broad; Armenian competence high
- **License / gating:** Anthropic ToS + per-token cost
- **Integration surface:** `patmir/eval/candidates/translation/claude_cloud.py`; reads `ANTHROPIC_API_KEY` from env
- **Status in pipeline:** **disabled** per the [local-first strict](../../decisions/2026-05-24-local-first-strict.md) policy
- **Notes:**
  - Same gate as `gpt_cloud`. Activation requires a new ADR.
  - Also under consideration as the path to OpenRouter integration if multiple frontier models are needed for comparison.
