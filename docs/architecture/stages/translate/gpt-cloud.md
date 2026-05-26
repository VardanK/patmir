# GPT cloud (translate)

- **Repo / version:** OpenAI GPT-4o / 4.x via API
- **Apple Silicon / CUDA:** n/a (cloud)
- **Languages:** broad; Armenian competence high
- **License / gating:** OpenAI ToS + per-token cost
- **Integration surface:** `patmir/eval/candidates/translation/gpt_cloud.py`; reads `OPENAI_API_KEY` from env
- **Status in pipeline:** **disabled** per the [local-first strict](../../decisions/2026-05-24-local-first-strict.md) policy
- **Notes:**
  - Enable only after the local set (NLLB-3.3B, Qwen3-Instruct-MLX, HyGPT, MADLAD) is judged inadequate for a specific stage.
  - Activation requires a new ADR in `decisions/` naming the cloud model, the local-exhaustion evidence, and the revisit trigger for going back to local.
