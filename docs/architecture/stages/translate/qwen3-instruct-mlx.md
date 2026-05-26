# Qwen3-Instruct-MLX (translate)

- **Repo / version:** Qwen3-Instruct via MLX port
- **Parameters / size / runtime memory:** depends on the chosen size (typically 7-14B params, fits 32 GB Mac at fp16 / q8)
- **Quantization available:** q8 (MLX), q4
- **Apple Silicon (MLX):** ✅ native
- **CUDA / CPU:** PyTorch builds available but the MLX variant is what's registered
- **Languages:** broad multilingual; Eastern Armenian quality unverified at the Phase 0 bar
- **License / gating:** Apache 2.0
- **Integration surface:** `patmir/eval/candidates/translation/qwen3_instruct_mlx.py`. System prompt: **"You preserve tone, register, and emotion."** User message template currently inserts source text + target language directive.
- **Status in pipeline:** **registered, not active.** Awaits ear-judging vs NLLB-3.3B on the masha clip.
- **Last evaluated:** Phase 1.x exploratory
- **Our experience:** Reasoning suppression had to be added (commit `c1b02c5`) — Qwen3 models tend to emit chain-of-thought interleaved with the translation unless explicitly suppressed.
- **Notes:**
  - **Emotion-aware translation** lives here. The Phase 2.x emotion-transfer spec adds a per-sentence emotion tag injected into the user message template: `"Translate this {src} sentence into Eastern Armenian, with emotion '{emotion}': {text}"`. Activates automatically when Qwen3 is the chosen translator; NLLB ignores it.
  - **Activation path:** flip `[pipeline].translate = "qwen3_instruct_mlx"` in `patmir.toml`. Listening test required first (recorded in `decisions.md` if the swap is made permanent).

## Revisit trigger

When ear-judged to beat NLLB-3.3B on a representative clip, it becomes the new active translator — at which point the Phase 2.x emotion plumbing automatically becomes load-bearing instead of dormant.
