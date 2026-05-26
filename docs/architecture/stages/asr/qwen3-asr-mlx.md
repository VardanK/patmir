# qwen3-asr-mlx (asr)

- **Repo / version:** experimental Qwen3-based ASR via MLX
- **Status in pipeline:** **registered, not active.** Alternative ASR architecture (LLM-style decoder). Kept as a candidate; Whisper-MLX is the proven baseline.
- **Last evaluated:** Phase 1.x exploratory
- **Our experience:** Qwen-style reasoning suppression had to be added (commit `c1b02c5`) because the model would emit chain-of-thought interleaved with transcription on long inputs.
- **Notes:**
  - Future-candidate. The bet is that LLM-architecture ASR might handle code-switched or accent-heavy speech better than encoder-decoder Whisper.
  - Not part of the Phase 2 verification path.
