# Bark (tts)

- **Repo / version:** `suno/bark`
- **Parameters / size / runtime memory:** ~3.5 GB
- **Quantization available:** fp16; small.bark community variants
- **Apple Silicon (MLX/MPS):** PyTorch only
- **CUDA / CPU:** both
- **Languages:** broad multilingual generative TTS; Armenian handling unverified
- **License / gating:** MIT (Suno)
- **Status in pipeline:** **deferred indefinitely**
- **Last evaluated:** 2026-05-24 (Phase 1 — decision: stays deferred to GPU re-eval; eval cost wasn't worth Phase 1 schedule slip)
- **Notes:**
  - Dialect handling is unverified. Bark sometimes produces unintelligible output on lower-resource languages; would need ear-check before any scoring.
  - Likely retired by the time the GPU workstation arrives in favor of newer engines.
