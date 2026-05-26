# HyGPT-10b-it (translate)

- **Repo / version:** `Gen2B/HyGPT-10b-it` (July 2025)
- **Parameters / size / runtime memory:** 10B params (Gemma-2-9b base + 10B Armenian tokens); ~20 GB VRAM at fp16
- **Quantization available:** fp16; quantized variants likely available
- **Apple Silicon (MPS):** unverified — 10B at fp16 on a 32 GB Mac is tight
- **CUDA:** ✅ recommended
- **Languages:** Armenian-centric (EN, RU, hy support via prompting); Flores 79.33% reported
- **License / gating:** check the model card; personal local use historically allowed
- **Status in pipeline:** **future-candidate** (registered, not active, gated on GPU workstation)
- **Last evaluated:** not yet — registered post-Phase 0, waiting for GPU re-eval
- **Notes:**
  - Likely the strongest local translator ever for this stack — *if* the GPU is available.
  - Dialect labeling: training corpus is Armenia-centric, so Eastern Armenian output is plausible but must be ear-verified.
  - On a 32 GB Mac, fp16 inference may swap; q4/q5 quantization would be the path.
