# Coqui XTTS-v2 (tts)

- **Repo / version:** Coqui XTTS-v2 (`coqui/XTTS-v2`)
- **Parameters / size / runtime memory:** ~1.8 GB; multi-GB at inference
- **Quantization available:** fp16
- **Apple Silicon (MLX/MPS):** PyTorch only; MPS workable
- **CUDA / CPU:** both
- **Languages:** **17 supported languages — en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi.** **Armenian is NOT among them.**
- **License / gating:** Coqui's CPML
- **Status in pipeline:** **rejected** (does not support Armenian — corrected from earlier tech-stack notes that mistakenly listed Armenian as supported)
- **Last evaluated:** 2026-05-24 (Phase 1 model scan; corrected error from earlier `tech-stack.md` claim)
- **Our experience:** No real-run experience — rejected after re-reading the actual model card.
- **Notes:**
  - Earlier `tech-stack.md` and `patmir.toml` claimed XTTS-v2 supported Armenian. **This was wrong.** The current model card lists 17 languages and Armenian is not one of them. Corrected in Phase 1's plan.
  - Entry kept in `patmir.toml` with `enabled = false` and an inline comment so future scans don't re-add it.
  - Path to use XTTS-v2 for EA: build an EA fine-tune (out of scope).
