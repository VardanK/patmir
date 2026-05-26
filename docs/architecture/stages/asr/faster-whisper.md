# faster-whisper (asr)

- **Repo / version:** `faster-whisper` PyPI package, model `large-v3` or `large-v3-turbo`
- **Parameters / size / runtime memory:** `large-v3` ~3 GB fp16, `large-v3-turbo` ~1.5 GB fp16; int8 halves
- **Quantization available:** int8 (CT2 quantization, default in our config)
- **Apple Silicon (MLX/MPS):** ❌ no MPS support; runs on CPU only on Mac
- **CUDA / CPU:** ✅ both (CUDA preferred when available)
- **Languages:** ~100 (Whisper coverage)
- **License / gating:** MIT
- **Integration surface:** `patmir/pipeline/stages/asr/faster_whisper.py`
- **Status in pipeline:** **registered, not active.** Was the Phase 1 default; superseded by `whisper-mlx-large-v3` in Phase 1.3 because MLX-native runs faster on the author's Mac.
- **Last evaluated:** 2026-05-25 (Phase 1.3)
- **Our experience:** Quality identical to OpenAI Whisper-large-v3 (same weights). CPU-only on Mac was the speed bottleneck.
- **Notes:**
  - Kept registered as a CUDA fallback for when the GPU workstation arrives — `large-v3` with int8 + `compute_type = "float16"` on CUDA is the right setup for high-throughput batch runs.
