# mlx-whisper large-v3 (asr)

- **Repo / version:** `mlx-community/whisper-large-v3-mlx`
- **Parameters / size / runtime memory:** ~1.5 GB weights; ~3-5 GB peak inference
- **Quantization available:** fp16 (active), fp32; the MLX port also offers int8 community builds
- **Apple Silicon (MLX):** ✅ first-class
- **CUDA / CPU:** n/a (this is the MLX port; use `faster-whisper` for CUDA/CPU)
- **Languages:** ~100 (Whisper-large coverage)
- **License / gating:** MIT (Whisper base) + community port
- **Integration surface:** see `patmir/pipeline/stages/asr/whisper_mlx.py`
- **Status in pipeline:** **active** since Phase 1.3 (commit `9c9f219`)
- **Last evaluated:** 2026-05-26 (Phase 2 verification on masha-2-5min)
- **Our experience:** Solid quality on EN and RU sources. The Phase 2 gap-filler + hallucination blocklist together recovered 11 dialogue lines in a 70 s laughter-dominated region the first pass missed.
- **Notes:**
  - Word-level timestamps work but require `word_timestamps=true`.
  - Detected source language is captured in `Transcript.language`; the runner uses this to route to the right translator language pair.
  - VAD post-filter (drop words outside `vad_intervals`) is the most important quality lever for cartoon content — Whisper hallucinates heavily on screams/laughs without it.
