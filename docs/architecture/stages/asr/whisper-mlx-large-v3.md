# mlx-whisper large-v3 (asr)

- **Repo / version:** `mlx-community/whisper-large-v3-mlx`
- **Parameters / size / runtime memory:** ~1.5 GB weights; ~3-5 GB peak inference
- **Quantization available:** fp16 (active), fp32; the MLX port also offers int8 community builds
- **Apple Silicon (MLX):** ✅ first-class
- **CUDA / CPU:** n/a (this is the MLX port; use `faster-whisper` for CUDA/CPU)
- **Languages:** ~100 (Whisper-large coverage)
- **License / gating:** MIT (Whisper base) + community port
- **Integration surface:** see `patmir/pipeline/stages/asr/whisper_mlx.py`
- **Status in pipeline:** **active** (Phase 1.3 → 2026-05-26 morning; cloud `openrouter_gpt_audio` activated 2026-05-26 morning, reverted 2026-05-26 evening after ear-test showed cloud was not materially better and occasionally truncated the first sentence — see ADR `2026-05-26-cloud-asr-via-openrouter.md`).
- **Last evaluated:** 2026-05-26 evening (masha-2-5min full pipeline run after revert).
- **Our experience:** Solid quality on EN and RU sources. The Phase 2 gap-filler + hallucination blocklist together recovered 11 dialogue lines in a 70 s laughter-dominated region the first pass missed. 2026-05-26 update: non-lexical filter extended to catch single-token "letter-runs" Whisper produces for long screams (a 200-char `'Аааааа...'` was killing downstream ElevenLabs TTS).
- **Notes:**
  - Word-level timestamps work but require `word_timestamps=true`.
  - Detected source language is captured in `Transcript.language`; the runner uses this to route to the right translator language pair.
  - VAD post-filter (drop words outside `vad_intervals`) is the most important quality lever for cartoon content — Whisper hallucinates heavily on screams/laughs without it.
