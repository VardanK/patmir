# pyannote 3.1 (diarize)

- **Repo / version:** `pyannote/speaker-diarization-3.1` (HF, license-gated)
- **Parameters / size / runtime memory:** model bundle ~30 MB on disk; ~1-2 GB peak during inference
- **Quantization available:** none
- **Apple Silicon (MLX/MPS):** partial — some ops fall back to CPU; full MPS coverage not guaranteed
- **CUDA / CPU:** both fully supported
- **Languages:** n/a (language-agnostic diarization)
- **License / gating:** HF-gated; requires accepting licenses on `pyannote/segmentation-3.0`, `pyannote/speaker-diarization-3.1`, and `pyannote/speaker-diarization-community-1` + an HF token
- **Integration surface:** `PyannoteDiarize.diarize(separated, ctx) -> Diarization` in `patmir/pipeline/stages/diarize/pyannote.py`. Runs on `SeparatedAudio.vocals`. Post-filters turns against `vad_intervals`. Calls `Annotation.get_overlap()` for overlap regions.
- **Status in pipeline:** **active**
- **Last evaluated:** 2026-05-26 (Phase 2)
- **Our experience:** Works well on multi-speaker cartoon dialogue. 5 speakers detected on `samples/masha-2-5min.mkv`. Some ghost speakers in laughter-dominated regions, filtered by VAD overlap check.
- **Notes:**
  - pyannote 3.1 returns a `DiarizeOutput` object with `.speaker_diarization` and `.exclusive_speaker_diarization` Annotations. We use `.speaker_diarization` and fall back to treating the result as an Annotation directly for older builds. See `patmir/pipeline/stages/diarize/pyannote.py:130-133`.
  - Always runs on the full `vocals.wav` (not VAD-masked) — pyannote needs soft onsets for accurate turn boundary placement.
  - Ref-clip slicing is NOT done in this stage — it's a runner-level step that fires after `emotion` runs, because the emotion bin of each turn is needed to pick the right ref per `(speaker, emotion)`.

## ADR

[2026-05-26 — pyannote 3.1 for diarization](../../decisions/2026-05-26-pyannote-3-1-for-diarization.md)
