# htdemucs (separate)

- **Repo / version:** `demucs` PyPI package, `htdemucs_ft` model
- **Parameters / size / runtime memory:** ~80 MB weights; runs comfortably on MPS or CPU
- **Quantization available:** none used (fp32)
- **Apple Silicon / CUDA / CPU:** all (`device = "auto"`)
- **Languages:** n/a (audio source separation)
- **License / gating:** MIT
- **Integration surface:** see `patmir/pipeline/stages/separate/htdemucs.py`. Combines Demucs separation with Silero VAD masking + cosine fades to produce `speech_vocals` and `non_speech_vocals`.
- **Status in pipeline:** **active**
- **Last evaluated:** 2026-05-26 (Phase 2)
- **Our experience:** Reliable. The VAD masking pattern (full vocals for ASR + diarize; speech-only for ref-clip slicing) is what makes Phase 2 multi-voice work cleanly.
- **Notes:**
  - Diarize consumes the **full vocals** stem (soft onsets matter for turn-boundary placement); ref_picker slices from the **speech_vocals** stem (cleaner).
  - The 20 ms cosine fades at VAD boundaries mean ref clips must be trimmed ≥100 ms inward to avoid hitting fade edges.
  - `non_speech_vocals` is mixed back into the accompaniment bed at mux time so atmospheric vocals (screams, laughs) survive into the final dub.
