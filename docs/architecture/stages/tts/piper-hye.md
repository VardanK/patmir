# Piper hy_AM-gor-medium (tts)

- **Repo / version:** `davit312/piper-TTS-Armenian`, voice `hy_AM-gor-medium`
- **Parameters / size / runtime memory:** ~63 MB ONNX, CPU only, trivially fast (~0.1 s/utterance)
- **Quantization available:** ONNX (no further quantization needed)
- **Apple Silicon / CUDA / CPU:** all (ONNX runtime)
- **Languages:** **Eastern Armenian only** (community-trained on EA data)
- **License / gating:** Piper's standard license, community-trained voice
- **Integration surface:** `patmir/eval/candidates/tts/piper_hye.py`
  ```python
  synthesize(text: str) -> bytes  # WAV bytes
  ```
  Note: signature does NOT take `ref_audio` / `ref_text` / `guidance_scale` — Piper has a fixed voice.
- **Status in pipeline:** **fallback** (was Phase 0 / Phase 1 active default; superseded by OmniVoice in Phase 1.3)
- **Last evaluated:** 2026-05-24 (Phase 0)
- **Phase 0 scores** (1-5 rubric, n=20):

  | metric                     | overall    | cartoon  | proper   | narration | idiom    |
  | -------------------------- | ---------- | -------- | -------- | --------- | -------- |
  | intelligibility            | 3.00       | 3.00     | 3.00     | 3.00      | 3.00     |
  | naturalness                | 3.00       | 3.00     | 3.00     | 3.00      | 3.00     |
  | prosody                    | 2.90       | 2.88     | 2.75     | 3.00      | 3.00     |
  | **child_would_watch_hours** | **20/20 Y** | 8/8 Y    | 4/4 Y    | 5/5 Y     | 3/3 Y    |

- **Our experience:**
  - Cleared the "watchable for hours?" bar on all 20/20 utterances. No `maybe` votes. Uniform 3/3/3 scores — consistent but no standout strengths.
  - One fixed voice; no character variety; no emotion control.
- **Notes:**
  - Phase 2.x retains Piper as the fallback when OmniVoice has no `(speaker, emotion)` ref for a sentence (rare; logged in the manifest).
  - The Phase 2.x TTS adapter signature gets uniform across candidates by having Piper accept-and-ignore the new `ref_audio` / `ref_text` / `guidance_scale` kwargs.

## ADRs

- [2026-05-24 — Piper hy_AM-gor-medium as TTS floor](../../decisions/2026-05-24-piper-hye-as-tts-floor.md)
- [2026-05-25 — OmniVoice supersedes Piper](../../decisions/2026-05-25-omnivoice-supersedes-piper.md)
