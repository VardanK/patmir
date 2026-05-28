# OmniVoice MLX (tts)

- **Repo / version:** `lightsofapollo/omnivoice-mlx-q8-g64` (8-bit MLX quantization of OmniVoice)
- **Parameters / size / runtime memory:** ~1 GB on disk (q8 + g64 group size); fp16 PyTorch variant would be ~3 GB
- **Quantization available:** q8 (active), fp16 (PyTorch original)
- **Apple Silicon (MLX):** ✅ native (this is the MLX port)
- **CUDA / CPU:** PyTorch original works; MLX port is Apple-only
- **Languages:** multilingual zero-shot cloning. **Eastern Armenian verified** — `hy_AM` ref clip in → `hy_AM` synthesized speech out, ear-judged on the masha clip.
- **License / gating:** see the `lightsofapollo` repo; personal local use OK
- **Integration surface:** see `patmir/eval/candidates/tts/omnivoice_mlx.py`. Key API:
  ```python
  synthesize(
      text: str,
      *,
      ref_audio: Path | str | None = None,   # per-call speaker ref
      ref_text:  str | None = None,           # per-call ref transcript
      guidance_scale: float | None = None,    # Phase 2.x — per-call emotion-driven strength
  ) -> bytes  # WAV bytes
  ```
- **Status in pipeline:** **active** (re-activated 2026-05-27 PM after ElevenLabs' captcha system blocked cloned voices of real celebrities on YouTube content — see `current.md`)
- **Last evaluated:** 2026-05-27 (re-activation; prior eval 2026-05-26 on `samples/masha-2-5min.mkv`)
- **Our experience:**
  - Produces watchable EA output via zero-shot ref-clip cloning. Phase 2 verification: 5 speakers, 3 cloned + 2 Piper-fallback, 36 dubbed sentences. Each character has a distinguishable Armenian voice.
  - q8 quant is ~25% faster than fp16 with no audible quality loss in ear-tests.
  - The reference clip's expressive content bleeds heavily into the synthesized output — which is *why* Phase 2.x routes emotion through per-`(speaker, emotion)` ref-clip selection rather than chasing engine swaps.
- **Notes:**
  - No explicit emotion / prosody / style knobs in the API. Emotion transfer in EA is bounded by what's audible in the source character's range. See the ADR for the accepted-constraint statement.
  - The underlying `mlx_audio.tts.utils.load_model` is what does the heavy lifting; we just thin-wrap it.
  - `model.generate(...)` is a generator yielding `GenerationResult` chunks — we concatenate them into a single mono PCM-16 buffer at the sample rate of the first chunk.

## ADRs

- [2026-05-25 — OmniVoice supersedes Piper](../../decisions/2026-05-25-omnivoice-supersedes-piper.md)
- [2026-05-26 — OmniVoice locked for Eastern Armenian](../../decisions/2026-05-26-omnivoice-locked-for-ea.md)
- [2026-05-26 — Emotion routed through ref-clip binning](../../decisions/2026-05-26-emotion-routed-through-ref-clip-binning.md)
