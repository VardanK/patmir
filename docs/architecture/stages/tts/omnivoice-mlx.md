# OmniVoice MLX (tts)

- **Repo / version:** `mlx-community/OmniVoice-bfloat16` (bf16 MLX OmniVoice) — **active since 2026-06-15**, switched from `lightsofapollo/omnivoice-mlx-q8-g64` (q8). Same `omnivoice_mlx` candidate, only `repo_id` changed.
- **Parameters / size / runtime memory:** bf16 ~1.23 GB on disk; q8 g64 ~0.6 GB; fp16 PyTorch ~3 GB
- **Quantization available:** bf16 (active), q8 (prior active — change `repo_id` to revert), fp16 (PyTorch original)
- **Ear-test + speed (hy line, same ref), 2026-06-16:** quality bf16@32 EXCELLENT > q8@32 > bf16@16 (slight nasal) > q8@16 (inter-word pauses). Speed (mlx-audio, RTF @8/16/32): bf16 0.37/0.46/0.89, q8 0.46/0.67/1.20; load bf16 1.7s vs q8 6.1s. bf16 wins on BOTH quality and speed. (ailuntx runtime, not adopted: fp16 0.35/0.97; q4 —/1.07/1.26.)
- **Rejected siblings:** `mlx-community/OmniVoice-8bit` (no quant config in config.json → won't load on mlx-audio; its card targets the `ailuntx/OmniVoice-MLX` runtime and disclaims reliable speaker-encoding) and `mlx-community/OmniVoice-bf16` (audio-tokenizer not wired by mlx-audio's load path → `HiggsAudioTokenizer required` on cloning). The `ailuntx/OmniVoice-MLX` runtime works + clones but is no faster (same OmniVoice diffusion), so not adopted.
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
