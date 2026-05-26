# OmniVoice MLX supersedes Piper as the active TTS

**Date:** 2026-05-25 (Phase 1.3 — commit `9c9f219`)
**Status:** locked as active TTS

## Context

Piper `hy_AM-gor-medium` cleared the Phase 0 quality floor (20/20 watchable) but has hard limits:

- One fixed speaker (no character variety).
- No prosody / emotion / style control.
- Uniform 3/3/2.9 scores — acceptable but not delightful.

Phase 2's multi-voice goal needed *something* that could give each character a distinct voice. The intent was originally a "preset pool" of 2-4 fixed Eastern-Armenian voices (per `tech-stack.md` §"Stage 4 voice variety"), with cloning treated as a stretch goal.

OmniVoice MLX (an MLX port of the OmniVoice TTS family) turned out to do zero-shot voice cloning from a 3-10 s reference clip — and the EA output, judged by ear on the masha clip, was both cleaner than Piper *and* automatically gave each character a distinct voice. The "preset pool first" plan was overridden by this happy accident.

## Decision

`omnivoice_mlx_q8` (8-bit quantized OmniVoice MLX) is the active TTS, called per-sentence with the appropriate `(speaker, [emotion])` reference clip. Piper stays as the fallback when no ref is available for a speaker.

API surface used:

```python
synthesize(text: str, *, ref_audio: Path|str|None = None, ref_text: str|None = None,
           guidance_scale: float|None = None) -> bytes
```

(See `patmir/eval/candidates/tts/omnivoice_mlx.py` for the implementation.)

## Consequences

- Phase 2's "preset pool first, cloning second" plan was scratched — OmniVoice cloning gives multi-voice for free.
- Per-call cloning means the ref-clip-selection logic in `diarize/ref_picker.py` is now the single most important determinant of voice quality.
- OmniVoice has no explicit emotion knob — see [emotion-routed-through-ref-clip-binning](2026-05-26-emotion-routed-through-ref-clip-binning.md) for how emotion is layered on top.
- Quantization (`q8`) fits the 32 GB Mac comfortably alongside the other models in the pipeline.

## Revisit trigger

- A meaningfully-better EA-capable local TTS appears. The current rejection survey is in `docs/architecture/stages/tts/f5-tts-mlx.md` and `indextts-2.md`.
- OmniVoice regresses or has license issues.
- Cloud TTS (ElevenLabs Multilingual v2) is approved under the local-first exception policy (would require its own ADR).

## Source

- Commit `9c9f219` (Phase 1.3)
- `agent-os/specs/2026-05-26-0215-multi-voice-diarization-cloning/plan.md` (Phase 2 plan which baked OmniVoice's per-call cloning in)
- `patmir/eval/candidates/tts/omnivoice_mlx.py`
