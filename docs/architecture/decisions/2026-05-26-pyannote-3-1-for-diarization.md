# pyannote 3.1 for diarization

**Date:** 2026-05-26 (Phase 2 — commit `3908e6c`)
**Status:** locked as active diarizer

## Context

Phase 2 introduced multi-voice support. The diarize stage emits per-speaker turns plus overlap regions, which feed into:

- ASR sentence tagging (`apply_speaker_tags`)
- Per-`(speaker, [emotion])` reference clip selection in `ref_picker`
- TTS per-sentence voice switching

Candidate diarizers considered:

- `pyannote/speaker-diarization-3.1` — SOTA accuracy + native overlap detection via `Annotation.get_overlap()`. License-gated.
- `speechbrain` ECAPA-TDNN — open license but lower accuracy + no native overlap detection.
- WhisperX bundled diarizer — wraps pyannote, no independent advantage.

## Decision

Use `pyannote/speaker-diarization-3.1`. Accept the HF-gated workflow:

1. Create an HF token at `huggingface.co/settings/tokens`.
2. Accept the license on three gated repos pyannote 3.1 pulls in:
   - `pyannote/segmentation-3.0`
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/speaker-diarization-community-1`
3. Either `export HF_TOKEN=hf_...` or `hf auth login` (cached token).

All inference is local after the one-time model download. Run on `SeparatedAudio.vocals` (the full Demucs vocal stem — pyannote needs soft onsets that the VAD-masked `speech_vocals` would clip). Post-filter pyannote turns against `vad_intervals` to drop ghost-speaker segments inside scream/laugh regions.

## Consequences

- HF token gate adds a one-time setup step (documented in `README.md` and the relevant stage docs).
- MPS coverage in pyannote 3.1 is partial — some ops fall back to CPU. ~1-3 min on a 20-min cartoon. Acceptable.
- Overlap regions come straight from `Annotation.get_overlap()`; no separate overlap-detection model needed.
- The "preset pool first" Phase 2 plan was overridden — pyannote turns plus OmniVoice ref cloning give automatic multi-voice without a preset cast.

## Revisit trigger

- A non-gated alternative reaches comparable accuracy.
- pyannote 3.2 / 4.x drops with relevant breaking changes.
- License terms change in a way that conflicts with the project's usage.

## Source

- `agent-os/specs/2026-05-26-0215-multi-voice-diarization-cloning/plan.md`
- Commit `3908e6c`
- `patmir/pipeline/stages/diarize/pyannote.py`
