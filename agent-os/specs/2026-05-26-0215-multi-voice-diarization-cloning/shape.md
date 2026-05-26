# Multi-voice (diarization + per-speaker cloning) — Shaping Notes

## Scope

Move the dub from a single global voice (Phase 1) to one cloned voice per detected source-language speaker. Concretely:

1. Add a diarization stage that splits the source's Demucs vocal stem into per-speaker turns with native overlap detection.
2. Tag every ASR sentence with a `speaker_id`; split sentences that cross a speaker boundary into per-speaker fragments when feasible, else assign to the dominant speaker.
3. For each detected speaker, automatically pick a clean reference clip and derive its `ref_text` from the ASR word stream.
4. Have the TTS stage (OmniVoice MLX, already a zero-shot cloner) swap `ref_audio`/`ref_text` per sentence based on `speaker_id`.

Outcome: drop a cartoon, get distinct Eastern-Armenian voices per character automatically.

## Decisions

- **Diarizer:** `pyannote-audio 3.1`. Best accuracy + native overlap detection + lets us keep `mlx-whisper` ASR untouched. One-time HF token setup; fully local inference after.
- **Reference selection:** fully automatic. No `patmir.toml` overrides this phase. Algorithm: drop turns intersecting `OverlapInterval`s; drop turns < `min_ref_seconds` (3.0 default); pick longest remaining; trim 100 ms inward each end.
- **Sentences spanning speakers:** split at boundary if both halves contain ≥1 ASR word; else assign to dominant speaker.
- **No noise / RMS-floor checks on reference clips** — user explicitly confirmed background bleeding is not a concern in this pipeline because Demucs + Silero VAD already produce a music-free, non-speech-free `speech_vocals` stem we slice from.
- **Local-first strict.** pyannote inference is fully local after one-time model download.
- **Eastern Armenian only.** Diarization is language-agnostic; the dialect constraint applies only to TTS (already satisfied by OmniVoice).

## Audio inputs (call-out)

The existing `separate` stage produces two relevant vocal stems:

- `vocals` — full Demucs vocal stem; preserves soft-onset speech.
- `speech_vocals` — same stem with Silero VAD applied per-sample + 20 ms cosine fades at boundaries; music-free and non-speech-vocalization-free.

We **diarize on `vocals`** (mirrors the ASR decision: full-stem input gives pyannote the soft onsets it needs for accurate boundaries; we post-filter the resulting turns against `vad_intervals` to drop ghost-speakers in scream/laugh regions).

We **slice reference clips from `speech_vocals`** (cleanest possible audio; the 100 ms inward trim keeps the cosine-fade edges out of the ref clip).

## Context

- **Visuals:** None.
- **References:** `omnivoice_mlx.py:48–75` (per-call cloning already works), `htdemucs.py:163–245` (VAD masking pattern we lean on for clean refs), `runner.py:131–177` (the `ctx.extra[...]` plumbing pattern we copy for handing diarization to ASR), `stage.py:54–101` (where the new `speaker_id` field lives).
- **Product alignment:** This is Phase 2 from `agent-os/product/roadmap.md`. The roadmap framed multi-voice as preset-pool default with cloning as stretch; we're going straight to cloning because OmniVoice has already demonstrated EA cloning in Phase 1 — the preset-pool fallback isn't needed.

## Standards Applied

`agent-os/standards/index.yml` is empty. No project standards to enforce yet.
