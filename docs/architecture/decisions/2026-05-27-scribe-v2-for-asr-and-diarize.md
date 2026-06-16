# ADR — Scribe v2 for ASR + diarize (2026-05-27)

## Status

Accepted, then **superseded for both stages on 2026-06-15** (ElevenLabs credits exhausted → moved local: ASR → `nemotron_mlx`, diarize → `sortformer_mlx`; see `2026-06-15-local-asr-diarize-via-mlx-audio.md`). The `elevenlabs_scribe_v2` candidates stay registered; this ADR's reasoning still applies if credits return and Scribe is reactivated.

## Context

Per-speaker voice cloning was fixed on 2026-05-27 (spec `2026-05-27-1100-voice-cloning-audit`). Ear-testing the resulting `out.mkv` flagged a residual problem: pyannote 3.1 — and after a swap, pyannote `speaker-diarization-community-1` — keeps bucketing short turns from distinct characters into a single `SPEAKER_NN`. Concrete failure:

- 59.44–61.96 s (male) + 62.20–63.16 s "Бонжорно!" (≈60 y/o female) + 63.34–64.96 s (same male as 59 s) → all assigned to `SPEAKER_00`.
- Community-1 partially helped (split off the 56–58 s male as `SPEAKER_01`) but the 1.0 s female "Бонжорно!" still merged with the male voices around it.

Forcing `num_speakers=4` on pyannote did not resolve the merge (clustering count was correct; embeddings still confused short turns). Pyannote's internal embedding model isn't accurate enough at sub-2-second segments on cartoon dialogue.

ElevenLabs released Scribe v2 in March 2026 with:
- 98 % speaker-label accuracy (advertised).
- Word-level timestamps + per-word `speaker_id` in a single API call.
- Up to 32 speakers; `num_speakers` constraint supported.
- 99 languages including Russian; auto-detected if `language_code` is omitted.

## Decision

Swap both `[pipeline].asr` and `[pipeline].diarize` to a new `elevenlabs_scribe_v2` candidate. One Scribe v2 HTTPS upload returns words + speakers; both stages consume the same response (cached to `<job_dir>/scribe/result.json`).

- New file `patmir/pipeline/stages/asr/elevenlabs_scribe.py` — owns `ScribeV2Client` (the shared API + cache helper) and the `ScribeV2ASR` candidate (parses words → `Transcript` + grouped sentences via the existing `group_into_sentences`).
- New file `patmir/pipeline/stages/diarize/elevenlabs_scribe.py` — `ScribeV2Diarize` candidate. Collapses consecutive same-speaker words into `SpeakerTurn`s. Normalizes Scribe's `speaker_0` IDs to pyannote-format `SPEAKER_00` so downstream artifacts (`refs.json`, `voice_cloning_summary.json`) read identically regardless of diarizer.
- `patmir.toml`: new candidate entries for both stages; `[pipeline]` flipped to `elevenlabs_scribe_v2`. Local Whisper (`whisper_large_v3_mlx_fp16`) and pyannote (`pyannote_3_1`) stay registered as alternates — flip `[pipeline]` to revert.

## Trade-offs

- **No overlap-interval output.** Pyannote 3.1 emits `OverlapInterval` regions used by `ref_picker` to skip overlapped windows. Scribe doesn't surface this; the diarize candidate returns `overlaps=[]`. In practice the ref picker still gets clean clips because Scribe attributes each word to its dominant speaker and the per-`(speaker, emotion)` cell picks the longest single-speaker turn. If overlap-aware ref picking matters for a future clip, keep pyannote as the diarizer and use Scribe only for ASR.
- **Cloud dependency on a critical-path stage.** Diarize used to be fully local. ASR was already cloud-allowed (OpenRouter) but the active candidate was `whisper_large_v3_mlx_fp16` (local). With Scribe v2 active, diarize + ASR both require `ELEVENLABS_API_KEY` and network. Mitigation: both candidates are swappable via `[pipeline]`; local fallbacks (pyannote, mlx-whisper) remain registered.
- **No `transcribe_slice` support.** The runner's diarization-driven gap-filler (which re-transcribes individual short clips Whisper missed) checks `getattr(stage_obj, "transcribe_slice", None)` and silently skips when absent. Scribe v2's per-API-call latency makes per-slice re-transcription cost-prohibitive; if a gap-filler is needed in the future, a separate slice-cache path would have to be added.
- **Pricing.** Scribe v2 is ~$0.22 per audio hour at the standard tier (per ElevenLabs 2026 pricing); a 5-minute clip is ~$0.02. Comparable to the OpenRouter GPT-Audio ASR cost we'd otherwise be paying.

## Revisit

- If Scribe v2 ear-quality regresses vs Whisper on a representative episode (Russian transcription accuracy, especially in noisy / overlapping regions).
- If overlap-aware ref picking becomes meaningful and the alternative of "Scribe ASR + pyannote diarize" is preferred — that's a config-only revert on the diarize side.
- If cost runs hot on full-episode dubbing.
- If a stronger local diarizer ships (NeMo Sortformer, WeSpeaker-based pyannote successor) — the swap is a `[pipeline].diarize` flip.
