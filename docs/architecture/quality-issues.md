# Quality issues — live tracker

Observed problems with the current pipeline output, with owner stage, status, and which spec is tackling them. Updated whenever a fix lands or a regression is observed.

Status values:

- `open` — observed, not yet scheduled
- `in-progress: <spec>` — actively being worked on
- `addressed (<date> — spec: <spec>)` — fixed; ear-verified on a representative clip
- `partial (<date> — spec: <spec>)` — fix shipped but didn't fully resolve; deeper work scheduled
- `deferred (<date>)` — closed without resolution; ear-quality is good enough for now. Reopen by moving back to Active and bumping status to `open` or `in-progress`.
- `regression-watch` — was addressed, but a regression is suspected

## Active issues

*(none active — all known issues closed or deferred on 2026-05-27 after the Scribe v2 + voice-cloning-audit + loudness-match work landed and the masha-2-5min and Fallout-S02E01-5min ear-tests passed at "good enough for now". Add new rows when fresh ear-tests surface specific regressions.)*

| # | Title | Owner stage | Status | Notes |
| - | ----- | ----------- | ------ | ----- |

## Closed issues

| # | Title                                | Owner stage | Status                                                                                          | Notes                                                                                                                       |
| - | ------------------------------------ | ----------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1 | Voice cloning fidelity inconsistent  | tts         | `deferred (2026-05-27)` — was `partial (2026-05-27 — spec: 2026-05-27-1100-voice-cloning-audit)` | 2026-05-27 audit on `runs/20260526-183305-masha-2-5min/` found a deterministic cause: `backfill_ref_text` was deleting per-`(speaker, emotion)` refs whose ASR text came back empty, and the TTS adapter required non-empty `ref_text` to consider a ref usable — even though ElevenLabs IVC ignores `ref_text` entirely. Net (before fix): 1 of 6 picked ref WAVs survived → 4 of 5 detected speakers silently fell back to the default "Rachel" voice. Fix: (a) `backfill_ref_text` no longer deletes; (b) adapter gates on `inner.ref_text_optional` (ElevenLabs sets `True`); (c) per-job `tts/voice_cloning_summary.json`; (d) `min_ref_seconds` 3.0 → 1.5; (e) per-sentence TTS timeout no longer kills the run (silent-WAV fallback). The Scribe v2 swap (ADR `2026-05-27-scribe-v2-for-asr-and-diarize`) further split the merged short-turn buckets that pyannote couldn't resolve. Author-confirmed good enough on masha + Fallout 2026-05-27. Reopen if a fresh clip shows characters merging or fallback-to-default. |
| 2 | Translation quality not there yet    | translate   | `addressed (2026-05-26 — spec: emotion-transfer-and-quality-docs)`                              | Escalated to OpenRouter / Gemini 3.5 Flash; ear-confirmed improvement over NLLB-3.3B. Claude Opus 4.7 + GPT-5 alternates registered for future A/B. 2026-05-27: benchmarked DeepSeek V4 Pro (rejected — 4/10 thinking-leak rate, ~10 s/sentence) and TranslateGemma 27B/12B/4B (rejected — no register-control surface, literary output). Both stay registered as `enabled=false` in `patmir.toml` with rationale. ADRs: `2026-05-26-cloud-translate-via-openrouter`. |
| 3 | Emotions not transferred             | tts (+emotion stage) | `deferred (2026-05-27)` — was `in-progress: 2026-05-26-emotion-transfer-and-quality-docs` | OmniVoice `guidance_scale` tuning ear-tested flat (two rounds, low 0.6-0.9 then high 1.8-3.0). Escalated to ElevenLabs `eleven_v3` with per-emotion `voice_settings` mapping (stability + style per of `neutral`/`happy`/`sad`/`angry`/`fearful`). Bounded by what's audible in the source character's range; a calm character can't be made angry. Author-accepted current behavior on masha + Fallout. Reopen with a specific ear-test failure if a sentence's emotion sounds wrong despite the source carrying it. |
| 4 | Original voice leaks at sentence boundaries | mux  | `addressed (2026-05-27)` — was `in-progress: 2026-05-26-emotion-transfer-and-quality-docs`     | Per-sentence 50 ms `afade` in/out on the synthesized track (`sentence_afade_seconds=0.05`); ducked across `[source_start, source_end] ∪ [tts_start, tts_end]`. Reopen if leak returns at boundaries. |
| 5 | Overlap handling unclear             | mux (+diarize) | `deferred (2026-05-27)` — was `open`                                                         | Pyannote detects overlap intervals; Scribe v2 (current active diarizer) does NOT emit overlap intervals — see ADR `2026-05-27-scribe-v2-for-asr-and-diarize` (accepted regression). Each word attributes to its dominant speaker. Author-accepted for cartoon/movie content where simultaneous-speaker handling is rare. Reopen with a specific clip + window if a real overlap is being mangled. |
| 6 | Cloud ASR fidelity vs local Whisper baseline | asr | `addressed (2026-05-27)` — was `partial (2026-05-26 — spec: 2026-05-26-2204-cloud-asr-openrouter)` | Moot now: 2026-05-27 swap from `openrouter_gpt_audio` (+ MMS-FA forced alignment) to `elevenlabs_scribe_v2`. Scribe v2 returns native word timestamps; no CTC alignment, no non-lexical-token flood, no 250 s dead zones. ADR: `2026-05-27-scribe-v2-for-asr-and-diarize`. |
| 7 | Cloud ASR word durations occasionally inflated | asr | `addressed (2026-05-27)` — was `open (carved out of #6)`                                  | Same Scribe v2 swap eliminated the MMS-FA padding artifact that produced the 18.9 s "Спокойно," issue. Scribe emits acoustically-grounded per-word boundaries directly. |

## How to keep this file useful

- When a spec starts, set the relevant rows to `in-progress: <spec-folder-name>`.
- When verification passes, flip to `addressed (<date> — spec: <spec>)`.
- When a partial fix ships, flip to `partial` and add a note explaining what's left.
- When an issue is good-enough-for-now without a hard fix, flip to `deferred (<date>)` and move it down to Closed. Reopen by moving back up and bumping status.
- If a closed issue regresses, change the status to `regression-watch` and move it back to Active with a note + reproducing case.
- Add new rows whenever a quality observation is logged (commit messages, dogfood notes, ear-tests).
