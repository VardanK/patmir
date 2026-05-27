# Quality issues — live tracker

Observed problems with the current pipeline output, with owner stage, status, and which spec is tackling them. Updated whenever a fix lands or a regression is observed.

Status values:

- `open` — observed, not yet scheduled
- `in-progress: <spec>` — actively being worked on
- `addressed (<date> — spec: <spec>)` — fixed; ear-verified on a representative clip
- `partial (<date> — spec: <spec>)` — fix shipped but didn't fully resolve; deeper work scheduled
- `regression-watch` — was addressed, but a regression is suspected

## Active issues

| # | Title                                | Owner stage | Status                                                                                          | Notes                                                                                                                       |
| - | ------------------------------------ | ----------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1 | Voice cloning fidelity inconsistent  | tts         | `partial (2026-05-27 — spec: 2026-05-27-1100-voice-cloning-audit)` — pipeline-verified, ear-test pending | Reported 2026-05-26 by author. 2026-05-27 audit on `runs/20260526-183305-masha-2-5min/` found a deterministic cause: `backfill_ref_text` was deleting per-`(speaker, emotion)` refs whose ASR text came back empty, and the TTS adapter required non-empty `ref_text` to consider a ref usable — even though ElevenLabs IVC ignores `ref_text` entirely. Net: 1 of 6 picked ref WAVs survived → 4 of 5 detected speakers silently fell back to the default "Rachel" voice. Fix: (a) `backfill_ref_text` no longer deletes; (b) adapter gates on `inner.ref_text_optional` (ElevenLabs sets `True`); (c) per-job `tts/voice_cloning_summary.json`; (d) `min_ref_seconds` 3.0 → 1.5; (e) per-sentence TTS timeout no longer kills the run (silent-WAV fallback). Full post-fix rerun: 5/5 speakers extracted, 4/4 speakers-with-sentences cloned with distinct IVC voice_ids, 33/36 sentences match `(speaker, emotion)` exactly (rest via same-speaker fallback), 0 default-voice fallbacks, 0 silent-WAV fallbacks. `partial` until the author ear-tests the resulting `out.mkv`. |
| 2 | Translation quality not there yet    | translate   | `addressed (2026-05-26 — spec: emotion-transfer-and-quality-docs)`                              | Escalated to OpenRouter / Gemini 2.5 Flash; ear-confirmed improvement over NLLB-3.3B. Claude Opus 4.7 + GPT-5 alternates registered for future A/B. ADR: 2026-05-26-cloud-translate-via-openrouter. Revisit if cost / latency / privacy concerns surface, or if a frontier-quality local Armenian LLM appears. |
| 3 | **Emotions not transferred**         | tts (+emotion stage) | `in-progress: 2026-05-26-emotion-transfer-and-quality-docs` (escalated to cloud TTS)            | Two rounds of OmniVoice-side `guidance_scale` tuning (low 0.6-0.9, then high 1.8-3.0) ear-tested flat. OmniVoice's zero-shot ref cloning normalizes prosody too aggressively. Escalated to ElevenLabs Multilingual v2 + IVC per the local-first cloud opt-in policy. ADR: 2026-05-26-elevenlabs-tts-cloud-opt-in. Ear-test pending. |
| 4 | Original voice leaks at sentence boundaries | mux  | `in-progress: 2026-05-26-emotion-transfer-and-quality-docs`                                     | Folded into spec #3. Fix attempt: per-sentence 50 ms afade in/out on synthesized track. Bumped to its own spec if the simple fix doesn't land. |
| 5 | Overlap handling unclear             | mux (+diarize) | `open`                                                                                       | Pyannote detects overlap intervals; downstream handling of simultaneous speakers is not defined. Needs its own spec — likely involves ducking the lower-priority voice or channel-separating. |
| 6 | Cloud ASR fidelity vs local Whisper baseline | asr | `partial (2026-05-26 — spec: 2026-05-26-2204-cloud-asr-openrouter)`                            | First ear-test of `openrouter_gpt_audio` on masha-2-5min regressed vs the local Whisper baseline — chunked transcription produced 800+ repeated `А-а-а!` tokens that crushed MMS-FA's CTC alignment, leaving a 250 s dead zone in the dub. Fixed in-spec: (a) verbatim-transcribe prompt now instructs the model to skip non-lexical vocalizations, (b) `_strip_nonlexical_lines` filters them from the transcript before alignment. Post-fix run: 33 sentences spread across the full 5 min, zero overlaps. Residual: a few words still align to 10–20 s windows (MMS-FA pads short tokens across silence). Needs its own spec if the residual matters. |
| 7 | Cloud ASR word durations occasionally inflated | asr | `open` (carved out of #6)                                                                  | Post-alignment, occasional words/sentences end up with absurdly long durations (e.g. one-word "Спокойно," → 18.9 s) when MMS-FA can't find a clean acoustic match and pads across silence. Downstream effect: the mux ±20 % stretch cap can't fit the synthesized line, so the AR voice plays briefly then silence. Possible fix: cap aligned word durations at e.g. 1.5 s + nudge sentence end inward; or align against `speech_vocals.wav` (VAD-masked) instead of full `vocals.wav`. |

## Closed issues

(none yet — keep a section here so closed entries don't crowd the active list)

## How to keep this file useful

- When a spec starts, set the relevant rows to `in-progress: <spec-folder-name>`.
- When verification passes, flip to `addressed (<date> — spec: <spec>)`.
- When a partial fix ships, flip to `partial` and add a note explaining what's left.
- If a closed issue regresses, change the status to `regression-watch` and add a note with the reproducing case.
- Add new rows whenever a quality observation is logged (commit messages, dogfood notes, ear-tests).
