# Voice cloning audit — shaping notes

## Scope

Phase 2.x quality issue #1 ("Voice cloning fidelity inconsistent"). Confirmed via on-disk audit of the most recent masha-2-5min run that voice cloning is silently failing for ≥4 of 5 detected speakers — the WAV refs are picked and written to disk, but `backfill_ref_text` deletes them before the TTS adapter sees them, so every sentence for those speakers ends up synthesized with the global default "Rachel" voice.

In-scope:
- One bug fix (the `ref_text` coupling).
- One observability addition (per-job voice_cloning_summary.json).
- One config nudge (min_ref_seconds 3 → 1.5).
- Tests for ref_picker, backfill, and adapter fallback.
- A standalone audit script so the issue is detectable from artifacts alone in future runs.

Out of scope:
- Ref-picker scoring overhauls (SNR, multi-turn concatenation). Revisit if the audit shows duration is no longer the dominant rejection reason.
- Overlap-handling changes (issue #5 — its own spec).
- Adding a non-ElevenLabs IVC candidate.

## Locked decisions

1. **`backfill_ref_text` stops deleting refs.** Returns to a purely-fill role. The deletion was masking a bug, not solving one.
2. **Candidate-level capability flag `ref_text_optional`.** Per-candidate boolean (default `False`), set `True` by ElevenLabs (and any future IVC-style candidate). The adapter — not the picker or backfill — owns the "is this ref usable" decision.
3. **No speaker-level voice persistence across jobs.** Each job re-uploads IVC clones and cleans them up on exit (existing behavior). Voice-id reuse across jobs is a future feature, not part of this spec.
4. **Summary is opt-in via `summarize()` duck-typed on the candidate.** Adapter writes the JSON only when the candidate provides the method. Keeps non-cloning candidates (Piper) clean.
5. **`min_ref_seconds` = 1.5.** ElevenLabs IVC accepts ≥ 1 s; 1.5 s gives the picker more options without taking on noisy short turns.

## Decisions deferred

- Whether to write deleted-ref WAVs out (no — picker no longer writes a WAV unless the picker keeps the ref; backfill no longer deletes).
- Cross-job voice_id reuse (would let us keep a stable voice for a recurring character across episodes).
- Bumping OmniVoice / Piper to also use the new flag — they correctly keep `ref_text_optional = False`.

## Context / inputs

- Audit performed against `runs/20260526-183305-masha-2-5min/` during the shaping pass:
  - 5 speakers, 139 turns
  - 6 WAVs written, 1 surviving ref → 5 speakers silently un-cloned
- Quality-issues row #1 theory ("many 'wrong character' reports are emotion-mismatched refs") is partially correct but missed the larger issue — most speakers are getting *no* clone at all, so emotion-mismatch is downstream of identity-loss.

## Out-of-scope risks

- If listen-test still flags wrong-character lines after the fix, the cause is no longer the gating bug; likely candidates are (a) ElevenLabs IVC quality on short refs, (b) overlap-region refs leaking through, (c) speaker-merge errors from pyannote. Track in a follow-up spec.

## Visuals

None.

## References

- `patmir/pipeline/stages/asr/speaker_tag.py:179-200` — backfill_ref_text (the bug)
- `patmir/pipeline/stages/tts/adapter.py:39-123` — adapter fallback chain
- `patmir/eval/candidates/tts/elevenlabs.py:169-227` — synth flow, ref_text unused
- `patmir/pipeline/stages/diarize/ref_picker.py` — picker (unchanged behavior)
- `patmir/pipeline/runner.py:144-234` — orchestration (unchanged)
- Prior spec `2026-05-26-1800-emotion-transfer-and-quality-docs/` (origin of per-(speaker, emotion) binning)
