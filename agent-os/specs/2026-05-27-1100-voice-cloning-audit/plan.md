# Voice cloning — audit & fix

## Context

The user asked us to "make sure voice cloning is in place and working." The path is wired end-to-end (pyannote diarization → `pick_refs` → ElevenLabs IVC), but an audit on the most recent masha-2-5min run uncovered a concrete failure mode that prevents it from working for most of the dialogue.

### Audit findings on `runs/20260526-183305-masha-2-5min/`

| signal | value |
| --- | --- |
| speakers detected (pyannote) | 5 (`SPEAKER_00..04`) |
| diarization turns | 139 |
| turns belonging to `SPEAKER_00` (main character) | 86 (≈ 60 % of the dialogue) |
| ref WAVs written to disk by `pick_refs` | **6** (covering all 5 speakers) |
| ref entries surviving in `diarize/refs.json` | **1** (only `SPEAKER_03/happy`) |

Result: 5 of 6 picked refs are dropped by `backfill_ref_text`. The TTS adapter has no usable ref for `SPEAKER_00`, `SPEAKER_01`, `SPEAKER_02`, `SPEAKER_04` and falls through to the global default ("Rachel") for ~85 % of the lines. Per-speaker voice cloning is **silently disabled** for most of the cast.

### Root cause: `ref_text` gates ref usability even when the active candidate doesn't need it

1. `backfill_ref_text` (`patmir/pipeline/stages/asr/speaker_tag.py:179-200`) deletes refs whose backfilled text is empty or non-lexical. The intent — from when OmniVoice was active — was to avoid feeding OmniVoice an empty `ref_text`.
2. The TTS adapter's `_is_usable` check (`patmir/pipeline/stages/tts/adapter.py:76-77`) requires `ref.ref_text.strip() != ""` at every fallback step.

ElevenLabs (the active TTS) **does not use `ref_text` at all** (`patmir/eval/candidates/tts/elevenlabs.py:185`: `del ref_text, guidance_scale`). IVC only needs the WAV. So we throw out perfectly-good audio refs because of a constraint that no longer applies.

Secondary effects fixed in the same pass:

- **No observability**: nothing surfaces "speaker X got voice_id Y, speaker Z fell back to default."
- **`min_ref_seconds = 3.0`** plus 100 ms edge trim excludes many short-but-clean turns on cartoon-pace dialogue.
- **No tests** cover this code path.

### Intended outcome

The masha-2-5min run should show every detected speaker mapped to its own IVC clone, the run dir should contain a machine-readable summary of voice-cloning decisions, and the regression should be guarded by unit tests.

---

## Plan

### Task 1 — Save spec documentation

This folder. Plus flip `docs/architecture/quality-issues.md` row #1 to `in-progress: 2026-05-27-1100-voice-cloning-audit`.

### Task 2 — One-shot audit reporter

`patmir/eval/voice_cloning_audit.py`: given a run dir, print and write a per-speaker + per-sentence cloning report. Re-uses `SentenceTTS._lookup_ref` so it mirrors the live fallback chain. Snapshot output as `audit-baseline.md`.

### Task 3 — Fix the ref-text coupling bug

- `backfill_ref_text` only fills `ref_text` (leaves `""` when empty), never deletes.
- `SentenceTTS.__init__` adds `_needs_ref_text = self._supports_ref and not getattr(inner, "ref_text_optional", False)`.
- `_is_usable(r) = r is not None and (not self._needs_ref_text or r.ref_text.strip())`.
- ElevenLabs gets `ref_text_optional = True`.

### Task 4 — Per-job voice-cloning summary

ElevenLabs exposes `summarize()` that returns the per-speaker decision dict. Adapter calls it at end of synth and writes `<job_dir>/tts/voice_cloning_summary.json`.

### Task 5 — Lower `min_ref_seconds` 3.0 → 1.5

`patmir.toml [pipeline.options]` + `docs/architecture/current-models.md` options table.

### Task 6 — Tests

`patmir/tests/` (new dir, pytest):

- `test_ref_picker.py` — synthetic turns + overlaps, cell selection, overlap rejection, min-duration filtering, longest-window tie-breaking.
- `test_backfill_ref_text.py` — refs are never removed; `ref_text` is set even when empty.
- `test_tts_adapter_lookup.py` — exact / speaker-neutral / any-speaker / none paths; `ref_text_optional` branch.

### Task 7 — Verify on masha-2-5min end-to-end

1. `patmir dub <masha source> --resume-from diarize`.
2. Expect `diarize/refs.json` entries for ≥ 4 of 5 speakers.
3. Expect `tts/voice_cloning_summary.json` with `status: cloned` for those speakers.
4. Re-run the audit reporter → `audit-after.md`, diff vs baseline.
5. Listen-test. Flip quality-issues row #1 to `addressed` or `partial`.

---

## Critical files

- `patmir/pipeline/stages/asr/speaker_tag.py:179-200` — the bug.
- `patmir/pipeline/stages/tts/adapter.py:39-89` — gating logic to relax.
- `patmir/eval/candidates/tts/elevenlabs.py` — declares `ref_text_optional`, exposes `summarize()`.
- `patmir/pipeline/stages/diarize/ref_picker.py` — no logic changes.
- `patmir/pipeline/runner.py:144-234` — no logic changes.
- `patmir.toml` `[pipeline.options].min_ref_seconds`.
- `docs/architecture/current-models.md`, `docs/architecture/quality-issues.md` — maintenance-contract updates.

## Verification checklist

- [ ] `audit-baseline.md` shows 1/5 speakers cloned on the existing run.
- [ ] `audit-after.md` shows ≥ 4/5 speakers cloned.
- [ ] `tts/voice_cloning_summary.json` exists per-job.
- [ ] `pytest patmir/tests/` passes.
- [ ] `docs/architecture/quality-issues.md` row #1 flipped to `addressed` or `partial`.
