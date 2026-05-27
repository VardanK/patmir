# References

## Code

- **`patmir/pipeline/stages/asr/speaker_tag.py`** — `backfill_ref_text` (lines 179-200) is the deletion site; we change its contract so it only fills text.
- **`patmir/pipeline/stages/tts/adapter.py`** — `SentenceTTS.__init__` + `_lookup_ref` (lines 39-89). Owns the new `_needs_ref_text` gate and the optional `summarize()` write-out hook.
- **`patmir/eval/candidates/tts/elevenlabs.py`** — declares `ref_text_optional = True`, exposes `summarize()` over its existing `_speaker_voice` / `_ivc_failed_speakers` state.
- **`patmir/pipeline/stages/diarize/ref_picker.py`** — picker unchanged; cited as the upstream producer of `(speaker, emotion) → SpeakerRef` entries.
- **`patmir/pipeline/runner.py`** (lines 144-234) — orchestration unchanged; verify the existing post-stage `_write_refs_json` re-run after backfill still works without the deletion path.
- **`patmir/pipeline/context.py`** + **`patmir/pipeline/config.py`** — `JobOptions.min_ref_seconds` default lowered to 1.5.

## Prior specs

- **`agent-os/specs/2026-05-26-0215-multi-voice-diarization-cloning/`** — original Phase 2 spec that introduced per-speaker IVC. Read for the original ref-text rationale (OmniVoice-era).
- **`agent-os/specs/2026-05-26-1800-emotion-transfer-and-quality-docs/`** — introduced per-`(speaker, emotion_bin)` keying; the backfill-deletion behavior we're changing was added by this spec.

## Docs

- **`docs/architecture/stages/tts/current.md`** + **`docs/architecture/stages/tts/elevenlabs-multilingual-v2.md`** — describe the intended cloning lifecycle. Stay accurate after this spec (no change).
- **`docs/architecture/quality-issues.md`** — row #1 is the issue being addressed.
- **`docs/architecture/current-models.md`** — Pipeline options table needs `min_ref_seconds` update.

## Run artifacts cited

- `runs/20260526-183305-masha-2-5min/diarize/refs.json` — 1 surviving entry (smoking gun)
- `runs/20260526-183305-masha-2-5min/diarize/refs/` — 6 WAVs on disk for 5 speakers
- `runs/20260526-183305-masha-2-5min/diarize/turns.json` — 139 turns, 5 speakers
