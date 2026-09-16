# References for "A translator outage must fail the dub, not mute it"

## The swallow chain (before this change)

- `patmir/eval/candidates/translation/openrouter_batch.py` — `_translate_chunk` raised
  `_BatchError` on a non-retryable status; `translate_batch` caught it and called
  `_fallback_chunk`; `_fallback_chunk._one` caught every exception and returned `""`.
- `patmir/eval/candidates/translation/openrouter.py` — `_post_with_retry`, `_RETRYABLE`;
  raised a plain `RuntimeError` on non-retryable statuses (the per-sentence path was never
  affected; only the batched candidate swallowed).
- `patmir/pipeline/stages/translate/adapter.py` — the `blank` count and the "came back empty"
  warning; "never fail the run" comment.
- `patmir/pipeline/stages/tts/adapter.py:283-285` — skips empty `target_text`.
- `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py:188, 353-356` — `duck_mask` all-ones when
  the sentence loop never runs; output is `music_bed_gain*accompaniment + vocals_bed_gain*vocals`.

## Propagation

- `patmir/pipeline/context.py` — `JobCancelled`, the precedent for a typed control-flow
  exception every layer lets through; `TranslatorUnavailable` sits next to it.
- `patmir/pipeline/runner.py:214-242` — stage loop; a raising stage is marked `failed` and
  re-raised unchanged.
- `patmir/pipeline/streaming/pipeline.py:153-170` — `process()`; translate raises straight out.
- `patmir/ui/stream.py` — `_process` (the `except Exception` that skipped a section),
  `_run_locked` (status/error), `_emit`, `_section_timing`, `sections_manifest` overlay.
- `patmir/ui/jobs.py:446-461` — quality-mode status/error.

## Cache and telemetry

- `patmir/ui/cache.py:append_section` — additive `timing` merged into the manifest entry.
- `patmir/pipeline/streaming/timing.py:section_timing` — the one reader shared with bench.
- `patmir/pipeline/streaming/section.py` — `SectionResult`, `speak_section`,
  `dub_prelude_window`.

## Extension

- `extension/content.js` — `pollWatchAlongSections` (read only `waiting` off the overlay),
  `stopWatchAlong`, the stall-abandon branch (the pattern for releasing the video).
- `extension/lib/session.js` — `deriveView` ordering (`pb.playing` before `failed`), `pollJob`
  (no action on `failed`), `NULL_PLAYER`, `stop()`.
- `extension/lib/sync.js` — pure decision helpers (`shouldAbandonStream` precedent).
- Tests: `extension/tests/session.test.js` (`makeFake`, `base`, `kindOf`),
  `extension/tests/sync.test.js`.

## Tests extended

- `patmir/tests/test_translate_batch.py::RetryPolicy` (FakeClient, `_FakeResp`, `_items_from`).
- `patmir/tests/test_audience.py::EmptyLineGuard` (`_RecordingCtx`, `_StubTranscript`).
- `patmir/tests/test_cancellation.py::_managers`, `patmir/tests/test_accelerator_lock.py`
  (`_job_record`, `_stream_record`, `_storage`) — how to drive `JobManager`/`StreamManager`
  with the pipeline body replaced.

## Docs

- `docs/architecture/decisions/2026-08-28-audience-age-guardrail.md:108-119, 163-164` — why
  empties were counted rather than raised, and the revisit trigger this incident fired.
- `docs/setup/windows-wsl2-cuda.md` Phase 7 — the environment rebuild procedure used.
