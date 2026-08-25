# Standards for the extension volume + seek-sync fix

`agent-os/standards/index.yml` contains only a header comment
(`# Agent OS Standards Index`) with no entries, so **no standards apply to this
work**.

The governing conventions come from `CLAUDE.md` instead:

- **The maintenance contract.** "Address or regress a quality issue" requires
  updating `quality-issues.md` (status + last-touched date) in the same change, and
  "lock a non-trivial decision" requires a new `decisions/<date>-<slug>.md` plus a
  one-line entry in `decisions.md`. Task 10 exists to satisfy the first; the ADR is
  conditional on the drift-correction design proving non-trivial once measured.
- **`decisions.md` is checked before investigating.** The relevant prior decision is
  `2026-05-27-low-friction-translation-d1` (which specified the drift-correction loop
  this work finally implements) and `2026-06-13-near-realtime-d3-spike` (which
  removed the scheduler's backfill, the reason a post-seek gap is never filled). No
  revisit trigger has fired; this work implements what those ADRs already called for.
- **`patmir.toml` is the single source of truth** for stage selection. Untouched here
  — nothing in this change swaps a model or adds a knob to `[pipeline.options]`.
- **Eastern Armenian (`hy_AM`) only.** Not at risk; no translate or TTS change.
- **A polished UI is not a goal.** This is why the fix stays inside the existing
  player wiring rather than adding a volume control of the extension's own: the page
  already has a slider, and the bug is that we ignore it.

## Conventions borrowed from the existing code

- **The programmatic-write guard pattern.** `waProgrammatic`
  (`extension/content.js:388`, `:570-575`, `:602`) already exists to stop the
  extension's own `video.pause()` being read as a user pause. The new mute guard is
  the same idea, made exact (a counter instead of a 400 ms timeout) because the
  volume case can distinguish its own writes precisely and the pause case could not.
- **Keyword-only optional parameters threaded through the streaming call chain.** PR
  #4 added `cancel: threading.Event | None = None` to `run_prelude`,
  `dub_prelude_window` and `process_section` in exactly this shape; the new `max_end`
  parameter follows it.
- **Python tests live in `patmir/tests/test_*.py`** as `unittest` classes. The server
  half extends `test_streaming_coverage.py` and `test_stream_cache.py` rather than
  adding a new module.
