# References for watch-along startup + stalls

Existing code to reuse rather than reinvent. Line numbers are against
`origin/main` @ `1dadb82`.

## Per-section telemetry — the shape already exists

### `_section_record`

- **Location:** `patmir/bench.py:366`
- **Relevance:** Task 3 needs exactly this — it already reads a section's
  `manifest.json` back and produces
  `start/end/section_s/wall_s/analysis_wall_s/dub_wall_s/build_s/rtf/stage_wall_s/new_speakers`.
- **Key patterns:** `ANALYSIS_STAGES` / `DUB_STAGES` (`bench.py:50-51`) split
  the two `run_job` halves of a section, which is the split the live path also
  wants. `_stage_wall` (`bench.py:121`) and `_stage_timings` (`bench.py:266`)
  are the readers.
- **Plan:** lift it out of `bench.py` into a shared helper so the live path
  (Task 3) and bench (Task 4) cannot drift. Do **not** write a second manifest
  reader.
- **Gap to close:** `wall_s - sum(stage_wall_s)` is the only handle on the
  ECAPA / age-gender / ffmpeg-slice cost, and nothing currently surfaces it.
  Task 3 names it `other_s`.

### `Manifest` / `StageStatus`

- **Location:** `patmir/pipeline/manifest.py:17-29` (fields),
  `patmir/pipeline/runner.py:175/199` (written)
- **Relevance:** the only per-stage timing on disk. `stages` is a **dict keyed
  by stage name**, not a list — easy to get wrong.
- **Caveat to respect:** timings use `time.time()`, not `perf_counter`, so a
  suspended run yields impossible rows (`performance.md:557-562`). Task 4's
  `time_to_first_audio_s` should use `perf_counter` deltas where it can.

## The `stage_cache` contract

- **Location:** `patmir/pipeline/runner.py:184-194`, key builder
  `stage_cache_key` at `runner.py:42`
- **Relevance:** Task 7's B3 turns on the key being `(stage, candidate_id)` —
  which is precisely why prewarming with the batch cfg misses the two ids
  `streaming_config()` overrides.
- **Key patterns:** `prewarm_stage_cache` (`runner.py:55`) is best-effort and
  idempotent, printing and continuing on failure — the right shape to copy when
  adding ECAPA / age-gender prewarm, since neither should be able to break boot.
- **Existing tests:** `patmir/tests/test_stage_cache.py` already pins key shape
  and prewarm idempotence.

## The `sections.json` schema

- **Location:** `patmir/ui/cache.py:377-403` (`append_section`, and the
  init/finalize helpers around it)
- **Relevance:** Task 3 extends the per-section entry; Task 6 (C1) changes how
  entries are replaced.
- **Key patterns:** replacement is by `start` within +/-0.5 s, **not** by index —
  which is exactly how a restarted stream produced duplicate indices pointing at
  one overwritten file. Any fix has to make that shape unrepresentable, not just
  avoid producing it.
- **Existing tests:** `patmir/tests/test_stream_cache.py` covers init -> append
  -> read -> finalize, out-of-order replacement and sort order. Extend it rather
  than starting a new file.

## Test patterns to copy

### `run_job` stub

- **Location:** `patmir/tests/test_bench.py:77-94`, artifact faker `_write_job`
  at `:40-68`
- **Relevance:** every new test in Tasks 4-6 needs a `run_job` that honours the
  cache contract without loading a model.
- **Key patterns:** it deliberately writes real-looking `manifest.json` files so
  the readers under test are exercised for real.

### Scheduler tests with fake callables

- **Location:** `patmir/tests/test_streaming_coverage.py`
- **Relevance:** Task 6's paused-scheduler and restart-index cases belong here.
- **Key patterns:** `StreamScheduler` is driven with fake `process_fn` /
  `playhead_fn`, so scheduling policy is testable with no audio at all.

### Source-inspection assertions

- **Location:** `patmir/tests/test_bench.py:333-344` (`WatchAlongOptOutTest`)
- **Relevance:** it asserts via `inspect.getsource` that the cache dict is
  created **inside** the repeat loop. Task 4 changes that region; this test will
  need updating deliberately rather than being tripped over.

### Batch equivalence

- **Location:** `patmir/tests/test_higgs_torch_batch.py`
- **Relevance:** constrains Task 7's B1. Batch-vs-single equivalence is verified
  bitwise on a tiny CPU model, and per-row generator seeding (ADR
  `2026-08-24-tts-sampling-reproducibility`) means **batch shape must not change
  a line's audio** — so "did I break it" is decidable, not a judgement call.

## The code being changed

| Concern | Location |
| --- | --- |
| section-end snap (A1) | `patmir/pipeline/streaming/prelude.py:59` `_snap_window_end`, called at `section.py:225` with `slice_len - END_GUARD_SECONDS` |
| scheduler idle / abandon (A3) | `patmir/pipeline/streaming/scheduler.py:115-120` `_default_idle`; `Playhead.paused` set at `:55`, never read |
| stream restart / dedupe (A3, C1) | `patmir/ui/stream.py:162` `submit`, accepted-status set at `:169-171`; index counter at `:341` |
| live run body (Task 3) | `patmir/ui/stream.py:281` `_run_blocking` |
| silent download (B5) | `patmir/ui/stream.py:289` calls `jobs._download_url` **without `rec`**; hook no-ops at `patmir/ui/jobs.py:395-396` |
| registry growth (A4) | `patmir/pipeline/streaming/section.py:96-100` appends; `registry.py:194-199` linear scan |
| prewarm (B2, B3) | `patmir/ui/app.py:84`; lazy loads at `streaming/embeddings.py:53` and `streaming/speaker_profile.py:38` |
| streaming candidate set (B3) | `streaming/runner.py:66-68` overrides, `patmir.toml:776-777` cuda streaming layer |
| bench watch-along (Task 4) | `patmir/bench.py:390` `bench_watch_along`, forward loop at `:441`, `None`s at `:488-489`; CLI at `patmir/cli.py:273` |
| client buffering (A2, C1) | `extension/content.js:392` lookahead, `:416-417` poll/tick, `:459-470` section map, `:538-556` freeze path |

## Dead code encountered

`stream_sections` (`patmir/pipeline/streaming/runner.py:83`) is unused —
`StreamManager` drives `StreamScheduler` and bench calls `process_section`
directly. It carries its own `stage_cache = {}`, so it would silently miss the
warm cache if anyone wired it up. Worth deleting or documenting while in here.
