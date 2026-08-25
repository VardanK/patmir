# Watch-along: slow first chunk + periodic freezes — measure, then fix

## Context

Dubbing `https://www.youtube.com/watch?v=os-jXiYRihI` in watch-along mode from the
Chrome extension on the WSL2 + RTX 3090 box: the first chunk takes too long and
playback periodically freezes.

The session is still on disk, so this spec starts from measurements rather than
hypotheses. Evidence reconstructed before planning:

- `~/.patmir/cache/d5af5653d745ed66/sections.json` — the stream's own manifest
- `/home/vardan/patmir/runs/20260825-1821*…1826*` — 10 prelude/section job dirs,
  each with `manifest.json` (per-stage `started_at`/`finished_at`/`build_s`) and
  `progress.jsonl`

### What actually happened, 18:21:43 → 18:27:37

| wall clock | job dir | window produced | stage walls (s) |
| --- | --- | --- | --- |
| 18:21:43→18:22:56 | `182143-prelude` | 0–40.0 | ing .12 sep 3.41 dia .65 emo .34 asr 5.50 · **[14.3 gap]** · tr 4.09 **tts 41.6** mux 1.07 |
| 18:22:56→18:23:31 | `182256-section` | 154.9–190.3 | asr 6.58 tr 6.22 **tts 17.96** |
| 18:23:32→18:24:04 | `182332-section` | 190.3–**198.4** | asr 9.08 tr 4.21 tts 15.51 — 32.8 s of work for **8.1 s** of dub |
| 18:24:05→18:24:13 | `182405-prelude` | 0–40.0 | analysis only, **discarded** |
| 18:24:13→18:24:56 | `182413-prelude` | 0–40.0 | full re-run, **tts 28.3** |
| 18:24:56→18:27:37 | 5 × `-section` | 198.5–334.1 | walls 22.0 / 30.0 / 31.9 / 37.5 / 38.3 |

**Time to first audio: 72.4 s** for a 40 s opening window (RTF 1.81), on top of a
blocking full yt-dlp download that reports no progress in watch-along mode.

**Steady-state production RTF 0.58–1.0, mean ≈ 0.87**, with `IDLE_BEFORE`
between sections of 0.4 s — strictly serial, no lookahead. A buffer that grows
at 13 % of playback cannot absorb one bad section.

## Findings

Every number below is from the session above. Ranked within each group.

### Freeze causes

**A1 — the VAD end-snap collapses a section when speech is continuous.**
`_snap_window_end` (`patmir/pipeline/streaming/prelude.py:59`) takes
`max(e for _s, e in vad_intervals if e <= target)` with `target = 38.0 − 1.5`.
Section `182332`'s intervals were `[[0.0, 6.3], [6.3, 8.1], [8.1, 38.0]]` — the
final utterance runs past the guard, so the only eligible end is **8.1**. The
section analyzed 38 s, emitted 8.1 s, and threw away 8 of its 10 ASR sentences.
RTF 4.0. Two of the session's seven sections were truncated this way (8.1 s and
13.6 s). On continuous narration this is the normal case, not an edge case, and
it is the structural reason the buffer never builds.

**A2 — no minimum pre-buffer before playback starts.** Playback begins the
moment section 0 lands (`extension/content.js:538`); the buffer is exactly the
prelude window. `watchAlongTick` pauses the video whenever `sectionAt()` has no
decoded buffer (`content.js:548-556`) — that is the freeze the author sees.
Nothing gates playback on N seconds of coverage.

**A3 — pausing for >45 s permanently kills the stream.** `_default_idle`
(`scheduler.py:115-120`) ends the run when `now − playhead.updated_at > 45`;
`_run_blocking` returns, the record goes `done`, and a later `submit()` returns
that same `done` record (`stream.py:169-171`) instead of restarting. The viewer
is then frozen at the buffer edge with no recovery. `Playhead.paused` is set but
**never read** by the scheduler, contradicting its own docstring
(`scheduler.py:5-6`).

**A4 — the ECAPA registry grows without bound.** `_resolve_identities`
(`section.py:96-100`) appends every ref clip of every matched speaker to
`_embeddings[spk]` each section, and `registry.resolve` (`registry.py:194-199`)
linearly scans all of them. Per-section cost rises with section index — a
mechanism for stalls that get worse the longer you watch.

**A5 — serial production, no pipelining.** One thread runs prelude and every
section (`stream.py:281`); within a section the two `run_job` halves are
sequential. `performance.md:393-400` measured analysis 0.578 RTF and dub 0.992
RTF; the pipelined ceiling is ~0.99 against a serial 1.71. That analysis
concluded the win was small **on MLX**, which serializes on one GPU stream — it
was never re-tested on CUDA, where cloud translate alone is pure idle GPU.

### Startup cost

**B1 — TTS batch tail (largest single win).** `higgs_v3_torch_bf16` chunks at
`max_batch=8` and a chunk costs `max(steps)`, so the remainder batch is nearly
as expensive as a full one:

| job | batching | tts wall |
| --- | --- | --- |
| prelude, 9 sentences | 8 + **1** | 24.3 + **17.3** = 41.6 s |
| `182659`, 12 sentences | 8 + **4** | 10.5 + **12.2** = 22.8 s |
| `182332`, 2 sentences | 2 | 15.5 s |

One batch of 9 instead of 8+1 removes ~17 s — **~23 % of time-to-first-audio**.
`max_batch=8` is documented as unswept.

**B2 — 14.3 s of unattributable time between ASR and translate in the prelude.**
No stage owns it, no progress event covers it, no manifest records it. It does
not recur on the second prelude, so it is a one-time model load: ECAPA
(`embeddings.py:53`, speechbrain, CPU, `@lru_cache`) and the wav2vec2 age-gender
model (`speaker_profile.py:38`). **Neither lives in `stage_cache` and neither is
prewarmed**, so both load lazily inside the prelude. 20 % of time-to-first-audio.

**B3 — `prewarm_stage_cache` misses the streaming candidate set.** `app.py:84`
prewarms with the **batch** cfg, and the cache key is `(stage, candidate_id)`, so
the two ids `streaming_config()` overrides are misses on the first stream after
every server start:

| stage | prewarmed | streaming needs | hit? |
| --- | --- | --- | --- |
| separate | `htdemucs_vocals` | **`htdemucs_fast`** | no |
| translate | `openrouter_gemini_flash` | **`openrouter_gemini_flash_batch`** | no |

Cost here was `separate build_s 1.4` inside the prelude; the larger consequence
is that both Demucs models end up resident, which the uncommitted
`2026-08-25-one-accelerator-at-a-time` ADR already counts against the 24 GB card.

**B4 — per-section fixed overheads that belong in the prelude.** ASR re-runs
language detection every section (`asr detected en`, 3.5–4.6 s, 12–15 % of
section wall) though the prelude already knows it. Translate shows a 3.4–4.2 s
gap before `sentence 1/N` in every section while the sentences themselves finish
in <0.1 s — the cost is the round trip, not the work.

**B5 — download is blocking, whole-file, and silent.** `_run_blocking`
(`stream.py:289`) calls `jobs._download_url(...)` **without `rec`**, so the
progress hook no-ops (`jobs.py:395-396`) and a URL stream reports nothing while
downloading. The planned `yt_dlp` ingest candidate does not exist
(`patmir/pipeline/stages/ingest/` has only `ffmpeg.py`).

### Correctness damage

**C1 — the section index counter resets on resume and overwrites cache files.**
The restart at 18:24 correctly re-seeded `Coverage` from `sections.json` (the
next section starts at 198.5, past the existing coverage) but restarted the
index counter at 0 (`stream.py:341`). `append_section` replaces by `start`
(±0.5 s), not by index, so the manifest now holds duplicate indices pointing at
one overwritten file:

```
index 1 → 154.9–190.3 → section_001.wav
index 2 → 190.3–198.4 → section_002.wav
index 1 → 198.5–229.8 → section_001.wav   ← same file, different window
index 2 → 229.8–243.4 → section_002.wav   ← same file, different window
```

Wrong audio at 154.9 s and 190.3 s, live and on every rewatch. The client keys
`waSections` by index and never deletes (`content.js:459-470`), so the stale
entry also wins `sectionAt()` in insertion order.

### Telemetry — the author's explicit ask, and it does not exist

**D1 — a live stream persists no timing whatsoever.** `sections.json` records
`{index, start, end, file}` (`ui/cache.py:377-403`); `StreamRecord.to_public()`
(`stream.py:80`) exposes `status`/`sections_done`/`window_end`/`duration`. There
is no per-section wall, no per-stage breakdown, no buffer-ahead figure, no
record of client buffering. All per-section numbers in `performance.md` come
from bench re-reading `manifest.json` after the fact. `stream.py` has no timing
instrumentation at all.

**D2 — `bench --mode watch-along` does not measure the thing that is slow.** It
bypasses `StreamManager` and `StreamScheduler` entirely (`bench.py:390`) — no
playhead, no coverage, no lookahead, no idle, no download, no cache writes, no
client. It also never passes `extra={"bench": True}`, so `tts`/`translate`
latency traces are hard-coded `None` (`bench.py:488-489`), and it takes a local
`clip` only — **there is no `--url`**, so the reported video cannot be
benchmarked as submitted. It creates a fresh cache per repeat, so its section-0
numbers overstate what a prewarmed server pays.

**D3 — the untracked slice.** `wall_s − Σ stage_wall_s` per section is the only
handle on ffmpeg slice/trim, ECAPA embedding and age-gender profiling, and
nothing surfaces it. `_resolve_identities` runs ~15–25 CPU forward passes per
section, all invisible.

### Secondary

- `_free_accelerator_cache()` runs after **every** stage including warm hits
  (`runner.py:226`) — ~8 calls/section, measured as consistent 0.2 s inter-stage
  gaps ≈ 1.8 s (≈6 %) per section. Its justification in the ADR is an MLX
  unified-memory finding, never a CUDA measurement.
- Diarization reports **1 speaker** on the prelude and every section of this
  video, so the 72 s cast-establishment prelude and the ECAPA registry buy
  almost nothing on this content.
- `stream_sections` (`streaming/runner.py:83`) is dead in production — nothing
  imports it; it carries its own `stage_cache = {}`.

## Baseline: rebase onto `origin/main` first

The code the server ran during the measured session is now **on `origin/main`**
(`1dadb82`, PR #4 "accelerator-lock-and-job-cancellation"): cooperative job
cancellation and a shared accelerator lock, with ADRs
`2026-08-25-cooperative-job-cancellation.md` and
`2026-08-25-one-accelerator-at-a-time.md`, touching `ui/stream.py`, `ui/jobs.py`,
`ui/app.py`, `pipeline/runner.py`, `pipeline/context.py`, `streaming/*.py` and
`stages/tts/adapter.py` — the exact files this spec edits.

This worktree is 3 commits behind at `5942acb`. Verified: the diff
`HEAD..origin/main` over those files matches the working-tree diff I read from
the server checkout file for file and line for line, and `/home/vardan/patmir`
is now clean at `57e5187`. So `origin/main` **is** what produced the measured
session — the numbers above stand against it, and every task below is written
against it.

---

## Task 1: Save spec documentation

Create `agent-os/specs/2026-08-25-1931-watchalong-startup-and-stalls/` with:

- **plan.md** — this plan
- **shape.md** — scope, the three shaping decisions (instrument *and* fix;
  freeze before startup; build on the uncommitted server work), and the
  reconstruction method so the numbers are reproducible
- **references.md** — `patmir/bench.py:366` `_section_record` as the shape for
  per-section telemetry; `runner.py:184-194` for the `stage_cache` contract;
  `ui/cache.py:377-403` for the `sections.json` schema; `test_bench.py:77-94` for
  the `run_job` stub pattern to reuse in new tests
- **measurements/** — the extracted session tables above, plus the raw
  `manifest.json` / `progress.jsonl` timings, so the before/after is anchored

No standards directory exists in this repo (`agent-os/standards/` is absent), so
no `standards.md`. No visuals.

## Task 2: Rebase onto `origin/main`

Rebase this branch onto `origin/main` (`1dadb82`) — 3 commits, already verified
to be the code that produced the measured session. Re-run the existing suite,
especially the tests that came with it (`test_cancellation.py`,
`test_accelerator_lock.py`) plus `test_streaming_*.py` and `test_stage_cache.py`.

Nothing below is valid against `5942acb`. Two things to reconcile while rebasing,
because both interact with fixes here:

- The accelerator lock is held for a **whole stream**, so Task 8's pipelining
  option and any concurrent-work idea must be checked against it rather than
  assumed available.
- `JobCancelled` must keep propagating out of `process_section`
  (`stream.py`, the `except JobCancelled: raise` guard) — Task 6 edits the same
  restart/stop paths and must not re-swallow it.

## Task 3: Instrument the live stream path (D1, D3)

Make the running server report what it is doing, since today it reports nothing.

- Time each section in `StreamManager._run_blocking` around the
  `process_section` call and persist a record next to the audio — extend the
  `sections.json` entry (`ui/cache.py:377-403`) with `wall_s`, `stage_wall_s`
  (read back from the section's `manifest.json`, exactly as
  `bench.py:_section_record` already does), `build_s`, `produced_at`, and
  `other_s = wall_s − Σ stage_wall_s` so the ECAPA/profiling/ffmpeg slice is
  visible rather than inferred.
- Record `buffer_ahead_s` at each section boundary (`coverage` end minus
  `playhead.t`) — the single number that predicts a freeze.
- Time the download separately and pass `rec` into `_download_url` so watch-along
  gets the progress hook it already has (B5).
- Surface the same on `StreamRecord.to_public()` so the popup can show it.
- Reuse `_section_record` rather than writing a second reader; lift it out of
  `bench.py` into a shared helper both call.

Keep the schema additive — `cache_version` stays readable by an older client.

## Task 4: Make `patmir bench --mode watch-along` measure the real path (D2)

- Add `--url`, resolving through the same `_resolve_source_id` +
  `_download_url` the server uses, so the reported video can be benchmarked as
  submitted.
- Drive `StreamScheduler` with a synthetic playhead instead of the plain forward
  loop at `bench.py:441`, so coverage, lookahead, idle and the abandon timer are
  in the measurement. Keep the existing forward loop behind a flag for
  comparability with the stored 2026-08-24 baselines.
- Pass `extra={"bench": True}` through the streaming `run_job` calls so
  `tts`/`translate` `timings.jsonl` traces exist in watch-along, filling the
  hard-coded `None`s at `bench.py:488-489`.
- Report `other_s` per section and a `time_to_first_audio_s` top-level figure —
  the author's actual complaint currently has no metric.
- Add a `--prewarm` mode matching what a live server does, since the fresh
  per-repeat cache overstates section 0.

**Capture the baseline on `os-jXiYRihI` before any fix lands.** Everything below
is measured against it.

## Task 5: Fix the section-end snap (A1) — the freeze's structural cause

In `_snap_window_end` (`prelude.py:59`), a section whose speech runs past the
guard currently falls back to the last early boundary and collapses. Change the
contract: when no VAD end sits in `[minimum_useful, target]`, cut at `target`
and let the transcript trim handle the partial utterance, rather than emitting
8.1 s for 38 s of work. Introduce a floor (e.g. `MIN_SECTION_SECONDS`) below
which a snap candidate is rejected outright.

The pad exists to find a silence; when there is no silence, paying full
analysis cost for a fifth of a section is strictly worse than a slightly rough
cut. Section boundaries already get a 50 ms fade at mux
(`sentence_afade_seconds`).

Unit-test with the exact measured intervals `[[0.0, 6.3], [6.3, 8.1], [8.1, 38.0]]`
and assert the section is not truncated to 8.1.

## Task 6: Stop the stream dying on pause and on restart (A3, C1)

- **A3:** honour `Playhead.paused` in `_default_idle` (`scheduler.py:115`) —
  a paused viewer should keep filling forward, which is what the scheduler's
  docstring already claims. Separate "viewer paused" from "viewer abandoned":
  the abandon timer should key off the client having stopped polling, not off
  the playhead having stopped moving.
- Make `submit()` restart a `done` stream whose coverage is incomplete
  (`stream.py:169-171`) instead of handing back a finished record.
- **C1:** seed the section index counter from the existing `sections.json` on
  resume (`stream.py:341`) so a restart cannot overwrite `section_NNN.wav`.
  Make `append_section` replace by `start` **and** reconcile the index, so the
  duplicate-index shape in the measured manifest becomes unrepresentable.
- Have the client drop superseded entries rather than accumulating them
  (`content.js:459-470`).
- Add a repair path (or at minimum a warning) for the already-corrupted
  `d5af5653d745ed66` manifest.

## Task 7: Cut time-to-first-audio (B1, B2, B3, B4)

In measured-win order:

- **B1, ~17 s.** Fix the TTS batch tail in `higgs_v3_torch_bf16` — avoid a
  remainder batch of 1–4 when a single batch would cost the same `max(steps)`.
  Sweep `max_batch` (documented as unswept) on the prelude's sentence counts.
  The existing bitwise batch-vs-single equivalence test
  (`test_higgs_torch_batch.py`) constrains the change; per-row generator seeding
  from ADR `2026-08-24-tts-sampling-reproducibility` means batch shape must not
  alter a line's audio, which makes this verifiable rather than a judgement call.
- **B2, ~14 s.** Prewarm ECAPA (`embeddings.py:53`) and the age-gender model
  (`speaker_profile.py:38`) on the FastAPI startup hook alongside
  `prewarm_stage_cache`. Both are `@lru_cache`d already — the fix is calling
  them at boot, not restructuring them.
- **B3.** Prewarm the **streaming** candidate set too: call
  `prewarm_stage_cache` with `streaming_config(...)` as well as the batch cfg
  (`app.py:84`). Then reconsider whether watch-along still needs
  `htdemucs_fast` when batch's `htdemucs_vocals` is already resident and
  measured faster (8.84 s batch) — dropping the
  `[pipeline.profiles.cuda.streaming]` override would remove a whole resident
  model, which the accelerator-lock ADR wants.
- **B4.** Pin the ASR language from the prelude instead of re-detecting per
  section (3.5–4.6 s × every section).

Re-measure after each, not once at the end.

## Task 8: Buffer policy (A2, A5) — only if Tasks 5–7 leave a gap

Deliberately last: Tasks 5–7 remove wasted work, and the buffer arithmetic
changes once they land. Re-measure first, then pick from:

- A minimum pre-buffer before the client starts playback — trade a slightly
  later start for no mid-scene freeze. Requires a number from Task 3's
  `buffer_ahead_s`, not a guess.
- Overlap the analysis half of section N+1 with the dub half of section N
  (A5). The MLX-based dismissal in `performance.md:393-400` does not transfer to
  CUDA, where the ~4 s cloud-translate round trip per section is idle GPU.
- Cap the ECAPA embedding list per speaker (A4) so per-section cost stops
  drifting upward on a feature-length video.
- Re-test `_free_accelerator_cache()` on warm hits under CUDA (~6 %/section),
  whose justification is an MLX-only measurement.

Also worth a decision here, informed by Task 3's data: diarization sees **one
speaker** on this whole video, so a 40 s cast-establishment prelude is nearly
pure latency. A shorter `prelude_window_seconds` for low-speaker-count openings
is the cheapest remaining startup win, but it needs the telemetry first.

## Verification

1. **Baseline, then delta.** `patmir bench --mode watch-along --url
   'https://www.youtube.com/watch?v=os-jXiYRihI' --profile cuda --report
   runs/bench/os-jXiYRihI-watch-before.json`, captured at the end of Task 4
   before any fix. Repeat after each of Tasks 5, 6, 7. Targets from the measured
   session: time-to-first-audio 72.4 s → under 40 s; no section emitting less
   than half its analyzed window; steady-state RTF meaningfully below the
   measured 0.87 mean.
2. **The real path, not just bench.** Re-dub the same URL from the Chrome
   extension with the server running under `PATMIR_PROFILE=cuda`, then read the
   new per-section timing straight out of `sections.json` — that is the point of
   Task 3.
3. **Freeze regression, explicitly.** Pause for >45 s and resume; confirm the
   stream keeps filling and playback recovers (A3). Seek forward into undubbed
   territory and confirm recovery. Reload mid-stream and confirm
   `sections.json` gains no duplicate index and no `section_NNN.wav` is
   overwritten (C1).
4. **Unit tests.** New cases in `test_streaming_coverage.py` (paused scheduler,
   restart index seeding), `test_stream_cache.py` (index reconciliation on
   resume), a snap test in a new `test_streaming_snap.py` pinned to the measured
   VAD intervals, and `test_bench.py` for `--url` + scheduler-driven mode. Reuse
   the `run_job` stub at `test_bench.py:77-94`.
5. **Ear check.** One full watch-along of the video end to end — the fixes
   change section boundaries (Task 5) and TTS batch shape (Task 7), both of
   which are audible if wrong.

## Docs to update in the same change (CLAUDE.md maintenance contract)

- `performance.md` — new watch-along baselines, and correct the
  `sections 1–9 = RTF 0.831` framing: that figure came from a run without the
  snap collapse, the restart, or the batch tail.
- `quality-issues.md` — new rows for the corrupted `sections.json` (C1) and the
  section-snap collapse (A1); both are output-affecting, not just slow.
- `pipeline.md` — the watch-along control-flow block, once pause/restart
  semantics change.
- New ADR for whichever of Tasks 5/6/8 changes a contract (section-end policy
  and pause semantics both qualify), plus a one-line entry in `decisions.md`.
