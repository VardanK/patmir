# References for the extension volume + seek-sync fix

## The file the bugs live in

### `extension/content.js` (649 lines)

The entire client. One IIFE, two playback engines sharing module-level `audioCtx`,
`video`, `wasMuted`, `originalVolume`:

- **Quality mode**, `:172-364` — `startDub` / `stopDub` / `startSourceAt`, the
  `handlers` event map at `:267-293`.
- **Watch-along mode**, `:366-637` — `startWatchAlong` / `stopWatchAlong`,
  `pollWatchAlongSections`, `ensureBuffer`, `sectionAt`, `scheduleNode`,
  `ensureScheduled`, `stopAllNodes`, `watchAlongTick`, the `waHandlers` map at
  `:594-621`.

Sites that matter to this work:

| What | Where |
| --- | --- |
| The only mute mechanism | `:209-212` (quality), `:407-409` (watch-along) |
| The mute/volume restore on stop | `:227-231`, `:436-440` |
| Graph connection (no GainNode exists) | `:246`, `:510` |
| Dead drift anchors, written and never read | `:33-34`, `:249-250` |
| Quality-mode seek handling, in full | `:278-280` |
| Watch-along drift check (the only one that exists) | `:519-528`, `WA_DRIFT` at `:393` |
| Section lookup, first-insertion-order | `:491-496` |
| Silent bail when the offset exceeds the buffer | `:501` |
| In-place section mutation without buffer invalidation | `:462` |
| The 250 ms tick and 2 s poll | `:416-417` |

Notably absent from both handler maps: `volumechange`, `seeking`, `waiting`,
`stalled`, `playing`.

## The pattern to reuse for the mute guard

### `waProgrammatic`

- **Location:** `extension/content.js:388` (declaration), `:570-575` (`waPause`
  setting it), `:602` (the `pause` handler consuming it).
- **Relevance:** the same problem one layer over — telling the extension's own write
  to the `<video>` apart from the viewer's action. Its comment explains the async
  nature of the event that forces the guard to be consumed in the handler rather than
  reset synchronously.
- **What to borrow, and what not to.** Borrow the consume-in-the-handler shape. Do
  **not** borrow the 400 ms timeout fallback: the mute case can be exact, because
  `forceMute` only writes when `muted` is currently `false` and that write emits
  exactly one `volumechange` (assigning `true` to an already-`true` `muted` fires
  nothing).

## The spec that already designed the drift loop

### `agent-os/specs/2026-05-27-1200-low-friction-translation-d1/plan.md`

- **Location:** `:145-158`, "Web Audio sync algorithm".
- **Relevance:** it specifies the loop that was never implemented — 1 s interval,
  `expected`/`actual` computed exactly as this plan does, hard-resync above 80 ms,
  soft ±0.5 % `playbackRate` nudge between 30 and 80 ms. This work implements it,
  with two changes justified by what the code turned out to do: a 250 ms tick (reusing
  watch-along's existing interval) and a wider/hysteretic band, because
  `video.currentTime` is coarsely quantized and a hard re-anchor is an audible cut.
- **Also there:** `:149` is the decision to mute rather than zero the volume — the
  line the volume bug descends from — and `:157` notes that mux keeps total dub
  duration within ~ε of source, which is what makes quality mode's single global
  offset mapping valid.

## Server-side: the watch-along seek half

### `patmir/pipeline/streaming/scheduler.py`

- **`_next_start`, `:102-113`** — the entire seek/pause/backfill decision, twelve
  lines. Returns the raw playhead-derived uncovered point with no clamp against the
  next covered run. `Playhead.paused` is set but read nowhere in this file.
- **`run`, `:124-144`** — the coverage-fudge paths at `:135-141` (out of scope, worth
  a quality-issues row).
- **The docstring at `:15-18`** describes a backfill fallback that no longer exists;
  it was removed by the 2026-06-15 cost guard.

### `patmir/pipeline/streaming/coverage.py`

- **Relevance:** `Coverage` exposes `add`, `covers`, `first_uncovered`, `is_complete`,
  `intervals` — and no "start of the next covered run", which is the query the clamp
  needs. `_MERGE_EPSILON = 0.5` at `:28`.
- Tests to extend: `patmir/tests/test_streaming_coverage.py` (already covers
  `first_uncovered` and the seek reprioritization at `:113`).

### `patmir/pipeline/streaming/section.py` + `prelude.py`

- **`process_section`, `section.py:195-259`** — slices `target_seconds + pad_seconds`
  (38 s) and VAD-snaps the end via `_snap_window_end`.
- **`_snap_window_end`, `prelude.py:58-67`** — already takes the target as a
  parameter, which is why the clamp is a one-line change: pass
  `min(slice_len - END_GUARD_SECONDS, max_end - start)`.
- **The `cancel` keyword added by PR #4** to both files is the shape `max_end` should
  copy.

### `patmir/ui/stream.py`

- **`:402`** — `index_counter = {"n": opening.index}`, seeded from section 0 on every
  run including a resume, while prior coverage is restored a few lines below. The
  index-reuse bug.
- **`set_playhead`** — write-only, no seek event, no clamp to duration, no
  monotonicity check. The server cannot tell a seek from normal advance.
- **`sections_manifest` / `section_file`** — the manifest overlay and the
  index → file resolution that returns the first matching index.

### `patmir/ui/cache.py`

- **`append_section`, `:377-403`** — writes `section_NNN.wav` and the four-field
  manifest entry (`index`, `start`, `end`, `file`; no duration). De-duplicates only
  starts within 0.5 s, which is why overlapping sections both survive.

## Where the drift measurement data already is

### `mux_placements.json`

- **Written by:** `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py:187-193`,
  `:246-249`, per section.
- **Relevance:** carries `source_start`, `placed_start`, `placed_end` and `drift` per
  line — everything needed to compare a seek-initiated section against a
  forward-streamed one without adding instrumentation.
- **The knobs that produce the drift:** `voice_no_overlap`, `stretch_cap=0.20`,
  `voice_max_compress=1.5` (`patmir/pipeline/context.py`), with `_placement_start`
  and `_stretch_to_window` at `ffmpeg_pad_stretch.py:253-277`.

## Spec formatting precedent

### `agent-os/specs/2026-08-24-1300-cuda-baseline-and-asr-segmentation/`

- **Relevance:** the most recent spec folder; this one copies its four-file layout and
  its handling of the empty standards index (`standards.md` records "no standards
  apply" and cites `CLAUDE.md` instead).
