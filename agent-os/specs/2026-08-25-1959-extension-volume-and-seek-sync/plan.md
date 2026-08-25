# Chrome extension: volume un-mutes the original, and seeking desyncs the dub

> **This is the plan as approved, kept unrewritten as the record.** It was
> executed against base `1dadb82`; `origin/main` then moved to `7f8d4a9` and
> subsumed part of it. Three items below did NOT ship as written — the section-index
> seeding (done upstream instead), the out-of-scope pause bug (fixed upstream), and
> the quality-issue numbers (20/21/22 were taken upstream; ours became 25/24/23).
> See the "Rebase note" at the end of `shape.md` for the reconciliation.

## Context

Two bugs reported by the author against the MV3 extension in `extension/`, both in
the in-page Web Audio overlay that plays the Armenian dub against YouTube's
`<video>`. Both affect **both** playback modes (author-confirmed).

**Bug A — changing the volume mixes the original audio back in.** The extension
silences the original with a single `video.muted = true` at start-up
(`extension/content.js:212` quality, `:409` watch-along) and never re-asserts it.
YouTube's slider and speaker icon call the player's `unMute()`, which clears `muted`
on the element — so the original track returns *underneath* the still-playing dub.
There is no `volumechange` listener anywhere in the extension, and no `GainNode`
either: the graph is `AudioBufferSourceNode → audioCtx.destination`, so there is
currently nowhere for a viewer volume to be applied even if we observed it.

**Bug B — seeking desyncs the dub.** Different mechanism per mode:

- *Quality mode has no drift correction at all.* `audioStartCtxTime` /
  `audioStartOffset` (`content.js:33-34`, written at `:249-250`) are dead state with
  no reader. The only seek handling is `seeked() → startSourceAt(video.currentTime)`
  (`:278-280`) via `src.start(0, offset)` — "start now" — which fires while the
  picture is still re-buffering, so the audio runs ahead by the re-buffer duration
  and nothing ever pulls it back. There is no `seeking`, `waiting`, `stalled` or
  `playing` listener. The 1 s drift loop that
  `agent-os/specs/2026-05-27-1200-low-friction-translation-d1/plan.md:153-157`
  specified (hard-resync > 80 ms, soft ±0.5 % nudge 30–80 ms) was never implemented;
  `extension/README.md` still admits "slice 1 has no active correction … toggle
  Stop / Play to resync."
- *Watch-along re-anchors, but on assumptions a seek breaks.* `StreamScheduler._next_start`
  (`patmir/pipeline/streaming/scheduler.py:102-113`) hands `process_section` the raw
  playhead float and **nothing clamps the resulting ~35 s window against an
  already-covered run** — `Coverage` has no "next covered start" query
  (`coverage.py:75`), and `append_section` only de-duplicates starts within 0.5 s
  (`patmir/ui/cache.py:398`). So a forward seek followed by normal play emits sections
  whose spans overlap. The client's `sectionAt` (`content.js:491-496`) then returns
  the **first insertion-order** match out of a `Map` keyed by arrival order, so the
  same wall-clock second can resolve to either rendering — a repeated or skipped
  line. Separately, `scheduleNode` silently returns when
  `offset >= sec.buffer.duration` (`:501`), dropping the tail of any section whose dub
  came out short, and `pollWatchAlongSections` mutates `start`/`end`/`file` in place
  (`:462`) **without invalidating the already-decoded `buffer`**.

Intended outcome: the original audio is inaudible for as long as a dub is active no
matter what the page player does to `muted`; the page volume slider and mute button
drive the dub; and the dub tracks `video.currentTime` within a bounded tolerance
across seeks, re-buffers and rate changes without needing a Stop/Play cycle.

## Decisions locked with the author

1. Both modes are in scope.
2. The page volume slider drives the **dub's** loudness (GainNode).
3. YouTube's mute button mutes the **dub**; the element stays muted throughout.
4. Continuous drift correction, not seek-only.
5. Tests via Node's built-in runner (`node --test`), no npm dependencies — the repo
   has zero JS test infrastructure today (28 Python test modules, no `package.json`).
   There is no Linux `node` in WSL; only the Windows `node.exe` (v26.7.0), which I
   verified runs `--test <file>` from a WSL cwd but fails `--test <directory>`
   ("Could not find '.'"). So **Node gets installed in WSL** as a prerequisite.
6. The watch-along server-side root cause (overlapping post-seek sections, resumed
   index reuse) is in scope. VAD-snapping the seek start is **not** committed to —
   it gets measured first.

## Base revision

This branch (`VardanK/minor-bug-fix`) is at `5942acb`; `origin/main` has since moved
to `1dadb82` (accelerator lock + cooperative job cancellation, PR #4). **Rebase onto
`1dadb82` before touching anything** — it changes three of the files this plan edits:

- `patmir/ui/stream.py` (+87): `index_counter = {"n": opening.index}` moves from
  `:338` to **`:402`** and is still unfixed. `StreamManager` gained an
  `accelerator_lock`, a `_cancels` token map, `_run_locked`, and `cancelling` /
  `cancelled` / `waiting` statuses.
- `patmir/pipeline/streaming/section.py` and `prelude.py`: `process_section`,
  `dub_prelude_window` and `run_prelude` each gained a
  `cancel: threading.Event | None = None` keyword. The new `max_end` parameter should
  be added the same way — keyword-only, defaulting to `None`.
- `extension/popup.js`: two new `"waiting"` status lines. Irrelevant here.

`extension/content.js` is **untouched** on main, so every `content.js` line reference
in this plan is still accurate.

## Approach

### 1. Extract the decision logic into a testable pure module

`extension/lib/sync.js`, a classic script with a UMD-style shim so the same file
serves as an MV3 content script (assigns `globalThis.PatmirSync`) and as a Node
`require()` target. `manifest.json` gets `"js": ["lib/sync.js", "content.js"]` —
content scripts load in order into the same isolated world, so `content.js` reads
`globalThis.PatmirSync`. No build step, no ES module syntax.

Pure functions to live there (no DOM, no Web Audio, no timers):

- `resolveVolumeEvent({ volume, muted, pendingForcedWrites, viewer })` →
  `{ viewer: {volume, muted}, consumedEcho, needsForceMute }`
- `dubGainFor(viewerVolume, viewerMuted)` → linear gain
- `shouldScheduleAudio({ paused, seeking, readyState, advanced })` → bool
- `driftDecision({ samples, rate, thresholds })` → `{ action: "none"|"nudge"|"reanchor", rateMultiplier }`
- `pickSection(sections, c)` → the section to play at time `c`

### 2. Bug A — dub gain + mute enforcement (`content.js`, both modes)

**Graph.** Create `dubGain = audioCtx.createGain()` once next to the `audioCtx`
(`content.js:188` / `:398`) and `connect(audioCtx.destination)`. Both
`startSourceAt` (`:246`) and `scheduleNode` (`:510`) connect their source to
`dubGain` instead of `destination`.

**Discriminating our own mute write from the viewer's.** A synchronous
pending-write counter, exact rather than time-based — better than the 400 ms
`waProgrammatic` timeout already in the file (`:388`, `:570-575`), and better than
comparing the written `(volume, muted)` pair (which cannot tell our forced
`muted = true` from the viewer pressing M):

```
forceMute(): if video.muted is already true -> no write, no event, return.
             else pendingForcedWrites++, video.muted = true
onVolumeChange(): if pendingForcedWrites > 0 && video.muted -> consume one, return
                  else viewer.{volume,muted} = video.{volume,muted}
                       applyGain(); forceMute()
```

The counter stays balanced because `forceMute` only writes when `muted` is
currently `false`, and that write always emits exactly one `volumechange`
(assigning `true` to an already-`true` `muted` fires nothing).

**Gain.** `dubGain.gain` follows `viewer.muted ? 0 : viewer.volume`, applied with a
~20 ms ramp (`setTargetAtTime`) so slider drags don't click. Linear from
`video.volume` deliberately: YouTube already maps its slider curve into
`video.volume`, so a linear read reproduces the loudness the original would have had.

**Teardown, and the `originalVolume` bug.** `originalVolume` is a module-level
singleton guarded by `if (originalVolume === null)` (`:211`, `:408`) that is **never
reset to null** in `stopDub`/`stopWatchAlong` — so after an SPA navigation a later
stop writes the *first* video's volume onto a different element. Fix by deleting the
volume save/restore entirely: we never write `video.volume`, so the element already
holds whatever the viewer chose. Restore `video.muted = viewer.muted` (viewer intent)
rather than the stale `wasMuted`, and reset the per-video state on stop.

`volumechange` joins both `handlers` (`:267-293`) and `waHandlers` (`:594-621`).

### 3. Bug B, client — one shared drift engine for both modes

**Only play audio while the picture is actually moving.** Gate every schedule on
`shouldScheduleAudio`: not paused, not `seeking`, `readyState >= HAVE_FUTURE_DATA`,
**and** `currentTime` advanced since the previous tick. The advance check is the
load-bearing one — it does not depend on YouTube firing `waiting`/`stalled`
reliably, and it is exactly what the current code lacks. New listeners: `seeking`
(stop audio, mark pending), `waiting`, `stalled`, `playing`; `seeked` records the
target but no longer starts audio itself — the tick does, once the video moves.

**Measure, then correct.** One 250 ms interval serves both modes (watch-along
already has one at `:417`). Quality mode finally reads its anchors:
`expected = video.currentTime`, `actual = audioStartOffset + (audioCtx.currentTime −
audioStartCtxTime) × rate`. Watch-along keeps its per-section form (`:523-524`).

**Anti-thrash, because a hard re-anchor is an audible cut** (`scheduleNode` calls
`stopAllNodes()` first) and `video.currentTime` is coarsely quantized — a naive
80 ms threshold against a quantized clock re-anchors spuriously. `driftDecision`
therefore uses: the median of the last 5 samples, samples taken only on ticks where
the video progressed, a soft band (~40–250 ms) corrected by nudging
`playbackRate` ±0.5 % (≈8 cents of pitch, inaudible), a hard band (>250 ms)
re-anchoring only after 3 consecutive confirmations, and no action inside 40 ms.
`WA_DRIFT = 0.08` (`:393`) is replaced by this soft/hard pair.

**Watch-along section selection.** `pickSection` iterates in **start order** and
returns the greatest `start` that contains `c` — the most recently produced covering
section — instead of the first arrival. `pollWatchAlongSections` invalidates
`sec.buffer` when `start`/`end`/`file` change (`:462`) and drops entries no longer in
the manifest. `scheduleNode`'s `offset >= buffer.duration` bail (`:501`) falls through
to the next section rather than going silent.

### 4. Bug B, server — stop emitting overlapping sections

`patmir/pipeline/streaming/`:

- Add `Coverage.next_covered_start(after)` (`coverage.py`) — the query that does not
  exist today.
- `StreamScheduler._next_start` (`scheduler.py:102-113`) returns a `(start, max_end)`
  pair; `process_section` (`section.py:195-259`) takes `max_end`. The clamp is one
  line: `_snap_window_end(job_dir, target)` (`prelude.py:58-67`) already takes the
  target as a parameter, so `target` becomes
  `min(slice_len - END_GUARD_SECONDS, max_end - start)` — a seek-initiated window
  then snaps to a VAD boundary at or before the next covered run instead of running
  through it.
- `patmir/ui/stream.py:402` (post-rebase) is `index_counter = {"n": opening.index}` — seeded from
  section 0 on **every** run including a resume, even though prior *coverage* is
  restored a few lines below. So a resumed run's section 1 overwrites the
  previous run's `section_001.wav`, which sat at a different global `start`, and it is
  served `immutable, max-age=31536000`. Seed from
  `max(opening.index, max(int(s["index"]) for s in prior sections))`. Reachable via
  the popup's "Resume watch-along" (`popup.js:188`) after a server restart.

**Measure before touching the mux.** A seek also starts a section mid-utterance with
no leading VAD silence to absorb `voice_no_overlap` ripple, which is a plausible
second cause of progressive post-seek lag. `mux_placements.json` is already written
per section with a `drift` column (`ffmpeg_pad_stretch.py:187-193`). Compare a
seek-initiated section against a forward-streamed one on the same video **before**
deciding whether to VAD-snap the seek start; do not change the mux on a hypothesis.

### 5. Out of scope (found while investigating, file as issues, do not fix here)

The `scheduler.py:135-141` coverage fudge (marks 5 s/30 s covered when nothing was
emitted → a permanent hole the client freezes on); backfill removed but still
documented at `scheduler.py:15-18`, so `is_complete` never fires after any forward
seek; a >45 s pause killing the scheduler irrecoverably (`scheduler.py:52`,
`:115-122`); boundary-straddling sentences dubbed twice (`section.py:104-118`);
orphaned `section_NNN.wav` files. Worth rows in `quality-issues.md`, not code in a
minor-bug-fix branch.

## Tasks

1. **Rebase onto `origin/main` (`1dadb82`)** and re-check the server-side line
   references above before editing.
2. **Save spec documentation** — `agent-os/specs/<stamp>-extension-volume-and-seek-sync/`
   with `plan.md`, `shape.md`, `standards.md`, `references.md` (no `visuals/` — none
   provided). `agent-os/standards/index.yml` is an empty header, so `standards.md`
   records "no standards apply" and cites `CLAUDE.md`'s maintenance contract instead,
   as the 2026-08-24 spec did.
3. **Install Node in WSL** — `sudo apt install -y nodejs` (or nvm), so
   `node --test extension/tests/` works with directory discovery. Prerequisite for
   task 5; nothing in the shipped extension depends on it.
4. **`extension/lib/sync.js`** — the pure module + UMD shim; wire into
   `manifest.json`.
5. **`extension/tests/sync.test.js`** — `node --test`, covering each pure function's
   edge cases (echo-vs-viewer mute, mute-during-dub, overlapping sections, quantized
   clock thrash, stalled-after-seek).
6. **Bug A in `content.js`** — `dubGain`, `volumechange` in both handler maps,
   `forceMute` counter, teardown/`originalVolume` fix.
7. **Bug B client in `content.js`** — shared tick + drift engine, new media
   listeners, `pickSection`, buffer invalidation.
8. **Bug B server** — `Coverage.next_covered_start`, `_next_start`/`process_section`
   clamp, `index_counter` seed; extend `patmir/tests/test_streaming_coverage.py` and
   `test_stream_cache.py`.
9. **Measure post-seek mux drift** — the `mux_placements.json` comparison above;
   record the numbers in the spec and only then decide on VAD-snapping the seek start.
10. **Docs, per the `CLAUDE.md` maintenance contract** — two new `quality-issues.md`
   rows (#20 volume un-mutes the original, #21 seek desync) plus the out-of-scope
   rows; refresh `extension/README.md`'s stale "Known gotchas / Drift" and "YouTube
   SPA navigation" bullets; an ADR only if the drift-correction design proves
   non-trivial once measured.

## Verification

**Automated:** `node --test extension/tests/` for the pure logic;
`python -m pytest patmir/tests/test_streaming_coverage.py patmir/tests/test_stream_cache.py`
for the server clamp and index seeding; full `python -m pytest patmir/tests/` for no
regressions.

**Manual, in the browser** (`patmir ui --port 8765`, load `extension/` unpacked, token
pasted in Options):

*Bug A, both modes:*
1. Dub a video, start playback. Drag YouTube's volume slider up and down — the
   Armenian dub's loudness must follow it and **no Russian/English speech must ever
   be audible under it**. (Before the fix, the first slider move brings the original
   back.)
2. Click YouTube's mute button → everything goes silent. Unmute → only the dub
   returns.
3. Stop the dub → the original audio returns at the volume the slider is showing.
4. Navigate to a second video via the SPA and stop again — the first video's volume
   must not be written onto it.

*Bug B, quality mode:*
5. Seek forward into an unbuffered region while playing. The dub must not start until
   the picture moves, and must be lip-locked when it does — not ahead by the
   re-buffer duration.
6. Seek backward, seek while paused then press play, and set playback rate to 0.5×
   and 2×. Sync holds in each case with no audible cut on a normal seek.
7. Let a 5-minute dub play through untouched: no accumulating drift, and no periodic
   audible re-anchor cuts (the anti-thrash check).

*Bug B, watch-along:*
8. Start watch-along, let it produce a few sections, then seek forward past the
   dubbed region. The video buffers (expected), then resumes in sync.
9. Seek back into the region the forward seek skipped, then let it play forward
   across the boundary between the pre- and post-seek sections. No line should repeat
   or be skipped — this is the overlapping-section case.
10. Restart `patmir ui` mid-stream and hit "Resume watch-along": no section's audio
    should play at a different video position than the one it was dubbed from.
