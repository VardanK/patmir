# 2026-08-25 — The dub owns the element's audio, and the picture drives the clock

**Status:** accepted, implemented. Client fixes await an ear/eye verdict.

## Context

Two bugs reported against the Chrome extension, both in the in-page Web Audio
overlay, both affecting quality mode and watch-along alike.

**Changing the volume brought the original audio back.** The extension silenced
the original with one `video.muted = true` at start-up and never re-asserted it.
YouTube's slider and speaker icon call the player's own `unMute()`, so the first
time the viewer touched the volume the original track returned underneath the
dub. There was no `volumechange` listener in the extension, and no `GainNode`
anywhere in it either — the graph was `AudioBufferSourceNode → destination`, so
even had we observed a viewer volume there was nowhere to apply it.

**Seeking desynced the dub.** Quality mode had no drift correction at all:
`audioStartCtxTime` / `audioStartOffset` were written on every anchor and read
by nothing. The loop that
`agent-os/specs/2026-05-27-1200-low-friction-translation-d1/plan.md:153-157`
specified — 1 s interval, hard-resync above 80 ms, ±0.5 % nudge between 30 and
80 — was never implemented, and `extension/README.md` still told the viewer to
toggle Stop / Play to resync. The only seek handling was
`seeked() → startSourceAt(video.currentTime)` with `src.start(0, offset)`: start
*now*. A seek is followed by a re-buffer during which the element reports
`paused === false` and a healthy `readyState` while the picture is frozen, so
the dub began playing into a still frame and stayed ahead by the length of the
stall, permanently.

Watch-along did re-anchor, on a flat 80 ms threshold, but chose its section with
`sectionAt`: the **first insertion-order** match from a Map keyed by poll-arrival
order. That only works if section spans never overlap, and they do — see below.

## Decision

### 1. The dub is the audio track; the element is held muted

Everything the dub plays goes through a `dubGain` node whose level follows
`video.volume`, and 0 when the viewer mutes. The element stays muted for as long
as a dub is active, re-asserted on every external `volumechange`.

Linear in `video.volume` on purpose: the player has already mapped its slider
curve into that value, so reading it linearly reproduces the loudness the
original would have had at the same slider position.

### 2. Our own mute write is identified by an exact count, not a timeout

Once we force `muted = true`, an event carrying `muted === true` is ambiguous —
our echo, or the viewer pressing mute. We keep a count of writes issued and not
yet echoed.

The count cannot mis-attribute, because `forceMute` writes only when `muted` is
currently `false`, and per the HTML spec that write fires exactly one
`volumechange` while re-asserting an already-`true` `muted` fires none. This is
deliberately **not** the pattern its neighbour `waProgrammatic` uses (a 400 ms
timeout), which was the right answer for the pause case — where the extension
cannot distinguish its own write precisely — and is the wrong one here, because
a 400 ms window can swallow a genuine viewer mute.

### 3. Audio only starts when the picture is confirmed moving

One 250 ms tick drives both modes. A source is (re)scheduled only when the video
is not paused, not seeking, at `readyState >= HAVE_FUTURE_DATA`, **and**
`currentTime` advanced since the previous tick.

The advance check is the load-bearing term. It needs no cooperation from
`waiting` / `stalled`, which YouTube does not fire reliably, and it is precisely
what the old code lacked. `seeked` no longer starts audio itself; it clears the
anchor and lets the tick decide.

### 4. Drift correction is banded and hysteretic, overriding the D1 spec's 80 ms

- below 40 ms: nothing;
- 40–250 ms: resample by ±0.5 % (~8 cents of pitch, inaudible) and let the error
  walk itself out;
- above 250 ms: hard re-anchor, but only after **3 consecutive agreeing samples**;
- all decisions on the **median of the last 5 samples**, taken only on ticks
  where the video actually advanced.

The D1 spec's flat 80 ms was written against an idealized clock.
`video.currentTime` is quantized in coarse steps, and a hard re-anchor stops and
restarts the buffer source — an audible cut. A tight threshold against a
quantized clock therefore buys a cut in exchange for noise. Widening the band
and requiring confirmation is what makes continuous correction tolerable to
listen to.

### 5. A section's span is clamped so no two renderings of the same second exist

`Coverage.next_covered_start(after)` is new, and `StreamScheduler._next_start`
now returns `(start, max_end)`. `process_section` takes `max_end` and snaps its
VAD-chosen end to the last silence at or before it — the clamp is one line,
because `_snap_window_end` already took its target as a parameter.

Forward streaming is never clamped (nothing is ahead of it). Only a window
opened at a seek target is, which is exactly where the overlap came from.

On the client, `pickSection` takes the **greatest** start covering the playhead
rather than the first arrival, so the choice no longer depends on poll order.

### 6. Not ours: the resumed-index collision

This work independently reached the same `index_counter` fix that
`2026-08-25-watchalong-startup-and-stall-causes` landed the same day, with
better evidence (a real corrupted cache key, `d5af5653d745ed66`). That version
is kept and ours was dropped on rebase — one fix, one implementation.

Worth being precise about why the clamp in §5 is still needed on top of it.
The index fix guarantees no two manifest entries share an **index**, and so no
two share a file. It cannot see two sections that overlap in **span**, because
those have different indices and different starts — which is exactly what a
seek produces, and what `append_section`'s 0.5 s start-dedup was never going to
catch. The two fixes are complementary: one stops a file being overwritten, the
other stops the same seconds being dubbed twice.

## Consequences

- The player's UI shows the video as muted while a dub plays. Deliberate, and
  the mute button still does the obvious thing (it mutes the dub).
- Audio resumes one tick (≤250 ms) after a play or a seek rather than instantly.
  That is the cost of waiting for proof the picture moved, and it is the right
  trade against a permanent offset.
- `WA_DRIFT` is gone; watch-along and quality now share one policy, so they
  cannot drift apart as designs.
- `process_fn` gained an argument. Six test fakes were updated. The signature
  now carries `bench`, `source_lang` and `max_end` from three separate changes
  landing the same day.
- Decision logic moved to `extension/lib/sync.js`, a classic script with a UMD
  shim, listed **before** `content.js` in the manifest. This introduces a second
  test runner (`node --test`) to a Python repo — accepted, because the mute
  arbitration and the drift hysteresis are exactly the kind of logic that is
  cheap to get subtly wrong and expensive to diagnose by ear.

## Alternatives rejected

**`createMediaElementSource(video)` + a zeroed gain**, instead of writing to
`muted` at all. It removes the write-back contest entirely and is the tidier
design on paper. Rejected: it is irreversible for the element's lifetime, the
routing outlives our teardown, and if our AudioContext were ever closed the
page's own video would be left permanently silent. Too much blast radius on
someone else's page for a problem a counter solves.

**Comparing the incoming `(volume, muted)` pair against the last pair we wrote**,
as the echo discriminator. Cannot tell our forced `muted = true` from the viewer
pressing mute, which is the one case that matters.

**Seek-time re-anchor only**, without a continuous loop. Fixes the reported
symptom and leaves slow clock drift over a feature-length film unfixed.

**Changing the mux so a seek-started section begins on a VAD boundary.** A seek
starts a section mid-utterance with no leading silence to absorb
`voice_no_overlap` ripple, which is a plausible second contributor to
progressive lag *within* a post-seek section. `mux_placements.json` already
carries the per-line `drift` needed to compare a seek-initiated section against
a forward one. That measurement has **not been taken**, so the mux was left
alone rather than changed on a hypothesis.

## A bug this design caused, and the fix

The `lib/sync.js` dependency check was first written mid-file, after the
message listener and the auto-play timers were registered. Returning from
there left a **half-initialized module**: function declarations hoist, so the
listener could still reach `startDub`, but every `let` below the return was
stranded in the temporal dead zone. Clicking Play reported
`Cannot access 'syncTickHandle' before initialization` — the guard produced
precisely the confusing failure it was added to prevent, and pointed at the
wrong thing.

Reported from a real browser session; reproduced in a stub world before being
touched. The check now runs at the very top of the IIFE, before anything is
registered, so a missing dependency leaves nothing callable and the only
symptom is one message naming the cause and the fix. `__patmirInjected` is
deliberately left unset on that path so a corrected reload injects cleanly
rather than being suppressed.

Then the same root cause surfaced a second time, in the place the first fix
made reachable. `popup.js` already had a self-healing fallback: if a tab has no
listener — normal right after the extension is reloaded, since content scripts
are injected at page load and an already-open tab is left orphaned — it calls
`chrome.scripting.executeScript` and retries. That fallback injected
**`content.js` alone**. Adding `lib/sync.js` updated `manifest.json` and missed
it, so the fallback injected a file whose first act is to check for `PatmirSync`
and bail; the retry then hit the very `Receiving end does not exist` it existed
to repair. Reported from the browser as exactly that message, with the server
running and the token set — nothing to do with either.

Fixed by deriving the list from `chrome.runtime.getManifest()`, so the two
injection paths cannot disagree again.

The general lesson, worth keeping: splitting a classic content script into two
files buys testability and costs a load-order dependency — and that dependency
has to be honoured everywhere the scripts are injected, not just in the
manifest. Both failures were the same mistake found in two places, and neither
was caught by the 34 unit tests covering the logic, because both were about
*loading* rather than about behaviour. Pinned by
`extension/tests/bootstrap.test.js`, which boots content.js with and without
`lib/sync.js`, asserts nothing registers on the failure path, and asserts the
manifest ordering and the popup's injected list agree with each other.

## Verification

42 JS unit tests (`node --test extension/tests/`) covering the echo-vs-viewer
arbitration, a full slider drag driven through the state machine, the
stalled-after-seek gate, quantization-outlier rejection, and overlapping /
short-buffer section selection. 8 new Python tests: `next_covered_start`
boundaries, the clamp end to end through the scheduler, a property test that no
two emitted sections overlap after two seeks, that the force-progress fudge
cannot claim past the clamp, and resumed index seeding including manifest
ordering and malformed entries. Full local suite: 16 modules pass, 0 fail (10
skip for missing `numpy` / `fastapi` in this environment, unrelated).

**Not yet verified by ear or eye.** The manual checklist is in the spec at
`agent-os/specs/2026-08-25-1959-extension-volume-and-seek-sync/plan.md`.

**Measured against the author's real cache** (5 keys, 47 cached sections),
which changed the evidence behind two claims here:

- The span clamp is **backed by observed data**. Key `d166323b1098fbf7` holds
  two sections overlapping by **27.75 s** (idx4 `[341.03, 370.53]` against idx1
  `[342.78, 378.88]`), and replaying the same-day index dedup over that manifest
  does not remove them — different indices, different files, both survive. That
  is §5's case, seen rather than argued. Bounding it honestly: that pair spans a
  *restart*, and within a single run it is unreachable, so the clamp fixes the
  single-run seek case while what fixes this instance is the deterministic
  `pickSection`. The cross-run route is a third mechanism neither closes; filed.
- `pickSection`'s exhausted-section branch, and the `isCovered` guard that keeps
  it from wedging, are **defensive with no observed backing**: 0 of 47 sections
  have a genuinely short WAV. The two that measured short are the overwrite
  artifact — each file matches the *later* entry's span to within 0.08 s, which
  independently confirms the other ADR's diagnosis.

**Rebased onto `7f8d4a9` after the watch-along startup work landed.** That
change rewrote `_snap_window_end`, which this ADR's clamp feeds. The interaction
was checked rather than assumed: its anti-collapse floor is a fraction of the
target it is *given*, so a clamped window scales the floor with the clamp
instead of measuring itself against the full 38 s slice, and a clamped window
with no qualifying silence falls back to the clamp itself — abutting the covered
run exactly, which is the one boundary that must not be crossed.

*Revisit:* the post-seek mux drift measurement lands and shows the ripple is
material; an ear-test finds the 250 ms nudge band audible as pitch or the
one-tick start latency objectionable; or the 4-speaker-cap work changes what
watch-along sections look like.
