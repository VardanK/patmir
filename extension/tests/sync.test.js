// Unit tests for extension/lib/sync.js -- the pure decision logic behind the
// content script's audio overlay.
//
//   node --test extension/tests/
//
// No dependencies: Node's built-in runner and assert only. Every case below is
// a behaviour the two reported bugs turned on, so a simplification that breaks
// one of them fails here rather than in someone's ears.

const { test } = require("node:test");
const assert = require("node:assert/strict");

const sync = require("../lib/sync.js");
const {
  resolveVolumeEvent, dubGainFor, shouldScheduleAudio, driftDecision,
  pickSection, isCovered, median, clamp, DEFAULT_THRESHOLDS, HAVE_FUTURE_DATA,
} = sync;

// --- helpers ---------------------------------------------------------

const viewerOf = (volume, muted) => ({ volume, muted });

// =====================================================================
// resolveVolumeEvent -- telling our own forced mute from the viewer's
// =====================================================================

test("a forced-mute echo is consumed and leaves viewer intent alone", () => {
  const before = viewerOf(0.7, false);
  const r = resolveVolumeEvent({ volume: 0.7, muted: true }, 1, before);
  assert.equal(r.consumedEcho, true);
  assert.equal(r.pendingForcedWrites, 0);
  assert.equal(r.needsForceMute, false);
  assert.deepEqual(r.viewer, before, "an echo must not be read as viewer intent");
});

test("the viewer pressing mute is NOT swallowed as an echo when none is pending", () => {
  // The failure this pins: treating every muted===true event as our own would
  // leave the dub audible while the page UI shows muted.
  const r = resolveVolumeEvent({ volume: 0.7, muted: true }, 0, viewerOf(0.7, false));
  assert.equal(r.consumedEcho, false);
  assert.deepEqual(r.viewer, { volume: 0.7, muted: true });
  assert.equal(dubGainFor(r.viewer.volume, r.viewer.muted), 0, "mute must silence the dub");
  assert.equal(r.needsForceMute, false, "already muted -- no write, so no echo to expect");
});

test("the page unmuting (what a slider drag does) is re-muted and drives the dub", () => {
  // This is bug A in one assertion: YouTube's setVolume() calls unMute(), so
  // the element goes audible underneath the dub unless we re-assert.
  const r = resolveVolumeEvent({ volume: 0.42, muted: false }, 0, viewerOf(0.7, true));
  assert.equal(r.needsForceMute, true);
  assert.deepEqual(r.viewer, { volume: 0.42, muted: false });
  assert.equal(dubGainFor(r.viewer.volume, r.viewer.muted), 0.42);
});

test("pending writes never go negative and a non-muted event never consumes one", () => {
  const r = resolveVolumeEvent({ volume: 0.5, muted: false }, 2, viewerOf(0.5, false));
  assert.equal(r.consumedEcho, false);
  assert.equal(r.pendingForcedWrites, 2, "only a muted===true event can be our echo");
  const z = resolveVolumeEvent({ volume: 0.5, muted: true }, 0, viewerOf(0.5, false));
  assert.equal(z.pendingForcedWrites, 0);
});

test("a full slider drag settles with the dub tracking it and the element muted", () => {
  // Drive the state machine the way the player does: unmute + volume, then our
  // forced re-mute echoing back, repeatedly.
  let pending = 0;
  let viewer = viewerOf(0.0, true);
  for (const v of [0.2, 0.5, 0.9]) {
    let r = resolveVolumeEvent({ volume: v, muted: false }, pending, viewer);
    viewer = r.viewer; pending = r.pendingForcedWrites;
    assert.equal(r.needsForceMute, true);
    pending += 1;                                    // caller writes muted = true
    r = resolveVolumeEvent({ volume: v, muted: true }, pending, viewer);   // echo
    viewer = r.viewer; pending = r.pendingForcedWrites;
    assert.equal(r.consumedEcho, true);
    assert.equal(pending, 0, "every forced write is accounted for");
  }
  assert.deepEqual(viewer, { volume: 0.9, muted: false });
  assert.equal(dubGainFor(viewer.volume, viewer.muted), 0.9);
});

test("volume is clamped and NaN-safe", () => {
  assert.equal(resolveVolumeEvent({ volume: 1.5, muted: false }, 0, viewerOf(1, false)).viewer.volume, 1);
  assert.equal(resolveVolumeEvent({ volume: -1, muted: false }, 0, viewerOf(1, false)).viewer.volume, 0);
  assert.equal(clamp(NaN, 0, 1), 0);
  assert.equal(dubGainFor(NaN, false), 0);
});

// =====================================================================
// dubGainFor
// =====================================================================

test("muted always wins over volume", () => {
  assert.equal(dubGainFor(1, true), 0);
  assert.equal(dubGainFor(0, false), 0);
  assert.equal(dubGainFor(0.33, false), 0.33);
});

// =====================================================================
// shouldScheduleAudio -- the stalled-after-seek guard
// =====================================================================

test("audio does not start while the picture is frozen after a seek", () => {
  // The element lies here: paused === false and readyState healthy, but
  // currentTime is not moving. Starting now is bug B in quality mode.
  assert.equal(shouldScheduleAudio({
    paused: false, seeking: false, readyState: HAVE_FUTURE_DATA, advanced: false,
  }), false);
});

test("audio starts once the clock actually moves", () => {
  assert.equal(shouldScheduleAudio({
    paused: false, seeking: false, readyState: HAVE_FUTURE_DATA, advanced: true,
  }), true);
});

test("no audio while seeking, paused, or under-buffered", () => {
  const ok = { paused: false, seeking: false, readyState: 4, advanced: true };
  assert.equal(shouldScheduleAudio({ ...ok, seeking: true }), false);
  assert.equal(shouldScheduleAudio({ ...ok, paused: true }), false);
  assert.equal(shouldScheduleAudio({ ...ok, readyState: 2 }), false);
  assert.equal(shouldScheduleAudio({ ...ok, readyState: undefined }), false);
});

test("an unknown advance (first tick) costs one tick rather than starting blind", () => {
  assert.equal(shouldScheduleAudio({
    paused: false, seeking: false, readyState: 4, advanced: null,
  }), false);
});

// =====================================================================
// driftDecision -- correct real drift, ignore quantization noise
// =====================================================================

test("small drift inside the soft band does nothing", () => {
  const r = driftDecision([0.01, 0.02, -0.01, 0.03, 0.0]);
  assert.equal(r.action, "none");
  assert.equal(r.rateMultiplier, 1);
});

test("audio behind the picture speeds up; ahead slows down", () => {
  const behind = driftDecision([0.10, 0.11, 0.09, 0.10, 0.10]);
  assert.equal(behind.action, "nudge");
  assert.ok(behind.rateMultiplier > 1, "behind -> play faster");

  const ahead = driftDecision([-0.10, -0.11, -0.09, -0.10, -0.10]);
  assert.equal(ahead.action, "nudge");
  assert.ok(ahead.rateMultiplier < 1, "ahead -> play slower");
});

test("the nudge is inaudibly small", () => {
  const r = driftDecision([0.1, 0.1, 0.1]);
  assert.ok(Math.abs(r.rateMultiplier - 1) <= 0.01,
    "a resampling nudge big enough to hear as pitch is worse than the drift");
});

test("a single quantization outlier does NOT buy an audible re-anchor", () => {
  // The thrash case: video.currentTime steps coarsely, so one sample can spike
  // well past `hard` while the true drift is nil. A re-anchor stops and
  // restarts the source -- an audible cut -- so one sample must not cause it.
  const r = driftDecision([0.0, 0.01, 0.9, 0.01, 0.0]);
  assert.notEqual(r.action, "reanchor");
});

test("sustained large drift re-anchors after the required confirmations", () => {
  const r = driftDecision([0.02, 0.5, 0.6, 0.7, 0.8]);
  assert.equal(r.action, "reanchor");
});

test("large drift that has not yet confirmed holds off", () => {
  // Median is over `hard`, but the tail does not all agree yet.
  const r = driftDecision([0.6, 0.7, 0.8, 0.02, 0.0]);
  assert.notEqual(r.action, "reanchor");
});

test("large drift flipping sign does not re-anchor", () => {
  const r = driftDecision([0.6, -0.7, 0.8, -0.9, 0.7]);
  assert.notEqual(r.action, "reanchor");
});

test("no samples yields no action", () => {
  assert.equal(driftDecision([]).action, "none");
  assert.equal(driftDecision([]).drift, null);
});

test("thresholds are overridable and the defaults are ordered sanely", () => {
  const r = driftDecision([0.05, 0.05, 0.05], { soft: 0.01, hard: 0.02, confirmations: 3 });
  assert.equal(r.action, "reanchor");
  assert.ok(DEFAULT_THRESHOLDS.soft < DEFAULT_THRESHOLDS.hard);
  assert.ok(DEFAULT_THRESHOLDS.soft > 0);
});

test("median handles even and odd lengths", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.equal(median([]), null);
});

// =====================================================================
// pickSection -- overlapping sections and short buffers
// =====================================================================

test("a plain non-overlapping timeline resolves to the containing section", () => {
  const secs = [
    { index: 0, start: 0, end: 40 },
    { index: 1, start: 40, end: 75 },
  ];
  assert.equal(pickSection(secs, 50).index, 1);
  assert.equal(pickSection(secs, 10).index, 0);
});

test("overlapping sections resolve to the later start, not to arrival order", () => {
  // Bug B, watch-along: seeking to 170 while [190,226] was already dubbed makes
  // the scheduler emit [170,206], which overlaps it. Stored in arrival order,
  // "first match" at t=195 could return either. The later start is the one
  // whose neighbours line up.
  const arrivedOutOfOrder = [
    { index: 4, start: 190, end: 226 },
    { index: 7, start: 170, end: 206 },
  ];
  assert.equal(pickSection(arrivedOutOfOrder, 195).index, 4);
  assert.equal(pickSection(arrivedOutOfOrder, 175).index, 7);
});

test("the choice does not depend on the order the sections were polled in", () => {
  const a = [{ index: 4, start: 190, end: 226 }, { index: 7, start: 170, end: 206 }];
  const b = [{ index: 7, start: 170, end: 206 }, { index: 4, start: 190, end: 226 }];
  assert.equal(pickSection(a, 195).index, pickSection(b, 195).index);
});

test("a section whose audio is exhausted is skipped for the next one", () => {
  // The WAV can be shorter than end-start (the accompaniment gets truncated in
  // vocals-only separation and nothing pads it back). Playing past the buffer
  // is silence, so fall through instead.
  const secs = [
    { index: 0, start: 0, end: 40, duration: 30 },
    { index: 1, start: 40, end: 75, duration: 35 },
  ];
  const picked = pickSection(secs, 35);
  assert.equal(picked.index, 1, "exhausted section must not win; schedule the next ahead");
});

test("a section with audio left is still chosen", () => {
  const secs = [{ index: 0, start: 0, end: 40, duration: 30 }];
  assert.equal(pickSection(secs, 20).index, 0);
});

test("sections with no decoded buffer yet are still eligible", () => {
  const secs = [{ index: 3, start: 10, end: 45 }];   // no `duration`
  assert.equal(pickSection(secs, 20).index, 3);
});

test("a gap ahead of the playhead reports the next section, which does not cover it", () => {
  const secs = [{ index: 0, start: 0, end: 40 }, { index: 5, start: 120, end: 155 }];
  const picked = pickSection(secs, 60);
  assert.equal(picked.index, 5);
  assert.ok(picked.start > 60);
  // The caller must not treat this as playable -- isCovered is what says so.
  assert.equal(isCovered(secs, 60), false);
});

// =====================================================================
// isCovered -- "no audio yet" vs "no audio ever", which must not be confused
// =====================================================================

test("an uncovered playhead is an underrun: the caller pauses to buffer", () => {
  assert.equal(isCovered([{ start: 0, end: 40 }], 60), false);
  assert.equal(isCovered([], 0), false);
  assert.equal(isCovered(null, 0), false);
});

test("a covered playhead is covered even when its section has no audio left", () => {
  // The wedge this prevents: a section's WAV can be shorter than its span.
  // pickSection rightly refuses to play it, but pausing to "buffer" would hang
  // forever -- the playhead has to move PAST the short section for the next one
  // to become current, and a paused playhead never moves.
  const secs = [{ index: 0, start: 0, end: 40, duration: 30 }];
  assert.equal(pickSection(secs, 35), null, "nothing playable here");
  assert.equal(isCovered(secs, 35), true, "...but it is covered, so play on silently");
});

test("isCovered honours the same 50 ms lead as pickSection", () => {
  assert.equal(isCovered([{ start: 80, end: 115 }], 79.98), true);
  assert.equal(isCovered([{ start: 80, end: 115 }], 79.0), false);
});

test("the end of a span is exclusive", () => {
  assert.equal(isCovered([{ start: 0, end: 40 }], 40), false);
  assert.equal(isCovered([{ start: 0, end: 40 }], 39.999), true);
});

test("nothing left to play returns null", () => {
  assert.equal(pickSection([{ index: 0, start: 0, end: 40 }], 100), null);
  assert.equal(pickSection([], 5), null);
  assert.equal(pickSection(null, 5), null);
});

test("the 50 ms lead lets a section start marginally early", () => {
  const secs = [{ index: 2, start: 80, end: 115 }];
  assert.equal(pickSection(secs, 79.98).index, 2, "within the lead");
  assert.equal(pickSection(secs, 79.0).index, 2, "before it: scheduled ahead, same section");
});


// --- dead-stream release ------------------------------------------------
//
// Watch-along pauses the <video> on a genuine underrun and resumes when the
// section lands. If the server-side stream DIES no section ever lands, and the
// client re-pauses on every 250 ms tick — so pressing play stops the video a
// quarter second later, forever, with no error anywhere. The author's report:
// "video hang up on me and won't continue, no errors, if I press play it
// automatically stops."

test("a starved stream is abandoned once nothing has arrived for the limit", () => {
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, msSinceProgress: 61000 }), true);
});

test("a merely slow section is not abandoned", () => {
  // Production runs ~2.6x realtime over ~36 s sections, so one lands roughly
  // every 14 s. Twenty seconds of silence is a slow section, not a dead one.
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, msSinceProgress: 20000 }), false);
});

test("exactly at the limit counts as abandoned", () => {
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, msSinceProgress: sync.STREAM_STALL_MS }), true);
});

test("a viewer who is not starved is never abandoned", () => {
  // Playing happily, or paused by the viewer: there is nothing to give up on.
  assert.equal(sync.shouldAbandonStream(
    { buffering: false, msSinceProgress: 999999 }), false);
});

test("a complete stream is never abandoned", () => {
  // Everything is on disk; a gap here is a short section, not a dead server.
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, complete: true, msSinceProgress: 999999 }), false);
});

test("missing timing information does not abandon", () => {
  // Better to keep waiting than to drop the dub on a bookkeeping gap.
  assert.equal(sync.shouldAbandonStream({ buffering: true }), false);
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, msSinceProgress: NaN }), false);
});

test("the limit is overridable, so a caller can be stricter", () => {
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, msSinceProgress: 5000, limitMs: 4000 }), true);
});

test("no state at all is safe", () => {
  assert.equal(sync.shouldAbandonStream(), false);
  assert.equal(sync.shouldAbandonStream(null), false);
});

test("a stream queued behind another job is not abandoned", () => {
  // Only one pipeline fits on the card, so a second stream reports `waiting`
  // and produces nothing until the first finishes. The 60 s rule read that as
  // a dead stream and abandoned it, the revive re-submitted, and it was
  // abandoned again — submit/stop pairs seconds apart, and a video that
  // stopped mid-translation. Two machines watching at once surfaced it.
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, msSinceProgress: 70000, waiting: true }), false);
});

test("a genuinely dead stream is still abandoned", () => {
  // The waiting exemption must not disable the release it guards.
  assert.equal(sync.shouldAbandonStream(
    { buffering: true, msSinceProgress: 70000, waiting: false }), true);
});
