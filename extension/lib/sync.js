// Patmir sync primitives.
//
// The decision logic behind the content script's audio overlay, extracted as
// pure functions so it can be unit-tested without a DOM, a Web Audio graph or a
// timer. `content.js` owns all the side effects; this file only answers
// questions about what those side effects should be.
//
// Loaded two ways:
//   - as an MV3 content script listed BEFORE content.js in manifest.json, where
//     it assigns `globalThis.PatmirSync` (content scripts share one isolated
//     world and load in manifest order);
//   - as a CommonJS module under `node --test`.
// Hence the shim below. Do not convert this file to an ES module — content
// scripts are classic scripts and `import`/`export` would fail to parse.

(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PatmirSync = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // HTMLMediaElement.HAVE_FUTURE_DATA. Named rather than inlined because this
  // file never touches the DOM and so cannot read the constant off an element.
  var HAVE_FUTURE_DATA = 3;

  // Section lookup tolerance: a section may be scheduled up to this far before
  // its nominal start (matches the pre-existing 50 ms slack in `sectionAt`).
  var SECTION_LEAD = 0.05;

  // Drift bands, in seconds of |video.currentTime - audio position|.
  //
  // SOFT is deliberately well above the ~80 ms the D1 spec asked for. That
  // figure was written against an idealized clock; `video.currentTime` is
  // quantized by the player in coarse steps, so an 80 ms hard threshold
  // re-anchors on quantization noise alone -- and a re-anchor stops and
  // restarts the buffer source, which is audible. Below SOFT we do nothing;
  // between SOFT and HARD we resample by NUDGE (~8 cents of pitch, inaudible)
  // and let the error walk back; only above HARD do we take the audible cut,
  // and only after CONFIRMATIONS consecutive agreeing samples.
  var DEFAULT_THRESHOLDS = {
    soft: 0.04,
    hard: 0.25,
    nudge: 0.005,
    confirmations: 3,
  };

  function clamp(v, lo, hi) {
    if (typeof v !== "number" || v !== v) return lo;   // NaN-safe
    return v < lo ? lo : (v > hi ? hi : v);
  }

  /** Median of a numeric array. Returns null for an empty array. */
  function median(values) {
    if (!values || !values.length) return null;
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /**
   * Decide what a `volumechange` event means.
   *
   * The ambiguity this resolves: once we force `video.muted = true`, an event
   * carrying `muted === true` could be our own write echoing back, or the
   * viewer pressing the page's mute button. We tell them apart with a count of
   * writes we have issued and not yet seen echo.
   *
   * The count is exact -- not a timeout, unlike the `waProgrammatic` guard in
   * content.js -- because the caller only ever writes `muted = true` when it is
   * currently `false`, and per the HTML spec that write fires exactly one
   * `volumechange`. Assigning `true` to an already-`true` `muted` fires nothing,
   * so the count can never run ahead of the events.
   *
   * @param {{volume:number, muted:boolean}} event     the element's state now
   * @param {number} pendingForcedWrites               unechoed forced writes
   * @param {{volume:number, muted:boolean}} viewer    last known viewer intent
   * @returns {{viewer:{volume:number,muted:boolean}, pendingForcedWrites:number,
   *            consumedEcho:boolean, needsForceMute:boolean}}
   */
  function resolveVolumeEvent(event, pendingForcedWrites, viewer) {
    var pending = pendingForcedWrites > 0 ? pendingForcedWrites : 0;
    if (pending > 0 && event.muted === true) {
      return {
        viewer: viewer,
        pendingForcedWrites: pending - 1,
        consumedEcho: true,
        needsForceMute: false,
      };
    }
    return {
      viewer: { volume: clamp(event.volume, 0, 1), muted: event.muted === true },
      pendingForcedWrites: pending,
      consumedEcho: false,
      // Anything the viewer (or the page player) did that left the element
      // audible has to be undone: the dub is the audio track now.
      needsForceMute: event.muted !== true,
    };
  }

  /**
   * Gain to apply to the dub, from the viewer's intent for the page player.
   *
   * Linear in `video.volume` on purpose: the player has already mapped its
   * slider curve into that value, so reading it linearly reproduces the
   * loudness the original track would have had at the same slider position.
   */
  function dubGainFor(volume, muted) {
    return muted ? 0 : clamp(volume, 0, 1);
  }

  /**
   * May we (re)schedule audio right now?
   *
   * `advanced` -- did `currentTime` move since the previous tick -- is the
   * load-bearing term. After a seek the element reports `paused === false` and
   * a healthy `readyState` while the picture is still frozen re-buffering; if
   * we start audio then, it runs ahead by the stall and nothing pulls it back.
   * Waiting for the clock to actually move needs no cooperation from the
   * player's `waiting`/`stalled` events, which YouTube does not fire reliably.
   *
   * Pass `advanced: null` when there is no previous sample (first tick after
   * play, or right after a seek): treated as "not confirmed yet", costing one
   * tick before audio starts.
   *
   * This gates STARTING and RE-ANCHORING only. A node already sounding is left
   * alone -- stopping is driven by the explicit pause/seeking/waiting handlers.
   */
  function shouldScheduleAudio(state) {
    if (state.paused) return false;
    if (state.seeking) return false;
    if ((state.readyState || 0) < HAVE_FUTURE_DATA) return false;
    return state.advanced === true;
  }

  /**
   * Turn a window of drift samples into an action.
   *
   * `samples` are `expected - actual` in seconds, oldest first, taken only on
   * ticks where the video actually advanced. Positive means the audio is behind
   * the picture.
   *
   * Returns `rateMultiplier` to apply on top of `video.playbackRate` for the
   * "nudge" action; 1 otherwise.
   */
  function driftDecision(samples, thresholds) {
    var t = thresholds || DEFAULT_THRESHOLDS;
    var soft = t.soft != null ? t.soft : DEFAULT_THRESHOLDS.soft;
    var hard = t.hard != null ? t.hard : DEFAULT_THRESHOLDS.hard;
    var nudge = t.nudge != null ? t.nudge : DEFAULT_THRESHOLDS.nudge;
    var need = t.confirmations != null ? t.confirmations : DEFAULT_THRESHOLDS.confirmations;

    var d = median(samples);
    if (d === null) return { action: "none", rateMultiplier: 1, drift: null };

    if (Math.abs(d) > hard) {
      // Require the last `need` samples to all breach `hard` in the same
      // direction. A single outlier -- a quantization step, a GC pause -- must
      // not buy an audible cut.
      var tail = samples.slice(-need);
      var agree = tail.length >= need && tail.every(function (s) {
        return Math.abs(s) > hard && (s > 0) === (d > 0);
      });
      if (agree) return { action: "reanchor", rateMultiplier: 1, drift: d };
      return { action: "none", rateMultiplier: 1, drift: d };
    }

    if (Math.abs(d) > soft) {
      // Audio behind (d > 0) -> play it faster to catch up.
      return { action: "nudge", rateMultiplier: d > 0 ? 1 + nudge : 1 - nudge, drift: d };
    }

    return { action: "none", rateMultiplier: 1, drift: d };
  }

  /**
   * Choose the section to sound at global time `c`.
   *
   * Two things the naive "first entry that contains c" lookup got wrong:
   *
   *  1. Sections can OVERLAP. A seek makes the scheduler dub a window that runs
   *     through an already-covered run, and both land in the manifest. The
   *     client stores them in a Map keyed by index, populated in poll-arrival
   *     order, so "first match" resolved the same second to either rendering
   *     depending on arrival order -- heard as a repeated or skipped line. We
   *     take the GREATEST start that contains `c`: the most recently produced
   *     covering section, and the one whose neighbours will line up.
   *  2. A section's WAV can be shorter than its declared span. Playing from
   *     `c - start` past the end of the buffer is silence, so a section whose
   *     audio is already exhausted at `c` is not a candidate; we fall through
   *     to the earliest section that starts after `c` and let the caller
   *     schedule it ahead on the Web Audio clock.
   *
   * `sections` entries need `{start, end}` and may carry `duration` (the
   * decoded buffer length) -- omit it when the buffer has not been decoded yet.
   */
  function pickSection(sections, c) {
    var list = (sections || []).slice().sort(function (a, b) { return a.start - b.start; });
    var containing = null;
    var ahead = null;
    for (var i = 0; i < list.length; i++) {
      var s = list[i];
      if (s.start - SECTION_LEAD <= c && c < s.end) {
        var exhausted = s.duration != null && (c - s.start) >= s.duration - 0.001;
        if (!exhausted) containing = s;   // keep going: later starts win
      } else if (s.start > c && ahead === null) {
        ahead = s;
      }
    }
    if (containing) return containing;
    // `ahead` is the earliest start greater than c, because the list is sorted.
    return ahead;
  }

  /**
   * Does any section claim global time `c`, whether or not it has audio to give?
   *
   * The caller needs this apart from `pickSection` to tell two situations
   * apart. No section covers `c` -> the dub has not been produced yet, so pause
   * and wait for it. A section covers `c` but its audio is exhausted -> there
   * is nothing to play AND nothing to wait for, so play on in silence. Pausing
   * in the second case wedges: the playhead has to move past the short section
   * for the next one to become current, and a paused playhead never will.
   */
  function isCovered(sections, c) {
    var list = sections || [];
    for (var i = 0; i < list.length; i++) {
      if (list[i].start - SECTION_LEAD <= c && c < list[i].end) return true;
    }
    return false;
  }

  return {
    HAVE_FUTURE_DATA: HAVE_FUTURE_DATA,
    SECTION_LEAD: SECTION_LEAD,
    DEFAULT_THRESHOLDS: DEFAULT_THRESHOLDS,
    median: median,
    clamp: clamp,
    resolveVolumeEvent: resolveVolumeEvent,
    dubGainFor: dubGainFor,
    shouldScheduleAudio: shouldScheduleAudio,
    driftDecision: driftDecision,
    pickSection: pickSection,
    isCovered: isCovered,
  };
});
