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

  // Ad auto-skip. A seek to the end of an unskippable ad is the only thing that
  // shortens it, and it is the fragile half of the feature -- YouTube sometimes
  // restarts or stalls the ad in response -- so it is capped rather than
  // retried forever, and skipped entirely when the ad is nearly over.
  var DEFAULT_AD_SEEK_ATTEMPTS = 3;
  var AD_SEEK_MIN_REMAINING = 0.5;
  // Longest thing we will accept as "an ad" before refusing to seek it. YouTube
  // ad pods top out well under this; a media element reporting more than three
  // minutes is the content video, and seeking it is the failure this bound
  // exists to stop. A backstop for a detector bug, not a tuning knob.
  var MAX_AD_SECONDS = 180;

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
    // An ad is playing in the same element, so `currentTime` belongs to the ad
    // rather than the content and every anchor computed from it would be wrong.
    // Checked before readyState because during an ad the element is perfectly
    // healthy -- it is just playing something else.
    if (state.adShowing) return false;
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

  // --- YouTube ads ---------------------------------------------------------
  //
  // Ads share the page's single <video> element. Two consequences, and both
  // used to be bugs:
  //
  //  1. `currentTime` during an ad is the AD's timeline. Anchoring the dub to it
  //     puts the audio wherever the ad happens to be.
  //  2. The src swap fires `emptied`/`loadstart`, which the navigation guard
  //     read as "the viewer moved to another video" and answered with a full
  //     teardown. So an unskippable mid-roll killed the dub for the rest of the
  //     video, with no way back short of pressing play again.
  //
  // Detection is by DOM signal rather than by inference from the media element,
  // because the element looks entirely healthy during an ad.

  /**
   * Is an ad on screen, and can it be skipped yet?
   *
   * `signals`, all optional:
   *   playerClasses     -- className (or class list) of #movie_player
   *   hasAdOverlay      -- an ad overlay node is present AND has content
   *   hasAdPreview      -- the "skip in N" countdown is present AND has content
   *   skipButtonPresent -- a skip control exists in the DOM
   *   skipButtonEnabled -- ...and it is clickable right now
   *
   * ONLY `ad-showing` / `ad-interrupting` on #movie_player can establish that an
   * ad is playing. That restriction is not caution, it is a bug fix.
   *
   * The first version treated ANY signal as sufficient, reasoning that "a false
   * positive costs a moment of silence, a false negative costs the desync".
   * That reasoning was wrong twice over. First, the overlay nodes it trusted
   * (`.ytp-ad-module`, `.ytp-ad-player-overlay`) are **persistent empty
   * containers that YouTube ships on every watch page**, so the predicate
   * returned true on every ordinary video, forever. Second, a false positive
   * does NOT merely cost silence: it authorizes the auto-skip seek, which then
   * jumped every video the viewer opened straight to its end. Presence of a
   * container is not evidence of an ad, and a signal that can authorize a
   * destructive action does not get to be liberal.
   *
   * The overlay/preview signals are kept as CORROBORATION -- reported so a
   * caller can require more than the class -- but they can no longer originate
   * `showing` on their own, and the caller must pass "present AND non-empty"
   * rather than "the selector matched".
   */
  function classifyAd(signals) {
    var s = signals || {};
    var classes = s.playerClasses || "";
    if (Array.isArray(classes)) classes = classes.join(" ");
    classes = String(classes);
    var byClass = /(^|\s)ad-showing(\s|$)/.test(classes);
    var interrupting = /(^|\s)ad-interrupting(\s|$)/.test(classes);
    var showing = byClass || interrupting;
    return {
      showing: showing,
      // The player class said so, as opposed to anything inferred. This is the
      // ONLY thing allowed to authorize a seek.
      classConfirmed: showing,
      interrupting: interrupting,
      corroborated: showing && (!!s.hasAdOverlay || !!s.hasAdPreview),
      // "Skippable" means a skip control exists at all -- the countdown implies
      // one is coming. Whether it can be pressed yet is `skipButtonEnabled`.
      skippable: showing && (!!s.skipButtonPresent || !!s.hasAdPreview),
      skipReady: showing && !!s.skipButtonPresent && !!s.skipButtonEnabled,
    };
  }

  /**
   * What to do about the ad currently on screen.
   *
   * Returns `{action, reason}` where action is "click" (press the skip control),
   * "seek" (jump to the end of the ad), or "none".
   *
   * The two actions are NOT equally safe and are no longer treated as though
   * they were. "click" presses a control that only exists during a skippable
   * ad -- it is the viewer's own gesture a second early, and its worst failure
   * is doing nothing. "seek" writes `currentTime` on a shared media element,
   * and if the ad detection is wrong it jumps the viewer's actual video to the
   * end. That is not a hypothetical: the first version shipped with an ad
   * predicate that was true on every watch page, and it did exactly that to
   * every video the author opened.
   *
   * So "seek" now requires, all of them:
   *   - `allowSeek`, which is OFF by default and a separate opt-in from
   *     auto-skip, because it is the only destructive thing here;
   *   - `classConfirmed`, i.e. #movie_player itself says an ad is playing --
   *     never a corroborating node, never an inference;
   *   - a finite, plausible ad duration (ads are short; a long duration is
   *     evidence we are looking at the content video and should stop);
   *   - attempts remaining, so a stalled ad cannot become a loop.
   */
  function adSkipPlan(state) {
    var s = state || {};
    if (!s.enabled) return { action: "none", reason: "disabled" };
    if (!s.showing) return { action: "none", reason: "no-ad" };
    if (s.skipReady) return { action: "click", reason: "skip-control" };
    if (!s.allowSeek) return { action: "none", reason: "seek-not-enabled" };
    // Only the player's own class may authorize a seek.
    if (!s.classConfirmed) return { action: "none", reason: "not-class-confirmed" };
    var max = s.maxAttempts != null ? s.maxAttempts : DEFAULT_AD_SEEK_ATTEMPTS;
    if ((s.attempts || 0) >= max) return { action: "none", reason: "attempts-exhausted" };
    var dur = s.duration;
    if (dur == null || !isFinite(dur) || dur <= 0) {
      return { action: "none", reason: "no-duration" };
    }
    // A duration this long is not an ad. If we get here the detection is wrong,
    // and the thing we are about to seek is the video the viewer came to watch.
    if (dur > MAX_AD_SECONDS) return { action: "none", reason: "too-long-for-an-ad" };
    var t = s.currentTime || 0;
    // Close enough to the end that a seek buys nothing and risks a restart.
    if (dur - t <= AD_SEEK_MIN_REMAINING) {
      return { action: "none", reason: "almost-over" };
    }
    return { action: "seek", reason: "unskippable", target: dur };
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

  // How long the picture may stay frozen waiting for audio that never comes
  // before the dub gives up and hands the video back.
  //
  // Watch-along pauses the <video> on a genuine underrun and resumes when the
  // section lands. If the server-side stream DIES, no section ever lands: the
  // client keeps polling, keeps finding nothing at the playhead, and re-pauses
  // on every 250 ms tick — so pressing play stops the video a quarter second
  // later, forever. That is what a dead stream looked like to the author, with
  // no error anywhere, and nothing in the client noticed.
  //
  // 60 s because production runs ~2.6x realtime over ~36 s sections, so one
  // lands roughly every 14 s; four missed in a row is not a slow section.
  var STREAM_STALL_MS = 60000;

  /**
   * Should the dub give up and let the original audio play?
   *
   * True only while actually starved AND nothing new has arrived for
   * `limitMs`. A paused viewer is not starved, and a complete stream has
   * nothing left to wait for.
   */
  function shouldAbandonStream(state) {
    var s = state || {};
    if (!s.buffering) return false;              // not waiting on anything
    if (s.complete) return false;                // everything is on disk
    // Queued behind another job is NOT dead. Only one pipeline fits on the
    // card, so a second stream reports `waiting` and produces nothing until
    // the first finishes — which looked exactly like a dead stream to the
    // 60 s rule, so it was abandoned, revived, and abandoned again. Two
    // machines watching at once is what surfaced it: submit/stop pairs
    // seconds apart and a video that stopped mid-translation.
    if (s.waiting) return false;
    var limit = typeof s.limitMs === "number" ? s.limitMs : STREAM_STALL_MS;
    var since = s.msSinceProgress;
    if (typeof since !== "number" || !isFinite(since)) return false;
    return since >= limit;
  }

  return {
    STREAM_STALL_MS: STREAM_STALL_MS,
    shouldAbandonStream: shouldAbandonStream,
    HAVE_FUTURE_DATA: HAVE_FUTURE_DATA,
    SECTION_LEAD: SECTION_LEAD,
    DEFAULT_AD_SEEK_ATTEMPTS: DEFAULT_AD_SEEK_ATTEMPTS,
    AD_SEEK_MIN_REMAINING: AD_SEEK_MIN_REMAINING,
    MAX_AD_SECONDS: MAX_AD_SECONDS,
    classifyAd: classifyAd,
    adSkipPlan: adSkipPlan,
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
