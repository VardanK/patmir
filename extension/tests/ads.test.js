// Ad detection and auto-skip — the pure half.
//
// Both functions exist because ads share the page's single <video> element, so
// during an ad `currentTime` belongs to the ad and every dub anchor computed
// from it is wrong. Before this, a mid-roll did worse than desync: the src swap
// looked like a navigation and tore the dub down for the rest of the video.

const { test } = require("node:test");
const assert = require("node:assert/strict");
const S = require("../lib/sync.js");

// --- classifyAd -------------------------------------------------------------

test("ad-showing on the player is enough on its own", () => {
  const r = S.classifyAd({ playerClasses: "html5-video-player ad-showing playing-mode" });
  assert.equal(r.showing, true);
  assert.equal(r.interrupting, false);
});

test("ad-interrupting also counts", () => {
  const r = S.classifyAd({ playerClasses: "ad-interrupting ad-showing" });
  assert.equal(r.showing, true);
  assert.equal(r.interrupting, true);
});

test("a class merely containing the substring does not match", () => {
  // "no-ad-showing-here" and "ytp-ad-showing-thing" must not trip it; the
  // regex is word-bounded on whitespace for exactly this reason.
  assert.equal(S.classifyAd({ playerClasses: "no-ad-showing-here" }).showing, false);
  assert.equal(S.classifyAd({ playerClasses: "xad-showing" }).showing, false);
});

test("the class list may be an array", () => {
  assert.equal(
    S.classifyAd({ playerClasses: ["html5-video-player", "ad-showing"] }).showing, true);
});

test("an overlay node ALONE is not an ad — this is the bug that shipped", () => {
  // The first version said "any signal is enough: a false positive costs a
  // moment of silence". Both halves were wrong. `.ytp-ad-module` and
  // `.ytp-ad-player-overlay` are containers YouTube leaves in the DOM on every
  // watch page, so this predicate was true on every ordinary video — and a
  // false positive does not cost silence, it authorizes the auto-skip seek,
  // which then sent every video the author opened straight to its end.
  assert.equal(S.classifyAd({ hasAdOverlay: true }).showing, false);
  assert.equal(S.classifyAd({ hasAdPreview: true }).showing, false);
  assert.equal(S.classifyAd({ hasAdOverlay: true, hasAdPreview: true }).showing, false);
});

test("overlay nodes corroborate the class but cannot originate it", () => {
  assert.equal(S.classifyAd({ playerClasses: "ad-showing" }).corroborated, false);
  assert.equal(
    S.classifyAd({ playerClasses: "ad-showing", hasAdOverlay: true }).corroborated, true);
  assert.equal(S.classifyAd({ hasAdOverlay: true }).corroborated, false);
});

test("classConfirmed is set only by the player's own class", () => {
  assert.equal(S.classifyAd({ playerClasses: "ad-showing" }).classConfirmed, true);
  assert.equal(S.classifyAd({ playerClasses: "ad-interrupting" }).classConfirmed, true);
  assert.equal(S.classifyAd({ hasAdOverlay: true, hasAdPreview: true,
                              skipButtonPresent: true }).classConfirmed, false);
});

test("no signals at all means no ad", () => {
  assert.equal(S.classifyAd({}).showing, false);
  assert.equal(S.classifyAd().showing, false);
  assert.equal(S.classifyAd({ playerClasses: "html5-video-player playing-mode" }).showing, false);
});

test("skippable and skipReady are distinct", () => {
  const present = S.classifyAd({ playerClasses: "ad-showing", skipButtonPresent: true });
  assert.equal(present.skippable, true);
  assert.equal(present.skipReady, false, "present but not yet enabled");

  const ready = S.classifyAd({
    playerClasses: "ad-showing", skipButtonPresent: true, skipButtonEnabled: true,
  });
  assert.equal(ready.skipReady, true);
});

test("a skip control with no ad on screen is not skipReady", () => {
  // Stale DOM after an ad ends must not make us click anything.
  const r = S.classifyAd({ skipButtonPresent: true, skipButtonEnabled: true });
  assert.equal(r.showing, false);
  assert.equal(r.skipReady, false);
  assert.equal(r.skippable, false);
});

test("an ordinary watch page with all the empty containers reads as no ad", () => {
  // The exact shape that broke it: YouTube's persistent ad containers present,
  // the player class absent. Whatever else changes, this must stay false.
  const r = S.classifyAd({
    playerClasses: "html5-video-player ytp-hide-controls playing-mode",
    hasAdOverlay: true, hasAdPreview: true,
  });
  assert.equal(r.showing, false);
  assert.equal(r.classConfirmed, false);
});

// --- adSkipPlan -------------------------------------------------------------

// `allowSeek` is opt-in, so every seek case has to ask for it explicitly.
const AD = {
  enabled: true, showing: true, classConfirmed: true, allowSeek: true,
  duration: 30, currentTime: 2,
};

test("disabled does nothing regardless of state", () => {
  const r = S.adSkipPlan({ ...AD, enabled: false, skipReady: true });
  assert.equal(r.action, "none");
  assert.equal(r.reason, "disabled");
});

test("no ad does nothing", () => {
  const r = S.adSkipPlan({ ...AD, showing: false });
  assert.equal(r.action, "none");
  assert.equal(r.reason, "no-ad");
});

test("an ordinary video is never seeked, by four independent routes", () => {
  // The failure that actually happened, blocked at every level it could be.
  // Any ONE of these returning "seek" is the bug back.
  const ordinary = {
    enabled: true, allowSeek: true, duration: 1200, currentTime: 10, attempts: 0,
  };
  const cls = S.classifyAd({
    playerClasses: "html5-video-player playing-mode",
    hasAdOverlay: true, hasAdPreview: true,
  });
  assert.equal(
    S.adSkipPlan({ ...ordinary, showing: cls.showing, classConfirmed: cls.classConfirmed }).action,
    "none", "predicate says no ad");
  assert.equal(
    S.adSkipPlan({ ...ordinary, showing: true, classConfirmed: false }).action,
    "none", "corroboration alone must not authorize it");
  assert.equal(
    S.adSkipPlan({ ...ordinary, showing: true, classConfirmed: true }).action,
    "none", "a 20-minute clip is not an ad");
  assert.equal(
    S.adSkipPlan({ ...ordinary, showing: true, classConfirmed: true, allowSeek: false }).action,
    "none", "and it is opt-in anyway");
});

test("a ready skip control is clicked", () => {
  const r = S.adSkipPlan({ ...AD, skipReady: true });
  assert.equal(r.action, "click");
});

test("clicking is preferred over seeking", () => {
  // The click is the viewer's own gesture a second early; the seek is the
  // fragile fallback. Never take the fragile route when the safe one exists.
  const r = S.adSkipPlan({ ...AD, skipReady: true, attempts: 0 });
  assert.equal(r.action, "click");
});

test("an unskippable ad is seeked to its end", () => {
  const r = S.adSkipPlan(AD);
  assert.equal(r.action, "seek");
  assert.equal(r.target, 30);
});

test("the seek is OFF unless explicitly allowed", () => {
  // A separate opt-in from auto-skip, because clicking a control that only
  // exists during a skippable ad is safe and writing currentTime is not.
  const r = S.adSkipPlan({ ...AD, allowSeek: false });
  assert.equal(r.action, "none");
  assert.equal(r.reason, "seek-not-enabled");
});

test("the seek refuses without the player's own class", () => {
  // Corroborating signals may never authorize it. This is the guard whose
  // absence let a broken predicate move the viewer's video.
  const r = S.adSkipPlan({ ...AD, classConfirmed: false });
  assert.equal(r.action, "none");
  assert.equal(r.reason, "not-class-confirmed");
});

test("the seek refuses anything too long to be an ad", () => {
  // A backstop for a detector bug: if we believe a 20-minute clip is an ad, we
  // are wrong, and the thing we would move is what the viewer came to watch.
  const r = S.adSkipPlan({ ...AD, duration: S.MAX_AD_SECONDS + 1, currentTime: 5 });
  assert.equal(r.action, "none");
  assert.equal(r.reason, "too-long-for-an-ad");
  assert.equal(S.adSkipPlan({ ...AD, duration: S.MAX_AD_SECONDS }).action, "seek");
});

test("clicking still works when the seek is disabled", () => {
  // The safe half must not be coupled to the risky half.
  const r = S.adSkipPlan({ ...AD, allowSeek: false, skipReady: true });
  assert.equal(r.action, "click");
});

test("the seek is capped so a stall cannot become a loop", () => {
  assert.equal(S.adSkipPlan({ ...AD, attempts: 2 }).action, "seek");
  const spent = S.adSkipPlan({ ...AD, attempts: S.DEFAULT_AD_SEEK_ATTEMPTS });
  assert.equal(spent.action, "none");
  assert.equal(spent.reason, "attempts-exhausted");
});

test("maxAttempts is overridable", () => {
  assert.equal(S.adSkipPlan({ ...AD, attempts: 1, maxAttempts: 1 }).action, "none");
});

test("an unknown or infinite duration is never seeked", () => {
  // A live ad, or an element that has not reported duration yet. Seeking to
  // NaN/Infinity is undefined behaviour on the element.
  for (const duration of [undefined, null, NaN, Infinity, 0, -1]) {
    const r = S.adSkipPlan({ ...AD, duration });
    assert.equal(r.action, "none", `duration=${duration}`);
    assert.equal(r.reason, "no-duration");
  }
});

test("an ad that is nearly over is left alone", () => {
  const r = S.adSkipPlan({ ...AD, currentTime: 29.8 });
  assert.equal(r.action, "none");
  assert.equal(r.reason, "almost-over");
});

// --- interaction with audio scheduling --------------------------------------

test("shouldScheduleAudio refuses while an ad shows, however healthy the element", () => {
  // During an ad the element reports paused:false, a full readyState and an
  // advancing clock — it is simply playing something else. So this must be
  // checked independently of every other term.
  const healthy = { paused: false, seeking: false, readyState: 4, advanced: true };
  assert.equal(S.shouldScheduleAudio(healthy), true);
  assert.equal(S.shouldScheduleAudio({ ...healthy, adShowing: true }), false);
});

test("adShowing absent or false leaves the old behaviour untouched", () => {
  const healthy = { paused: false, seeking: false, readyState: 4, advanced: true };
  assert.equal(S.shouldScheduleAudio({ ...healthy, adShowing: false }), true);
  assert.equal(S.shouldScheduleAudio(healthy), true);
});
