// Bootstrap tests for content.js — does it come up, and does it fail safely?
//
// These exist because of a real bug. The `lib/sync.js` dependency check
// originally sat in the middle of the file, AFTER the message listener and the
// auto-play timers were registered. Returning from there left a half-initialized
// module: function declarations hoist, so the listener could still reach
// `startDub`, but every `let` below the return was stranded in the temporal dead
// zone. Clicking Play reported "Cannot access 'syncTickHandle' before
// initialization" — the guard produced exactly the confusing failure it was
// added to prevent.
//
// content.js is a classic content script, not a module, so it is evaluated
// against a stub of the pieces of the extension world it touches.

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const CONTENT = fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8");
const SYNC = fs.readFileSync(path.join(__dirname, "..", "lib", "sync.js"), "utf8");

/** Evaluate content.js in a fresh sandbox, optionally loading lib/sync.js first
 *  (which is what the manifest's content_scripts order does in Chrome). */
function boot(opts) {
  const { withSync } = opts;
  const state = { handler: null, errors: [], timers: [] };
  const sandbox = {
    console: { error: (...a) => state.errors.push(a.join(" ")), warn() {}, log() {} },
    setTimeout: (fn) => { state.timers.push(fn); return state.timers.length; },
    clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    location: { href: "https://www.youtube.com/watch?v=abc" },
    document: {
      querySelectorAll: () => opts.videos || [],
      // Deliberately absent by default (see the DOM-globals guard below); a
      // caller opts in when it wants to drive the ad detector.
      querySelector: opts.querySelector,
      addEventListener() {}, contains: () => true,
    },
    addEventListener() {},
    AudioContext: function () {},
    atob: (b) => Buffer.from(b, "base64").toString("binary"),
    Date,
    chrome: {
      runtime: {
        onMessage: { addListener: (fn) => { state.handler = fn; } },
        sendMessage: async () => ({ ok: false }),
      },
      storage: {
        local: { get: async () => ({}), set: async () => {} },
        onChanged: { addListener() {} },
      },
    },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  const ctx = vm.createContext(sandbox);
  if (withSync) vm.runInContext(SYNC, ctx);
  vm.runInContext(CONTENT, ctx);
  return { state, sandbox };
}

test("with lib/sync.js loaded, the content script comes up and registers", () => {
  const { state, sandbox } = boot({ withSync: true });
  assert.equal(state.errors.length, 0, `unexpected errors: ${state.errors.join(" | ")}`);
  assert.ok(state.handler, "the message listener must be registered");
  assert.equal(sandbox.window.__patmirInjected, true);
  assert.ok(sandbox.PatmirSync, "lib/sync.js must publish PatmirSync into the shared world");
});

test("without lib/sync.js, it bails BEFORE registering anything", () => {
  const { state, sandbox } = boot({ withSync: false });
  assert.equal(state.handler, null,
    "nothing may be registered — a live listener over a half-initialized module " +
    "is what produced the 'Cannot access X before initialization' error");
  assert.equal(state.timers.length, 0, "no auto-play timers either");
  assert.equal(sandbox.window.__patmirInjected, undefined,
    "the injected flag must stay unset so a corrected reload can inject cleanly");
});

test("the missing-dependency message names the cause and the fix", () => {
  const { state } = boot({ withSync: false });
  assert.equal(state.errors.length, 1, "exactly one message, not a stack trace per event");
  const msg = state.errors[0];
  assert.match(msg, /lib\/sync\.js/);
  assert.match(msg, /Reload/i, "must tell the reader a manifest change needs a full reload");
});

test("a second injection is suppressed once the first succeeded", () => {
  const { sandbox } = boot({ withSync: true });
  const ctx = vm.createContext(sandbox);
  let reRegistered = false;
  sandbox.chrome.runtime.onMessage.addListener = () => { reRegistered = true; };
  vm.runInContext(CONTENT, ctx);
  assert.equal(reRegistered, false, "__patmirInjected must guard re-injection");
});

test("the manifest loads lib/sync.js before content.js", () => {
  // The ordering the guard depends on, pinned so a manifest edit cannot
  // silently reintroduce the failure above.
  const m = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "manifest.json"), "utf8"));
  const js = m.content_scripts[0].js;
  assert.ok(js.indexOf("lib/sync.js") >= 0, "lib/sync.js must be listed");
  assert.ok(js.indexOf("lib/sync.js") < js.indexOf("content.js"),
    `lib/sync.js must come first, got ${JSON.stringify(js)}`);
});

test("the manifest's load order satisfies all three dependencies", () => {
  // Content scripts share one isolated world and load in order, so the manifest
  // IS the dependency graph. Three edges, each of which produces a loud bail
  // rather than a subtle failure if broken — but only if the order is right:
  //   lib/sync.js    before content.js  (PatmirSync)
  //   lib/session.js before overlay.js  (PatmirSession)
  //   content.js     before overlay.js  (PatmirPlayer, published at the end)
  const m = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "manifest.json"), "utf8"));
  const js = m.content_scripts[0].js;
  const at = (f) => {
    const i = js.indexOf(f);
    assert.ok(i >= 0, `${f} must be listed, got ${JSON.stringify(js)}`);
    return i;
  };
  assert.ok(at("lib/sync.js") < at("content.js"));
  assert.ok(at("lib/session.js") < at("overlay.js"));
  assert.ok(at("content.js") < at("overlay.js"),
    "overlay.js consumes the PatmirPlayer facade content.js publishes");
});

test("content.js publishes the PatmirPlayer facade, and only after full init", () => {
  const { sandbox } = boot({ withSync: true });
  const facade = sandbox.PatmirPlayer;
  assert.ok(facade, "overlay.js depends on this being published");
  for (const k of ["play", "playWatchAlong", "stop", "status", "findVideo",
                   "canonicalUrl", "onUrlChange", "adState", "applyAdAction"]) {
    assert.equal(typeof facade[k], "function", `PatmirPlayer.${k}`);
  }
});

test("without lib/sync.js, no facade is published either", () => {
  // Same reasoning as the listener: a facade over a half-initialized module
  // would let overlay.js reach into the temporal dead zone.
  const { sandbox } = boot({ withSync: false });
  assert.equal(sandbox.PatmirPlayer, undefined);
});

test("content.js does not claim the channel for messages it does not own", () => {
  // overlay.js registers its own listener in the same isolated world. This
  // listener used to `return true` for EVERY message, including the
  // patmir:session-* ones it never answers — leaving a response port open that
  // nothing here closes.
  const { state } = boot({ withSync: true });
  assert.equal(state.handler({ type: "patmir:session-state" }, {}, () => {}), undefined);
  assert.equal(state.handler({ type: "nonsense" }, {}, () => {}), undefined);
  assert.equal(state.handler(null, {}, () => {}), undefined);
  // ...but still does for its own.
  assert.equal(state.handler({ type: "patmir:status" }, {}, () => {}), true);
});

test("content.js still touches NO DOM-construction globals at boot", () => {
  // The regression guard that keeps button code out of this file. The sandbox
  // above deliberately provides no createElement, no MutationObserver and no
  // CSSStyleSheet; if someone moves the overlay's DOM work in here, this fails
  // immediately and says why.
  const { state, sandbox } = boot({ withSync: true });
  assert.equal(state.errors.length, 0, state.errors.join(" | "));
  assert.equal(typeof sandbox.document.createElement, "undefined");
  assert.equal(typeof sandbox.MutationObserver, "undefined");
  assert.equal(typeof sandbox.CSSStyleSheet, "undefined");
});

// --- the two injection paths must agree ------------------------------
//
// Content scripts reach a page two ways: the manifest's declarative
// `content_scripts` entry, and popup.js's `chrome.scripting.executeScript`
// fallback for tabs that have no listener (normal right after the extension is
// reloaded — scripts are injected at page load, so an already-open tab is left
// orphaned). Those two lists disagreed when `lib/sync.js` was added: the
// fallback still injected `content.js` alone, which bails on the missing
// dependency, so the retry hit the same "Receiving end does not exist" it was
// meant to repair. The list is now derived from the manifest; these tests stop
// it being hardcoded again.

const POPUP = fs.readFileSync(path.join(__dirname, "..", "popup.js"), "utf8");

test("popup.js derives the injected file list from the manifest", () => {
  assert.match(POPUP, /getManifest\(\)\.content_scripts/,
    "the injection fallback must read the manifest, not carry its own list");
});

test("popup.js does not inject content.js on its own", () => {
  assert.doesNotMatch(POPUP, /files:\s*\[\s*["']content\.js["']\s*\]/,
    "injecting content.js without lib/sync.js makes it bail and the retry fail");
});

test("the fallback's hardcoded default still matches the manifest", () => {
  // `contentScriptFiles()` keeps a literal default for the case where the
  // manifest cannot be read. If it drifts from the real manifest it is a trap.
  const m = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "manifest.json"), "utf8"));
  const expected = m.content_scripts[0].js;
  const fallback = POPUP.match(/return \(cs && cs\.js\) \|\| (\[[^\]]*\])/);
  assert.ok(fallback, "expected a literal fallback list in contentScriptFiles()");
  assert.deepEqual(JSON.parse(fallback[1].replace(/'/g, '"')), expected,
    "the fallback default must match manifest content_scripts[0].js");
});


// --- the ad facade ----------------------------------------------------
//
// Ads share the page's single <video> element, so during an ad `currentTime` is
// the AD's timeline and the src swap looks like a navigation. Before this,
// an unskippable mid-roll therefore killed the dub for the REST of the video —
// the src-change guard tore it down and nothing brought it back. These pin the
// two halves of the fix that content.js owns: reading the signal, and refusing
// to act on a stale one.

function bootWithPlayer(playerClass, extra) {
  const nodes = Object.assign({ "#movie_player": { className: playerClass } }, extra || {});
  return boot({ withSync: true, querySelector: (sel) => nodes[sel] || null });
}

test("adState reads ad-showing off the player", () => {
  const { sandbox } = bootWithPlayer("html5-video-player ad-showing playing-mode");
  assert.equal(sandbox.PatmirPlayer.adState().showing, true);
});

test("adState reports no ad on an ordinary player", () => {
  const { sandbox } = bootWithPlayer("html5-video-player playing-mode");
  assert.equal(sandbox.PatmirPlayer.adState().showing, false);
});

test("adState survives a DOM with no querySelector at all", () => {
  // It runs from the sync tick on every page, including ones mid-teardown.
  const { sandbox } = boot({ withSync: true });
  assert.doesNotThrow(() => sandbox.PatmirPlayer.adState());
  assert.equal(sandbox.PatmirPlayer.adState().showing, false);
});

test("adState sees a skip control only when it is actually clickable", () => {
  const skip = { disabled: false, offsetParent: {}, click() {} };
  const ready = bootWithPlayer("ad-showing", { ".ytp-skip-ad-button": skip });
  assert.equal(ready.sandbox.PatmirPlayer.adState().skipReady, true);

  skip.disabled = true;
  const notYet = bootWithPlayer("ad-showing", { ".ytp-skip-ad-button": skip });
  assert.equal(notYet.sandbox.PatmirPlayer.adState().skipReady, false);
  assert.equal(notYet.sandbox.PatmirPlayer.adState().skippable, true,
    "present-but-counting-down is still 'skippable'");
});

test("applyAdAction clicks a live skip control", () => {
  let clicked = 0;
  const skip = { disabled: false, offsetParent: {}, click() { clicked += 1; } };
  const { sandbox } = bootWithPlayer("ad-showing", { ".ytp-skip-ad-button": skip });
  assert.equal(sandbox.PatmirPlayer.applyAdAction("click"), true);
  assert.equal(clicked, 1);
});

test("applyAdAction REFUSES to seek when no ad is on screen", () => {
  // The single destructive failure available in this feature: a seek that lands
  // on the content video. The guard is re-checked here, against live DOM,
  // immediately before acting — not inherited from whatever the caller decided
  // a tick ago.
  const video = { duration: 100, currentTime: 5, currentSrc: "blob:x" };
  const { sandbox } = boot({
    withSync: true,
    videos: [video],
    querySelector: (sel) => (sel === "#movie_player" ? { className: "html5-video-player" } : null),
  });
  assert.equal(sandbox.PatmirPlayer.applyAdAction("seek"), false);
  assert.equal(video.currentTime, 5, "the content video must not move");
});

test("applyAdAction seeks to the end of an ad that IS on screen", () => {
  const video = { duration: 30, currentTime: 2, currentSrc: "blob:ad" };
  const { sandbox } = boot({
    withSync: true,
    videos: [video],
    querySelector: (sel) => (sel === "#movie_player" ? { className: "ad-showing" } : null),
  });
  assert.equal(sandbox.PatmirPlayer.applyAdAction("seek"), true);
  assert.equal(video.currentTime, 30);
});

test("adState is false on an ordinary page carrying YouTube's empty ad containers", () => {
  // The shipped bug, at the layer that produced it: `!!querySelector(...)` was
  // true because YouTube leaves `.ytp-ad-module` in the DOM with no ad running,
  // so the predicate said "ad" on every video and the seek moved every video.
  // The container has to have CONTENT to count for anything now.
  const empty = { children: [], textContent: "" };
  const { sandbox } = boot({
    withSync: true,
    querySelector: (sel) => (sel === "#movie_player"
      ? { className: "html5-video-player playing-mode" }
      : empty),
  });
  const st = sandbox.PatmirPlayer.adState();
  assert.equal(st.showing, false);
  assert.equal(st.classConfirmed, false);
});

test("applyAdAction refuses to seek a clip too long to be an ad", () => {
  // Backstop for a detector bug: if the class says "ad" but the element reports
  // twenty minutes, the class is wrong and this is the viewer's video.
  const video = { duration: 1200, currentTime: 10, currentSrc: "blob:x" };
  const { sandbox } = boot({
    withSync: true,
    videos: [video],
    querySelector: (sel) => (sel === "#movie_player" ? { className: "ad-showing" } : null),
  });
  assert.equal(sandbox.PatmirPlayer.applyAdAction("seek"), false);
  assert.equal(video.currentTime, 10);
});

test("applyAdAction refuses to seek the duration it recorded as the content", () => {
  // Last line of defence, and independent of the class: whatever we believe, an
  // element reporting the length of the video the viewer opened is that video.
  const video = { duration: 42, currentTime: 3, currentSrc: "blob:x" };
  let ad = false;
  const { sandbox } = boot({
    withSync: true,
    videos: [video],
    querySelector: (sel) => (sel === "#movie_player"
      ? { className: ad ? "ad-showing" : "html5-video-player" } : null),
  });
  sandbox.PatmirPlayer.adState();      // no ad yet -> records duration 42
  ad = true;                            // now claim an ad, same element/duration
  assert.equal(sandbox.PatmirPlayer.applyAdAction("seek"), false);
  assert.equal(video.currentTime, 3);
});

test("applyAdAction ignores anything that is not a known action", () => {
  const { sandbox } = bootWithPlayer("ad-showing");
  for (const a of ["stop", "", null, undefined, "SEEK"]) {
    assert.equal(sandbox.PatmirPlayer.applyAdAction(a), false, String(a));
  }
});

