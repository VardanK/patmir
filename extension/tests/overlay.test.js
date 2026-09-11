// overlay.js — mounting, rendering, and the popup's window onto the session.
//
// No jsdom: a ~70-line fake DOM, the same hand-rolled-stubs idiom
// bootstrap.test.js uses for content.js. That is enough because overlay.js only
// creates elements, walks parents, and sets text — the *policy* (which mount
// point, which label) is pure and already tested in session.test.js.

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const OVERLAY = fs.readFileSync(path.join(__dirname, "..", "overlay.js"), "utf8");

// --- mini DOM ---------------------------------------------------------------

class FakeEl {
  constructor(tag) {
    this.tagName = String(tag || "div").toUpperCase();
    this.id = "";
    this.className = "";
    this.children = [];
    this.parentElement = null;
    this.attributes = {};
    this.listeners = {};
    this.disabled = false;
    this.offsetParent = {};
    this._text = "";
    this.textWrites = 0;
    this.styles = {};
    this.shadow = null;
    this.classList = {
      add: (c) => { if (!this.className.includes(c)) this.className = (this.className + " " + c).trim(); },
      remove: (c) => { this.className = this.className.split(/\s+/).filter((x) => x && x !== c).join(" "); },
      contains: (c) => this.className.split(/\s+/).includes(c),
      toggle: (c, on) => (on ? this.classList.add(c) : this.classList.remove(c)),
    };
    this.style = { setProperty: (k, v) => { this.styles[k] = v; } };
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); this.textWrites += 1; }
  get firstChild() { return this.children[0] || null; }
  get nextSibling() {
    if (!this.parentElement) return null;
    const i = this.parentElement.children.indexOf(this);
    return this.parentElement.children[i + 1] || null;
  }
  appendChild(c) { c.parentElement = this; this.children.push(c); return c; }
  insertBefore(c, ref) {
    c.parentElement = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
    return c;
  }
  remove() {
    if (!this.parentElement) return;
    const i = this.parentElement.children.indexOf(this);
    if (i >= 0) this.parentElement.children.splice(i, 1);
    this.parentElement = null;
  }
  attachShadow() { this.shadow = new FakeEl("shadow-root"); this.shadow.host = this; return this.shadow; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); }
  click() {
    // Browsers do not dispatch click on a disabled button; the fake must not
    // either, or a test would "pass" against behaviour the DOM never allows.
    if (this.disabled) return;
    for (const fn of this.listeners.click || []) fn({});
  }
  closest() { return this._closest || null; }
  querySelector(sel) {
    // Only used inside the shadow root, for ".label".
    const want = String(sel).replace(/^\./, "");
    const walk = (n) => {
      for (const c of n.children) {
        if (c.className.split(/\s+/).includes(want)) return c;
        const hit = walk(c);
        if (hit) return hit;
      }
      return null;
    };
    return walk(this);
  }
}

function makeDoc(selectorMap) {
  const body = new FakeEl("body");
  const doc = {
    body,
    _map: { ...selectorMap },
    createElement: (t) => new FakeEl(t),
    querySelector: (sel) => doc._map[sel] || null,
    getElementById: (id) => {
      const walk = (n) => {
        for (const c of n.children) {
          if (c.id === id) return c;
          const hit = walk(c);
          if (hit) return hit;
        }
        return null;
      };
      return walk(body) || Object.values(doc._map).find((e) => e && e.id === id) || null;
    },
    contains: (node) => {
      let n = node;
      while (n) { if (n === body) return true; n = n.parentElement; }
      return false;
    },
    addEventListener() {},
  };
  return doc;
}

// --- harness ----------------------------------------------------------------

const SESSION = require("../lib/session.js");
const SYNC = require("../lib/sync.js");

function boot(opts = {}) {
  const state = {
    errors: [], intervals: [], timeouts: [], msgHandlers: [],
    storageHandlers: [], winListeners: {}, urlCallbacks: [],
    playerCalls: [], adActions: [],
  };
  const doc = opts.doc || makeDoc({
    "ytd-watch-flexy #primary-inner #below": (() => {
      const el = new FakeEl("div"); el.id = "below"; return el;
    })(),
  });
  // The mount target has to be reachable from body for `document.contains`.
  for (const el of Object.values(doc._map)) {
    if (el && !el.parentElement && el !== doc.body) doc.body.appendChild(el);
  }

  const sessionStub = opts.session || null;
  const sandbox = {
    console: { error: (...a) => state.errors.push(a.join(" ")), warn() {}, log() {} },
    setInterval: (fn) => { state.intervals.push(fn); return state.intervals.length; },
    clearInterval() {},
    setTimeout: (fn) => { state.timeouts.push(fn); return state.timeouts.length; },
    clearTimeout() {},
    Date, JSON, Promise, Error, Set, Object, Array, String,
    location: { href: opts.href || "https://www.youtube.com/watch?v=abc" },
    document: doc,
    addEventListener: (ev, fn) => { (state.winListeners[ev] = state.winListeners[ev] || []).push(fn); },
    CSSStyleSheet: opts.noConstructableSheets ? undefined : function CSSStyleSheet() {
      this.replaceSync = () => {};
    },
    chrome: {
      runtime: {
        onMessage: { addListener: (fn) => state.msgHandlers.push(fn) },
        sendMessage: async (m) => { state.playerCalls.push(m); return { ok: true, data: {} }; },
      },
      storage: {
        local: {
          get: async (keys) => {
            const want = Array.isArray(keys) ? keys : [keys];
            const out = {};
            for (const k of want) if (k in (opts.store || {})) out[k] = opts.store[k];
            return out;
          },
          set: async () => {},
        },
        onChanged: { addListener: (fn) => state.storageHandlers.push(fn) },
      },
    },
  };
  if (sandbox.CSSStyleSheet) sandbox.CSSStyleSheet.prototype.replaceSync = () => {};
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.PatmirSession = opts.omitSession ? undefined : (sessionStub ? patchedSession(sessionStub) : SESSION);
  sandbox.PatmirSync = SYNC;
  sandbox.PatmirPlayer = opts.omitPlayer ? undefined : makePlayer(state, opts);

  vm.runInContext(OVERLAY, vm.createContext(sandbox));
  return { state, sandbox, doc };
}

/** Wrap the real module but hand back a caller-supplied session object. */
function patchedSession(stub) {
  return Object.assign({}, SESSION, { createSession: () => stub });
}

function makePlayer(state, opts) {
  return {
    play: async (k) => { state.playerCalls.push({ type: "play", k }); return { ok: true }; },
    playWatchAlong: async (k) => { state.playerCalls.push({ type: "playWatchAlong", k }); return { ok: true }; },
    stop: async () => { state.playerCalls.push({ type: "stop" }); return { ok: true }; },
    status: async () => ({ ok: true, playing: false, hasVideo: true }),
    findVideo: () => opts.video || null,
    canonicalUrl: (h) => h,
    onUrlChange: (cb) => { state.urlCallbacks.push(cb); return () => {}; },
    adState: () => opts.ad || {
      showing: false, classConfirmed: false, skipReady: false, skippable: false,
    },
    applyAdAction: (a) => { state.adActions.push(a); return true; },
  };
}

/** A session stub that records commands and reports a fixed view. */
function stubSession(over = {}) {
  const calls = [];
  const st = {
    phase: "idle", mode: "quality", lookup: { cached: false }, playback: { playing: false, hasVideo: true },
    mountedAt: 0, nowMs: 1000, ...(over.state || {}),
  };
  return {
    calls,
    getState: () => st,
    view: () => SESSION.deriveView(st),
    onChange: () => () => {},
    refresh: async () => { calls.push("refresh"); return st; },
    primary: async () => { calls.push("primary"); return st; },
    start: async () => { calls.push("start"); return st; },
    stop: async () => { calls.push("stop"); return st; },
    regenerate: async () => { calls.push("regenerate"); return st; },
    setMode: async (m) => { calls.push("set-mode:" + m); return st; },
    setAdShowing: (v) => { calls.push("ad:" + v); return true; },
    notePopupOpen: () => { calls.push("popupOpen"); },
    tick: async () => { calls.push("tick"); return { changed: true }; },
  };
}

const flush = () => new Promise((r) => setImmediate(r));

// --- dependency guards ------------------------------------------------------

test("bails loudly, and registers nothing, without lib/session.js", () => {
  const { state, sandbox } = boot({ omitSession: true });
  assert.equal(state.msgHandlers.length, 0);
  assert.equal(state.intervals.length, 0);
  assert.equal(sandbox.window.__patmirOverlayInjected, undefined,
    "leaving the flag unset lets a fixed reload inject cleanly");
  assert.equal(state.errors.length, 1);
  assert.match(state.errors[0], /lib\/session\.js/);
  assert.match(state.errors[0], /Reload/);
});

test("bails loudly without the PatmirPlayer facade, naming the manifest order", () => {
  const { state, sandbox } = boot({ omitPlayer: true });
  assert.equal(state.msgHandlers.length, 0);
  assert.equal(sandbox.window.__patmirOverlayInjected, undefined);
  assert.match(state.errors[0], /content\.js/);
  assert.match(state.errors[0], /BEFORE overlay\.js/);
});

test("a second injection is suppressed", () => {
  const s = stubSession();
  const { sandbox } = boot({ session: s });
  const ctx = vm.createContext(sandbox);
  let re = 0;
  sandbox.chrome.runtime.onMessage.addListener = () => { re += 1; };
  vm.runInContext(OVERLAY, ctx);
  assert.equal(re, 0);
});

// --- mounting ---------------------------------------------------------------

async function mounted(opts = {}) {
  const s = opts.session || stubSession(opts);
  const r = boot({ ...opts, session: s });
  await flush(); await flush(); await flush();
  const root = r.doc.getElementById("patmir-toggle-root");
  return { ...r, session: s, root };
}

test("mounts as the FIRST child of #below on a watch page", async () => {
  const below = new FakeEl("div"); below.id = "below";
  const existing = new FakeEl("div");
  below.appendChild(existing);
  const doc = makeDoc({ "ytd-watch-flexy #primary-inner #below": below });
  const { root } = await mounted({ doc });
  assert.ok(root, "a root must be created");
  assert.equal(below.children[0], root,
    "prepending is what puts the button immediately under the player");
  assert.equal(below.children[1], existing);
});

test("falls down the mount list as selectors disappear", async () => {
  const cases = [
    ["ytd-watch-flexy #primary-inner > #player", "after"],
    ["ytd-watch-metadata", "before"],
    ["#primary-inner", "prepend"],
  ];
  for (const [selector, position] of cases) {
    const target = new FakeEl("div");
    const parent = new FakeEl("div");
    parent.appendChild(target);
    const doc = makeDoc({ [selector]: target });
    doc.body.appendChild(parent);
    const { root } = await mounted({ doc });
    assert.ok(root, `${selector} must be usable`);
    if (position === "prepend") assert.equal(target.children[0], root);
    else assert.equal(root.parentElement, parent);
  }
});

test("uses the selector-free video ancestor when every id is gone", async () => {
  // The last resort, in the spirit of findVideo(): survives a full id rename.
  const box = new FakeEl("div");
  const parent = new FakeEl("div");
  parent.appendChild(box);
  const video = new FakeEl("video");
  video._closest = box;
  const doc = makeDoc({});
  doc.body.appendChild(parent);
  const { root } = await mounted({ doc, video });
  assert.ok(root);
  assert.equal(root.parentElement, parent);
});

test("mounts nothing, and does not throw, when there is nowhere to put it", async () => {
  // Deliberately not document.body: guessing a parent on someone else's page is
  // worse than showing no button, and the popup still works.
  const { root, state } = await mounted({ doc: makeDoc({}) });
  assert.equal(root, null);
  assert.equal(state.errors.length, 0);
});

test("Shorts gets the floating pill instead of an inline block", async () => {
  const { root } = await mounted({
    doc: makeDoc({}), href: "https://www.youtube.com/shorts/abc",
  });
  assert.ok(root, "the pill needs no page structure at all");
  assert.equal(root.styles.position, "fixed");
  assert.ok(root.classList.contains("pill"));
  assert.equal(root.styles["pointer-events"], "none",
    "the host must not swallow taps meant for the page");
});

test("no button at all on a page that is neither watch nor shorts", async () => {
  const { root } = await mounted({ href: "https://www.youtube.com/results?search_query=x" });
  assert.equal(root, null);
});

test("the button lives inside a shadow root", async () => {
  // A shadow boundary REMOVES the possibility of style bleed in both
  // directions, where scoped class names only lower the odds.
  const { root } = await mounted();
  assert.ok(root.shadow, "attachShadow must be used");
  const button = root.shadow.children[0].children[0];
  assert.equal(button.tagName, "BUTTON");
  assert.equal(button.type, "button");
});

test("falls back to a <style> element when constructable sheets are missing", async () => {
  const { root } = await mounted({ noConstructableSheets: true });
  const tags = root.shadow.children.map((c) => c.tagName);
  assert.ok(tags.includes("STYLE"), tags.join(","));
});

// --- re-mounting ------------------------------------------------------------

test("re-mounts when YouTube replaces the parent's children", async () => {
  const below = new FakeEl("div"); below.id = "below";
  const doc = makeDoc({ "ytd-watch-flexy #primary-inner #below": below });
  const { root, state } = await mounted({ doc });
  assert.ok(root);
  root.remove();                       // detached, as if #below were rebuilt
  assert.equal(doc.contains(root), false);
  state.intervals[0]();                // one tick
  await flush();
  const again = doc.getElementById("patmir-toggle-root");
  assert.ok(again, "the tick's document.contains check must re-mount");
  assert.equal(below.children.filter((c) => c.id === "patmir-toggle-root").length, 1,
    "exactly one root, never two");
});

test("a URL change tears down, re-mounts and re-looks-up", async () => {
  const { state, session, doc } = await mounted();
  session.calls.length = 0;
  state.urlCallbacks[0]("https://www.youtube.com/watch?v=other");
  await flush(); await flush();
  assert.ok(session.calls.includes("refresh"));
  assert.ok(doc.getElementById("patmir-toggle-root"));
});

test("it registers on content.js's URL poll rather than starting its own", async () => {
  // content.js's 500 ms poll is the only navigation signal that fires while
  // nothing is playing; a second identical poll would be pure duplication.
  const { state } = await mounted();
  assert.equal(state.urlCallbacks.length, 1);
  assert.equal(state.intervals.length, 1, "exactly one owned timer");
});

test("yt-navigate-finish schedules a re-check", async () => {
  const { state } = await mounted();
  assert.ok(state.winListeners["yt-navigate-finish"]);
  state.winListeners["yt-navigate-finish"][0]();
  assert.ok(state.timeouts.length >= 1);
});

// --- rendering --------------------------------------------------------------

test("an undubbed page reads Թարգմանել", async () => {
  const { root } = await mounted();
  const button = root.shadow.children[0].children[0];
  assert.equal(button.children[0].textContent, "Թարգմանել");
});

test("a playing page reads Չթարգմանել", async () => {
  const { root } = await mounted({
    state: { playback: { playing: true, hasVideo: true } },
  });
  const button = root.shadow.children[0].children[0];
  assert.equal(button.children[0].textContent, "Չթարգմանել");
  assert.ok(button.className.includes("stop"));
});

test("a busy job with nothing measurable shows the sweeping band, not a grey button", async () => {
  // Watch-along start-up (waiting for the GPU, prelude) has no percentage;
  // the fill used to be hidden for all of it, so the button looked dead.
  for (const job of [
    { status: "waiting", job_id: "j" },
    { status: "prelude", job_id: "j" },
    { status: "streaming", job_id: "j", dubbed_s: 0, duration: 600 },   // 0% == nothing
  ]) {
    const { root } = await mounted({ state: { phase: "job", job } });
    const button = root.shadow.children[0].children[0];
    const fill = button.children.find((c) => c.className.startsWith("fill"));
    assert.equal(fill.className, "fill indeterminate", job.status);
    assert.equal(button.attributes["aria-busy"], "true", job.status);
  }
});

test("a measurable job shows the proportional fill; an idle page shows none", async () => {
  const busy = await mounted({
    state: { phase: "job", job: { status: "streaming", job_id: "j", dubbed_s: 180, duration: 600 } },
  });
  const fill = busy.root.shadow.children[0].children[0].children.find((c) => c.className.startsWith("fill"));
  assert.equal(fill.className, "fill");
  assert.equal(fill.style.width, "30%");

  const idle = await mounted();
  const idleFill = idle.root.shadow.children[0].children[0].children.find((c) => c.className.startsWith("fill"));
  assert.equal(idleFill.className, "fill hidden");
});

test("render is diffed: an unchanged state does not rewrite the DOM", async () => {
  // Unconditional writes at 2 Hz inside YouTube's layout, on a page decoding
  // video, is what shows up as dropped frames and gets blamed on the dub.
  const { root, state } = await mounted();
  const label = root.shadow.children[0].children[0].children[0];
  const before = label.textWrites;
  state.intervals[0]();
  state.intervals[0]();
  await flush();
  assert.equal(label.textWrites, before);
});

test("pressing the button calls primary() exactly once", async () => {
  const { root, session } = await mounted();
  session.calls.length = 0;
  root.shadow.children[0].children[0].click();
  await flush();
  assert.deepEqual(session.calls, ["primary"]);
});

test("a disabled button does not act", async () => {
  // Belt and braces: the DOM will not dispatch the click, and `session.primary`
  // independently re-derives the view and returns early when it is disabled
  // (covered in session.test.js). Either alone would be enough; both is cheap.
  const { root, session } = await mounted({ state: { lookup: null } });
  const button = root.shadow.children[0].children[0];
  assert.equal(button.disabled, true);
  session.calls.length = 0;
  button.click();
  await flush();
  assert.deepEqual(session.calls, []);
});

test("player shortcut keys are stopped from reaching the page", async () => {
  // A focused button plus YouTube's document-level handlers means space would
  // both activate the button AND toggle playback.
  const { root } = await mounted();
  const button = root.shadow.children[0].children[0];
  let stopped = 0;
  let prevented = 0;
  for (const key of [" ", "k", "ArrowLeft"]) {
    button.listeners.keydown[0]({
      key, stopPropagation: () => { stopped += 1; }, preventDefault: () => { prevented += 1; },
    });
  }
  assert.equal(stopped, 3);
  assert.equal(prevented, 0, "preventDefault on space would break the button itself");
});

// --- ads --------------------------------------------------------------------

test("an ad is reported to the session", async () => {
  const { state, session } = await mounted({ ad: { showing: true, classConfirmed: true, skipReady: false } });
  session.calls.length = 0;
  state.intervals[0]();
  await flush();
  assert.ok(session.calls.includes("ad:true"));
});

test("a ready skip control is clicked", async () => {
  const video = new FakeEl("video");
  video.currentTime = 2; video.duration = 30;
  const { state } = await mounted({ ad: { showing: true, classConfirmed: true, skipReady: true }, video });
  state.adActions.length = 0;
  state.intervals[0]();
  await flush();
  assert.deepEqual(state.adActions, ["click"]);
});

test("an unskippable ad is NOT seeked unless the viewer opted in", async () => {
  // The default. Clicking a skip control is safe; writing currentTime is the
  // one thing here that can move the viewer's video, so it is a separate
  // opt-in — a lesson from shipping it on by default with a broken predicate.
  const video = new FakeEl("video");
  video.currentTime = 2; video.duration = 30;
  const { state } = await mounted({
    ad: { showing: true, classConfirmed: true, skipReady: false }, video,
  });
  state.adActions.length = 0;
  for (let i = 0; i < 10; i++) { state.intervals[0](); await flush(); }
  assert.deepEqual(state.adActions, []);
});

test("with the seek opted in, it fires a bounded number of times", async () => {
  const video = new FakeEl("video");
  video.currentTime = 2; video.duration = 30;
  const { state } = await mounted({
    ad: { showing: true, classConfirmed: true, skipReady: false },
    video, store: { patmirSkipAdSeek: true },
  });
  state.adActions.length = 0;
  for (let i = 0; i < 10; i++) { state.intervals[0](); await flush(); }
  assert.equal(state.adActions.length, SYNC.DEFAULT_AD_SEEK_ATTEMPTS,
    "a stalled ad must not become a loop");
});

test("even opted in, a seek needs the player's own class", async () => {
  const video = new FakeEl("video");
  video.currentTime = 2; video.duration = 30;
  const { state } = await mounted({
    ad: { showing: true, classConfirmed: false, skipReady: false },
    video, store: { patmirSkipAdSeek: true },
  });
  state.adActions.length = 0;
  for (let i = 0; i < 5; i++) { state.intervals[0](); await flush(); }
  assert.deepEqual(state.adActions, []);
});

test("one throwing concern does not disable the others for good", async () => {
  // They used to share a single try/catch, so a mount failure silently stopped
  // ad handling and rendering on every subsequent tick, with nothing said.
  const video = new FakeEl("video");
  video.currentTime = 2; video.duration = 30;
  const { state, doc } = await mounted({
    ad: { showing: true, classConfirmed: true, skipReady: true }, video,
  });
  doc.querySelector = () => { throw new Error("hostile DOM"); };
  state.adActions.length = 0;
  state.intervals[0]();
  await flush();
  assert.deepEqual(state.adActions, ["click"], "ad handling must survive a mount throw");
});

test("skipping honours the stored setting", async () => {
  const video = new FakeEl("video");
  video.currentTime = 2; video.duration = 30;
  const { state } = await mounted({
    ad: { showing: true, classConfirmed: true, skipReady: true }, video,
    store: { patmirSkipAds: false },
  });
  state.adActions.length = 0;
  state.intervals[0]();
  await flush();
  assert.deepEqual(state.adActions, []);
});

test("nothing is skipped when no ad is showing", async () => {
  const video = new FakeEl("video");
  video.currentTime = 2; video.duration = 30;
  const { state } = await mounted({ video });
  state.adActions.length = 0;
  for (let i = 0; i < 5; i++) { state.intervals[0](); await flush(); }
  assert.deepEqual(state.adActions, [],
    "a seek with no ad on screen would move the CONTENT video");
});

// --- the popup's window -----------------------------------------------------

function overlayHandler(state) {
  // content.js's listener is not present here, so the overlay's is the only one.
  return state.msgHandlers[state.msgHandlers.length - 1];
}

function ask(state, msg) {
  return new Promise((resolve) => {
    const kept = overlayHandler(state)(msg, {}, resolve);
    assert.equal(kept, true);
  });
}

test("session-state returns a snapshot and notes that the popup is watching", async () => {
  const { state, session } = await mounted();
  const resp = await ask(state, { type: "patmir:session-state", popupOpen: true });
  assert.equal(resp.ok, true);
  assert.equal(resp.state.mode, "quality");
  assert.ok(session.calls.includes("popupOpen"));
});

test("every session-command action reaches the session", async () => {
  const { state, session } = await mounted();
  for (const [action, expected] of [
    ["refresh", "refresh"], ["primary", "primary"], ["start", "start"],
    ["stop", "stop"], ["regenerate", "regenerate"],
  ]) {
    session.calls.length = 0;
    const resp = await ask(state, { type: "patmir:session-command", action });
    assert.equal(resp.ok, true, action);
    assert.ok(session.calls.includes(expected), `${action} -> ${session.calls}`);
    assert.ok(resp.state, "a command replies with the post-action snapshot");
  }
});

test("set-mode carries the mode through", async () => {
  const { state, session } = await mounted();
  session.calls.length = 0;
  await ask(state, { type: "patmir:session-command", action: "set-mode", mode: "watch-along" });
  assert.ok(session.calls.includes("set-mode:watch-along"));
});

test("messages the overlay does not own are left for content.js", async () => {
  const { state } = await mounted();
  assert.equal(overlayHandler(state)({ type: "patmir:status" }, {}, () => {}), undefined);
  assert.equal(overlayHandler(state)({ type: "patmir:play" }, {}, () => {}), undefined);
  assert.equal(overlayHandler(state)(null, {}, () => {}), undefined);
});

test("a mode change in storage is pushed into the session", async () => {
  const { state, session } = await mounted();
  session.calls.length = 0;
  state.storageHandlers[0]({ patmirMode: { newValue: "watch-along" } }, "local");
  await flush();
  assert.ok(session.calls.includes("set-mode:watch-along"));
});

test("an audience change forces a fresh lookup, not just a repaint", async () => {
  // The level is part of the cache key, so the previous lookup answer is wrong
  // rather than merely stale.
  const { state, session } = await mounted();
  session.calls.length = 0;
  state.storageHandlers[0]({ patmirAudience: { newValue: "child" } }, "local");
  await flush();
  assert.ok(session.calls.includes("refresh"));
});

test("an unrelated storage key does no work", async () => {
  const { state, session } = await mounted();
  session.calls.length = 0;
  state.storageHandlers[0]({ somethingElse: { newValue: 1 } }, "local");
  await flush();
  assert.deepEqual(session.calls, []);
});

test("a sync-area storage change is ignored", async () => {
  const { state, session } = await mounted();
  session.calls.length = 0;
  state.storageHandlers[0]({ patmirMode: { newValue: "watch-along" } }, "sync");
  await flush();
  assert.deepEqual(session.calls, []);
});


// --- progress fill ------------------------------------------------------
//
// The control doubles as its own progress bar. Reported problem: while
// translating, the button read `Չթարգմանել` ("do not translate") with the
// percentage buried in 12 px of subtitle, so it looked like a statement that
// nothing was being translated. Now the word is `Չեղարկել` and the work shows
// as green filling the grey.

const RUNNING_STATE = {
  phase: "job",
  job: {
    status: "running", job_id: "j",
    stage_states: { a: "done", b: "done", c: "running", d: "queued", e: "queued" },
  },
};

function fillOf(root) {
  const button = root.shadow.children[0].children[0];
  return button.children.find((c) => (c.className || "").includes("fill"));
}

test("a translating page reads Չեղարկել, not Չթարգմանել", async () => {
  const { root } = await mounted({ state: RUNNING_STATE });
  const button = root.shadow.children[0].children[0];
  assert.equal(button.children[0].textContent, "Չեղարկել");
});

test("the fill is shown and sized while a job runs", async () => {
  const { root } = await mounted({ state: RUNNING_STATE });
  const fill = fillOf(root);
  assert.ok(fill, "the button must carry a fill element");
  assert.ok(!(fill.className || "").includes("hidden"), "fill must be visible");
  // 2 of 5 stages done, the running one contributing nothing without live_pct.
  assert.equal(fill.style.width, "40%");
});

test("the fill is hidden when there is no job to measure", async () => {
  const { root } = await mounted();
  const fill = fillOf(root);
  assert.ok(fill, "the fill element exists in every state");
  assert.ok((fill.className || "").includes("hidden"));
  assert.equal(fill.style.width, "0%");
});

test("the fill is hidden while merely playing a cached dub", async () => {
  // Playback is not progress: a bar that sat at 100% during playback would be
  // claiming work that is not happening.
  const { root } = await mounted({
    state: { playback: { playing: true, hasVideo: true } },
  });
  assert.ok((fillOf(root).className || "").includes("hidden"));
});

test("the label stays children[0] so the fill cannot displace it", async () => {
  // The fill is appended last and layered with z-index for exactly this
  // reason: inserting it first blanked the label, because the button's shape
  // is read positionally here and in the popup-parity checks.
  const { root } = await mounted({ state: RUNNING_STATE });
  const button = root.shadow.children[0].children[0];
  assert.ok((button.children[0].className || "").includes("label"));
  assert.ok((button.children[1].className || "").includes("sub"));
});

test("the fill is decorative, so the percentage is announced once", async () => {
  // The subtitle is aria-live; a second announcing element would say the
  // percentage twice on every tick.
  const { root } = await mounted({ state: RUNNING_STATE });
  // The stub element records setAttribute into `attributes` and has no
  // getAttribute, so read the map it keeps.
  assert.equal(fillOf(root).attributes["aria-hidden"], "true");
});

test("a translating button keeps the neutral chrome", async () => {
  // Cancel shares Stop's styling: both mean "there is something to
  // interrupt", and the difference between them is the word, not the colour.
  const { root } = await mounted({ state: RUNNING_STATE });
  const button = root.shadow.children[0].children[0];
  assert.ok(button.className.includes("stop"));
});
