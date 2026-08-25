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
function boot({ withSync }) {
  const state = { handler: null, errors: [], timers: [] };
  const sandbox = {
    console: { error: (...a) => state.errors.push(a.join(" ")), warn() {}, log() {} },
    setTimeout: (fn) => { state.timers.push(fn); return state.timers.length; },
    clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    location: { href: "https://www.youtube.com/watch?v=abc" },
    document: { querySelectorAll: () => [], addEventListener() {}, contains: () => true },
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
