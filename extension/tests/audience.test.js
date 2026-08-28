// The service worker is the only place that knows the audience age level.
//
// This matters because two different calls have to agree on it: `cache:lookup`
// decides whether the button says "already dubbed", and `jobs:submit` decides
// which key the pipeline writes. If they resolve the level differently, the
// viewer is told "not dubbed" for a dub that exists — or offered a play button
// for a key nothing will ever write. Attaching the level inside sw.js, rather
// than at each call site, makes disagreement unrepresentable, and means
// content.js's auto-play lookup gets it right without knowing it exists.
//
// sw.js is a classic script, so it is evaluated against a stub of the extension
// world exactly as bootstrap.test.js does for content.js.

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SW = fs.readFileSync(path.join(__dirname, "..", "sw.js"), "utf8");

/** Boot sw.js with a canned storage and a recording fetch. */
function boot(stored) {
  const state = { handler: null, fetches: [] };
  const sandbox = {
    console: { error() {}, warn() {}, log() {} },
    setTimeout: (fn) => fn, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    Date, JSON, Promise, Error, encodeURIComponent, Uint8Array, Buffer,
    btoa: (s) => Buffer.from(s, "binary").toString("base64"),
    atob: (b) => Buffer.from(b, "base64").toString("binary"),
    fetch: async (url, opts) => {
      state.fetches.push({ url, opts });
      return {
        ok: true,
        status: 200,
        json: async () => ({ cached: false, cache_key: "deadbeefdeadbeef" }),
        text: async () => "",
        arrayBuffer: async () => new ArrayBuffer(0),
      };
    },
    chrome: {
      runtime: {
        onMessage: { addListener: (fn) => { state.handler = fn; } },
      },
      storage: {
        local: {
          get: async (keys) => {
            const want = Array.isArray(keys) ? keys : [keys];
            const out = {};
            for (const k of want) if (k in stored) out[k] = stored[k];
            return out;
          },
          set: async () => {},
        },
      },
    },
  };
  sandbox.self = sandbox;
  sandbox.globalThis = sandbox;
  vm.runInContext(SW, vm.createContext(sandbox));
  return state;
}

/** Drive one message through the registered listener and await its reply. */
function send(state, msg) {
  return new Promise((resolve) => {
    const kept = state.handler(msg, {}, resolve);
    assert.equal(kept, true, "the listener must keep the channel open for async work");
  });
}

const TOKEN = { patmirToken: "abc123" };

test("a stored level reaches the lookup query string", async () => {
  const state = boot({ ...TOKEN, patmirAudience: "child" });
  await send(state, { type: "cache:lookup", url: "https://youtu.be/xyz" });
  assert.equal(state.fetches.length, 1);
  assert.match(state.fetches[0].url, /[?&]audience=child(&|$)/);
});

test("a stored level reaches the submit body", async () => {
  const state = boot({ ...TOKEN, patmirAudience: "teenager" });
  await send(state, { type: "jobs:submit", url: "https://youtu.be/xyz" });
  assert.equal(state.fetches.length, 1);
  const body = JSON.parse(state.fetches[0].opts.body);
  assert.equal(body.audience, "teenager");
  assert.equal(body.source_url, "https://youtu.be/xyz");
});

test("with nothing stored, both calls default to teenager", async () => {
  // Not "adult": a softened register is what you get by not choosing. (The
  // cache key's no-suffix level IS "adult" — a deliberately different constant,
  // because that is the one whose prompt matches every pre-guardrail dub. See
  // patmir/pipeline/context.py.) This must stay in step with DEFAULT_AUDIENCE
  // there; test_audience.py pins the Python side of the same value.
  const state = boot({ ...TOKEN });
  await send(state, { type: "cache:lookup", url: "https://youtu.be/xyz" });
  await send(state, { type: "jobs:submit", url: "https://youtu.be/xyz" });
  assert.match(state.fetches[0].url, /[?&]audience=teenager(&|$)/);
  assert.equal(JSON.parse(state.fetches[1].opts.body).audience, "teenager");
});

test("the two client copies of the default agree with each other", async () => {
  // sw.js and options.js each carry the constant (one attaches it to requests,
  // the other seeds the picker). If they drift, the options page shows one level
  // while submissions use another — and the cache key follows the submission, so
  // the picker would be quietly lying.
  const fs = require("node:fs");
  const path = require("node:path");
  const read = (f) => fs.readFileSync(path.join(__dirname, "..", f), "utf8");
  const grab = (src) => (src.match(/DEFAULT_AUDIENCE\s*=\s*"([a-z]+)"/) || [])[1];
  assert.equal(grab(read("sw.js")), grab(read("options.js")));
  assert.equal(grab(read("sw.js")), "teenager");
});

test("lookup and submit resolve to the SAME level", async () => {
  // The whole reason the level is attached here and not by the caller.
  const state = boot({ ...TOKEN, patmirAudience: "teenager" });
  await send(state, { type: "cache:lookup", url: "https://youtu.be/xyz", mode: "quality" });
  await send(state, { type: "jobs:submit", url: "https://youtu.be/xyz", mode: "quality" });
  const fromUrl = new URL(state.fetches[0].url).searchParams.get("audience");
  const fromBody = JSON.parse(state.fetches[1].opts.body).audience;
  assert.equal(fromUrl, fromBody);
});

test("mode and audience coexist in the lookup query string", async () => {
  // Regression guard: a naive `path += ...` that overwrote rather than appended
  // would drop one of them, and the watch-along cache would be looked up under
  // the quality key.
  const state = boot({ ...TOKEN, patmirAudience: "child" });
  await send(state, {
    type: "cache:lookup", url: "https://youtu.be/xyz", mode: "watch-along",
  });
  const params = new URL(state.fetches[0].url).searchParams;
  assert.equal(params.get("mode"), "watch-along");
  assert.equal(params.get("audience"), "child");
  assert.equal(params.get("url"), "https://youtu.be/xyz");
});

test("an unknown stored value is passed through for the server to reject", async () => {
  // Deliberately not sanitized here: the options page constrains the choice to
  // a <select>, and the server validates and answers 400. Silently rewriting a
  // bad value to "child" in the client would hide a real bug — and the two
  // sides must agree on what was asked for, whatever it was.
  const state = boot({ ...TOKEN, patmirAudience: "toddler" });
  await send(state, { type: "jobs:submit", url: "https://youtu.be/xyz" });
  assert.equal(JSON.parse(state.fetches[0].opts.body).audience, "toddler");
});

test("jobs:cancel posts to the cancel endpoint", async () => {
  // Added alongside the in-page toggle: without it, switching translation off
  // mid-job stops the UI while the GPU keeps grinding.
  const state = boot({ ...TOKEN });
  await send(state, { type: "jobs:cancel", job_id: "job-1-abcdef" });
  assert.equal(state.fetches.length, 1);
  assert.match(state.fetches[0].url, /\/jobs\/job-1-abcdef\/cancel$/);
  assert.equal(state.fetches[0].opts.method, "POST");
});
