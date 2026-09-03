// Unit tests for extension/options.js -- the save path.
//
//   node --test "extension/tests/*.test.js"
//
// The options page is where a broken save is invisible: if `save()` throws,
// the click handler's promise rejects silently, "Saved." never appears, and
// NOTHING is written -- so the extension has no server and no token and
// cannot connect. That failure looks exactly like a server problem, which is
// how 2026-09-03 went. These tests run the real file against a stub DOM.

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.join(__dirname, "..");

function el(value = "") {
  return {
    value, textContent: "", className: "", checked: false,
    _handlers: {},
    addEventListener(kind, fn) { this._handlers[kind] = fn; },
  };
}

/** Load lib/server.js + options.js against a stub DOM and chrome API.
 *
 * Async on purpose: options.js calls `load()` at the bottom, and that awaits
 * chrome.storage. Its continuation would otherwise land AFTER a test assigns
 * the input fields and clobber them with the stored (empty) values.
 */
async function boot({ stored = {}, granted = false, requestAnswers = true,
                fetchStatus = 200, fetchThrows = false } = {}) {
  const els = {
    server: el(), token: el(), autoplay: el(), audience: el(),
    skipads: el(), skipadseek: el(), save: el(), status: el(),
  };
  const writes = [];
  const requested = [];
  const fetched = [];
  const sandbox = {
    console,
    // save() now asks the server whether it accepts the token before storing
    // it, so the harness has to answer.
    fetch: async (url) => {
      fetched.push(url);
      if (fetchThrows) throw new TypeError("Failed to fetch");
      return {
        status: fetchStatus,
        ok: fetchStatus >= 200 && fetchStatus < 300,
        text: async () => "{}",
      };
    },
    // A fresh vm context has none of the web platform: without URL,
    // normalizeServerUrl catches the ReferenceError and reports every address
    // as unusable, which looks exactly like the bug being tested for.
    URL, URLSearchParams,
    document: { getElementById: (id) => els[id] },
    setTimeout: () => {},
    chrome: {
      storage: { local: {
        get: async () => ({ ...stored }),
        set: async (obj) => { writes.push(obj); },
      } },
      permissions: {
        contains: async () => granted,
        request: async (req) => { requested.push(req); return requestAnswers; },
      },
    },
  };
  sandbox.globalThis = sandbox;
  const ctx = vm.createContext(sandbox);
  for (const f of ["lib/server.js", "options.js"]) {
    vm.runInContext(fs.readFileSync(path.join(ROOT, f), "utf8"), ctx,
                    { filename: f });
  }
  // Let load()'s continuation run before the caller touches the fields.
  await new Promise((resolve) => process.nextTick(resolve));
  return { els, writes, requested, fetched,
           save: () => els.save._handlers.click() };
}

test("the library the save path depends on is actually in scope", () => {
  // options.html loads lib/server.js BEFORE options.js. If that ordering is
  // ever broken, `PatmirServer` is undefined and save() throws on its first
  // line -- with no visible error anywhere.
  const html = fs.readFileSync(path.join(ROOT, "options.html"), "utf8");
  const libAt = html.indexOf("lib/server.js");
  const optAt = html.indexOf("options.js\"");
  assert.ok(libAt > -1, "options.html must load lib/server.js");
  assert.ok(libAt < optAt, "lib/server.js must come before options.js");
});

test("a loopback server saves with no permission prompt", async () => {
  const h = await boot();
  h.els.server.value = "http://127.0.0.1:8765";
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.requested.length, 0, "loopback is already in the manifest");
  assert.equal(h.writes.length, 1);
  assert.equal(h.writes[0].patmirServer, "http://127.0.0.1:8765");
  assert.equal(h.writes[0].patmirToken, "deadbeef");
  assert.equal(h.els.status.className, "ok");
  assert.match(h.els.status.textContent, /token accepted/);
});

test("an empty server field falls back to loopback", async () => {
  const h = await boot();
  h.els.server.value = "";
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.writes[0].patmirServer, "http://127.0.0.1:8765");
});

test("a LAN server requests the host permission and then saves", async () => {
  const h = await boot({ granted: false, requestAnswers: true });
  h.els.server.value = "http://192.168.10.164:8765";
  h.els.token.value = "deadbeef";
  await h.save();
  // Compare the value, not the object: it was created inside the vm context,
  // so its prototype is from another realm and deepEqual fails on that alone.
  assert.equal(h.requested.length, 1);
  assert.equal(h.requested[0].origins[0], "http://192.168.10.164/*");
  assert.equal(h.writes.length, 1);
  assert.equal(h.writes[0].patmirServer, "http://192.168.10.164:8765");
});

test("an already-granted LAN server saves without asking again", async () => {
  const h = await boot({ granted: true });
  h.els.server.value = "http://192.168.10.164:8765";
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.requested.length, 0);
  assert.equal(h.writes.length, 1);
});

test("a declined permission saves NOTHING and says so", async () => {
  const h = await boot({ granted: false, requestAnswers: false });
  h.els.server.value = "http://192.168.10.164:8765";
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.writes.length, 0, "an address we cannot reach must not be stored");
  assert.equal(h.els.status.className, "err");
  assert.match(h.els.status.textContent, /did not grant/);
});

test("a missing token is refused before anything else", async () => {
  const h = await boot();
  h.els.server.value = "http://192.168.10.164:8765";
  h.els.token.value = "";
  await h.save();
  assert.equal(h.writes.length, 0);
  assert.equal(h.requested.length, 0);
});

test("an unusable URL is refused rather than stored", async () => {
  const h = await boot();
  h.els.server.value = "192.168.10.164:8765";      // no scheme
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.writes.length, 0);
  assert.equal(h.els.status.className, "err");
});

test("a trailing slash is stripped before storing", async () => {
  // A stored trailing slash builds `//jobs` once sw.js appends a path.
  const h = await boot();
  h.els.server.value = "http://127.0.0.1:8765/";
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.writes[0].patmirServer, "http://127.0.0.1:8765");
});


test("a token the server rejects is NOT saved", async () => {
  // A 401 is definitive: the server is there and refuses this token, so
  // storing it would leave a configuration that cannot work. A malformed
  // token pasted with a line break is what actually broke this on
  // 2026-09-03, after the bind, the permission and WSL networking had all
  // been chased first.
  const h = await boot({ fetchStatus: 401 });
  h.els.server.value = "http://127.0.0.1:8765";
  h.els.token.value = "wrong";
  await h.save();
  assert.equal(h.writes.length, 0);
  assert.equal(h.els.status.className, "err");
  assert.match(h.els.status.textContent, /rejected that token/);
});

test("an unreachable server still saves, with a warning", async () => {
  // Inconclusive, not wrong: configuring before starting the server is a
  // legitimate order of operations, so this must not block.
  const h = await boot({ fetchThrows: true });
  h.els.server.value = "http://127.0.0.1:8765";
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.writes.length, 1);
  assert.equal(h.els.status.className, "warn");
  assert.match(h.els.status.textContent, /did not respond/);
});

test("an odd HTTP status saves and reports the status", async () => {
  const h = await boot({ fetchStatus: 502 });
  h.els.server.value = "http://127.0.0.1:8765";
  h.els.token.value = "deadbeef";
  await h.save();
  assert.equal(h.writes.length, 1);
  assert.equal(h.els.status.className, "warn");
  assert.match(h.els.status.textContent, /502/);
});

test("a token pasted with a line break is cleaned before use", async () => {
  // The real failure: 64 hex characters wrapped across two terminal lines.
  const h = await boot();
  h.els.server.value = "http://127.0.0.1:8765";
  h.els.token.value = "ff50333ba01cddd0a4665afa3972656d\n47280ea128e3b488ec63b2f286f9d9ee";
  await h.save();
  assert.equal(h.writes[0].patmirToken,
    "ff50333ba01cddd0a4665afa3972656d47280ea128e3b488ec63b2f286f9d9ee");
  // ...and the field shows what was stored, so the viewer is not left
  // looking at text that differs from the saved value.
  assert.equal(h.els.token.value, h.writes[0].patmirToken);
});

test("a copied `Bearer ` prefix is stripped", async () => {
  const h = await boot();
  h.els.server.value = "http://127.0.0.1:8765";
  h.els.token.value = "Bearer deadbeef";
  await h.save();
  assert.equal(h.writes[0].patmirToken, "deadbeef");
});
