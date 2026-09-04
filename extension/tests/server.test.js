// Unit tests for extension/lib/server.js -- the server URL and the host
// permission it needs.
//
//   node --test "extension/tests/*.test.js"
//
// Every case is something that actually cost time: a manifest pinned to
// 127.0.0.1 made a LAN server unreachable inside the browser, and the symptom
// was indistinguishable from a server that was down.

const { test } = require("node:test");
const assert = require("node:assert/strict");

const {
  normalizeServerUrl, hostPermissionPattern, isLoopbackServer,
} = require("../lib/server.js");

test("normalize keeps an explicit port", () => {
  assert.equal(normalizeServerUrl("http://192.168.1.50:8765"),
               "http://192.168.1.50:8765");
});

test("normalize trims whitespace and a trailing slash", () => {
  // A stored trailing slash builds `//jobs` once sw.js appends a path.
  assert.equal(normalizeServerUrl("  http://127.0.0.1:8765/  "),
               "http://127.0.0.1:8765");
});

test("normalize drops a path", () => {
  assert.equal(normalizeServerUrl("http://127.0.0.1:8765/jobs"),
               "http://127.0.0.1:8765");
});

test("normalize rejects rubbish rather than storing it", () => {
  for (const bad of ["", "   ", null, undefined, "not a url", "127.0.0.1:8765",
                     "ftp://127.0.0.1", "javascript:alert(1)"]) {
    assert.equal(normalizeServerUrl(bad), "", `expected "" for ${bad}`);
  }
});

test("the permission pattern has no port", () => {
  // Chrome's match-pattern grammar has no port component. A host-only pattern
  // is unambiguous and covers whichever port the server runs on.
  assert.equal(hostPermissionPattern("http://192.168.1.50:8765"),
               "http://192.168.1.50/*");
  assert.equal(hostPermissionPattern("http://patmir.lan:8000"),
               "http://patmir.lan/*");
});

test("the permission pattern keeps the scheme", () => {
  assert.equal(hostPermissionPattern("https://dub.example.com:8443"),
               "https://dub.example.com/*");
});

test("no pattern for an unusable URL", () => {
  assert.equal(hostPermissionPattern("nonsense"), "");
});

test("loopback is recognised in its several spellings", () => {
  for (const s of ["http://127.0.0.1:8765", "http://localhost:8765",
                   "http://[::1]:8765"]) {
    assert.equal(isLoopbackServer(s), true, s);
  }
});

test("a LAN address is not loopback", () => {
  assert.equal(isLoopbackServer("http://192.168.1.50:8765"), false);
  assert.equal(isLoopbackServer("http://10.0.0.7:8765"), false);
});

test("an unusable URL is not loopback either", () => {
  // Must not report true and skip the permission request.
  assert.equal(isLoopbackServer("nonsense"), false);
});
