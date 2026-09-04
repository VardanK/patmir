// Pure helpers for the configured server URL and the host permission it needs.
//
// The extension ships `host_permissions` for `http://127.0.0.1:8765/*` only,
// so pointing it at a server on another machine used to fail inside the
// browser before any request left — and the failure looks like "the server is
// down", which cost a long debugging session on 2026-09-03. The server URL is
// already user-configurable, so the permission has to follow it: the options
// page requests it at save time (a user gesture, which `permissions.request`
// requires) and refuses to save an address it cannot reach.
//
// No `chrome.*` here and no DOM: this module is pure so it can be tested with
// `node --test extension/tests/`.

/** Trim and normalize a server URL. Returns "" when it is not usable. */
function normalizeServerUrl(raw) {
  const text = String(raw == null ? "" : raw).trim();
  if (!text) return "";
  let u;
  try {
    u = new URL(text);
  } catch (_) {
    return "";
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") return "";
  if (!u.hostname) return "";
  // Origin only: the server is addressed by origin + path built in sw.js, and
  // a stored trailing slash or path would produce `//jobs` style URLs.
  return u.port ? `${u.protocol}//${u.hostname}:${u.port}` : `${u.protocol}//${u.hostname}`;
}

/**
 * The match pattern to request for a server URL, or "" if there is none.
 *
 * Deliberately WITHOUT the port. Chrome's match-pattern grammar does not
 * include a port component, and rather than depend on how a given Chrome
 * version treats one that is present, a host-only pattern is unambiguous and
 * covers whatever port the server is on. It is also what the viewer would
 * expect from "let the extension talk to this machine".
 */
function hostPermissionPattern(raw) {
  const origin = normalizeServerUrl(raw);
  if (!origin) return "";
  const u = new URL(origin);
  return `${u.protocol}//${u.hostname}/*`;
}

/**
 * Clean up a pasted bearer token.
 *
 * The server prints 64 hex characters. What arrives in the field can carry a
 * trailing newline, a line break from a wrapped terminal, surrounding quotes,
 * or a `Bearer ` prefix copied along with a curl example — and every one of
 * those is stored happily and then fails as a 401 that nothing surfaces. A
 * malformed token is what actually broke this on 2026-09-03, after the bind,
 * the host permission and the WSL networking had all been chased first.
 */
function normalizeToken(raw) {
  let text = String(raw == null ? "" : raw).trim();
  text = text.replace(/^["'`]+|["'`]+$/g, "");
  text = text.replace(/^\s*Bearer\s+/i, "");
  // A token pasted from a wrapped terminal line arrives with the break in it.
  text = text.replace(/\s+/g, "");
  return text;
}

/**
 * True when a token has the shape the server issues: 64 hex characters.
 *
 * Advisory only — the server is the authority, and `ensure_token` could change
 * format. Used to warn, never to refuse, so a future format cannot lock the
 * viewer out of their own options page.
 */
function looksLikeToken(raw) {
  return /^[0-9a-f]{64}$/i.test(normalizeToken(raw));
}

/** True when a server URL points at this machine, which needs no permission. */
function isLoopbackServer(raw) {
  const origin = normalizeServerUrl(raw);
  if (!origin) return false;
  const host = new URL(origin).hostname;
  return host === "127.0.0.1" || host === "localhost" || host === "[::1]"
    || host === "::1";
}

const PatmirServer = {
  normalizeServerUrl,
  hostPermissionPattern,
  isLoopbackServer,
  normalizeToken,
  looksLikeToken,
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = PatmirServer;
}
if (typeof globalThis !== "undefined") {
  globalThis.PatmirServer = PatmirServer;
}
