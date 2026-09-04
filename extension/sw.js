// Patmir service worker.
//
// All HTTP/SSE calls to the local Patmir server originate here, not in
// content scripts — YouTube's mixed-content policy blocks
// `http://127.0.0.1` fetches from the page context, but the SW runs in
// `chrome-extension://<id>` origin and is exempt.
//
// Messages handled:
//   { type: "cache:lookup", url, mode? }       → { cached, cache_key, ... }
//   { type: "jobs:submit",  url? | source_path?, mode? } → { job_id, status, ... }
//   { type: "jobs:get",     job_id }           → JobRecord
//   { type: "jobs:cancel",  job_id }           → JobRecord
//   { type: "options:open" }                   → opens the options page
//
// The audience age level is attached HERE, from chrome.storage.local, rather
// than by each caller — `cache:lookup` and `jobs:submit` must agree on it or the
// client is told "not dubbed" for a dub that exists, and content.js's auto-play
// lookup would otherwise have to remember to pass it too.
//   { type: "cache:dub-wav", cache_key }       → { bytesB64 }
//   { type: "stream:sections", cache_key }     → { sections, complete, ... }
//   { type: "stream:section-wav", cache_key, index } → { bytesB64 }
//   { type: "stream:playhead", cache_key, t, paused } → { ok }

const DEFAULT_SERVER = "http://127.0.0.1:8765";
const DEFAULT_AUDIENCE = "teenager";

async function getToken() {
  const { patmirToken } = await chrome.storage.local.get("patmirToken");
  if (!patmirToken) {
    throw new Error("No bearer token set. Open the Patmir extension options page and paste the token printed by `patmir ui`.");
  }
  return patmirToken;
}

async function getServer() {
  const { patmirServer } = await chrome.storage.local.get("patmirServer");
  return patmirServer || DEFAULT_SERVER;
}

/** Age register the translator is steered toward, from the options page.
 *
 * Defaults to "teenager", matching DEFAULT_AUDIENCE in
 * patmir/pipeline/context.py: a softened register is what you get by not
 * choosing. The server's own default is the same, but we always send a value
 * explicitly — the extension wins over patmir.toml, and being explicit means
 * the lookup and the submit cannot resolve differently.
 */
async function getAudience() {
  const { patmirAudience } = await chrome.storage.local.get("patmirAudience");
  return patmirAudience || DEFAULT_AUDIENCE;
}

/** Host match pattern for a server URL.
 *
 * Mirrors `hostPermissionPattern` in lib/server.js, which cannot be imported
 * here — the service worker is a classic script, not a module. Kept to one
 * line so the duplication is obvious rather than subtle.
 */
function hostPatternFor(server) {
  try {
    const u = new URL(server);
    return `${u.protocol}//${u.hostname}/*`;
  } catch (_) {
    return "";
  }
}

async function api(path, opts = {}) {
  const [token, server] = await Promise.all([getToken(), getServer()]);
  const headers = Object.assign({}, opts.headers || {}, {
    "Authorization": "Bearer " + token,
  });
  if (opts.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const method = opts.method || "GET";
  let res;
  try {
    res = await fetch(server + path, { ...opts, headers });
  } catch (cause) {
    // `fetch` rejects the same way for a server that is down and for a host
    // this extension has no permission for, and the message does not
    // distinguish them — which is exactly how a LAN server spent an evening
    // looking like a dead server on 2026-09-03. So ask.
    let hint = "";
    const pattern = hostPatternFor(server);
    if (pattern) {
      const allowed = await chrome.permissions
        .contains({ origins: [pattern] })
        .catch(() => null);
      if (allowed === false) {
        hint = ` The extension has no permission for ${pattern} — open the`
             + ` options page, re-save the server address, and accept Chrome's`
             + ` prompt.`;
      }
    }
    // This module had ZERO console calls, so every failure below was silent
    // and the only symptom was "it does not connect".
    console.warn(`[patmir] ${method} ${server}${path} failed: ${cause.message}.${hint}`);
    const err = new Error(
      `Cannot reach the Patmir server at ${server}: ${cause.message}.${hint}`);
    err.cause = cause;
    throw err;
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    console.warn(`[patmir] ${method} ${server}${path} -> HTTP ${res.status}`
                 + (res.status === 401
                    ? " (token mismatch: re-copy the one `patmir ui` printed)"
                    : "")
                 + ` ${text.slice(0, 200)}`);
    const err = new Error("HTTP " + res.status + " " + path + ": " + text.slice(0, 200));
    err.status = res.status;
    throw err;
  }
  return res;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === "cache:lookup") {
        let path = "/cache/lookup?url=" + encodeURIComponent(msg.url);
        if (msg.mode) path += "&mode=" + encodeURIComponent(msg.mode);
        path += "&audience=" + encodeURIComponent(await getAudience());
        const r = await api(path);
        sendResponse({ ok: true, data: await r.json() });
      } else if (msg.type === "jobs:submit") {
        const body = { source_url: msg.url };
        if (msg.source_lang) body.source_lang = msg.source_lang;
        if (msg.force) body.force = true;
        if (msg.mode) body.mode = msg.mode;
        body.audience = await getAudience();
        const r = await api("/jobs", { method: "POST", body: JSON.stringify(body) });
        sendResponse({ ok: true, data: await r.json() });
      } else if (msg.type === "jobs:get") {
        const r = await api("/jobs/" + encodeURIComponent(msg.job_id));
        sendResponse({ ok: true, data: await r.json() });
      } else if (msg.type === "options:open") {
        // A content script has no chrome.runtime.openOptionsPage, so the
        // in-page button's "no token" state routes through here.
        await chrome.runtime.openOptionsPage();
        sendResponse({ ok: true });
      } else if (msg.type === "jobs:cancel") {
        // The endpoint is idempotent, so a cancel racing a completion is fine.
        // Without this the in-page Չթարգմանել would stop the UI while the GPU
        // kept grinding — the waste `stream:stop` already prevents for
        // watch-along.
        const r = await api(
          "/jobs/" + encodeURIComponent(msg.job_id) + "/cancel",
          { method: "POST", body: JSON.stringify({}) },
        );
        sendResponse({ ok: true, data: await r.json() });
      } else if (msg.type === "cache:dub-wav") {
        const r = await api("/cache/" + encodeURIComponent(msg.cache_key) + "/dub.wav");
        const buf = await r.arrayBuffer();
        // chrome.runtime.sendMessage's behavior with typed-array transfer
        // has been inconsistent across Chrome versions; base64 over the
        // message channel is reliable. ~33% overhead is acceptable for
        // ~30-60 MB WAVs on a local loopback round trip.
        sendResponse({ ok: true, bytesB64: bytesToBase64(new Uint8Array(buf)) });
      } else if (msg.type === "stream:sections") {
        const r = await api("/stream/" + encodeURIComponent(msg.cache_key) + "/sections");
        sendResponse({ ok: true, data: await r.json() });
      } else if (msg.type === "stream:section-wav") {
        const r = await api("/stream/" + encodeURIComponent(msg.cache_key) +
          "/section/" + encodeURIComponent(msg.index) + ".wav");
        const buf = await r.arrayBuffer();
        sendResponse({ ok: true, bytesB64: bytesToBase64(new Uint8Array(buf)) });
      } else if (msg.type === "stream:playhead") {
        const r = await api("/stream/" + encodeURIComponent(msg.cache_key) + "/playhead", {
          method: "POST",
          body: JSON.stringify({ t: msg.t, paused: !!msg.paused }),
        });
        sendResponse({ ok: true, data: await r.json() });
      } else if (msg.type === "stream:stop") {
        const r = await api("/stream/" + encodeURIComponent(msg.cache_key) + "/stop", {
          method: "POST", body: "{}",
        });
        sendResponse({ ok: true, data: await r.json() });
      } else if (msg.type === "ping") {
        sendResponse({ ok: true, pong: true });
      } else {
        sendResponse({ ok: false, error: "unknown message type: " + msg.type });
      }
    } catch (e) {
      sendResponse({ ok: false, error: String((e && e.message) || e) });
    }
  })();
  return true;  // keep the message channel open for async sendResponse
});

// Optional: surface job completion as a desktop notification (slice 2 will
// drive this via SSE; for slice 1 the popup just polls).

function bytesToBase64(uint8) {
  // Chunked encoder — `btoa(String.fromCharCode(...uint8))` blows the stack
  // on large arrays.
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < uint8.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, uint8.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}
