const serverEl = document.getElementById("server");
const tokenEl = document.getElementById("token");
const autoplayEl = document.getElementById("autoplay");
const audienceEl = document.getElementById("audience");
const skipAdsEl = document.getElementById("skipads");
const skipAdSeekEl = document.getElementById("skipadseek");
const saveEl = document.getElementById("save");
const statusEl = document.getElementById("status");

// Age register the translator is steered toward. "teenager" by default: a
// softened register is what you get by not choosing, and it is the one that is
// wrong in the smaller direction across a mixed library. Mirrors
// DEFAULT_AUDIENCE in patmir/pipeline/context.py, and sw.js attaches this value
// to every lookup and submit.
const AUDIENCE_LEVELS = ["child", "teenager", "adult"];
const DEFAULT_AUDIENCE = "teenager";

async function load() {
  const {
    patmirServer, patmirToken, patmirTranslateEnabled, patmirAudience, patmirSkipAds,
    patmirSkipAdSeek,
  } = await chrome.storage.local.get([
    "patmirServer", "patmirToken", "patmirTranslateEnabled", "patmirAudience",
    "patmirSkipAds", "patmirSkipAdSeek",
  ]);
  serverEl.value = patmirServer || "http://127.0.0.1:8765";
  tokenEl.value = patmirToken || "";
  // Default ON.
  // One remembered choice for every video, shared with the in-page button:
  // pressing Թարգմանել sets it, pressing Չթարգմանել / Չեղարկել clears it.
  // Default OFF, because ON means every video you open submits GPU work.
  autoplayEl.checked = patmirTranslateEnabled === true;
  audienceEl.value = AUDIENCE_LEVELS.includes(patmirAudience)
    ? patmirAudience : DEFAULT_AUDIENCE;
  // Default ON: clicking the skip control is just the viewer's own gesture a
  // second early. Default OFF for the seek: it is the only setting that can
  // move the video, and it shipped once with a broken ad predicate that sent
  // every video to its end.
  skipAdsEl.checked = patmirSkipAds !== false;
  skipAdSeekEl.checked = patmirSkipAdSeek === true;
}

/**
 * Make sure the extension is allowed to talk to `server`.
 *
 * The manifest grants `http://127.0.0.1:8765/*` and nothing else, so a server
 * on another machine is blocked inside the browser before any request leaves —
 * and that failure is indistinguishable from a server that is down. So the
 * permission follows the configured address: requested here, at save time,
 * because `permissions.request` needs a user gesture and the Save click is one.
 *
 * Returns true when the address is usable.
 */
async function ensureServerPermission(server) {
  if (PatmirServer.isLoopbackServer(server)) return true;   // already in the manifest
  const pattern = PatmirServer.hostPermissionPattern(server);
  if (!pattern) return true;                                 // caller already rejected it
  try {
    if (await chrome.permissions.contains({ origins: [pattern] })) return true;
    return await chrome.permissions.request({ origins: [pattern] });
  } catch (err) {
    statusEl.textContent = `Could not request access to ${pattern}: ${err.message}`;
    statusEl.className = "err";
    return false;
  }
}

/**
 * Ask the server whether it accepts this token, before storing it.
 *
 * Three outcomes, and the distinction matters: a **401 is definitive** — the
 * server is there and rejects this token, so storing it would leave a
 * configuration that cannot work. Anything else is inconclusive: the server
 * may simply not be running yet, which is a legitimate thing to configure
 * around, so we save and say so rather than blocking.
 *
 * A malformed token is what actually broke this on 2026-09-03 — pasted with a
 * line break — after the server bind, the host permission and the WSL
 * networking had all been chased first. Nothing checked the one value nobody
 * suspected.
 */
async function verifyToken(server, token) {
  const url = `${server}/cache/lookup?url=`
            + encodeURIComponent("https://youtu.be/patmir-probe");
  try {
    const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (res.status === 401) return { ok: false, kind: "rejected" };
    if (res.ok) return { ok: true, kind: "verified" };
    return { ok: true, kind: "status", status: res.status };
  } catch (err) {
    return { ok: true, kind: "unreachable", detail: err.message };
  }
}

async function save() {
  const raw = serverEl.value.trim();
  const server = PatmirServer.normalizeServerUrl(raw) || (raw ? null : "http://127.0.0.1:8765");
  if (!server) {
    // Refuse rather than store something that cannot work. A saved-but-broken
    // address is the state that wastes an evening.
    statusEl.textContent = `Not a usable server URL: ${raw}`;
    statusEl.className = "err";
    return;
  }
  // Normalize before anything looks at it: a trailing newline, a break from a
  // wrapped terminal line, quotes, or a copied `Bearer ` prefix all arrive
  // here and all produce a 401 later.
  const token = PatmirServer.normalizeToken(tokenEl.value);
  tokenEl.value = token;               // show the viewer what will be stored
  if (!token) {
    statusEl.textContent = "Token is required.";
    statusEl.className = "err";
    return;
  }
  if (!await ensureServerPermission(server)) {
    // Same reasoning: do NOT save a server the extension may not contact.
    statusEl.textContent =
      `Chrome did not grant access to ${server}. Nothing saved — the `
      + `extension cannot reach a host it has no permission for.`;
    statusEl.className = "err";
    return;
  }
  const audience = AUDIENCE_LEVELS.includes(audienceEl.value)
    ? audienceEl.value : DEFAULT_AUDIENCE;
  const check = await verifyToken(server, token);
  if (!check.ok) {
    statusEl.textContent =
      `${server} rejected that token (401). Nothing saved — re-copy the value `
      + `\`patmir ui\` printed at startup.`;
    statusEl.className = "err";
    return;
  }

  await chrome.storage.local.set({
    patmirServer: server,
    patmirToken: token,
    patmirTranslateEnabled: autoplayEl.checked,
    patmirAudience: audience,
    patmirSkipAds: skipAdsEl.checked,
    patmirSkipAdSeek: skipAdSeekEl.checked,
  });

  if (check.kind === "verified") {
    statusEl.textContent = "Saved — server reachable and token accepted.";
    statusEl.className = "ok";
    setTimeout(() => { statusEl.textContent = ""; }, 4000);
    return;
  }
  // Saved, but we could not confirm it. Say which, and do not clear it: an
  // unconfirmed save is exactly the state worth leaving on screen.
  const shape = PatmirServer.looksLikeToken(token)
    ? "" : " The token is also not the 64 hex characters the server issues.";
  statusEl.textContent = check.kind === "unreachable"
    ? `Saved, but ${server} did not respond — is the server running?${shape}`
    : `Saved, but ${server} answered HTTP ${check.status} instead of 200.${shape}`;
  statusEl.className = "warn";
}

saveEl.addEventListener("click", save);
// Called directly, NOT bound to DOMContentLoaded: this script tag sits at the
// end of <body>, so by the time it runs the event may already have fired and
// the listener would never see it. (The old code had this latent bug; the
// fields happened to be populated because Chrome fires it late enough often
// enough.)
load();
