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
    patmirServer, patmirToken, patmirAutoPlayEnabled, patmirAudience, patmirSkipAds,
    patmirSkipAdSeek,
  } = await chrome.storage.local.get([
    "patmirServer", "patmirToken", "patmirAutoPlayEnabled", "patmirAudience",
    "patmirSkipAds", "patmirSkipAdSeek",
  ]);
  serverEl.value = patmirServer || "http://127.0.0.1:8765";
  tokenEl.value = patmirToken || "";
  // Default ON.
  autoplayEl.checked = patmirAutoPlayEnabled !== false;
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
  const token = tokenEl.value.trim();
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
  await chrome.storage.local.set({
    patmirServer: server,
    patmirToken: token,
    patmirAutoPlayEnabled: autoplayEl.checked,
    patmirAudience: audience,
    patmirSkipAds: skipAdsEl.checked,
    patmirSkipAdSeek: skipAdSeekEl.checked,
  });
  statusEl.textContent = "Saved.";
  statusEl.className = "ok";
  setTimeout(() => { statusEl.textContent = ""; }, 2000);
}

saveEl.addEventListener("click", save);
// Called directly, NOT bound to DOMContentLoaded: this script tag sits at the
// end of <body>, so by the time it runs the event may already have fired and
// the listener would never see it. (The old code had this latent bug; the
// fields happened to be populated because Chrome fires it late enough often
// enough.)
load();
