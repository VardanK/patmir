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

async function save() {
  const server = serverEl.value.trim() || "http://127.0.0.1:8765";
  const token = tokenEl.value.trim();
  if (!token) {
    statusEl.textContent = "Token is required.";
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
