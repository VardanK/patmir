// Patmir popup.
//
// Two dub modes (toggle at top, persisted in chrome.storage.local):
//   "Quality"     — D1 batch pipeline → cached dub.wav → best quality, one-time
//                   wait, instant rewatch. Unchanged from the original flow.
//   "Watch-along" — D3 streaming: a first-chunk prelude then per-section dub
//                   streamed behind a short buffer while you watch. Sections
//                   are written to the cache as they land, so a rewatch is free.
//
// The mode is sent on every cache lookup + job submit and folded into the cache
// key server-side, so the two modes cache independently.

const urlEl = document.getElementById("url");
const statusEl = document.getElementById("status");
const ctaEl = document.getElementById("cta");
const secondaryEl = document.getElementById("secondary");
const errorEl = document.getElementById("error");
const openOptions = document.getElementById("open-options");
const modesEl = document.getElementById("modes");
const modeNoteEl = document.getElementById("mode-note");

const MODE_NOTES = {
  "quality": "Best quality. One-time wait, then instant.",
  "watch-along": "Stream the dub while you watch (~20–30s to first audio).",
};

let mode = "quality";

openOptions.addEventListener("click", async (e) => {
  e.preventDefault();
  try {
    await chrome.runtime.openOptionsPage();
  } catch (err) {
    try {
      const url = chrome.runtime.getURL("options.html");
      await chrome.tabs.create({ url });
    } catch (err2) {
      setError("Could not open options page: " + String(err2.message || err2));
    }
  }
});

let activeTabId = null;
let activeUrl = null;
let currentJobId = null;
let pollHandle = null;

async function sw(msg) {
  const resp = await chrome.runtime.sendMessage(msg);
  if (!resp || !resp.ok) {
    const err = resp && resp.error ? resp.error : "Unknown error.";
    throw new Error(err);
  }
  return resp.data || resp;
}

async function tabMessage(tabId, msg) {
  try {
    return await chrome.tabs.sendMessage(tabId, msg);
  } catch (e) {
    const m = String(e.message || e);
    if (!/Could not establish connection|Receiving end does not exist/i.test(m)) {
      throw e;
    }
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    } catch (inj) {
      throw new Error("Content script could not be injected: " + String(inj.message || inj));
    }
    return await chrome.tabs.sendMessage(tabId, msg);
  }
}

function setError(text) {
  if (text) { errorEl.textContent = text; errorEl.hidden = false; }
  else { errorEl.textContent = ""; errorEl.hidden = true; }
}

function setCta(label, disabled, handler, primary = true) {
  ctaEl.textContent = label;
  ctaEl.disabled = disabled;
  ctaEl.className = primary ? "primary" : "";
  ctaEl.onclick = handler || null;
}

function setSecondary(label, handler) {
  if (!label) { secondaryEl.hidden = true; secondaryEl.onclick = null; return; }
  secondaryEl.hidden = false;
  secondaryEl.textContent = label;
  secondaryEl.onclick = handler || null;
}

function renderModeToggle() {
  for (const btn of modesEl.querySelectorAll("button")) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
    btn.onclick = async () => {
      if (mode === btn.dataset.mode) return;
      mode = btn.dataset.mode;
      await chrome.storage.local.set({ patmirMode: mode });
      stopPolling();
      renderModeToggle();
      refresh();
    };
  }
  modeNoteEl.textContent = MODE_NOTES[mode] || "";
}

function stopPolling() {
  if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
  currentJobId = null;
}

async function refresh() {
  setError(null);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) { statusEl.textContent = "No active tab."; return; }
  activeTabId = tab.id;
  activeUrl = tab.url || "";
  urlEl.textContent = activeUrl;

  if (!/^https?:\/\//.test(activeUrl)) {
    statusEl.textContent = "Patmir works on web video pages.";
    setCta("—", true, null);
    setSecondary(null, null);
    return;
  }

  let lookup;
  try {
    lookup = await sw({ type: "cache:lookup", url: activeUrl, mode });
  } catch (e) {
    statusEl.textContent = "Server unreachable.";
    setError(String(e.message || e));
    setCta("Retry", false, refresh);
    return;
  }

  if (mode === "watch-along") return refreshWatchAlong(lookup);
  return refreshQuality(lookup);
}

// ---- Quality mode (D1, unchanged) ------------------------------------

async function refreshQuality(lookup) {
  if (lookup.cached) {
    let status = null;
    try { status = await tabMessage(activeTabId, { type: "patmir:status" }); }
    catch (_) { /* content script may not be loaded */ }
    if (status && status.ok && status.playing) {
      statusEl.textContent = "Dub is playing.";
      setCta("Stop dub", false, () => stopInTab());
      setSecondary("Regenerate dub", () => regenerateJob());
    } else {
      const disabled = status && status.disabled;
      statusEl.textContent = disabled
        ? "Dub cached — auto-play is off for this video."
        : "Dub is ready (cached).";
      setCta("Play dub", false, () => playInTab(lookup.cache_key));
      setSecondary("Regenerate dub", () => regenerateJob());
    }
  } else {
    statusEl.textContent = "Not dubbed yet.";
    setCta("Translate this", false, () => submitJob());
    setSecondary(null, null);
  }
}

// ---- Watch-along mode (D3) -------------------------------------------

async function refreshWatchAlong(lookup) {
  let status = null;
  try { status = await tabMessage(activeTabId, { type: "patmir:status" }); }
  catch (_) { /* content script may not be loaded */ }
  if (status && status.ok && status.playing && status.mode === "watch-along") {
    statusEl.textContent = "Watch-along playing.";
    setCta("Stop", false, () => stopInTab());
    setSecondary(null, null);
    return;
  }
  if (lookup.cached) {
    statusEl.textContent = "Watch-along ready (cached).";
    setCta("Play watch-along", false, () => startWatchAlong(lookup.cache_key));
    setSecondary("Regenerate", () => regenerateJob());
  } else if (lookup.partial) {
    statusEl.textContent = "Partially streamed (" + (lookup.sections || 0) + " sections cached).";
    setCta("Resume watch-along", false, () => startWatchAlong(lookup.cache_key));
    setSecondary(null, null);
  } else {
    statusEl.textContent = "Not dubbed yet.";
    setCta("Start watch-along", false, () => startWatchAlong(null));
    setSecondary(null, null);
  }
}

/** Submit (if needed) + tell the content script to begin streaming playback. */
async function startWatchAlong(cacheKey) {
  setError(null);
  setCta("Starting…", true, null);
  setSecondary(null, null);
  try {
    const rec = await sw({ type: "jobs:submit", url: activeUrl, mode });
    currentJobId = rec.job_id;
    cacheKey = cacheKey || rec.cache_key;
    const resp = await tabMessage(activeTabId, { type: "patmir:play-watchalong", cache_key: cacheKey });
    if (!resp || !resp.ok) throw new Error(resp && resp.error ? resp.error : "Could not start playback.");
    setCta("Stop", false, () => stopInTab());
    if (rec.status === "done") {
      statusEl.textContent = "Watch-along playing (cached).";
    } else {
      startPolling(rec);
    }
  } catch (e) {
    setError(String(e.message || e));
    setCta("Start watch-along", false, () => startWatchAlong(cacheKey));
  }
}

// ---- job submit + polling (shared) -----------------------------------

async function submitJob(force = false) {
  setError(null);
  setCta(force ? "Regenerating…" : "Submitting…", true, null);
  setSecondary(null, null);
  try {
    if (force) {
      try { await tabMessage(activeTabId, { type: "patmir:stop" }); } catch (_) { /* ok */ }
    }
    const rec = await sw({ type: "jobs:submit", url: activeUrl, force, mode });
    currentJobId = rec.job_id;
    if (mode === "quality" && rec.status === "done" && rec.cache_key && !force) {
      statusEl.textContent = "Dub is ready (cached).";
      setCta("Play dub", false, () => playInTab(rec.cache_key));
      setSecondary("Regenerate dub", () => regenerateJob());
      return;
    }
    startPolling(rec);
  } catch (e) {
    setError(String(e.message || e));
    setCta(force ? "Regenerate dub" : "Translate this", false, () => submitJob(force));
  }
}

let regenArmedAt = 0;
const REGEN_ARM_WINDOW_MS = 3000;
function regenerateJob() {
  const now = Date.now();
  if (now - regenArmedAt > REGEN_ARM_WINDOW_MS) {
    regenArmedAt = now;
    setSecondary("Click again to confirm regenerate", regenerateJob);
    setError("This will delete the current dub and rerun the pipeline (several minutes).");
    setTimeout(() => {
      if (Date.now() - regenArmedAt >= REGEN_ARM_WINDOW_MS && secondaryEl && !secondaryEl.hidden) {
        setSecondary("Regenerate dub", regenerateJob);
        setError(null);
      }
    }, REGEN_ARM_WINDOW_MS + 100);
    return;
  }
  regenArmedAt = 0;
  setError(null);
  submitJob(true);
}

function startPolling(initialRec) {
  renderJob(initialRec);
  if (pollHandle) clearInterval(pollHandle);
  pollHandle = setInterval(pollOnce, 2000);
}

async function pollOnce() {
  if (!currentJobId) return;
  let rec;
  try {
    rec = await sw({ type: "jobs:get", job_id: currentJobId });
  } catch (e) {
    setError(String(e.message || e));
    return;
  }
  renderJob(rec);
  if (rec.status === "done") {
    clearInterval(pollHandle);
    pollHandle = null;
    if (mode === "watch-along") {
      setCta("Stop", false, () => stopInTab());
    } else {
      setCta("Play dub", false, () => playInTab(rec.cache_key));
    }
  } else if (rec.status === "failed") {
    clearInterval(pollHandle);
    pollHandle = null;
    setError(rec.error || "Job failed.");
    setCta("Retry", false, () => (mode === "watch-along" ? startWatchAlong(rec.cache_key) : submitJob()));
  }
}

function renderJob(rec) {
  if (rec.mode === "watch-along") return renderStreamJob(rec);
  const stages = rec.stage_states || {};
  const allOrder = ["download", "ingest", "separate", "diarize", "emotion", "asr", "translate", "tts", "mux", "output"];
  const order = allOrder.filter((s) => s in stages);
  const running = order.find((s) => stages[s] === "running");
  const done = order.filter((s) => stages[s] === "done").length;

  let line;
  if (rec.status === "queued") {
    line = "Queued…";
  } else if (running) {
    let detail = running;
    if (running === "download" && typeof rec.download_pct === "number") {
      detail = "download " + Math.round(rec.download_pct * 100) + "%";
    } else if (rec.live_stage === running && typeof rec.live_pct === "number") {
      detail = running + " " + Math.round(rec.live_pct * 100) + "%";
    }
    if (rec.live_msg && rec.live_stage === running) detail += " — " + rec.live_msg;
    line = "Running — " + detail + "  (" + done + "/" + order.length + ")";
  } else {
    line = "Running…  (" + done + "/" + order.length + ")";
  }
  statusEl.textContent = line;
  setCta("Translating…", true, null);
  setSecondary(null, null);
}

function renderStreamJob(rec) {
  let line;
  if (rec.status === "queued") line = "Queued…";
  else if (rec.status === "prelude") line = "Analyzing opening (prelude)…";
  else if (rec.status === "streaming") {
    line = "Streaming — " + (rec.sections_done || 0) + " sections dubbed";
  } else if (rec.status === "done") line = "Watch-along complete (cached).";
  else line = rec.status;
  statusEl.textContent = line;
  // Keep the Stop control live while streaming so the user can bail.
  setCta("Stop", false, () => stopInTab());
  setSecondary(null, null);
}

async function playInTab(cacheKey) {
  setError(null);
  setCta("Loading dub…", true, null);
  try {
    const resp = await tabMessage(activeTabId, { type: "patmir:play", cache_key: cacheKey });
    if (!resp || !resp.ok) throw new Error(resp && resp.error ? resp.error : "Playback failed.");
    setCta("Stop dub", false, () => stopInTab());
    statusEl.textContent = "Playing dub.";
  } catch (e) {
    setError("Playback failed: " + String(e.message || e));
    setCta("Play dub", false, () => playInTab(cacheKey));
  }
}

async function stopInTab() {
  stopPolling();
  try { await tabMessage(activeTabId, { type: "patmir:stop" }); } catch (_) { /* tab navigated away */ }
  refresh();
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (tabId === activeTabId && changeInfo.url) refresh();
});

document.addEventListener("DOMContentLoaded", async () => {
  const { patmirMode = "quality" } = await chrome.storage.local.get("patmirMode");
  mode = (patmirMode === "watch-along") ? "watch-along" : "quality";
  renderModeToggle();
  refresh();
});
