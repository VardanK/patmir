// Patmir popup.
//
// Three states (UI):
//   "Translate this"   — no cache entry, click to enqueue
//   "Translating… N%"  — job in flight; polls every 2 s
//   "Play dub"         — cache hit; toggles playback in the active tab

const urlEl = document.getElementById("url");
const statusEl = document.getElementById("status");
const ctaEl = document.getElementById("cta");
const secondaryEl = document.getElementById("secondary");
const errorEl = document.getElementById("error");
const openOptions = document.getElementById("open-options");

openOptions.addEventListener("click", async (e) => {
  e.preventDefault();
  try {
    await chrome.runtime.openOptionsPage();
  } catch (err) {
    // Fallback for the case where the options page isn't declared correctly
    // in the manifest yet — happens until the user reloads the extension
    // after manifest changes.
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

/** Send a message to a tab's content script, lazily injecting the script
 * if it's not already running there (the case for any YouTube tab that
 * was open before the extension was last reloaded). */
async function tabMessage(tabId, msg) {
  try {
    return await chrome.tabs.sendMessage(tabId, msg);
  } catch (e) {
    const m = String(e.message || e);
    if (!/Could not establish connection|Receiving end does not exist/i.test(m)) {
      throw e;
    }
    // Content script absent — inject it now, then retry once.
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["content.js"],
      });
    } catch (inj) {
      throw new Error("Content script could not be injected: " + String(inj.message || inj));
    }
    return await chrome.tabs.sendMessage(tabId, msg);
  }
}

function setError(text) {
  if (text) {
    errorEl.textContent = text;
    errorEl.hidden = false;
  } else {
    errorEl.textContent = "";
    errorEl.hidden = true;
  }
}

function setCta(label, disabled, handler, primary = true) {
  ctaEl.textContent = label;
  ctaEl.disabled = disabled;
  ctaEl.className = primary ? "primary" : "";
  ctaEl.onclick = handler || null;
}

function setSecondary(label, handler) {
  if (!label) {
    secondaryEl.hidden = true;
    secondaryEl.onclick = null;
    return;
  }
  secondaryEl.hidden = false;
  secondaryEl.textContent = label;
  secondaryEl.onclick = handler || null;
}

async function refresh() {
  setError(null);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    statusEl.textContent = "No active tab.";
    return;
  }
  activeTabId = tab.id;
  activeUrl = tab.url || "";
  urlEl.textContent = activeUrl;

  if (!/^https?:\/\//.test(activeUrl)) {
    statusEl.textContent = "Patmir works on web video pages.";
    setCta("—", true, null);
    return;
  }

  let lookup;
  try {
    lookup = await sw({ type: "cache:lookup", url: activeUrl });
  } catch (e) {
    statusEl.textContent = "Server unreachable.";
    setError(String(e.message || e));
    setCta("Retry", false, refresh);
    return;
  }

  if (lookup.cached) {
    // Is the content script currently playing the dub?
    let status = null;
    try {
      status = await tabMessage(activeTabId, { type: "patmir:status" });
    } catch (_) { /* content script may not be loaded on this URL */ }
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

async function submitJob(force = false) {
  setError(null);
  setCta(force ? "Regenerating…" : "Submitting…", true, null);
  setSecondary(null, null);
  try {
    // Make sure any in-tab playback stops before we drop the cache entry.
    if (force) {
      try { await tabMessage(activeTabId, { type: "patmir:stop" }); } catch (_) { /* ok */ }
    }
    const rec = await sw({ type: "jobs:submit", url: activeUrl, force });
    currentJobId = rec.job_id;
    if (rec.status === "done" && rec.cache_key && !force) {
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

// Two-click arm-and-fire confirmation. `confirm()` / `alert()` are silently
// blocked in MV3 popups, so we can't use a native dialog.
let regenArmedAt = 0;
const REGEN_ARM_WINDOW_MS = 3000;
function regenerateJob() {
  const now = Date.now();
  if (now - regenArmedAt > REGEN_ARM_WINDOW_MS) {
    regenArmedAt = now;
    setSecondary("Click again to confirm regenerate", regenerateJob);
    setError("This will delete the current dub and rerun the pipeline (several minutes).");
    setTimeout(() => {
      // If still armed and the user didn't click, revert label.
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
    setCta("Play dub", false, () => playInTab(rec.cache_key));
  } else if (rec.status === "failed") {
    clearInterval(pollHandle);
    pollHandle = null;
    setError(rec.error || "Job failed.");
    setCta("Retry", false, submitJob);
  }
}

function renderJob(rec) {
  const stages = rec.stage_states || {};
  const allOrder = ["download", "ingest", "separate", "diarize", "emotion", "asr", "translate", "tts", "mux", "output"];
  // Only render stages that exist on this job (download is URL-only).
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
    if (rec.live_msg && rec.live_stage === running) {
      detail += " — " + rec.live_msg;
    }
    line = "Running — " + detail + "  (" + done + "/" + order.length + ")";
  } else {
    line = "Running…  (" + done + "/" + order.length + ")";
  }
  statusEl.textContent = line;
  setCta("Translating…", true, null);
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
  try {
    await tabMessage(activeTabId, { type: "patmir:stop" });
  } catch (_) { /* tab navigated away */ }
  refresh();
}

// Re-render whenever the active tab's URL changes underneath us (e.g.
// YouTube SPA navigation between Shorts while the popup is open).
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (tabId === activeTabId && changeInfo.url) {
    refresh();
  }
});

document.addEventListener("DOMContentLoaded", refresh);
