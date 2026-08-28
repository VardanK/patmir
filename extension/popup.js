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
//
// This window holds NO dub state of its own on a YouTube tab. The content
// script's session (lib/session.js, driven by overlay.js) is the single owner
// there, and this file renders it over two messages — which is what makes the
// popup and the in-page Թարգմանել / Չթարգմանել button agree by construction
// rather than by reconciliation.
//
// It cannot delegate unconditionally, though: the popup works on any http(s)
// page and content scripts are only injected on YouTube. So there is a second
// path running the SAME state machine locally with no player attached, which
// reproduces the pre-button behaviour off-YouTube exactly.
//
// The job protocol itself — which statuses are terminal, that `waiting` means
// the accelerator lock, the download_pct special case — used to live here and
// now lives in lib/session.js, in one copy. See the note on
// `contentScriptFiles` below for what duplicating knowledge across these two
// surfaces cost last time.

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
let remote = false;          // is the tab's own session driving?
let localSession = null;     // used only where no content script exists
let pollHandle = null;

async function sw(msg) {
  const resp = await chrome.runtime.sendMessage(msg);
  if (!resp || !resp.ok) {
    const err = resp && resp.error ? resp.error : "Unknown error.";
    throw new Error(err);
  }
  return resp.data || resp;
}

/** The content-script files, in manifest order.
 *
 * Read from the manifest rather than hardcoded, because there are TWO paths
 * that inject them — the declarative `content_scripts` entry and the fallback
 * below — and a hardcoded list silently disagreed with the manifest the moment
 * `lib/sync.js` was added. The fallback then injected `content.js` alone, whose
 * first act is to check for `PatmirSync` and bail, so the retry hit the very
 * "Receiving end does not exist" it was trying to repair. One source of truth.
 */
function contentScriptFiles() {
  const cs = (chrome.runtime.getManifest().content_scripts || [])[0];
  return (cs && cs.js) || ["lib/sync.js", "lib/session.js", "content.js", "overlay.js"];
}

async function tabMessage(tabId, msg) {
  try {
    return await chrome.tabs.sendMessage(tabId, msg);
  } catch (e) {
    const m = String(e.message || e);
    if (!/Could not establish connection|Receiving end does not exist/i.test(m)) {
      throw e;
    }
    // No listener in the tab. Normal after the extension is reloaded: content
    // scripts are injected at page load, so already-open tabs are left with an
    // orphaned script and no receiver. Inject and retry rather than making the
    // viewer reload the page.
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: contentScriptFiles() });
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
      renderModeToggle();
      // The overlay watches `patmirMode` in storage and re-looks-up on its own,
      // so on a YouTube tab this is already in flight; command it anyway so the
      // popup does not have to wait for the storage event to round-trip.
      await command({ action: "set-mode", mode });
      refresh();
    };
  }
  modeNoteEl.textContent = MODE_NOTES[mode] || "";
}

/** The content script's session is the single owner of dub state for its tab.
 *
 * The popup cannot simply delegate unconditionally: it also works on any
 * http(s) page, and content scripts are only injected on YouTube. So there are
 * two paths, both running the SAME state machine from lib/session.js — one
 * remotely over two messages, one locally with no player. That is why there is
 * no reconciliation protocol anywhere: on a YouTube tab there is exactly one
 * state, and this window renders it.
 */
async function getState() {
  if (remote) {
    const resp = await tabMessage(activeTabId, {
      type: "patmir:session-state", popupOpen: true,
    });
    if (!resp || !resp.ok) throw new Error(resp && resp.error ? resp.error : "No session.");
    return resp.state;
  }
  if (!localSession) return null;
  return localSession.getState();
}

async function command(msg) {
  try {
    if (remote) {
      const resp = await tabMessage(activeTabId, { type: "patmir:session-command", ...msg });
      if (!resp || !resp.ok) throw new Error(resp && resp.error ? resp.error : "Command failed.");
      return resp.state;
    }
    if (!localSession) return null;
    if (msg.action === "primary") await localSession.primary();
    else if (msg.action === "start") await localSession.start(!!msg.force);
    else if (msg.action === "stop") await localSession.stop();
    else if (msg.action === "regenerate") await localSession.regenerate();
    else if (msg.action === "set-mode") await localSession.setMode(msg.mode);
    else if (msg.action === "refresh") await localSession.refresh();
    return localSession.getState();
  } catch (e) {
    setError(String(e.message || e));
    return null;
  }
}

function makeLocalSession() {
  return PatmirSession.createSession({
    mode,
    // Unchanged from the popup's historical behaviour: a finished job offers
    // "Play dub", it does not start playing by itself.
    autoPlayOnDone: false,
    transport: {
      sw,
      player: PatmirSession.NULL_PLAYER,
      getUrl: () => activeUrl,
      storage: chrome.storage.local,
      now: () => Date.now(),
      openOptions: () => chrome.runtime.openOptionsPage(),
    },
  });
}

function stopPolling() {
  if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
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

  // Probe for the tab's own session. `tabMessage` re-injects the content
  // scripts if they are missing (normal after an extension reload), so on a
  // YouTube tab this effectively always succeeds.
  remote = false;
  try {
    const resp = await tabMessage(activeTabId, {
      type: "patmir:session-state", popupOpen: true,
    });
    remote = !!(resp && resp.ok);
  } catch (_) { remote = false; }

  if (!remote && !localSession) localSession = makeLocalSession();
  if (!remote) await localSession.refresh();
  else await command({ action: "refresh" });

  render();
  stopPolling();
  // 1 s, and only re-reading: the polling itself happens in whichever session
  // owns the job, so this is a view refresh rather than a second poll loop.
  pollHandle = setInterval(pull, 1000);
}

async function pull() {
  let state;
  try { state = await getState(); } catch (e) { setError(String(e.message || e)); return; }
  if (!state) return;
  if (!remote && localSession) await localSession.tick();
  render(state);
}

// Last state we rendered, so a command's reply can be rendered without a
// second round trip.
let lastState = null;

function render(state) {
  const st = state || lastState;
  if (!st) return;
  lastState = st;
  const v = PatmirSession.deriveView(st);
  setError(v.errorText || (st.errorKind ? st.error : null));
  statusEl.textContent = PatmirSession.formatSubtitle(v, PatmirSession.STRINGS_EN);
  setCta(
    PatmirSession.formatLabel(v, PatmirSession.STRINGS_EN),
    v.disabled,
    () => { command({ action: "primary" }).then((s) => render(s)); },
  );
  // Regenerate stays a popup-only, two-click-armed control: it deletes a
  // finished dub and reruns a multi-minute pipeline, which has no business
  // being one press away on the page itself.
  const canRegen = v.kind === "cached-idle" || v.kind === "playing";
  setSecondary(canRegen ? "Regenerate dub" : null, canRegen ? regenerateJob : null);
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
  command({ action: "regenerate" }).then((s) => render(s));
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
