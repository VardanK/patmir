// Patmir in-page control — the button under the YouTube player.
//
// `Թարգմանել` / `Չթարգմանել`: one control that turns the Armenian audio track
// on and off, doing everything the popup CTA does (cache lookup, job submit,
// progress, playback) without the viewer leaving the video.
//
// WHY THIS IS A SEPARATE FILE from content.js:
//   content.js owns the Web Audio graph, the volume/mute arbitration and the
//   drift tick, and its bootstrap guard exists because a listener registered
//   over a half-initialized module reported "Cannot access X before
//   initialization" instead of the real problem. Keeping every `document`
//   mutation out of that file means its sandbox test still boots with no DOM
//   globals at all — which is the regression guard that keeps button code from
//   creeping back in.
//
// This file therefore touches the DOM and nothing else: all state lives in
// lib/session.js (pure, transport-injected), all playback goes through the
// four-method `PatmirPlayer` facade content.js publishes, and all HTTP stays in
// sw.js. It owns exactly one timer.
//
// SYNC WITH THE POPUP is by construction rather than by protocol: on a YouTube
// tab this script owns the only session, and the popup renders it over two
// messages. There is no second state to reconcile.
(function () {
  if (window.__patmirOverlayInjected) return;

  // Both dependencies are checked BEFORE anything is registered, for the same
  // reason content.js does it: a half-live overlay is harder to diagnose than
  // no overlay.
  const Session = globalThis.PatmirSession;
  if (!Session) {
    console.error(
      "[patmir] lib/session.js did not load, so the in-page button is disabled. " +
      "manifest.json must list it before overlay.js in content_scripts.js — " +
      "if you just updated the extension, hit Reload on chrome://extensions " +
      "(a manifest change needs a full reload, not a page refresh).");
    return;
  }
  const Player = globalThis.PatmirPlayer;
  if (!Player) {
    console.error(
      "[patmir] content.js did not publish PatmirPlayer, so the in-page button " +
      "is disabled. manifest.json must list content.js BEFORE overlay.js in " +
      "content_scripts.js. Hit Reload on chrome://extensions.");
    return;
  }
  const Sync = globalThis.PatmirSync;

  window.__patmirOverlayInjected = true;

  const ROOT_ID = "patmir-toggle-root";
  const TICK_MS = 500;
  // Every 4th tick drives the session, so job polling stays at ~2 s while the
  // cheap mount/render check runs at 500 ms.
  const SESSION_EVERY = 4;

  let root = null;
  let shadow = null;
  let buttonEl = null;
  let subtitleEl = null;
  let fillEl = null;
  let labelEl = null;
  let session = null;
  let tickHandle = null;
  let tickCount = 0;
  let lastRendered = null;
  let floating = false;
  let adSeekAttempts = 0;
  let adWasShowing = false;
  let dead = false;

  // --- transport -------------------------------------------------------

  /** Unwrap the service worker's {ok, data} envelope, or throw its error. */
  async function sw(msg) {
    const resp = await chrome.runtime.sendMessage(msg);
    if (!resp || !resp.ok) {
      throw new Error(resp && resp.error ? resp.error : "Unknown error.");
    }
    return resp.data || resp;
  }

  function makeSession(mode) {
    return Session.createSession({
      mode,
      // The point of an in-page control is not to need a second press once the
      // dub is ready. (The popup keeps its two-step behaviour.)
      autoPlayOnDone: true,
      transport: {
        sw,
        player: {
          play: (k) => Player.play(k),
          playWatchAlong: (k) => Player.playWatchAlong(k),
          stop: () => Player.stop(),
          status: () => Player.status(),
        },
        getUrl: () => window.location.href,
        storage: chrome.storage.local,
        now: () => Date.now(),
        openOptions: () => {
          // A content script cannot open the options page itself.
          chrome.runtime.sendMessage({ type: "options:open" }).catch(() => {});
        },
      },
    });
  }

  // --- mounting --------------------------------------------------------

  function insertionPoint() {
    const present = {};
    for (const m of Session.MOUNTS) {
      if (!m.selector) continue;
      try { present[m.name] = !!document.querySelector(m.selector); } catch (_) {}
    }
    // The selector-free last resort: walk up from the <video> itself, in the
    // spirit of findVideo(). Survives a full id rename.
    let ancestor = null;
    try {
      const v = Player.findVideo();
      const box = v && v.closest ? v.closest("#player, ytd-player, #movie_player") : null;
      ancestor = box && box.parentElement ? box : null;
    } catch (_) {}
    present["video-ancestor"] = !!ancestor;

    const plan = Session.mountPlan(present);
    if (!plan) return null;
    let node = ancestor;
    if (plan.selector) {
      try { node = document.querySelector(plan.selector); } catch (_) { node = null; }
    }
    if (!node) return null;
    return { node, position: plan.position, name: plan.name };
  }

  function ensureMounted() {
    if (dead) return;
    const kind = Session.pageKind(window.location.href);
    const wantFloating = Session.isFloatingPage(kind);
    const wantInline = Session.isInlinePage(kind);
    if (!wantFloating && !wantInline) { unmount(); return; }

    if (root && document.contains(root) && floating === wantFloating) return;

    // A stale root can survive an executeScript re-injection whose interval was
    // already cleared, so clear by id rather than by our own reference.
    unmount();
    const existing = document.getElementById(ROOT_ID);
    if (existing && existing.remove) { try { existing.remove(); } catch (_) {} }

    floating = wantFloating;
    if (wantFloating) {
      mountFloating();
    } else {
      const at = insertionPoint();
      if (!at) return;   // no place for it; the popup still works
      mountInline(at);
    }
    render(true);
  }

  function buildRoot() {
    const el = document.createElement("div");
    el.id = ROOT_ID;
    // The host's own box is the only thing outside the shadow boundary, so it
    // is the only place YouTube's cascade could touch — set every property that
    // matters explicitly rather than inheriting.
    el.style.setProperty("display", "block");
    el.style.setProperty("all", "initial");
    el.style.setProperty("display", "block");
    shadow = el.attachShadow ? el.attachShadow({ mode: "open" }) : null;
    if (!shadow) return null;
    applyStyles(shadow);
    shadow.appendChild(buildUi());
    return el;
  }

  function mountInline(at) {
    const el = buildRoot();
    if (!el) return;
    el.style.setProperty("margin", "12px 0");
    try {
      if (at.position === "prepend" && at.node.firstChild) {
        at.node.insertBefore(el, at.node.firstChild);
      } else if (at.position === "prepend") {
        at.node.appendChild(el);
      } else if (at.position === "before") {
        at.node.parentElement.insertBefore(el, at.node);
      } else {
        at.node.parentElement.insertBefore(el, at.node.nextSibling);
      }
    } catch (_) { return; }
    root = el;
  }

  function mountFloating() {
    // Shorts is a recycler with no stable region under the player, and an embed
    // is somebody else's article layout — neither can take a full-width block.
    // So: a compact pill pinned over the bottom-right of the viewport, which
    // needs no cooperation from the page's structure at all.
    const el = buildRoot();
    if (!el) return;
    el.style.setProperty("position", "fixed");
    el.style.setProperty("right", "16px");
    el.style.setProperty("bottom", "88px");
    el.style.setProperty("z-index", "2000");
    // Only the pill itself takes clicks; the rest of the host is inert so it
    // cannot swallow a scroll or a tap meant for the page.
    el.style.setProperty("pointer-events", "none");
    try { document.body.appendChild(el); } catch (_) { return; }
    root = el;
  }

  function unmount() {
    if (root && root.remove) { try { root.remove(); } catch (_) {} }
    root = null; shadow = null; buttonEl = null; subtitleEl = null;
    lastRendered = null;
  }

  const CSS = `
    :host { all: initial; }
    .wrap {
      font-family: "Noto Sans Armenian", "Arial Armenian", system-ui, -apple-system, sans-serif;
      display: block; pointer-events: auto;
    }
    button {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 2px; width: 100%; min-height: 52px; padding: 8px 16px; cursor: pointer;
      border: none; border-radius: 12px; font: inherit; font-size: 16px; font-weight: 600;
      background: var(--yt-spec-brand-button-background, #065fd4);
      color: var(--yt-spec-static-brand-white, #fff);
      transition: filter 120ms ease;
    }
    button:hover:not(:disabled) { filter: brightness(1.08); }
    button:disabled { cursor: default; opacity: 0.65; }
    button.stop {
      background: var(--yt-spec-badge-chip-background, rgba(0,0,0,0.05));
      color: var(--yt-spec-text-primary, #0f0f0f);
      box-shadow: inset 0 0 0 1px var(--yt-spec-10-percent-layer, rgba(0,0,0,0.1));
    }
    /* The control doubles as its own progress bar: green filling the grey
       left-to-right. The author's report was that the old button read as a
       status ("not translating") with the percentage buried in 12 px of
       subtitle, so the work now shows in the shape of the control itself.
       Text sits above the fill and stays dark, so contrast does not depend on
       how far along the job is. */
    button { position: relative; overflow: hidden; }
    .fill {
      position: absolute; left: 0; top: 0; bottom: 0; width: 0;
      background: rgba(30, 142, 62, 0.30);
      border-right: 1px solid rgba(30, 142, 62, 0.55);
      transition: width 300ms ease;
      pointer-events: none; z-index: 0;
    }
    .fill.hidden { display: none; }
    .label, .sub { position: relative; z-index: 1; }
    button.error { background: #c5221f; color: #fff; }
    button.warn  { background: #b06000; color: #fff; }
    .sub { font-size: 12px; font-weight: 400; opacity: 0.85; }
    .sub:empty { display: none; }
    :host(.pill) button {
      min-height: 0; width: auto; border-radius: 999px; font-size: 14px;
      padding: 8px 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    }
  `;

  function applyStyles(target) {
    // Constructable stylesheets are not inline styles, so YouTube's style-src
    // CSP does not apply; the <style> path is the fallback for anything older.
    try {
      if (typeof CSSStyleSheet === "function" && "replaceSync" in CSSStyleSheet.prototype) {
        const sheet = new CSSStyleSheet();
        sheet.replaceSync(CSS);
        target.adoptedStyleSheets = [sheet];
        return;
      }
    } catch (_) {}
    const style = document.createElement("style");
    style.textContent = CSS;
    target.appendChild(style);
  }

  function buildUi() {
    const wrap = document.createElement("div");
    wrap.className = "wrap";
    wrap.setAttribute("lang", "hy");

    buttonEl = document.createElement("button");
    buttonEl.type = "button";

    fillEl = document.createElement("div");
    fillEl.className = "fill hidden";
    // Decorative: the percentage is already spoken by the aria-live subtitle,
    // and a second announcing element would say everything twice.
    fillEl.setAttribute("aria-hidden", "true");

    labelEl = document.createElement("span");
    labelEl.className = "label";

    subtitleEl = document.createElement("span");
    subtitleEl.className = "sub";
    // The state changes under the viewer without them acting, so announce it.
    subtitleEl.setAttribute("aria-live", "polite");

    buttonEl.appendChild(labelEl);
    buttonEl.appendChild(subtitleEl);
    // Appended LAST but painted underneath via z-index, rather than inserted
    // first: the label stays `children[0]`, which the popup-parity tests and
    // anything else reading the button's shape already rely on.
    buttonEl.appendChild(fillEl);
    buttonEl.addEventListener("click", onPress);
    // A focused button plus YouTube's document-level shortcuts means space and
    // k would both activate the button AND toggle the player. Stop the bubble,
    // but never preventDefault on space — that would break the button itself.
    buttonEl.addEventListener("keydown", (e) => {
      if (["k", "j", "l", "f", "m", " ", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]
        .includes(e.key)) {
        e.stopPropagation();
      }
    });
    wrap.appendChild(buttonEl);
    return wrap;
  }

  // --- rendering -------------------------------------------------------

  function render(force) {
    if (!buttonEl || !session) return;
    const v = session.view();
    const label = Session.formatLabel(v, Session.STRINGS_HY);
    const sub = Session.formatSubtitle(v, Session.STRINGS_HY);
    // Progress joins the signature so the bar advances between ticks whose
    // text has not changed — `queued` and `streaming` carry no percentage in
    // their subtitle, and without this the fill would sit still.
    const pct = typeof v.progress === "number" ? Math.round(v.progress * 100) : null;
    const signature = [v.kind, label, sub, v.disabled, v.tone, pct].join("|");
    // Diffed on purpose: unconditional DOM writes at 2 Hz inside YouTube's
    // layout, on a page decoding video, is what shows up as dropped frames and
    // gets blamed on the dub.
    if (!force && signature === lastRendered) return;
    lastRendered = signature;

    labelEl.textContent = label;
    subtitleEl.textContent = sub;
    buttonEl.disabled = !!v.disabled;
    buttonEl.setAttribute("aria-busy", v.busy ? "true" : "false");
    // CANCEL shares STOP's neutral chrome: both are the "there is something to
    // interrupt" state, and the difference between them is the word, not the
    // colour.
    const neutral = v.label === Session.LABELS.STOP
                 || v.label === Session.LABELS.CANCEL;
    buttonEl.className =
      (neutral ? "stop" : "") +
      (v.tone === "error" ? " error" : v.tone === "warn" ? " warn" : "");

    if (pct === null) {
      fillEl.className = "fill hidden";
      fillEl.style.width = "0%";
    } else {
      fillEl.className = "fill";
      fillEl.style.width = pct + "%";
    }
    if (root) {
      // Pill styling is keyed off the host so the shadow CSS can select it.
      if (floating) root.classList.add("pill"); else root.classList.remove("pill");
    }
  }

  function onPress() {
    if (!session) return;
    session.primary().then(() => render()).catch(() => render());
  }

  // --- ads -------------------------------------------------------------

  function handleAds() {
    if (!Sync || !Player.adState) return;
    const st = Player.adState();
    if (session) session.setAdShowing(st.showing);
    // Attempt budget is per ad break, not per page.
    if (!st.showing) { adSeekAttempts = 0; adWasShowing = false; return; }
    if (!adWasShowing) { adSeekAttempts = 0; adWasShowing = true; }

    const v = Player.findVideo();
    const plan = Sync.adSkipPlan({
      enabled: skipAdsEnabled,
      // Separate opt-in, default OFF. Clicking a skip control that only exists
      // during a skippable ad is safe; writing `currentTime` on the shared media
      // element is not, and the first version of this feature proved it by
      // jumping every video the author opened to its end.
      allowSeek: skipAdSeekEnabled,
      showing: st.showing,
      classConfirmed: st.classConfirmed,
      skipReady: st.skipReady,
      currentTime: v ? v.currentTime : 0,
      duration: v ? v.duration : null,
      attempts: adSeekAttempts,
    });
    if (plan.action === "none") return;
    const applied = Player.applyAdAction(plan.action);
    if (plan.action === "seek" && applied) adSeekAttempts += 1;
  }

  // Default ON: pressing the skip control the moment it goes live is the
  // viewer's own gesture, a second early, and its worst failure is doing
  // nothing.
  let skipAdsEnabled = true;
  // Default OFF, and deliberately a SECOND setting rather than part of the one
  // above. Seeking is the only thing here that can damage playback, and it did:
  // shipped with a broken ad predicate it sent every video straight to its end.
  // It stays opt-in even now that the predicate is fixed and the seek carries
  // three independent guards.
  let skipAdSeekEnabled = false;

  // --- lifecycle -------------------------------------------------------

  function selfDestruct() {
    dead = true;
    if (tickHandle !== null) { clearInterval(tickHandle); tickHandle = null; }
    unmount();
  }

  /** Any chrome.* call in an orphaned content script throws this. A stale button
   *  left rendering "server unreachable" forever is worse than no button. */
  function isOrphaned(err) {
    return /Extension context invalidated|context invalidated/i.test(
      String((err && err.message) || err || ""));
  }

  /** Run one concern, and never let it take the others down with it.
   *
   * Each of these is independent, and they used to share a single try/catch —
   * which meant one throw inside `ensureMounted` (a page whose DOM is not what
   * the probe expects) silently stopped ad handling and rendering on EVERY
   * subsequent tick, with nothing said. An orphaned extension context is the
   * one error that legitimately stops everything.
   */
  function guarded(name, fn) {
    try { fn(); } catch (e) {
      if (isOrphaned(e)) { selfDestruct(); return; }
      if (!warned[name]) { warned[name] = true; console.warn("[patmir] " + name + ":", e); }
    }
  }
  const warned = {};

  function tick() {
    if (dead) return;
    guarded("mount", ensureMounted);
    guarded("ads", handleAds);
    guarded("render", () => render());
    tickCount += 1;
    if (session && (tickCount % SESSION_EVERY === 0)) {
      session.tick().then(() => render()).catch((e) => {
        if (isOrphaned(e)) selfDestruct();
      });
    }
  }

  async function boot() {
    const stored = await chrome.storage.local.get([
      "patmirMode", "patmirSkipAds", "patmirSkipAdSeek",
    ]);
    skipAdsEnabled = stored.patmirSkipAds !== false;
    skipAdSeekEnabled = stored.patmirSkipAdSeek === true;
    session = makeSession(
      stored.patmirMode === Session.WATCH_ALONG ? Session.WATCH_ALONG : Session.QUALITY);
    session.onChange(() => render());
    ensureMounted();
    session.refresh().then(() => render(true)).catch(() => render(true));
    tickHandle = setInterval(tick, TICK_MS);
  }

  // Re-mount on SPA navigation. Two signals, both already trusted by
  // content.js: its 500 ms URL poll (the only one that fires while nothing is
  // playing) and yt-navigate-finish. No MutationObserver — #below is YouTube's
  // churniest subtree, and a subtree observer there fires thousands of times per
  // navigation on the main thread of a video-decoding page. The tick's
  // `document.contains` check catches a replaced parent one frame later, which
  // is imperceptible for a button.
  Player.onUrlChange(() => {
    if (dead || !session) return;
    lastRendered = null;
    adSeekAttempts = 0;
    unmount();
    ensureMounted();
    session.refresh().then(() => render(true)).catch(() => render(true));
  });

  window.addEventListener("yt-navigate-finish", () => {
    setTimeout(() => { if (!dead) ensureMounted(); }, 250);
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || dead) return;
    if ("patmirSkipAds" in changes) {
      skipAdsEnabled = changes.patmirSkipAds.newValue !== false;
    }
    if ("patmirSkipAdSeek" in changes) {
      skipAdSeekEnabled = changes.patmirSkipAdSeek.newValue === true;
    }
    if ("patmirMode" in changes && session) {
      session.setMode(changes.patmirMode.newValue).then(() => render(true)).catch(() => {});
    }
    if (("patmirDisabledVideos" in changes || "patmirAudience" in changes) && session) {
      // The audience level is part of the cache key, so a level change makes the
      // current lookup answer wrong, not merely stale.
      session.refresh().then(() => render(true)).catch(() => {});
    }
  });

  // --- the popup's window onto this session ----------------------------
  //
  // Two messages, and no state of its own on the popup side. That is what makes
  // "whichever you touch, both agree" true by construction: on a YouTube tab
  // there is exactly one session, and the popup renders it.
  const OWN_MESSAGES = new Set(["patmir:session-state", "patmir:session-command"]);

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (!msg || !OWN_MESSAGES.has(msg.type)) return undefined;
    (async () => {
      try {
        if (!session) { sendResponse({ ok: false, error: "session not ready" }); return; }
        if (msg.type === "patmir:session-state") {
          if (msg.popupOpen) session.notePopupOpen();
          sendResponse({ ok: true, state: snapshot() });
          return;
        }
        if (msg.action === "refresh") await session.refresh();
        else if (msg.action === "primary") await session.primary();
        else if (msg.action === "start") await session.start(!!msg.force);
        else if (msg.action === "stop") await session.stop();
        else if (msg.action === "regenerate") await session.regenerate();
        else if (msg.action === "set-mode") await session.setMode(msg.mode);
        render();
        sendResponse({ ok: true, state: snapshot() });
      } catch (e) {
        sendResponse({ ok: false, error: String((e && e.message) || e) });
      }
    })();
    return true;
  });

  /** A structured-clone-safe copy of the session state for the popup. */
  function snapshot() {
    const st = session.getState();
    return {
      mode: st.mode, phase: st.phase, url: st.url,
      lookup: st.lookup, job: st.job, jobId: st.jobId,
      playback: st.playback, error: st.error, errorKind: st.errorKind,
      adShowing: st.adShowing, mountedAt: st.mountedAt, nowMs: st.nowMs,
    };
  }

  boot().catch((e) => {
    if (isOrphaned(e)) { selfDestruct(); return; }
    console.error("[patmir] in-page button failed to start:", e);
  });
})();
