// Patmir content script.
//
// Runs on YouTube pages. Owns the page-context Web Audio graph that plays
// the cached dub synced to the page <video> element. The service worker
// (sw.js) handles all HTTP — this script only talks to it via
// chrome.runtime.sendMessage.

(function () {
  if (window.__patmirInjected) return;

  // Pure decision logic, from lib/sync.js — which manifest.json lists BEFORE
  // this file, since content scripts share one isolated world and load in
  // order. This check has to happen HERE, before anything below is registered.
  //
  // It used to sit further down, after the message listener and the auto-play
  // timers. Returning from there left a half-initialized module: function
  // declarations hoist, so the listener could still reach `startDub`, but every
  // `let` below the return was stranded in the temporal dead zone. Clicking
  // Play then reported "Cannot access 'syncTickHandle' before initialization"
  // instead of the reason — the guard produced exactly the confusing failure it
  // was added to prevent. Bailing before any registration leaves nothing
  // callable, so the only symptom is the message below.
  //
  // `__patmirInjected` is deliberately NOT set on this path, so a reload with
  // the dependency present injects cleanly instead of being suppressed.
  const S = globalThis.PatmirSync;
  if (!S) {
    console.error(
      "[patmir] lib/sync.js did not load, so the dub overlay is disabled. " +
      "manifest.json must list it before content.js in content_scripts.js — " +
      "if you just updated the extension, hit Reload on chrome://extensions " +
      "(a manifest change needs a full reload, not a page refresh).");
    return;
  }

  window.__patmirInjected = true;

  // --- state ----------------------------------------------------------

  /** @type {AudioContext|null} */
  let audioCtx = null;
  /** @type {AudioBuffer|null} */
  let dubBuffer = null;
  /** @type {string|null} */
  let dubBufferCacheKey = null;
  /** @type {AudioBufferSourceNode|null} */
  let dubSource = null;
  /** @type {HTMLVideoElement|null} */
  let video = null;
  /** @type {string|null} */
  let videoSrcAtStart = null;
  /** @type {string|null} */
  let urlAtStart = null;
  /** @type {GainNode|null} */
  let dubGain = null;
  /** @type {number|null} */
  let navWatchHandle = null;
  // What the viewer wants from the page player, tracked via `volumechange`.
  // While a dub is active we keep the element hard-muted -- the dub IS the
  // audio track now -- so the element's own muted/volume stop expressing the
  // viewer's intent and this does instead.
  let viewer = { volume: 1, muted: false };
  // `video.muted = true` writes we have issued and not yet seen echo back as a
  // `volumechange`. See PatmirSync.resolveVolumeEvent for why this is a count
  // and not a timeout.
  let pendingForcedWrites = 0;
  let playing = false;
  // Quality-mode playback anchor: the dub buffer position `audioStartOffset`
  // was sounding at AudioContext time `audioStartCtxTime`, advancing at
  // `currentSourceRate`. Integrating from here gives the audio's position now,
  // which is what drift is measured against. (These three existed before but
  // nothing ever read them -- the correction loop the D1 spec called for was
  // never written.)
  let audioStartCtxTime = 0;
  let audioStartOffset = 0;
  let currentSourceRate = 1;

  // --- YouTube ads ----------------------------------------------------
  //
  // Ads play through the SAME <video> element as the content, which breaks the
  // dub in two ways, and the second one used to be permanent:
  //
  //   (a) `currentTime` during an ad is the AD's timeline, so every anchor
  //       computed from it is wrong.
  //   (b) the src swap fires `emptied`/`loadstart`, which `onVideoSourceChanged`
  //       read as "the viewer navigated to another video" and answered with a
  //       full `stopDub()`. An unskippable mid-roll therefore killed the dub for
  //       the rest of the video, with no way back short of pressing play again.
  //
  // So: detect the ad, silence the dub WITHOUT tearing it down, and let the
  // ordinary "picture confirmed moving" path re-anchor when the content
  // resumes. Detection lives in `adState()` and is polled from the sync tick
  // that already runs -- no new timer.
  let adShowing = false;
  // The content source, remembered across an ad break so the navigation guard
  // has something true to compare against once the ad ends.
  let contentSrc = null;
  // Duration of the video the viewer actually came to watch, sampled whenever
  // the player is NOT showing an ad. The last line of defence for the auto-skip
  // seek: if the element we are about to seek reports this duration, whatever
  // else we believe, it is the content and we must not touch it.
  let contentDuration = null;
  // How long a source change waits for the ad class before being believed to
  // be a navigation. Long enough for YouTube to stamp `#movie_player`, short
  // enough that a genuine Shorts auto-advance keeps dubbing the wrong video
  // for well under a second.
  const SRC_CHANGE_GRACE_MS = 1200;
  let srcChangeGrace = null;

  // --- message routing ------------------------------------------------

  // The playback primitives, extracted from the message handlers so the
  // in-page overlay reaches EXACTLY the same code path the popup does. One
  // implementation matters most for the opt-out bookkeeping below: a stop that
  // skipped `setVideoDisabled(url, true)` would be undone by the auto-play
  // listener within a tick, and the button would look broken.
  async function playQuality(cacheKey) {
    // Explicit play also removes any prior opt-out for this video so the next
    // visit auto-plays again.
    await setVideoDisabled(window.location.href, false);
    stopWatchAlong();
    await startDub(cacheKey);
    return { ok: true };
  }

  async function playWatchAlongFor(cacheKey) {
    await setVideoDisabled(window.location.href, false);
    stopDub();
    waRevivesUsed = 0;          // a deliberate start earns a fresh budget
    await startWatchAlong(cacheKey);
    return { ok: true };
  }

  /**
   * Re-submit a stream that stopped producing, once.
   *
   * Worth trying because the server RESUMES rather than restarting: `submit()`
   * on a `done`-but-incomplete stream picks up from the cached sections
   * (quality issue #21) and no longer re-dubs the cached opening (#31). So a
   * revive costs the rest of the video, not the whole of it.
   *
   * Bounded at one per user-initiated start, because the failure that made
   * this necessary — a NameError on the server's stream path — was
   * deterministic, and an unbounded retry against a deterministic failure is
   * a loop that interrupts the picture every minute.
   */
  async function reviveWatchAlong() {
    const url = window.location.href;
    waRevivesUsed += 1;
    console.warn(`[patmir] dub stream stopped producing — re-submitting `
      + `(attempt ${waRevivesUsed} of ${WA_MAX_REVIVES}).`);
    stopWatchAlong();
    let resp;
    try {
      resp = await chrome.runtime.sendMessage({
        type: "jobs:submit", url, mode: "watch-along",
      });
    } catch (_) { return false; }
    if (!resp || !resp.ok || !resp.data) return false;
    const key = resp.data.cache_key;
    if (!key) return false;
    if (window.location.href !== url) return false;   // navigated meanwhile
    stopDub();
    await startWatchAlong(key);
    return true;
  }

  // Switches the AUDIO TRACK off, nothing else: the dub source stops, the mute
  // control hands `video.muted` back to the viewer's own intent (so the original
  // comes back), and the picture keeps rolling. Never pauses, never seeks.
  async function stopTranslation() {
    // Explicit stop opts this video out of future auto-play — "off" has to stay
    // off, and it is also what stops the auto-play listener restarting the dub
    // the moment `patmirDisabledVideos` changes.
    await setVideoDisabled(window.location.href, true);
    stopPlayback();
    return { ok: true };
  }

  async function playbackStatus() {
    const url = window.location.href;
    const disabled = await isVideoDisabled(url);
    const isPlaying = playing || waActive;
    const mode = waActive ? "watch-along" : (playing ? "quality" : null);
    return {
      ok: true, playing: isPlaying, mode, hasVideo: !!findVideo(), disabled,
      buffering: waActive && waBuffering,
      adShowing: adState().showing,
    };
  }

  // Message types this script answers. Anything else must fall through WITHOUT
  // claiming the response channel: overlay.js registers its own listener in the
  // same isolated world, and returning `true` for its messages left a port open
  // that nothing here ever closed.
  const OWN_MESSAGES = new Set([
    "patmir:play", "patmir:play-watchalong", "patmir:stop",
    "patmir:status", "patmir:current-url",
  ]);

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (!msg || !OWN_MESSAGES.has(msg.type)) return undefined;
    (async () => {
      try {
        if (msg.type === "patmir:play") {
          sendResponse(await playQuality(msg.cache_key));
        } else if (msg.type === "patmir:play-watchalong") {
          sendResponse(await playWatchAlongFor(msg.cache_key));
        } else if (msg.type === "patmir:stop") {
          sendResponse(await stopTranslation());
        } else if (msg.type === "patmir:status") {
          sendResponse(await playbackStatus());
        } else if (msg.type === "patmir:current-url") {
          sendResponse({ ok: true, url: window.location.href });
        }
      } catch (e) {
        sendResponse({ ok: false, error: String((e && e.message) || e) });
      }
    })();
    return true;
  });

  // --- auto-play ------------------------------------------------------

  function canonicalUrl(href) {
    try {
      const u = new URL(href);
      const host = (u.hostname || "").toLowerCase();
      if (host.endsWith("youtube.com")) {
        const v = u.searchParams.get("v");
        if (v) return "yt:" + v;
        const m = u.pathname.match(/^\/(shorts|embed|v|live)\/([^\/]+)/);
        if (m) return "yt:" + m[2];
      } else if (host === "youtu.be") {
        const id = u.pathname.slice(1).split("/")[0];
        if (id) return "yt:" + id;
      }
      return u.origin + u.pathname;
    } catch (_) {
      return href;
    }
  }

  async function isVideoDisabled(href) {
    const key = canonicalUrl(href);
    const { patmirDisabledVideos = [] } = await chrome.storage.local.get("patmirDisabledVideos");
    return Array.isArray(patmirDisabledVideos) && patmirDisabledVideos.includes(key);
  }

  async function setVideoDisabled(href, disabled) {
    const key = canonicalUrl(href);
    const { patmirDisabledVideos = [] } = await chrome.storage.local.get("patmirDisabledVideos");
    const list = Array.isArray(patmirDisabledVideos) ? patmirDisabledVideos.slice() : [];
    const idx = list.indexOf(key);
    if (disabled && idx < 0) list.push(key);
    if (!disabled && idx >= 0) list.splice(idx, 1);
    // Bound the list so it doesn't grow forever.
    while (list.length > 500) list.shift();
    await chrome.storage.local.set({ patmirDisabledVideos: list });
  }

  async function autoPlayIfApplicable() {
    if (playing) return;
    const url = window.location.href;
    if (!/^https?:\/\//.test(url)) return;

    const { patmirAutoPlayEnabled = true } = await chrome.storage.local.get("patmirAutoPlayEnabled");
    if (!patmirAutoPlayEnabled) return;
    if (await isVideoDisabled(url)) return;

    let lookup;
    try {
      lookup = await chrome.runtime.sendMessage({ type: "cache:lookup", url });
    } catch (_) { return; }
    if (!lookup || !lookup.ok || !lookup.data || !lookup.data.cached) return;

    // Wait for the player <video> element to be ready (YouTube swaps it
    // in asynchronously after navigation).
    const found = await waitForVideo(8000);
    if (!found) return;
    // Re-check the URL didn't change while we waited.
    if (window.location.href !== url) return;

    try {
      await startDub(lookup.data.cache_key);
    } catch (e) {
      console.warn("[patmir] auto-play failed:", e);
    }
  }

  function waitForVideo(timeoutMs) {
    return new Promise((resolve) => {
      const start = Date.now();
      const tick = () => {
        if (findVideo()) return resolve(true);
        if (Date.now() - start > timeoutMs) return resolve(false);
        setTimeout(tick, 200);
      };
      tick();
    });
  }

  // Re-check on YouTube SPA navigation (covers watch-page transitions).
  window.addEventListener("yt-navigate-finish", () => {
    setTimeout(autoPlayIfApplicable, 250);
  });
  // Storage changes from the options page (e.g. user flips the global
  // toggle on) should re-evaluate too.
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    if ("patmirAutoPlayEnabled" in changes || "patmirDisabledVideos" in changes) {
      autoPlayIfApplicable();
    }
  });
  // Initial trigger on first script injection.
  setTimeout(autoPlayIfApplicable, 500);

  // --- core playback --------------------------------------------------

  // --- audio graph + volume ------------------------------------------
  //
  // Everything the dub plays goes through `dubGain`, whose level follows the
  // page player's volume slider. Before this existed the graph was
  // `BufferSource -> destination` and the only thing silencing the original was
  // a one-shot `video.muted = true`, which YouTube's own unMute() cleared the
  // moment the viewer touched the slider -- putting the original back
  // underneath the dub.

  function ensureAudioGraph() {
    if (!audioCtx) audioCtx = new AudioContext();
    if (!dubGain) {
      dubGain = audioCtx.createGain();
      dubGain.connect(audioCtx.destination);
    }
    return audioCtx;
  }

  /** Where every dub source connects. */
  function dubDestination() {
    return dubGain || (audioCtx && audioCtx.destination) || null;
  }

  function applyDubGain() {
    if (!dubGain || !audioCtx) return;
    const g = S.dubGainFor(viewer.volume, viewer.muted);
    // A slider drag fires a burst of events; stepping the gain discontinuously
    // on each one clicks, so ramp.
    try { dubGain.gain.setTargetAtTime(g, audioCtx.currentTime, 0.02); }
    catch (_) { dubGain.gain.value = g; }
  }

  /** Force the element muted, counting the write so its `volumechange` echo can
   * be told apart from the viewer touching the volume. Assigning `true` to an
   * already-`true` `muted` fires no event, so that case must not be counted. */
  function forceMute() {
    if (!video || video.muted === true) return;
    pendingForcedWrites++;
    video.muted = true;
  }

  /** Take over the element's audio: capture what the viewer wanted, silence the
   * original, point the dub's gain at that intent. */
  function beginMuteControl() {
    if (!video) return;
    viewer = { volume: video.volume, muted: video.muted };
    pendingForcedWrites = 0;
    applyDubGain();
    forceMute();
  }

  /** Hand the element's audio back, honouring whatever the viewer did to the
   * volume while we held it. We never write `video.volume`, so the element
   * already carries the right level -- only `muted` needs restoring, and to the
   * viewer's intent rather than to whatever it was before we started. */
  function endMuteControl() {
    if (video) video.muted = viewer.muted;
    pendingForcedWrites = 0;
  }

  function onVolumeChange() {
    if (!video) return;
    const r = S.resolveVolumeEvent(
      { volume: video.volume, muted: video.muted }, pendingForcedWrites, viewer);
    pendingForcedWrites = r.pendingForcedWrites;
    if (r.consumedEcho) return;
    viewer = r.viewer;
    applyDubGain();
    if (r.needsForceMute) forceMute();
  }

  // --- shared A/V sync engine ----------------------------------------
  //
  // One 250 ms tick drives both modes. Each tick asks two questions: is the
  // picture actually moving (if not, do not start audio into a frozen frame --
  // that is the seek bug), and how far has the audio drifted from the playhead.
  // The correction policy lives in PatmirSync.driftDecision.

  const SYNC_TICK_MS = 250;
  const DRIFT_WINDOW = 5;         // samples the median is taken over

  let syncTickHandle = null;
  let lastTickCurrentTime = null; // video.currentTime at the previous tick
  let driftSamples = [];
  let rateNudge = 1;              // multiplier applied on top of video.playbackRate

  /** Forget the drift history and re-seed the progress detector. Called from
   * every event that invalidates the anchor: play, pause, seek, stall, rate
   * change. Seeding from the CURRENT time rather than null means the very next
   * tick can already tell whether the picture moved -- one tick of latency
   * before audio resumes, not two. */
  function resetSyncWindow() {
    lastTickCurrentTime = (video && isFinite(video.currentTime)) ? video.currentTime : null;
    driftSamples = [];
    rateNudge = 1;
  }

  /** Did the picture move since the last tick? null when there is nothing to
   * compare against yet. */
  function sampleAdvance() {
    if (!video) return null;
    const c = video.currentTime;
    const prev = lastTickCurrentTime;
    lastTickCurrentTime = c;
    if (prev === null) return null;
    return c > prev;
  }

  function pushDrift(d) {
    driftSamples.push(d);
    if (driftSamples.length > DRIFT_WINDOW) driftSamples.shift();
  }

  /** Read the DOM signals that mean "an ad is on screen".
   *
   * Read rather than inferred from the media element, because during an ad the
   * element looks entirely healthy — it is just playing something else. The
   * classification itself is pure and lives in lib/sync.js.
   */
  function adState() {
    let playerClasses = "";
    let hasAdOverlay = false;
    let hasAdPreview = false;
    let skipButtonPresent = false;
    let skipButtonEnabled = false;
    try {
      const player = document.querySelector("#movie_player");
      if (player) playerClasses = player.className || "";
      // NON-EMPTY, not merely present. `.ytp-ad-module` and
      // `.ytp-ad-player-overlay` are containers YouTube ships on every watch
      // page and leaves in the DOM when there is no ad, so `!!querySelector(...)`
      // was true always — which made the ad predicate true always, and the
      // auto-skip seek then jumped every video to its end. These are
      // corroboration only now (see classifyAd), but they still have to be
      // true statements.
      hasAdOverlay = hasContent(document.querySelector(
        ".ytp-ad-player-overlay, .ytp-ad-player-overlay-layout, .ytp-ad-module"));
      hasAdPreview = hasContent(document.querySelector(".ytp-ad-preview-container"));
      const skip = findSkipControl();
      skipButtonPresent = !!skip;
      skipButtonEnabled = !!skip && !skip.disabled && isVisible(skip);
    } catch (_) { /* a hostile or half-built DOM must not break the tick */ }
    const st = S.classifyAd({
      playerClasses, hasAdOverlay, hasAdPreview,
      skipButtonPresent, skipButtonEnabled,
    });
    // Sampled here rather than only from the dub's sync tick, because the
    // overlay polls this on every page whether or not a dub is playing — and it
    // is the overlay that can ask for a seek.
    if (!st.showing) noteContentDuration();
    return st;
  }

  /** Does this node actually carry something, or is it an empty shell? */
  function hasContent(el) {
    if (!el) return false;
    if (el.children && el.children.length) return true;
    return !!(el.textContent && el.textContent.trim());
  }

  /** Visible enough to click.
   *
   * NOT `offsetParent !== null`: that is null for anything inside a
   * position:fixed ancestor even when it is plainly on screen, which would make
   * a live skip button look unclickable and push us onto the seek path instead.
   */
  function isVisible(el) {
    if (!el) return false;
    if (typeof el.getBoundingClientRect !== "function") return true;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  // YouTube has renamed this control repeatedly; try the modern names first.
  const SKIP_SELECTORS = [
    ".ytp-skip-ad-button",
    ".ytp-ad-skip-button-modern",
    ".ytp-ad-skip-button",
    ".ytp-ad-skip-button-slot button",
  ];

  function findSkipControl() {
    for (const sel of SKIP_SELECTORS) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return null;
  }

  /** Sample the content video's duration while no ad is on screen. */
  function noteContentDuration() {
    const v = findVideo();
    if (v && isFinite(v.duration) && v.duration > 1) contentDuration = v.duration;
  }

  /** Called from the tick. Returns true when the ad state changed. */
  function updateAdState() {
    const showing = adState().showing;
    if (showing === adShowing) return false;
    adShowing = showing;
    if (showing) {
      // Silence, do NOT tear down. Keeping the listeners, the decoded buffer,
      // `playing`, and the mute control is the whole difference between "the
      // dub pauses for the ad" and "the dub is gone for the rest of the video".
      stopDubSource();
      stopAllNodes();
      resetSyncWindow();
    } else {
      // Content is back and `currentTime` is its timeline again. Re-seed the
      // progress detector; the ordinary tick re-anchors on the next confirmed
      // advance, and the src we compare against is the content's.
      if (video) contentSrc = video.currentSrc || video.src || null;
      if (contentSrc) videoSrcAtStart = contentSrc;
      resetSyncWindow();
    }
    return true;
  }

  function videoIsProgressing(advanced) {
    if (!video) return false;
    return S.shouldScheduleAudio({
      paused: video.paused, seeking: video.seeking,
      readyState: video.readyState, advanced: advanced,
      adShowing: adShowing,
    });
  }

  function startSyncTick() {
    if (syncTickHandle === null) syncTickHandle = setInterval(syncTick, SYNC_TICK_MS);
  }

  function stopSyncTick() {
    if (syncTickHandle !== null) { clearInterval(syncTickHandle); syncTickHandle = null; }
    resetSyncWindow();
  }

  function syncTick() {
    // Polled here rather than from a new timer: this interval already runs
    // whenever a dub is active, which is exactly when an ad matters.
    updateAdState();
    if (adShowing) return;   // the element's clock belongs to the ad
    if (waActive) { watchAlongTick(); return; }
    if (playing) qualityTick();
  }

  function findVideo() {
    // YouTube's primary player is the largest, longest-duration <video> on
    // the page. The thumbnail-preview <video> elements are short and tiny.
    const candidates = Array.from(document.querySelectorAll("video"))
      .filter((v) => v.duration && isFinite(v.duration) && v.duration > 1);
    if (!candidates.length) return null;
    candidates.sort((a, b) => (b.duration || 0) - (a.duration || 0));
    return candidates[0];
  }

  async function startDub(cacheKey) {
    video = findVideo();
    if (!video) throw new Error("No playable <video> element found on this page.");

    ensureAudioGraph();

    // Refetch + redecode if the user is dubbing a different video than the
    // last one we loaded. Without this the same dub plays for any video the
    // SPA navigates to.
    if (!dubBuffer || dubBufferCacheKey !== cacheKey) {
      const resp = await chrome.runtime.sendMessage({ type: "cache:dub-wav", cache_key: cacheKey });
      if (!resp || !resp.ok) throw new Error(resp && resp.error ? resp.error : "Failed to fetch dub.wav.");
      const bytes = base64ToBytes(resp.bytesB64);
      dubBuffer = await audioCtx.decodeAudioData(bytes.buffer);
      dubBufferCacheKey = cacheKey;
    }

    // Remember which media source + URL this dub belongs to so we can
    // auto-stop on:
    //   (a) source swap on the same <video> (YouTube watch SPA reuse), and
    //   (b) Shorts scroll, which swaps to a different <video> element
    //       in a different URL — listeners on the old element never fire.
    videoSrcAtStart = video.currentSrc || video.src || null;
    contentSrc = videoSrcAtStart;
    urlAtStart = window.location.href;

    beginMuteControl();

    attachVideoListeners();
    attachNavigationListeners();
    playing = true;
    // Audio starts from the tick, once the picture is confirmed moving.
    resetSyncWindow();
    startSyncTick();
  }

  function stopDub() {
    stopSyncTick();
    stopDubSource();
    if (video) {
      detachVideoListeners();
      endMuteControl();
    }
    detachNavigationListeners();
    videoSrcAtStart = null;
    urlAtStart = null;
    playing = false;
  }

  function stopDubSource() {
    if (dubSource) {
      try { dubSource.stop(); } catch (_) { /* already stopped */ }
      dubSource = null;
    }
  }

  function startSourceAt(offset) {
    if (!dubBuffer || !audioCtx) return;
    stopDubSource();
    const src = audioCtx.createBufferSource();
    src.buffer = dubBuffer;
    // A hard re-anchor lands us exactly on the playhead, so any accumulated
    // nudge has done its job and goes back to unity.
    rateNudge = 1;
    currentSourceRate = (video && video.playbackRate) || 1.0;
    src.playbackRate.value = currentSourceRate;
    src.connect(dubDestination());
    const clipped = Math.max(0, Math.min(offset, dubBuffer.duration - 0.001));
    src.start(0, clipped);
    audioStartCtxTime = audioCtx.currentTime;
    audioStartOffset = clipped;
    dubSource = src;
    driftSamples = [];
  }

  /** Resample the dub slightly so a small error walks itself out, instead of
   * taking the audible cut a re-anchor costs. The anchor has to move with it:
   * `audioStartOffset` is integrated at the OLD rate up to now, then the clock
   * restarts from here at the new one. */
  function applyRateNudge(baseRate, multiplier) {
    if (!dubSource || !audioCtx) return;
    const now = audioCtx.currentTime;
    audioStartOffset = audioStartOffset + (now - audioStartCtxTime) * currentSourceRate;
    audioStartCtxTime = now;
    currentSourceRate = baseRate * multiplier;
    rateNudge = multiplier;
    try { dubSource.playbackRate.setValueAtTime(currentSourceRate, now); }
    catch (_) { /* ok */ }
  }

  function qualityTick() {
    if (!video || !audioCtx || !dubBuffer) return;
    const advanced = sampleAdvance();
    const rate = video.playbackRate || 1.0;

    if (!videoIsProgressing(advanced)) {
      // Paused, seeking, under-buffered, or the clock is frozen mid-stall.
      // Silence rather than let the dub run on past a still picture -- that is
      // precisely how a seek used to leave the two permanently offset.
      stopDubSource();
      driftSamples = [];
      return;
    }

    if (!dubSource) { startSourceAt(video.currentTime); return; }

    const actual = audioStartOffset +
      (audioCtx.currentTime - audioStartCtxTime) * currentSourceRate;
    pushDrift(video.currentTime - actual);

    const d = S.driftDecision(driftSamples);
    if (d.action === "reanchor") {
      startSourceAt(video.currentTime);
    } else if (d.action === "nudge") {
      applyRateNudge(rate, d.rateMultiplier);
    } else if (rateNudge !== 1) {
      applyRateNudge(rate, 1);      // back inside the dead band; stop nudging
    }
  }

  // --- video event wiring --------------------------------------------

  function onVideoSourceChanged() {
    // YouTube auto-advances Shorts (and reuses the same <video> element)
    // by swapping the media source. Either `emptied` or `loadstart` fires
    // when this happens — the dub belongs to the previous video, so stop.
    if (!playing) return;
    // ...UNLESS an ad is starting. An ad swaps the src on the same element for
    // the same video, and treating that as a navigation is what made an
    // unskippable mid-roll kill the dub for the rest of the video. The ad is a
    // suspension, handled by `updateAdState`; only a real source change is a
    // teardown.
    if (adState().showing) { updateAdState(); return; }
    const currentSrc = video ? (video.currentSrc || video.src || null) : null;
    if (currentSrc === videoSrcAtStart) return;
    // The src differs and no ad is showing YET -- which is not the same as no
    // ad. `emptied`/`loadstart` fire the instant the source is swapped, and
    // YouTube sets `ad-showing` on the player around the same moment with no
    // guaranteed ordering. Tearing down here is what put the video back in
    // English after a mid-roll (quality issue #26 was never confirmed on a
    // real ad break), and issue #27 made the race more likely by tightening
    // detection so ONLY the player class counts.
    //
    // So defer: re-ask once the class has had time to appear. A real
    // navigation is still caught -- by this same check a moment later, and by
    // the URL poll that owns navigation anyway.
    if (srcChangeGrace !== null) return;      // already waiting on one
    srcChangeGrace = setTimeout(() => {
      srcChangeGrace = null;
      if (!playing) return;
      if (adState().showing) { updateAdState(); return; }
      const nowSrc = video ? (video.currentSrc || video.src || null) : null;
      if (nowSrc !== videoSrcAtStart) stopDub();
    }, SRC_CHANGE_GRACE_MS);
  }

  // Every one of these invalidates the anchor, so they stop the audio and clear
  // the drift history; `qualityTick` restarts it once the picture is confirmed
  // moving again. `seeked` deliberately does NOT start audio itself -- the
  // element reports a healthy readyState there while the frame is still frozen,
  // and starting then is what left the dub permanently ahead.
  const handlers = {
    play() {
      if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
      resetSyncWindow();
    },
    pause() { stopDubSource(); resetSyncWindow(); },
    seeking() { stopDubSource(); resetSyncWindow(); },
    seeked() { resetSyncWindow(); },
    waiting() { stopDubSource(); resetSyncWindow(); },
    stalled() { stopDubSource(); resetSyncWindow(); },
    playing() { resetSyncWindow(); },
    ratechange() { stopDubSource(); resetSyncWindow(); },
    ended() {
      stopDub();
    },
    volumechange: onVolumeChange,
    emptied: onVideoSourceChanged,
    loadstart: onVideoSourceChanged,
  };

  function onYouTubeNavigate() {
    if (playing || waActive) stopPlayback();
  }

  /** Stop whichever player is active (quality single-buffer or watch-along
   * chunked feed). Used by every navigation/teardown path. */
  function stopPlayback() {
    stopDub();
    stopWatchAlong();
  }

  function base64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function attachVideoListeners() {
    if (!video) return;
    for (const [ev, fn] of Object.entries(handlers)) {
      video.addEventListener(ev, fn);
    }
  }

  function detachVideoListeners() {
    if (!video) return;
    for (const [ev, fn] of Object.entries(handlers)) {
      video.removeEventListener(ev, fn);
    }
  }

  function attachNavigationListeners() {
    // YouTube's SPA fires `yt-navigate-finish` on the window when route
    // changes — covers list-page → watch transitions reliably.
    window.addEventListener("yt-navigate-finish", onYouTubeNavigate);
    // Shorts scroll doesn't always fire yt-navigate-finish AND uses a
    // recycler-style pager with multiple <video> elements (so our
    // emptied/loadstart listeners attached to the original one never fire
    // when the next short starts playing). Poll location.href + the
    // currently-active <video> as a backstop.
    if (navWatchHandle === null) {
      navWatchHandle = setInterval(checkNavigation, 400);
    }
  }

  function detachNavigationListeners() {
    window.removeEventListener("yt-navigate-finish", onYouTubeNavigate);
    if (navWatchHandle !== null) {
      clearInterval(navWatchHandle);
      navWatchHandle = null;
    }
  }

  function checkNavigation() {
    if (!playing) return;
    if (urlAtStart && window.location.href !== urlAtStart) {
      stopPlayback();
      // Try auto-playing the new URL if it's also cached.
      setTimeout(autoPlayIfApplicable, 300);
      return;
    }
    // Recycler / element swap — the active video is no longer the one we
    // attached to (or the original was removed from the DOM).
    const active = findVideo();
    if (!active || (video && active !== video) || (video && !document.contains(video))) {
      stopPlayback();
      setTimeout(autoPlayIfApplicable, 300);
    }
  }

  // --- watch-along (D3) chunked playback ------------------------------
  //
  // Quality mode decodes one whole dub.wav and plays it at a single offset.
  // Watch-along instead consumes an offset-addressed feed: the server streams
  // VAD-snapped sections, each a slice-relative WAV that represents global
  // [section.start, section.end]. We poll the section list, fetch + decode
  // each as it lands, and schedule it on the same Web Audio clock as the
  // <video> playhead. When the playhead reaches audio that isn't produced yet
  // we pause the video (buffering) and resume once it's ready — that gate is
  // what lets playback start ~20–30s in while the rest streams behind it.

  let waActive = false;
  let waCacheKey = null;
  /** @type {Map<number,{index:number,start:number,end:number,file:string,buffer:AudioBuffer|null,fetching:boolean,node:AudioBufferSourceNode|null,startCtx:number,startOffset:number,rate:number}>} */
  let waSections = new Map();
  let waComplete = false;
  let waPollHandle = null;
  let waDesiredPlaying = false;   // does the user want playback right now?
  // Guard so our own buffering-pause isn't mistaken for a user pause. The
  // <video> 'pause' event fires ASYNCHRONOUSLY after video.pause(), so this is
  // consumed (cleared) inside the pause handler, never reset synchronously.
  let waProgrammatic = false;
  let waBuffering = false;        // want to play but the current section isn't ready yet
  // When the manifest last gained a section. A dead stream is indistinguishable
  // from a slow one except by how long nothing has arrived.
  let waLastProgressAt = 0;
  // Automatic revives used since the viewer last pressed start. ONE, and reset
  // only by an explicit start — not by progress. A per-stall budget would loop
  // forever on a stream that always dies after one section, and re-submitting
  // is not free: it wakes the GPU and interrupts the picture.
  let waRevivesUsed = 0;
  const WA_MAX_REVIVES = 1;
  let waLastPlayheadPost = 0;

  const WA_LOOKAHEAD = 15;        // seconds of sections to prefetch ahead of playhead
  // Drift tolerance is no longer a single flat number here -- it lives in
  // PatmirSync.driftDecision, shared with quality mode. A flat 80 ms against a
  // coarsely-quantized `currentTime` re-anchored on noise, and a re-anchor
  // stops and restarts the node, which is audible.

  async function startWatchAlong(cacheKey) {
    video = findVideo();
    if (!video) throw new Error("No playable <video> element found on this page.");
    ensureAudioGraph();

    waActive = true;
    waCacheKey = cacheKey;
    waSections = new Map();
    waComplete = false;
    urlAtStart = window.location.href;
    videoSrcAtStart = video.currentSrc || video.src || null;

    beginMuteControl();

    waDesiredPlaying = !video.paused;
    waBuffering = false;

    attachWatchAlongListeners();
    attachNavigationListeners();
    waLastProgressAt = Date.now();   // the clock starts when we start watching
    if (waPollHandle === null) waPollHandle = setInterval(pollWatchAlongSections, 2000);
    resetSyncWindow();
    startSyncTick();
    pollWatchAlongSections();
  }

  function stopWatchAlong() {
    if (!waActive) return;
    // Tell the server to stop dubbing immediately — otherwise its scheduler
    // keeps running Scribe + ElevenLabs TTS over the rest of the video (paid)
    // even though we've stopped watching. Best-effort, fire-and-forget.
    if (waCacheKey) {
      chrome.runtime.sendMessage({ type: "stream:stop", cache_key: waCacheKey }).catch(() => {});
    }
    for (const sec of waSections.values()) {
      if (sec.node) { try { sec.node.stop(); } catch (_) { /* ok */ } sec.node = null; }
    }
    waSections.clear();
    if (waPollHandle !== null) { clearInterval(waPollHandle); waPollHandle = null; }
    stopSyncTick();
    if (video) {
      detachWatchAlongListeners();
      endMuteControl();
    }
    detachNavigationListeners();
    waActive = false;
    waCacheKey = null;
    waDesiredPlaying = false;
    waBuffering = false;
    videoSrcAtStart = null;
    urlAtStart = null;
  }

  async function pollWatchAlongSections() {
    if (!waActive || !waCacheKey) return;
    let resp;
    try {
      resp = await chrome.runtime.sendMessage({ type: "stream:sections", cache_key: waCacheKey });
    } catch (_) { return; }
    if (!resp || !resp.ok || !resp.data) return;
    const data = resp.data;
    waComplete = !!data.complete;
    for (const s of data.sections || []) {
      const existing = waSections.get(s.index);
      if (existing) {
        // An index can be RE-POINTED: a resumed stream restarts its counter, so
        // section_001.wav can come back describing a different span entirely.
        // Keeping the already-decoded buffer then plays one part of the film
        // over another. Drop it and refetch.
        if (existing.start !== s.start || existing.end !== s.end || existing.file !== s.file) {
          if (existing.node) { try { existing.node.stop(); } catch (_) { /* ok */ } existing.node = null; }
          existing.buffer = null;
          existing.fetching = false;
        }
        existing.start = s.start; existing.end = s.end; existing.file = s.file;
      } else {
        waSections.set(s.index, {
          index: s.index, start: s.start, end: s.end, file: s.file,
          buffer: null, fetching: false, node: null, startCtx: 0, startOffset: 0, rate: 1,
        });
        waLastProgressAt = Date.now();
      }
    }
    // Evict entries the manifest no longer lists (deduplicated away server
    // side); otherwise they linger forever and can still be picked.
    const live = new Set((data.sections || []).map((x) => x.index));
    for (const idx of Array.from(waSections.keys())) {
      if (live.has(idx)) continue;
      const gone = waSections.get(idx);
      if (gone && gone.node) { try { gone.node.stop(); } catch (_) { /* ok */ } }
      waSections.delete(idx);
    }
    // Stop polling once the whole video is on disk (rewatch / completed stream).
    if (waComplete && waPollHandle !== null) {
      clearInterval(waPollHandle); waPollHandle = null;
    }
  }

  async function ensureBuffer(sec) {
    if (sec.buffer || sec.fetching) return;
    sec.fetching = true;
    try {
      const resp = await chrome.runtime.sendMessage({
        type: "stream:section-wav", cache_key: waCacheKey, index: sec.index,
      });
      if (resp && resp.ok && resp.bytesB64) {
        const bytes = base64ToBytes(resp.bytesB64);
        sec.buffer = await audioCtx.decodeAudioData(bytes.buffer);
      }
    } catch (_) { /* try again next tick */ }
    finally { sec.fetching = false; }
  }

  /** The section that should sound at global time `c`.
   *
   * Delegates the choice to PatmirSync.pickSection, which prefers the LATEST
   * start covering `c` (sections overlap after a seek, and the Map is in
   * poll-arrival order, so "first match" picked between two renderings of the
   * same dialogue at random) and skips a section whose decoded audio is already
   * exhausted at `c` (a section's WAV can be shorter than its declared span). */
  function sectionView() {
    const view = [];
    for (const sec of waSections.values()) {
      view.push({
        index: sec.index, start: sec.start, end: sec.end,
        duration: sec.buffer ? sec.buffer.duration : null,
      });
    }
    return view;
  }

  function pickPlayableSection(view, c) {
    const chosen = S.pickSection(view, c);
    return chosen ? waSections.get(chosen.index) : null;
  }

  function scheduleNode(sec, c, rate) {
    if (!sec.buffer || !audioCtx) return;
    const offset = Math.max(0, c - sec.start);
    if (offset >= sec.buffer.duration - 0.001) return;
    // Exactly one section sounds at a time. Stop EVERY live node (not just this
    // section's) before starting the new one, so a section's tail can never
    // overlap the next section at a boundary, and a re-anchor can't double up.
    stopAllNodes();
    const now = audioCtx.currentTime;
    const src = audioCtx.createBufferSource();
    src.buffer = sec.buffer;
    rateNudge = 1;                 // a hard re-anchor lands on the playhead exactly
    driftSamples = [];
    src.playbackRate.value = rate;
    src.connect(dubDestination());
    let when, startOffset;
    if (c >= sec.start) { when = now; startOffset = offset; }
    else { when = now + (sec.start - c) / rate; startOffset = 0; }
    try { src.start(when, startOffset); } catch (_) { return; }
    src.onended = () => { if (sec.node === src) sec.node = null; };
    sec.node = src; sec.startCtx = when; sec.startOffset = startOffset; sec.rate = rate;
  }

  /** Resample one section's node, moving its anchor with it -- the section
   * equivalent of `applyRateNudge`. */
  function nudgeNodeRate(sec, baseRate, multiplier) {
    if (!sec.node || !audioCtx) return;
    const now = audioCtx.currentTime;
    sec.startOffset = sec.startOffset + (now - sec.startCtx) * sec.rate;
    sec.startCtx = now;
    sec.rate = baseRate * multiplier;
    rateNudge = multiplier;
    try { sec.node.playbackRate.setValueAtTime(sec.rate, now); } catch (_) { /* ok */ }
  }

  function ensureScheduled(sec, c, rate) {
    if (!sec.buffer || !audioCtx) return;
    if (!sec.node) { scheduleNode(sec, c, rate); return; }
    if (sec.rate !== rate * rateNudge) { scheduleNode(sec, c, rate); return; }

    const expected = c - sec.start;
    const actual = sec.startOffset + (audioCtx.currentTime - sec.startCtx) * sec.rate;
    pushDrift(expected - actual);

    const d = S.driftDecision(driftSamples);
    if (d.action === "reanchor") {
      scheduleNode(sec, c, rate);
    } else if (d.action === "nudge") {
      nudgeNodeRate(sec, rate, d.rateMultiplier);
    } else if (rateNudge !== 1) {
      nudgeNodeRate(sec, rate, 1);
    }
  }

  function stopAllNodes() {
    for (const sec of waSections.values()) {
      if (sec.node) { try { sec.node.stop(); } catch (_) { /* ok */ } sec.node = null; }
    }
  }

  function watchAlongTick() {
    if (!waActive || !video) return;
    const c = video.currentTime;
    const rate = video.playbackRate || 1.0;
    const advanced = sampleAdvance();

    // Prefetch the current + upcoming sections.
    for (const sec of waSections.values()) {
      if (sec.end > c - 1 && sec.start < c + WA_LOOKAHEAD && !sec.buffer) ensureBuffer(sec);
    }

    if (!waDesiredPlaying) return;

    const view = sectionView();
    const cur = pickPlayableSection(view, c);
    const usable = cur && cur.buffer && cur.start - S.SECTION_LEAD <= c && c < cur.end;
    if (!usable) {
      stopAllNodes();
      driftSamples = [];
      if (S.isCovered(view, c)) {
        // A section claims this second but has no audio to give -- its WAV came
        // out shorter than its declared span. Nothing to play, but nothing to
        // wait for either: pausing would WEDGE, because the playhead has to
        // move past the short section before the next one becomes current.
        waBuffering = false;
        reportPlayhead(false);
      } else if (S.shouldAbandonStream({
        buffering: true, complete: waComplete,
        msSinceProgress: Date.now() - waLastProgressAt,
      })) {
        // Nothing has arrived for a minute: the stream is dead, not slow.
        // Without this the client re-pauses on every tick, so pressing play
        // stops the video a quarter second later — forever, with no error
        // anywhere. Hand the video back and play the original.
        const starvedFor = Math.round((Date.now() - waLastProgressAt) / 1000);
        if (waRevivesUsed < WA_MAX_REVIVES) {
          // Try once to get it going again; the server resumes from what is
          // already cached. `reviveWatchAlong` restarts the stall clock.
          reviveWatchAlong().then((ok) => {
            if (ok) return;
            console.warn("[patmir] re-submit failed — releasing the video.");
            stopWatchAlong();
            if (video && video.paused) { try { waPlay(); } catch (_) { /* ok */ } }
          }).catch(() => {
            stopWatchAlong();
            if (video && video.paused) { try { waPlay(); } catch (_) { /* ok */ } }
          });
          return;
        }
        console.warn(`[patmir] no new dub section for ${starvedFor}s and the `
          + `revive was already used — releasing the video.`);
        stopWatchAlong();
        if (video && video.paused) { try { waPlay(); } catch (_) { /* ok */ } }
      } else {
        // Genuine underrun: the dub for this point isn't produced yet. Freeze
        // the video; a later tick resumes once the section lands.
        waBuffering = true;
        waPause();
        reportPlayhead(true);
      }
      return;
    }
    waBuffering = false;
    // Resume if we (or a stalled autoplay) left the video paused while the
    // user still wants to watch.
    if (video.paused) { waPlay(); resetSyncWindow(); reportPlayhead(false); return; }
    if (!videoIsProgressing(advanced)) {
      // Just resumed, seeking, or re-buffering: the frame is still frozen.
      // Starting audio now offsets it by the length of the stall.
      stopAllNodes();
      driftSamples = [];
      reportPlayhead(false);
      return;
    }
    ensureScheduled(cur, c, rate);
    reportPlayhead(false);
  }

  /** Pause the <video> for buffering without the pause handler mistaking it
   * for a user pause. The 'pause' event is async, so the guard is consumed in
   * the handler; a timeout clears it if (somehow) no event fires. */
  function waPause() {
    if (!video || video.paused) return;
    waProgrammatic = true;
    setTimeout(() => { waProgrammatic = false; }, 400);
    video.pause();
  }

  function waPlay() {
    if (!video || !video.paused) return;
    if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
    video.play().catch(() => { /* autoplay gate — retried next tick */ });
  }

  function reportPlayhead(force) {
    if (!waActive || !waCacheKey || !video) return;
    // An ad's `currentTime` starts near zero. Posting it would tell the
    // scheduler the viewer seeked to the beginning, and it would start dubbing
    // sections that are already covered.
    if (adShowing) return;
    const now = Date.now();
    if (!force && now - waLastPlayheadPost < 1000) return;
    waLastPlayheadPost = now;
    chrome.runtime.sendMessage({
      type: "stream:playhead", cache_key: waCacheKey,
      t: video.currentTime, paused: video.paused,
    }).catch(() => {});
  }

  const waHandlers = {
    play() {
      if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
      waDesiredPlaying = true;
      resetSyncWindow();
      reportPlayhead(true);
    },
    pause() {
      if (waProgrammatic) { waProgrammatic = false; return; }  // our buffering-pause
      waDesiredPlaying = false;
      stopAllNodes();
      resetSyncWindow();
      reportPlayhead(true);
    },
    seeking() { stopAllNodes(); resetSyncWindow(); },
    seeked() {
      stopAllNodes();
      resetSyncWindow();
      reportPlayhead(true);
    },
    waiting() { stopAllNodes(); resetSyncWindow(); },
    stalled() { stopAllNodes(); resetSyncWindow(); },
    playing() { resetSyncWindow(); },
    ratechange() { stopAllNodes(); resetSyncWindow(); },
    ended() { stopWatchAlong(); },
    volumechange: onVolumeChange,
    emptied: onVideoSourceChangedWA,
    loadstart: onVideoSourceChangedWA,
  };

  function onVideoSourceChangedWA() {
    if (!waActive) return;
    // Same ad exemption as the quality path — see onVideoSourceChanged.
    if (adState().showing) { updateAdState(); return; }
    const currentSrc = video ? (video.currentSrc || video.src || null) : null;
    if (currentSrc !== videoSrcAtStart) stopWatchAlong();
  }

  function attachWatchAlongListeners() {
    if (!video) return;
    for (const [ev, fn] of Object.entries(waHandlers)) video.addEventListener(ev, fn);
  }

  function detachWatchAlongListeners() {
    if (!video) return;
    for (const [ev, fn] of Object.entries(waHandlers)) video.removeEventListener(ev, fn);
  }

  // Also poll the URL even when *not* currently playing — covers the
  // Shorts case where the user scrolls from a non-dubbed video to a
  // previously-dubbed one. Cheap (string compare every 500 ms).
  //
  // This is the ONLY navigation signal that fires while nothing is playing, so
  // overlay.js registers here (via `onUrlChange`) rather than starting a second
  // identical poll of its own.
  const urlChangeCallbacks = [];

  function onUrlChange(cb) {
    if (typeof cb === "function") urlChangeCallbacks.push(cb);
    return () => {
      const i = urlChangeCallbacks.indexOf(cb);
      if (i >= 0) urlChangeCallbacks.splice(i, 1);
    };
  }

  let lastSeenUrl = window.location.href;
  setInterval(() => {
    if (window.location.href !== lastSeenUrl) {
      lastSeenUrl = window.location.href;
      if (!playing) setTimeout(autoPlayIfApplicable, 300);
      for (const cb of urlChangeCallbacks.slice()) {
        try { cb(lastSeenUrl); } catch (_) { /* one bad subscriber, not all */ }
      }
    }
  }, 500);

  // --- facade for overlay.js -------------------------------------------
  //
  // Published LAST, after every `let` above is initialized. The bootstrap guard
  // at the top of this file exists because a listener registered over a
  // half-initialized module reported "Cannot access X before initialization"
  // instead of the real problem; publishing this facade earlier would reopen
  // exactly that window for the overlay.
  //
  // Deliberately four playback methods plus three read-only helpers, and NOT a
  // handle on the audio graph: `dubGain`, the mute arbitration and the drift
  // tick have one owner, and it is this file.
  globalThis.PatmirPlayer = {
    play: playQuality,
    playWatchAlong: playWatchAlongFor,
    stop: stopTranslation,
    status: playbackStatus,
    findVideo,
    canonicalUrl,
    onUrlChange,
    adState,
    /** Perform an ad-skip action decided by `PatmirSync.adSkipPlan`.
     *
     * Lives here, not in the overlay, because it is the one place that already
     * owns a reference to the media element — and because the `showing` guard
     * has to be re-checked against live DOM immediately before a seek. A seek
     * that lands on the CONTENT video is the single destructive failure
     * available in this feature.
     */
    applyAdAction(action) {
      if (action !== "click" && action !== "seek") return false;
      const st = adState();
      if (!st.showing) return false;
      if (action === "click") {
        const skip = findSkipControl();
        if (!skip || skip.disabled) return false;
        try { skip.click(); return true; } catch (_) { return false; }
      }
      // Everything below is the destructive path, and it is re-checked here
      // against live DOM rather than trusting the plan the caller computed a
      // tick ago. The first version checked only `showing`, which was true on
      // every watch page because of a broken predicate — and so it seeked the
      // viewer's actual video to the end, on every video. Three independent
      // conditions now have to agree, and any one of them failing is a refusal:
      //   1. #movie_player ITSELF says an ad is playing;
      //   2. the media element's duration is short enough to be an ad;
      //   3. it is not the duration we recorded for this page's content video.
      if (!st.classConfirmed) return false;
      const v = findVideo();
      if (!v || !isFinite(v.duration) || v.duration <= 0) return false;
      if (v.duration > S.MAX_AD_SECONDS) return false;
      if (contentDuration != null && Math.abs(v.duration - contentDuration) < 0.5) {
        return false;
      }
      try { v.currentTime = v.duration; return true; } catch (_) { return false; }
    },
  };
})();
