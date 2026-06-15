// Patmir content script.
//
// Runs on YouTube pages. Owns the page-context Web Audio graph that plays
// the cached dub synced to the page <video> element. The service worker
// (sw.js) handles all HTTP — this script only talks to it via
// chrome.runtime.sendMessage.

(function () {
  if (window.__patmirInjected) return;
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
  /** @type {number|null} */
  let navWatchHandle = null;
  let wasMuted = false;
  let originalVolume = null;
  let playing = false;
  let audioStartCtxTime = 0;
  let audioStartOffset = 0;

  // --- message routing ------------------------------------------------

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    (async () => {
      try {
        if (msg.type === "patmir:play") {
          // Explicit play also removes any prior opt-out for this video so
          // the next visit auto-plays again.
          await setVideoDisabled(window.location.href, false);
          stopWatchAlong();
          await startDub(msg.cache_key);
          sendResponse({ ok: true });
        } else if (msg.type === "patmir:play-watchalong") {
          await setVideoDisabled(window.location.href, false);
          stopDub();
          await startWatchAlong(msg.cache_key);
          sendResponse({ ok: true });
        } else if (msg.type === "patmir:stop") {
          // Explicit stop opts this video out of future auto-play.
          await setVideoDisabled(window.location.href, true);
          stopPlayback();
          sendResponse({ ok: true });
        } else if (msg.type === "patmir:status") {
          const url = window.location.href;
          const disabled = await isVideoDisabled(url);
          const isPlaying = playing || waActive;
          const mode = waActive ? "watch-along" : (playing ? "quality" : null);
          sendResponse({
            ok: true, playing: isPlaying, mode, hasVideo: !!findVideo(), disabled,
            buffering: waActive && waBuffering,
          });
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

    if (!audioCtx) audioCtx = new AudioContext();

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
    urlAtStart = window.location.href;

    // Mute (don't zero volume — leave that to the user).
    wasMuted = video.muted;
    if (originalVolume === null) originalVolume = video.volume;
    video.muted = true;

    attachVideoListeners();
    attachNavigationListeners();
    if (!video.paused) {
      startSourceAt(video.currentTime);
    }
    playing = true;
  }

  function stopDub() {
    if (dubSource) {
      try { dubSource.stop(); } catch (_) { /* already stopped */ }
      dubSource = null;
    }
    if (video) {
      detachVideoListeners();
      video.muted = wasMuted;
      if (originalVolume !== null) video.volume = originalVolume;
    }
    detachNavigationListeners();
    videoSrcAtStart = null;
    urlAtStart = null;
    playing = false;
  }

  function startSourceAt(offset) {
    if (!dubBuffer || !audioCtx) return;
    if (dubSource) {
      try { dubSource.stop(); } catch (_) { /* ok */ }
    }
    const src = audioCtx.createBufferSource();
    src.buffer = dubBuffer;
    src.playbackRate.value = (video && video.playbackRate) || 1.0;
    src.connect(audioCtx.destination);
    const clipped = Math.max(0, Math.min(offset, dubBuffer.duration - 0.001));
    src.start(0, clipped);
    audioStartCtxTime = audioCtx.currentTime;
    audioStartOffset = clipped;
    dubSource = src;
  }

  // --- video event wiring --------------------------------------------

  function onVideoSourceChanged() {
    // YouTube auto-advances Shorts (and reuses the same <video> element)
    // by swapping the media source. Either `emptied` or `loadstart` fires
    // when this happens — the dub belongs to the previous video, so stop.
    if (!playing) return;
    const currentSrc = video ? (video.currentSrc || video.src || null) : null;
    if (currentSrc !== videoSrcAtStart) {
      stopDub();
    }
  }

  const handlers = {
    play() {
      if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
      if (video) startSourceAt(video.currentTime);
    },
    pause() {
      if (dubSource) {
        try { dubSource.stop(); } catch (_) { /* ok */ }
        dubSource = null;
      }
    },
    seeked() {
      if (video && !video.paused) startSourceAt(video.currentTime);
    },
    ratechange() {
      if (dubSource && video && audioCtx) {
        try {
          dubSource.playbackRate.setValueAtTime(video.playbackRate, audioCtx.currentTime);
        } catch (_) { /* ok */ }
      }
    },
    ended() {
      stopDub();
    },
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
  let waTickHandle = null;
  let waDesiredPlaying = false;   // does the user want playback right now?
  // Guard so our own buffering-pause isn't mistaken for a user pause. The
  // <video> 'pause' event fires ASYNCHRONOUSLY after video.pause(), so this is
  // consumed (cleared) inside the pause handler, never reset synchronously.
  let waProgrammatic = false;
  let waBuffering = false;        // want to play but the current section isn't ready yet
  let waLastPlayheadPost = 0;

  const WA_LOOKAHEAD = 15;        // seconds of sections to prefetch ahead of playhead
  const WA_DRIFT = 0.08;          // ±80 ms sync tolerance before we re-anchor a node

  async function startWatchAlong(cacheKey) {
    video = findVideo();
    if (!video) throw new Error("No playable <video> element found on this page.");
    if (!audioCtx) audioCtx = new AudioContext();

    waActive = true;
    waCacheKey = cacheKey;
    waSections = new Map();
    waComplete = false;
    urlAtStart = window.location.href;
    videoSrcAtStart = video.currentSrc || video.src || null;

    wasMuted = video.muted;
    if (originalVolume === null) originalVolume = video.volume;
    video.muted = true;

    waDesiredPlaying = !video.paused;
    waBuffering = false;

    attachWatchAlongListeners();
    attachNavigationListeners();
    if (waPollHandle === null) waPollHandle = setInterval(pollWatchAlongSections, 2000);
    if (waTickHandle === null) waTickHandle = setInterval(watchAlongTick, 250);
    pollWatchAlongSections();
    watchAlongTick();
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
    if (waTickHandle !== null) { clearInterval(waTickHandle); waTickHandle = null; }
    if (video) {
      detachWatchAlongListeners();
      video.muted = wasMuted;
      if (originalVolume !== null) video.volume = originalVolume;
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
        existing.start = s.start; existing.end = s.end; existing.file = s.file;
      } else {
        waSections.set(s.index, {
          index: s.index, start: s.start, end: s.end, file: s.file,
          buffer: null, fetching: false, node: null, startCtx: 0, startOffset: 0, rate: 1,
        });
      }
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

  function sectionAt(c) {
    for (const sec of waSections.values()) {
      if (sec.start - 0.05 <= c && c < sec.end) return sec;
    }
    return null;
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
    src.playbackRate.value = rate;
    src.connect(audioCtx.destination);
    let when, startOffset;
    if (c >= sec.start) { when = now; startOffset = offset; }
    else { when = now + (sec.start - c) / rate; startOffset = 0; }
    try { src.start(when, startOffset); } catch (_) { return; }
    src.onended = () => { if (sec.node === src) sec.node = null; };
    sec.node = src; sec.startCtx = when; sec.startOffset = startOffset; sec.rate = rate;
  }

  function ensureScheduled(sec, c, rate) {
    if (!sec.buffer || !audioCtx) return;
    if (!sec.node) { scheduleNode(sec, c, rate); return; }
    // Drift / rate check against the video playhead — re-anchor if off.
    const expected = c - sec.start;
    const actual = sec.startOffset + (audioCtx.currentTime - sec.startCtx) * sec.rate;
    if (sec.rate !== rate || Math.abs(expected - actual) > WA_DRIFT) {
      scheduleNode(sec, c, rate);
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

    // Prefetch the current + upcoming sections.
    for (const sec of waSections.values()) {
      if (sec.end > c - 1 && sec.start < c + WA_LOOKAHEAD && !sec.buffer) ensureBuffer(sec);
    }

    if (!waDesiredPlaying) return;

    const cur = sectionAt(c);
    if (!cur || !cur.buffer) {
      // Underrun: the dub for the playhead isn't produced yet (still in the
      // prelude, or the stream hasn't reached here). Freeze the video and
      // silence any tail node; a later tick resumes once the section lands.
      waBuffering = true;
      stopAllNodes();
      waPause();
      reportPlayhead(true);
      return;
    }
    waBuffering = false;
    // Resume if we (or a stalled autoplay) left the video paused while the
    // user still wants to watch.
    if (video.paused) waPlay();
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
      reportPlayhead(true);
      watchAlongTick();
    },
    pause() {
      if (waProgrammatic) { waProgrammatic = false; return; }  // our buffering-pause
      waDesiredPlaying = false;
      stopAllNodes();
      reportPlayhead(true);
    },
    seeked() {
      stopAllNodes();
      reportPlayhead(true);
      watchAlongTick();
    },
    ratechange() {
      const c = video ? video.currentTime : 0;
      const rate = video ? (video.playbackRate || 1) : 1;
      const cur = sectionAt(c);
      if (cur) scheduleNode(cur, c, rate);
    },
    ended() { stopWatchAlong(); },
    emptied: onVideoSourceChangedWA,
    loadstart: onVideoSourceChangedWA,
  };

  function onVideoSourceChangedWA() {
    if (!waActive) return;
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
  let lastSeenUrl = window.location.href;
  setInterval(() => {
    if (window.location.href !== lastSeenUrl) {
      lastSeenUrl = window.location.href;
      if (!playing) setTimeout(autoPlayIfApplicable, 300);
    }
  }, 500);
})();
