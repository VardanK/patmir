// Patmir session — ONE copy of the job protocol, driving two surfaces.
//
// The popup and the in-page button both need the same sequence: look the URL up
// in the cache, submit a job if it is a miss, poll it to completion, then play.
// That is ~120 lines of protocol knowledge — which statuses are terminal, that
// `waiting` means the accelerator lock, that `download_pct` is special-cased,
// that `live_pct` is only meaningful when `live_stage` equals the running stage,
// that watch-along has `prelude` and `streaming` — and this repo already has a
// scar from duplicating knowledge across those two surfaces (see the comment on
// `contentScriptFiles` in popup.js: the injected-file list drifted from the
// manifest and produced a self-inflicted "Receiving end does not exist"). So
// there is one implementation, here.
//
// Following lib/sync.js exactly: NO DOM, NO timers, NO direct `chrome.*`.
// Everything reaches the world through an injected `transport`, which is what
// lets the whole state machine be driven by hand under `node --test`. The host
// owns the interval and calls `tick()`.
//
// Do NOT convert this to an ES module: it is loaded both as a content script
// (assigning `globalThis.PatmirSession`) and by `require` in the tests.
(function (root, factory) {
  var api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PatmirSession = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var QUALITY = "quality";
  var WATCH_ALONG = "watch-along";

  // Poll cadences, in ms. The job poll matches what the popup used; the status
  // poll is cheap (an in-process call on the overlay side) and only exists so
  // the label reacts to playback ending without waiting for a job tick.
  var JOB_POLL_MS = 2000;
  var JOB_POLL_IDLE_MS = 8000;   // nothing watching: back off
  var STATUS_POLL_MS = 500;

  // `findVideo()` filters on `duration > 1`, so a fresh watch page legitimately
  // has no video for a few hundred ms. Show "not ready" for this long before
  // falling through to the normal idle label, rather than flashing the wrong
  // one. Same budget `waitForVideo` uses in content.js.
  var VIDEO_GRACE_MS = 8000;

  // How long a `popupOpen` assertion keeps the fast poll cadence alive. The
  // popup re-asserts on every pull, and a transient popup has no close event we
  // could observe, so this is a lease rather than a flag.
  var POPUP_LEASE_MS = 4000;

  var ACTIVE_JOBS_KEY = "patmirActiveJobs";
  var ACTIVE_JOBS_MAX = 20;

  var TERMINAL = { done: true, failed: true, cancelled: true };

  // --- page classification -------------------------------------------------

  /**
   * Which kind of YouTube page is this, for mounting purposes?
   *
   * Kept here rather than in the overlay so it is testable without a DOM, and
   * so the popup can use the same answer to decide whether to delegate.
   */
  function pageKind(href) {
    var u;
    try { u = new URL(href); } catch (_) { return "other"; }
    var host = (u.hostname || "").toLowerCase();
    if (host === "youtu.be") return "short-link";
    if (!/(^|\.)youtube(-nocookie)?\.com$/.test(host)) return "other";
    var p = u.pathname || "";
    if (p === "/watch") return "watch";
    if (/^\/live\//.test(p)) return "live";
    if (/^\/shorts\//.test(p)) return "shorts";
    if (/^\/embed\//.test(p)) return "embed";
    return "other";
  }

  /** Page kinds that get the full-width button under the player. */
  function isInlinePage(kind) {
    return kind === "watch" || kind === "live";
  }

  /** Page kinds that get the compact floating pill instead. */
  function isFloatingPage(kind) {
    return kind === "shorts" || kind === "embed";
  }

  // --- mount points --------------------------------------------------------
  //
  // Ordered: first hit wins. Several entries rather than one because YouTube
  // rearranges this region, and the last is deliberately selector-free — in the
  // spirit of `findVideo()`, which avoids YouTube class names entirely.
  var MOUNTS = [
    // The column directly under the player, above the title. Prepending is what
    // puts the button *immediately* under the video.
    { name: "below", selector: "ytd-watch-flexy #primary-inner #below", position: "prepend" },
    // Same visual result one level up; survives a rename of #below.
    { name: "player-sibling", selector: "ytd-watch-flexy #primary-inner > #player", position: "after" },
    // The title/metadata block; inserting before it is again "under the player".
    { name: "metadata", selector: "ytd-watch-metadata", position: "before" },
    // Coarse but correct-ish column.
    { name: "primary-inner", selector: "#primary-inner", position: "prepend" },
    // Last resort: walk up from the <video> element. No YouTube ids involved.
    { name: "video-ancestor", selector: null, position: "after" },
  ];

  /**
   * Pick a mount from a map of which probes resolved.
   *
   * Pure so the *policy* is testable with plain booleans; the overlay does the
   * four `querySelector` calls and hands the answers in.
   */
  function mountPlan(present) {
    var p = present || {};
    for (var i = 0; i < MOUNTS.length; i++) {
      if (p[MOUNTS[i].name]) {
        return { name: MOUNTS[i].name, selector: MOUNTS[i].selector, position: MOUNTS[i].position };
      }
    }
    return null;
  }

  // --- progress ------------------------------------------------------------

  /**
   * Overall completion of a job record, 0..1, or null if not derivable.
   *
   * One number on purpose: the in-page button shows Armenian only, with no
   * English stage names, so there is nowhere to put "tts 40%". Stages are
   * weighted equally — crude (tts is over half the wall) but monotonic, which
   * is the property a progress readout actually needs.
   */
  function jobProgress(rec) {
    if (!rec) return null;
    var stages = rec.stage_states || {};
    var names = Object.keys(stages);
    if (!names.length) {
      // A watch-along record has no `stage_states` at all: it does not walk
      // the pipeline once, it dubs the video in sections. So the fraction is
      // how much of the video is dubbed. Without this the in-page progress bar
      // showed nothing in watch-along, which was invisible until that mode
      // became the default.
      if (typeof rec.dubbed_s === "number" && typeof rec.duration === "number"
          && rec.duration > 0) {
        return clamp01(rec.dubbed_s / rec.duration);
      }
      return null;
    }
    var done = 0, running = null;
    for (var i = 0; i < names.length; i++) {
      if (stages[names[i]] === "done") done++;
      else if (stages[names[i]] === "running") running = names[i];
    }
    var partial = 0;
    if (running === "download" && typeof rec.download_pct === "number") {
      partial = clamp01(rec.download_pct);
    } else if (running && rec.live_stage === running && typeof rec.live_pct === "number") {
      // `live_pct` belongs to `live_stage`; using it for a different stage is
      // how a progress bar goes backwards.
      partial = clamp01(rec.live_pct);
    }
    return clamp01((done + partial) / names.length);
  }

  function clamp01(v) {
    if (typeof v !== "number" || !isFinite(v)) return 0;
    return v < 0 ? 0 : v > 1 ? 1 : v;
  }

  // --- view derivation -----------------------------------------------------

  // The four labels. Progress NEVER enters a label: a 52 px full-width control
  // whose text changes every two seconds cannot be aimed at.
  var TRANSLATE = "translate";
  var STOP = "stop";
  // `cancel`, distinct from `stop`, because the two are different acts and the
  // words for them read differently. `Չթարգմանել` ("do not translate") is
  // right while a dub is PLAYING: it turns the Armenian track off. While a job
  // is being PREPARED it reads as a status — the author's report was that the
  // button "says that it's not translating, while in reality it does" — so
  // those states get `Չեղարկել` ("cancel"), which names the action instead.
  var CANCEL = "cancel";
  var RETRY = "retry";
  var SETTINGS = "settings";

  /**
   * Turn session state into everything a surface needs to render, minus the
   * words. Language-free so one derivation serves the Armenian in-page button
   * and the English popup.
   */
  function deriveView(state) {
    var s = state || {};
    var pb = s.playback || {};
    var job = s.job || null;
    var jobActive = !!(job && !TERMINAL[job.status]) || s.phase === "submitting";

    // Blocking configuration/connection problems come first: nothing else the
    // viewer could press will work.
    if (s.errorKind === "no-token") {
      return view("no-token", SETTINGS, { action: "options", tone: "warn" });
    }
    if (s.errorKind === "orphaned") {
      return view("orphaned", TRANSLATE, { disabled: true, tone: "warn" });
    }
    if (s.errorKind === "unreachable") {
      return view("server-down", RETRY, { action: "retry", tone: "error" });
    }

    // An ad owns the element's clock, so the dub is suspended whatever else is
    // true. Reported before playback so the viewer sees why it went quiet.
    if (s.adShowing && (pb.playing || jobActive)) {
      return view("ad", STOP, { action: "stop", tone: "active" });
    }
    if (pb.playing && pb.buffering) {
      return view("buffering", STOP, { action: "stop", tone: "active", busy: true });
    }
    if (s.phase === "stopping") {
      return view("stopping", STOP, { disabled: true, tone: "active", busy: true });
    }
    if (s.phase === "starting") {
      return view("starting", STOP, { disabled: true, tone: "active", busy: true });
    }

    // A job in flight. Watch-along plays *while* streaming, so this is checked
    // before plain playback — the section count is the more useful readout.
    if (jobActive) {
      var kind = "running";
      if (s.phase === "submitting" || !job) kind = "submitting";
      else if (job.status === "queued") kind = "queued";
      else if (job.status === "waiting") kind = "waiting-gpu";
      // A watch-along record has no `stage_states`, so `runningStage` cannot
      // see its download; it sat under "prelude" with no percentage and no
      // fill for the whole fetch (author: "the button stays gray while
      // loading"). The yt-dlp hook writes `live_stage`/`download_pct` onto
      // both record kinds, and `download_pct` stays 1.0 once the fetch is
      // done, so `< 1` is what separates "downloading" from "downloaded".
      else if (job.status === "prelude" && job.live_stage === "download"
               && typeof job.download_pct === "number" && job.download_pct < 1) {
        kind = "downloading";
      }
      else if (job.status === "prelude") kind = "prelude";
      else if (job.status === "streaming") kind = "streaming";
      else if (runningStage(job) === "download") kind = "downloading";
      // `Չեղարկել` for EVERY in-flight job, streaming included (author's
      // call). Watch-along does play while producing, which is why STOP was
      // used here at first — but the button's job is to name what pressing it
      // does, and while work is still queued behind the playhead that is
      // cancelling the work, not just muting a finished track. `Չթարգմանել`
      // is left to states where nothing is being produced any more.
      return view(kind, CANCEL, {
        action: "stop", tone: "active", busy: true,
        progress: jobProgress(job),
        // The download's OWN fraction, for the "Downloading… {pct}%" subtitle.
        // `progress` is whole-job completion, which for a batch job puts a 60%
        // download at 6% and for a stream is unknown until the first section.
        stagePct: kind === "downloading" && typeof job.download_pct === "number"
          ? clamp01(job.download_pct) : null,
        // `job` is null while phase is "submitting" — the request is in flight
        // and there is no record yet. Reading through it here threw, which took
        // out the button on the very first press.
        sections: job && typeof job.sections_done === "number" ? job.sections_done : null,
      });
    }

    // A failed job outranks "playing". Watch-along plays while it streams, so
    // a stream the server fails mid-way (translator unavailable, say) used to
    // render as "Playing dub." for as long as the content script had not yet
    // released the element — the one moment the viewer most needs the reason.
    if (job && job.status === "failed") {
      return view("job-failed", RETRY, {
        action: "retry", tone: "error", errorText: job.error || null,
      });
    }

    if (pb.playing) {
      return view("playing", STOP, { action: "stop", tone: "active" });
    }
    if (job && job.status === "cancelled") {
      return view("cancelled", TRANSLATE, { action: "start", tone: "idle" });
    }
    if (s.errorKind === "playback") {
      return view("playback-error", TRANSLATE, {
        action: "start", tone: "error", errorText: s.error || null,
      });
    }

    if (s.phase === "boot" || s.phase === "checking" || !s.lookup) {
      return view("loading", TRANSLATE, { disabled: true, tone: "idle", busy: true });
    }

    // "No video yet" is the normal state for the first few hundred ms of a
    // watch page. Latch it only briefly, then fall through — a permanently
    // disabled button on a page that simply has no player is worse than one
    // that fails when pressed.
    if (pb.hasVideo === false && withinGrace(s)) {
      return view("no-video", TRANSLATE, { disabled: true, tone: "idle" });
    }

    if (s.lookup.cached) {
      return view("cached-idle", TRANSLATE, {
        action: "start", tone: "idle", optedOut: !!pb.disabled,
      });
    }
    if (s.lookup.partial) {
      return view("partial", TRANSLATE, {
        action: "start", tone: "idle",
        sections: s.lookup.sections || 0,
      });
    }
    return view("not-cached", TRANSLATE, { action: "start", tone: "idle", mode: s.mode });
  }

  /**
   * Which mode a surface should use when the viewer has not chosen one.
   *
   * Watch-along by default (author's call): it starts playing within about
   * half a minute instead of after a full pass over the video, which is the
   * difference between pressing translate and watching, and pressing
   * translate and waiting. Quality mode stays one click away and an explicit
   * stored choice always wins.
   *
   * Mode is folded into the cache key, so this changes which dubs a lookup
   * finds: a video previously dubbed in quality mode is not the same cache
   * entry as the same video in watch-along.
   */
  function resolveMode(stored) {
    var st = stored || {};
    return st.patmirMode === QUALITY ? QUALITY : WATCH_ALONG;
  }

  /**
   * Resolve the remembered translate-everything choice from stored values.
   *
   * `patmirTranslateEnabled` replaced two older keys, and reading only the new
   * one silently reset everybody who already had translation working: the
   * author reloaded, opened a video and got the original language with a blue
   * `Թարգմանել` button, because the new flag had never been written.
   *
   * `patmirAutoPlayEnabled` is the legacy signal and it DEFAULTED ON, so a
   * stored `true` means "this viewer had automatic dubbing and expects it".
   * That is honoured once; anything explicit on the new key always wins.
   */
  function resolveTranslateEnabled(stored) {
    var st = stored || {};
    if (typeof st.patmirTranslateEnabled === "boolean") {
      return st.patmirTranslateEnabled;
    }
    return st.patmirAutoPlayEnabled === true;
  }

  /**
   * Should the surface press its own start button?
   *
   * `patmirTranslateEnabled` is a GLOBAL, remembered choice, not a per-video
   * one: "if enabled it should work for all videos, if disabled should not
   * translate". So this fires on any video whose view offers `start`, which
   * includes one that has never been dubbed — that is the point, and it means
   * enabling translation submits work.
   *
   * `retry` is deliberately NOT auto-pressed. A failed job that re-submits
   * itself is a loop, and the failure is usually the reason the viewer needs
   * to see the button.
   */
  function shouldAutoStart(v, opts) {
    var o = opts || {};
    if (!o.enabled) return false;
    if (o.tried) return false;
    if (!v || v.disabled) return false;
    return v.action === "start";
  }

  function view(kind, label, extra) {
    var v = {
      kind: kind,
      label: label,
      action: null,
      disabled: false,
      busy: false,
      tone: "idle",
      progress: null,
      stagePct: null,
      sections: null,
      errorText: null,
      optedOut: false,
      mode: null,
    };
    for (var k in extra) if (Object.prototype.hasOwnProperty.call(extra, k)) v[k] = extra[k];
    return v;
  }

  function runningStage(rec) {
    var stages = (rec && rec.stage_states) || {};
    for (var k in stages) {
      if (Object.prototype.hasOwnProperty.call(stages, k) && stages[k] === "running") return k;
    }
    return null;
  }

  function withinGrace(s) {
    // `== null`, not falsy: 0 is a real timestamp. Treating it as "unknown" kept
    // the grace window open forever, which left the button permanently disabled
    // on any page whose player never appears.
    if (s.mountedAt == null || s.nowMs == null) return true;
    return (s.nowMs - s.mountedAt) < VIDEO_GRACE_MS;
  }

  // --- strings -------------------------------------------------------------
  //
  // The in-page surface is Armenian only, with no English stage names: the
  // viewer is a child watching a video, not an operator reading a pipeline. The
  // popup keeps English because it is also the developer's window.

  var STRINGS_HY = {
    labels: {
      translate: "Թարգմանել",
      stop: "Չթարգմանել",
      cancel: "Չեղարկել",
      retry: "Կրկին փորձել",
      settings: "Կարգավորել",
    },
    loading: "Ստուգվում է…",
    "no-token": "Անհրաժեշտ է token",
    orphaned: "Հավելումը թարմացվել է — թարմացրու էջը",
    "server-down": "Սերվերը հասանելի չէ",
    "no-video": "Վիդեոն դեռ չի բացվել…",
    "not-cached": "",
    partial: "{n} հատված պահված է",
    submitting: "Ուղարկվում է…",
    queued: "Հերթում է…",
    "waiting-gpu": "Սպասում է GPU-ին…",
    downloading: "Ներբեռնվում է… {pct}%",
    prelude: "Սկիզբը վերլուծվում է…",
    running: "Թարգմանում… {pct}%",
    streaming: "Հեռարձակվում է — {n} հատված",
    "job-failed": "Չհաջողվեց",
    cancelled: "Ընդհատված է",
    "cached-idle": "Պատրաստ է",
    "cached-idle-optedout": "Պատրաստ է — ավտոմատ միացումն անջատված է",
    starting: "Բեռնվում է…",
    stopping: "Անջատվում է…",
    buffering: "Բեռնվում է…",
    playing: "",
    ad: "Գովազդ…",
    "playback-error": "Չհաջողվեց միացնել",
  };

  var STRINGS_EN = {
    labels: {
      translate: "Translate this",
      stop: "Stop dub",
      cancel: "Cancel",
      retry: "Retry",
      settings: "Open options",
    },
    loading: "Checking…",
    "no-token": "No bearer token set — open Options.",
    orphaned: "Extension reloaded — refresh the page.",
    "server-down": "Server unreachable.",
    "not-cached": "Not dubbed yet.",
    partial: "Partially streamed ({n} sections cached).",
    submitting: "Submitting…",
    queued: "Queued…",
    "waiting-gpu": "Waiting for the GPU (another job is running)…",
    downloading: "Downloading… {pct}%",
    prelude: "Analyzing opening (prelude)…",
    running: "Running… {pct}%",
    streaming: "Streaming — {n} sections dubbed",
    "job-failed": "Job failed.",
    cancelled: "Cancelled.",
    "cached-idle": "Dub is ready (cached).",
    "cached-idle-optedout": "Dub cached — auto-play is off for this video.",
    starting: "Loading dub…",
    stopping: "Stopping…",
    buffering: "Buffering…",
    playing: "Playing dub.",
    ad: "Ad playing — dub paused.",
    "playback-error": "Playback failed.",
    "no-video": "Waiting for the player…",
  };

  function formatLabel(v, strings) {
    var s = strings || STRINGS_HY;
    return (s.labels && s.labels[v.label]) || v.label;
  }

  function formatSubtitle(v, strings) {
    var s = strings || STRINGS_HY;
    var key = v.kind;
    if (key === "cached-idle" && v.optedOut) key = "cached-idle-optedout";
    var tpl = s[key];
    if (tpl == null) tpl = "";
    // A stage's own fraction wins over whole-job progress when the subtitle
    // names that stage (`downloading`); the fill keeps showing the whole job.
    var frac = typeof v.stagePct === "number" ? v.stagePct : v.progress;
    var pct = frac == null ? 0 : Math.round(frac * 100);
    var out = tpl.replace("{pct}", String(pct)).replace("{n}", String(v.sections || 0));
    if (v.errorText) out = out ? out + " — " + trim(v.errorText) : trim(v.errorText);
    return out;
  }

  function trim(text, max) {
    var t = String(text == null ? "" : text);
    var n = max || 120;
    return t.length > n ? t.slice(0, n - 1) + "…" : t;
  }

  // --- the session ---------------------------------------------------------

  var NULL_PLAYER = {
    play: function () { return Promise.resolve({ ok: false, error: "no player in this context" }); },
    playWatchAlong: function () { return Promise.resolve({ ok: false, error: "no player in this context" }); },
    stop: function () { return Promise.resolve({ ok: true }); },
    // `hasVideo` is deliberately absent rather than `false`: this context cannot
    // observe the page at all, so the answer is unknown. Reporting `false` made
    // the popup show "waiting for the player" on every non-YouTube page, where
    // the correct offer is still "Translate this".
    status: function () { return Promise.resolve({ ok: true, playing: false }); },
  };

  /**
   * Build a session.
   *
   * `transport`:
   *   sw(msg)      -> resolves the service worker's `data`, throws on error
   *   player       -> { play, playWatchAlong, stop, status }
   *   getUrl()     -> the URL this session is about
   *   storage      -> { get(keys), set(obj) }
   *   now()        -> ms
   *   openOptions? -> optional, for the SETTINGS action
   *
   * `autoPlayOnDone` — start playback as soon as a submitted job finishes. True
   * for the in-page button (the point of an in-page control is not to need a
   * second press); false for the popup, which keeps its two-step behaviour.
   */
  function createSession(opts) {
    var o = opts || {};
    var tx = o.transport || {};
    var player = tx.player || NULL_PLAYER;
    var now = tx.now || function () { return 0; };
    var autoPlayOnDone = !!o.autoPlayOnDone;
    var listeners = [];

    var st = {
      seq: 0,
      url: safeUrl(),
      mode: o.mode === WATCH_ALONG ? WATCH_ALONG : QUALITY,
      phase: "boot",
      lookup: null,
      job: null,
      jobId: null,
      playback: { playing: false, hasVideo: null, disabled: false, buffering: false, mode: null },
      error: null,
      errorKind: null,
      adShowing: false,
      lastPollAt: 0,
      lastStatusAt: 0,
      mountedAt: now(),
      nowMs: now(),
      popupLeaseUntil: 0,
    };

    function safeUrl() {
      try { return tx.getUrl ? tx.getUrl() : null; } catch (_) { return null; }
    }

    function emit() {
      st.nowMs = now();
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](st); } catch (_) { /* a bad listener must not break the loop */ }
      }
    }

    function getState() { st.nowMs = now(); return st; }

    /** Bump on every command. An awaited reply whose captured seq no longer
     *  matches is discarded — that is the double-click and the
     *  mode-flip-mid-submit guard in one. */
    function begin() { st.seq += 1; return st.seq; }

    /** Is this reply still the one we asked for? URL as well as seq, because a
     *  YouTube SPA navigation can land between request and response and the
     *  answer would then belong to another video. Same defence
     *  `autoPlayIfApplicable` already applies in content.js. */
    function fresh(seq) { return seq === st.seq && safeUrl() === st.url; }

    function fail(kind, err) {
      st.error = String((err && err.message) || err || "");
      st.errorKind = classifyError(st.error, kind);
      st.phase = "error";
    }

    function classifyError(msg, fallback) {
      // The exact wording sw.js throws when the options page has no token —
      // worth special-casing because the fix is a click, not a retry.
      if (/No bearer token set/i.test(msg)) return "no-token";
      if (/Extension context invalidated/i.test(msg)) return "orphaned";
      return fallback || "unreachable";
    }

    function clearError() { st.error = null; st.errorKind = null; }

    // --- reads -----------------------------------------------------------

    function refresh() {
      var seq = begin();
      st.url = safeUrl();
      st.phase = "checking";
      clearError();
      emit();
      return Promise.resolve()
        .then(function () {
          return tx.sw({ type: "cache:lookup", url: st.url, mode: st.mode });
        })
        .then(function (data) {
          if (!fresh(seq)) return null;
          st.lookup = data || {};
          st.phase = "idle";
          return readStatus();
        })
        .then(function () {
          if (!fresh(seq)) return null;
          // Re-adopt a job started elsewhere (the popup, or this tab before a
          // navigation). Cheap, and self-healing: the server dedupes by cache
          // key, so a stale entry just resolves to whatever it actually has.
          if (!st.jobId && st.lookup && !st.lookup.cached) return adoptJob();
          return null;
        })
        .then(function () { emit(); return st; })
        .catch(function (e) {
          if (!fresh(seq)) return st;
          fail(null, e);
          emit();
          return st;
        });
    }

    function readStatus() {
      return Promise.resolve()
        .then(function () { return player.status(); })
        .then(function (s) {
          st.lastStatusAt = now();
          if (s && s.ok !== false) {
            st.playback = {
              playing: !!s.playing,
              hasVideo: s.hasVideo === undefined ? null : !!s.hasVideo,
              disabled: !!s.disabled,
              buffering: !!s.buffering,
              mode: s.mode || null,
            };
            if (s.adShowing !== undefined) st.adShowing = !!s.adShowing;
          }
          return st.playback;
        })
        .catch(function () { return st.playback; });
    }

    function adoptJob() {
      var key = st.lookup && st.lookup.cache_key;
      if (!key || !tx.storage) return Promise.resolve(null);
      return Promise.resolve(tx.storage.get(ACTIVE_JOBS_KEY))
        .then(function (bag) {
          var map = (bag && bag[ACTIVE_JOBS_KEY]) || {};
          var jobId = map[key];
          if (!jobId) return null;
          return tx.sw({ type: "jobs:get", job_id: jobId }).catch(function () { return null; });
        })
        .then(function (rec) {
          if (!rec) return null;
          if (TERMINAL[rec.status]) return forgetJob(rec.cache_key || key);
          st.jobId = rec.job_id;
          st.job = rec;
          st.phase = "job";
          return rec;
        })
        .catch(function () { return null; });
    }

    function rememberJob(key, jobId) {
      if (!key || !jobId || !tx.storage) return Promise.resolve();
      return Promise.resolve(tx.storage.get(ACTIVE_JOBS_KEY)).then(function (bag) {
        var map = Object.assign({}, (bag && bag[ACTIVE_JOBS_KEY]) || {});
        map[key] = jobId;
        var keys = Object.keys(map);
        while (keys.length > ACTIVE_JOBS_MAX) delete map[keys.shift()];
        var out = {};
        out[ACTIVE_JOBS_KEY] = map;
        return tx.storage.set(out);
      }).catch(function () { /* storage is a convenience, never load-bearing */ });
    }

    function forgetJob(key) {
      if (!key || !tx.storage) return Promise.resolve(null);
      return Promise.resolve(tx.storage.get(ACTIVE_JOBS_KEY)).then(function (bag) {
        var map = Object.assign({}, (bag && bag[ACTIVE_JOBS_KEY]) || {});
        if (!(key in map)) return null;
        delete map[key];
        var out = {};
        out[ACTIVE_JOBS_KEY] = map;
        return tx.storage.set(out);
      }).catch(function () { return null; });
    }

    // --- commands --------------------------------------------------------

    function primary() {
      var v = deriveView(getState());
      if (v.disabled) return Promise.resolve(st);
      if (v.action === "start" || v.action === "retry") return start();
      if (v.action === "stop") return stop();
      if (v.action === "options") {
        if (tx.openOptions) { try { tx.openOptions(); } catch (_) {} }
        return Promise.resolve(st);
      }
      return Promise.resolve(st);
    }

    function start(force) {
      var seq = begin();
      var url = safeUrl();
      st.url = url;
      clearError();
      st.job = null;
      st.jobId = null;
      st.phase = "submitting";
      emit();

      var cachedKey = st.lookup && st.lookup.cached ? st.lookup.cache_key : null;
      // Quality mode with the dub already on disk: nothing to submit.
      if (st.mode === QUALITY && cachedKey && !force) {
        return playNow(seq, cachedKey);
      }

      var msg = { type: "jobs:submit", url: url, mode: st.mode };
      if (force) msg.force = true;
      return Promise.resolve(tx.sw(msg))
        .then(function (rec) {
          if (!fresh(seq)) return st;
          st.job = rec || null;
          st.jobId = rec && rec.job_id;
          st.phase = "job";
          if (rec && rec.cache_key) rememberJob(rec.cache_key, rec.job_id);
          // Watch-along starts playing immediately and buffers; quality has
          // nothing to play until the job finishes.
          if (st.mode === WATCH_ALONG) {
            var key = (rec && rec.cache_key) || cachedKey;
            return playNow(seq, key, true);
          }
          if (rec && rec.status === "done" && rec.cache_key && !force) {
            return autoPlayOnDone ? playNow(seq, rec.cache_key) : (emit(), st);
          }
          emit();
          return st;
        })
        .catch(function (e) {
          if (!fresh(seq)) return st;
          fail(null, e);
          emit();
          return st;
        });
    }

    function regenerate() { return start(true); }

    function playNow(seq, cacheKey, watchAlong) {
      if (!cacheKey) { emit(); return Promise.resolve(st); }
      st.phase = "starting";
      emit();
      var call = (watchAlong || st.mode === WATCH_ALONG)
        ? player.playWatchAlong(cacheKey)
        : player.play(cacheKey);
      return Promise.resolve(call)
        .then(function (resp) {
          if (!fresh(seq)) return st;
          if (resp && resp.ok === false) throw new Error(resp.error || "playback failed");
          st.phase = "playing";
          return readStatus();
        })
        .then(function () { if (fresh(seq)) emit(); return st; })
        .catch(function (e) {
          if (!fresh(seq)) return st;
          fail("playback", e);
          emit();
          return st;
        });
    }

    function stop() {
      var seq = begin();
      var job = st.job;
      st.phase = "stopping";
      emit();
      return Promise.resolve(player.stop())
        .then(function () {
          // Cancel the server-side work too. Quality jobs had no path to this
          // at all from the extension, so switching translation off left the
          // GPU grinding on a dub nobody was waiting for.
          if (!job || !job.job_id || TERMINAL[job.status]) return null;
          if (st.mode === WATCH_ALONG) {
            // The content script's `stopWatchAlong` sends `stream:stop` too,
            // but only once playback has ATTACHED (`waActive`). While the job
            // is still in prelude the button already reads Չեղարկել, and
            // pressing it stopped nothing on the server. `/stream/<key>/stop`
            // is idempotent, so the two never fight.
            if (!job.cache_key) return null;
            return tx.sw({ type: "stream:stop", cache_key: job.cache_key })
              .catch(function () { return null; });
          }
          return tx.sw({ type: "jobs:cancel", job_id: job.job_id })
            .catch(function () { return null; });
        })
        .then(function () {
          if (job && job.cache_key) forgetJob(job.cache_key);
          if (!fresh(seq)) return st;
          st.job = null;
          st.jobId = null;
          st.phase = "idle";
          clearError();
          return readStatus();
        })
        .then(function () { if (fresh(seq)) emit(); return st; })
        .catch(function (e) {
          if (!fresh(seq)) return st;
          fail("playback", e);
          emit();
          return st;
        });
    }

    function setMode(mode) {
      var next = mode === WATCH_ALONG ? WATCH_ALONG : QUALITY;
      if (next === st.mode) return Promise.resolve(st);
      st.mode = next;
      st.job = null;
      st.jobId = null;
      st.lookup = null;
      if (tx.storage) {
        try { tx.storage.set({ patmirMode: next }); } catch (_) {}
      }
      return refresh();
    }

    /** Report what the page's ad detector sees. Pure state; the host observes. */
    function setAdShowing(showing) {
      var next = !!showing;
      if (next === st.adShowing) return false;
      st.adShowing = next;
      emit();
      return true;
    }

    function notePopupOpen() { st.popupLeaseUntil = now() + POPUP_LEASE_MS; }

    // --- host-driven tick -------------------------------------------------

    /**
     * One step. The host owns the interval; this file owns no timers.
     * Returns a promise of `{changed}` so a host can skip a render.
     */
    function tick() {
      var t = now();
      st.nowMs = t;
      var work = [];
      var watched = t < st.popupLeaseUntil;
      var interval = watched ? JOB_POLL_MS : (st.job ? JOB_POLL_MS : JOB_POLL_IDLE_MS);

      if (st.jobId && (t - st.lastPollAt) >= interval) work.push(pollJob());
      if ((t - st.lastStatusAt) >= STATUS_POLL_MS) work.push(readStatus());
      if (!work.length) return Promise.resolve({ changed: false });
      return Promise.all(work).then(function () { emit(); return { changed: true }; });
    }

    function pollJob() {
      var seq = st.seq;
      var jobId = st.jobId;
      st.lastPollAt = now();
      return Promise.resolve(tx.sw({ type: "jobs:get", job_id: jobId }))
        .then(function (rec) {
          if (!fresh(seq) || st.jobId !== jobId) return null;
          st.job = rec || st.job;
          if (!rec) return null;
          if (TERMINAL[rec.status]) {
            if (rec.cache_key) forgetJob(rec.cache_key);
            st.jobId = null;
            if (rec.status === "failed" && st.playback.playing) {
              // The dub is still attached to a job the server has failed.
              // Release the element so the original audio comes back, then
              // re-read playback so the view flips off "playing". The
              // content script does the same off its own sections poll; this
              // is the belt for a poll it missed (popup-only surface, or a
              // page whose poll was already torn down).
              return Promise.resolve(player.stop())
                .catch(function () { return null; })
                .then(function () { return readStatus(); })
                .then(function () { return null; });
            }
            if (rec.status === "done") {
              // Re-read the cache so an idle button says "ready" rather than
              // "not dubbed" if we do not auto-play.
              st.lookup = Object.assign({}, st.lookup || {}, {
                cached: true, cache_key: rec.cache_key,
              });
              if (autoPlayOnDone && st.mode === QUALITY && !st.playback.playing) {
                return playNow(seq, rec.cache_key);
              }
            }
          }
          return null;
        })
        .catch(function (e) {
          if (!fresh(seq)) return null;
          fail(null, e);
          return null;
        });
    }

    return {
      getState: getState,
      onChange: function (cb) { listeners.push(cb); return function () {
        var i = listeners.indexOf(cb); if (i >= 0) listeners.splice(i, 1);
      }; },
      refresh: refresh,
      primary: primary,
      start: start,
      stop: stop,
      regenerate: regenerate,
      setMode: setMode,
      setAdShowing: setAdShowing,
      notePopupOpen: notePopupOpen,
      tick: tick,
      view: function () { return deriveView(getState()); },
    };
  }

  return {
    QUALITY: QUALITY,
    WATCH_ALONG: WATCH_ALONG,
    JOB_POLL_MS: JOB_POLL_MS,
    STATUS_POLL_MS: STATUS_POLL_MS,
    VIDEO_GRACE_MS: VIDEO_GRACE_MS,
    ACTIVE_JOBS_KEY: ACTIVE_JOBS_KEY,
    ACTIVE_JOBS_MAX: ACTIVE_JOBS_MAX,
    MOUNTS: MOUNTS,
    LABELS: { TRANSLATE: TRANSLATE, STOP: STOP, CANCEL: CANCEL, RETRY: RETRY,
              SETTINGS: SETTINGS },
    shouldAutoStart: shouldAutoStart,
    resolveTranslateEnabled: resolveTranslateEnabled,
    resolveMode: resolveMode,
    TRANSLATE_KEYS: ["patmirTranslateEnabled", "patmirAutoPlayEnabled"],
    STRINGS_HY: STRINGS_HY,
    STRINGS_EN: STRINGS_EN,
    NULL_PLAYER: NULL_PLAYER,
    pageKind: pageKind,
    isInlinePage: isInlinePage,
    isFloatingPage: isFloatingPage,
    mountPlan: mountPlan,
    jobProgress: jobProgress,
    deriveView: deriveView,
    formatLabel: formatLabel,
    formatSubtitle: formatSubtitle,
    trim: trim,
    createSession: createSession,
  };
});
