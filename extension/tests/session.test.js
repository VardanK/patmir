// The session state machine: one copy of the job protocol, two surfaces.
//
// Split in two halves, matching the module: the pure derivation (state object
// in, view object out — no transport, no clock) and the session itself, driven
// against a recording fake transport with a hand-turned clock. Nothing here
// needs a DOM.

const { test } = require("node:test");
const assert = require("node:assert/strict");
const S = require("../lib/session.js");

// --- pageKind ---------------------------------------------------------------

test("pageKind classifies every surface we care about", () => {
  const cases = {
    "https://www.youtube.com/watch?v=abc": "watch",
    "https://youtube.com/watch?v=abc&t=10": "watch",
    "https://www.youtube.com/live/abc": "live",
    "https://www.youtube.com/shorts/abc": "shorts",
    "https://www.youtube.com/embed/abc": "embed",
    "https://www.youtube-nocookie.com/embed/abc": "embed",
    "https://youtu.be/abc": "short-link",
    "https://www.youtube.com/": "other",
    "https://www.youtube.com/results?search_query=x": "other",
    "https://example.com/watch?v=abc": "other",
    "not a url": "other",
  };
  for (const [href, want] of Object.entries(cases)) {
    assert.equal(S.pageKind(href), want, href);
  }
});

test("watch and live get the inline button; shorts and embeds get the pill", () => {
  // Live shares ytd-watch-flexy, so it needs no separate handling. Shorts is a
  // recycler with no stable region under the player, and an embed is somebody
  // else's article layout — neither can take a full-width block.
  assert.ok(S.isInlinePage("watch") && S.isInlinePage("live"));
  assert.ok(!S.isInlinePage("shorts") && !S.isInlinePage("embed"));
  assert.ok(S.isFloatingPage("shorts") && S.isFloatingPage("embed"));
  assert.ok(!S.isFloatingPage("watch") && !S.isFloatingPage("other"));
});

// --- mountPlan --------------------------------------------------------------

test("mountPlan takes the first hit, in declared order", () => {
  const names = S.MOUNTS.map((m) => m.name);
  for (let i = 0; i < names.length; i++) {
    // Everything from i onward resolves; expect i, never a later one.
    const present = {};
    for (let j = i; j < names.length; j++) present[names[j]] = true;
    assert.equal(S.mountPlan(present).name, names[i], `from ${names[i]}`);
  }
});

test("mountPlan covers every subset without preferring a later entry", () => {
  const names = S.MOUNTS.map((m) => m.name);
  for (let bits = 0; bits < (1 << names.length); bits++) {
    const present = {};
    for (let i = 0; i < names.length; i++) if (bits & (1 << i)) present[names[i]] = true;
    const plan = S.mountPlan(present);
    const firstSet = names.find((n) => present[n]) || null;
    assert.equal(plan ? plan.name : null, firstSet, `bits=${bits}`);
  }
});

test("mountPlan returns null when nothing resolves", () => {
  // Deliberately null rather than falling back to document.body: guessing a
  // parent on someone else's page is worse than showing no button.
  assert.equal(S.mountPlan({}), null);
  assert.equal(S.mountPlan(), null);
});

test("the last mount is selector-free", () => {
  // In the spirit of findVideo(): a last resort that survives a full id rename.
  const last = S.MOUNTS[S.MOUNTS.length - 1];
  assert.equal(last.selector, null);
  assert.equal(last.name, "video-ancestor");
});

// --- jobProgress ------------------------------------------------------------

const stages = (o) => ({ stage_states: o });

test("jobProgress counts done stages", () => {
  assert.equal(S.jobProgress(stages({ a: "done", b: "pending" })), 0.5);
  assert.equal(S.jobProgress(stages({ a: "done", b: "done" })), 1);
  assert.equal(S.jobProgress(stages({ a: "pending", b: "pending" })), 0);
});

test("jobProgress folds in the running stage's own progress", () => {
  const rec = {
    ...stages({ a: "done", b: "running" }),
    live_stage: "b", live_pct: 0.5,
  };
  assert.equal(S.jobProgress(rec), 0.75);
});

test("live_pct is ignored when it belongs to a different stage", () => {
  // The real subtlety: `live_pct` is whatever the last progress line said, and
  // the last line can be from a stage that has since finished. Crediting it to
  // the current stage makes the number go backwards.
  const rec = {
    ...stages({ a: "done", b: "running" }),
    live_stage: "a", live_pct: 0.9,
  };
  assert.equal(S.jobProgress(rec), 0.5);
});

test("download_pct drives the download stage", () => {
  const rec = {
    ...stages({ download: "running", ingest: "pending" }),
    download_pct: 0.6,
  };
  assert.equal(S.jobProgress(rec), 0.3);
});

test("jobProgress is null with no stages and clamps nonsense", () => {
  assert.equal(S.jobProgress(null), null);
  assert.equal(S.jobProgress(stages({})), null);
  const rec = { ...stages({ a: "running" }), live_stage: "a", live_pct: 5 };
  assert.equal(S.jobProgress(rec), 1);
});

// --- deriveView -------------------------------------------------------------

const IDLE_PB = { playing: false, hasVideo: true, disabled: false, buffering: false };
const base = (over) => ({
  phase: "idle", mode: "quality", lookup: { cached: false }, playback: IDLE_PB,
  mountedAt: 0, nowMs: 100000, ...over,
});

function kindOf(over) { return S.deriveView(base(over)).kind; }

test("a fresh, undubbed watch page offers Թարգմանել", () => {
  const v = S.deriveView(base());
  assert.equal(v.kind, "not-cached");
  assert.equal(v.label, S.LABELS.TRANSLATE);
  assert.equal(v.action, "start");
  assert.equal(v.disabled, false);
  assert.equal(S.formatLabel(v, S.STRINGS_HY), "Թարգմանել");
});

test("a cached dub also offers Թարգմանել — one word for 'give me the Armenian'", () => {
  // A deliberate departure from the popup's Translate this / Play dub split:
  // from the viewer's side both are the same request, and whether the server
  // has to work for it belongs in the subtitle.
  const v = S.deriveView(base({ lookup: { cached: true, cache_key: "k" } }));
  assert.equal(v.kind, "cached-idle");
  assert.equal(v.label, S.LABELS.TRANSLATE);
  assert.equal(v.action, "start");
});

test("playing offers Չթարգմանել", () => {
  const v = S.deriveView(base({ playback: { ...IDLE_PB, playing: true } }));
  assert.equal(v.kind, "playing");
  assert.equal(v.label, S.LABELS.STOP);
  assert.equal(v.action, "stop");
  assert.equal(S.formatLabel(v, S.STRINGS_HY), "Չթարգմանել");
});

test("no token asks to open options rather than offering a doomed press", () => {
  const v = S.deriveView(base({ errorKind: "no-token" }));
  assert.equal(v.kind, "no-token");
  assert.equal(v.label, S.LABELS.SETTINGS);
  assert.equal(v.action, "options");
});

test("an unreachable server offers retry", () => {
  const v = S.deriveView(base({ errorKind: "unreachable" }));
  assert.equal(v.kind, "server-down");
  assert.equal(v.action, "retry");
  assert.equal(v.tone, "error");
});

test("an orphaned content script is disabled, not retryable", () => {
  // After an extension reload the old script cannot talk to anything; a retry
  // button would fail forever. The page needs a refresh.
  const v = S.deriveView(base({ errorKind: "orphaned" }));
  assert.equal(v.kind, "orphaned");
  assert.equal(v.disabled, true);
});

test("every job status maps to a state, and all of them offer the way out", () => {
  const cases = {
    queued: "queued",
    waiting: "waiting-gpu",
    prelude: "prelude",
    streaming: "streaming",
    running: "running",
  };
  for (const [status, kind] of Object.entries(cases)) {
    const v = S.deriveView(base({
      phase: "job", job: { status, job_id: "j", stage_states: { a: "running" } },
    }));
    assert.equal(v.kind, kind, status);
    // Every in-flight job reads `Չեղարկել`, streaming included (author's
    // call). Watch-along does play while producing, which is why STOP was
    // used here at first — but while work is still queued behind the playhead,
    // pressing the button cancels that work rather than muting a finished
    // track. `Չթարգմանել` is left to states where nothing is in flight.
    assert.equal(v.label, S.LABELS.CANCEL, status);
    assert.equal(v.action, "stop", `${status} must stay cancellable`);
  }
});

test("a running download is reported as downloading", () => {
  const v = S.deriveView(base({
    phase: "job",
    job: {
      status: "running", job_id: "j",
      stage_states: { download: "running", ingest: "pending" },
      download_pct: 0.4,
    },
  }));
  assert.equal(v.kind, "downloading");
  // Half the stages, 40% through the running one -> 20% overall.
  assert.equal(Math.round(v.progress * 100), 20);
});

test("a watch-along download is reported as downloading, with its own percentage", () => {
  // A stream record has no `stage_states`, so `runningStage` cannot see the
  // fetch; it used to read "prelude" with no fill and no number for the whole
  // download. The yt-dlp hook writes `live_stage`/`download_pct` on both
  // record kinds.
  const v = S.deriveView(base({
    phase: "job",
    job: { status: "prelude", job_id: "j", live_stage: "download", download_pct: 0.4 },
  }));
  assert.equal(v.kind, "downloading");
  assert.equal(v.progress, null, "whole-video progress is unknown before the first section");
  assert.equal(v.stagePct, 0.4);
  assert.equal(S.formatSubtitle(v, S.STRINGS_EN), "Downloading… 40%");
  // `download_pct` stays 1.0 after the fetch: that is the prelude proper.
  const done = S.deriveView(base({
    phase: "job",
    job: { status: "prelude", job_id: "j", live_stage: "download", download_pct: 1 },
  }));
  assert.equal(done.kind, "prelude");
});

test("the downloading subtitle shows the download's own fraction, not whole-job progress", () => {
  const v = S.deriveView(base({
    phase: "job",
    job: {
      status: "running", job_id: "j",
      stage_states: { download: "running", ingest: "pending" },
      download_pct: 0.6,
    },
  }));
  assert.equal(Math.round(v.progress * 100), 30);          // the fill
  assert.equal(S.formatSubtitle(v, S.STRINGS_EN), "Downloading… 60%");  // the words
});

test("failure offers retry and carries the error", () => {
  const v = S.deriveView(base({
    job: { status: "failed", job_id: "j", error: "boom" },
  }));
  assert.equal(v.kind, "job-failed");
  assert.equal(v.action, "retry");
  assert.equal(v.errorText, "boom");
});

test("a cancelled job returns to the idle offer", () => {
  // The old popup fell through to a bare stage count here, printing nothing
  // useful for a job the user had deliberately stopped.
  const v = S.deriveView(base({ job: { status: "cancelled", job_id: "j" } }));
  assert.equal(v.kind, "cancelled");
  assert.equal(v.action, "start");
});

test("an ad suspends the readout while playing", () => {
  const v = S.deriveView(base({
    adShowing: true, playback: { ...IDLE_PB, playing: true },
  }));
  assert.equal(v.kind, "ad");
  assert.equal(v.label, S.LABELS.STOP);
  assert.equal(S.formatSubtitle(v, S.STRINGS_HY), "Գովազդ…");
});

test("an ad with nothing playing and no job is not reported", () => {
  // Pre-roll on a page where translation is off: nothing to say about it.
  assert.equal(kindOf({ adShowing: true }), "not-cached");
});

test("buffering is distinguished from playing", () => {
  assert.equal(kindOf({ playback: { ...IDLE_PB, playing: true, buffering: true } }), "buffering");
});

test("no video yet is latched only briefly", () => {
  const noVideo = { ...IDLE_PB, hasVideo: false };
  // Within the grace window: say so rather than flash the wrong label.
  assert.equal(
    S.deriveView(base({ playback: noVideo, mountedAt: 1000, nowMs: 2000 })).kind,
    "no-video");
  // Past it: fall through, because a permanently disabled button is worse.
  assert.equal(
    S.deriveView(base({ playback: noVideo, mountedAt: 0, nowMs: S.VIDEO_GRACE_MS + 1 })).kind,
    "not-cached");
});

test("a partially streamed watch-along offers resume with its section count", () => {
  const v = S.deriveView(base({
    mode: "watch-along", lookup: { cached: false, partial: true, sections: 7 },
  }));
  assert.equal(v.kind, "partial");
  assert.equal(v.action, "start");
  assert.equal(S.formatSubtitle(v, S.STRINGS_HY), "7 հատված պահված է");
});

test("no lookup yet is a loading state, not an offer", () => {
  assert.equal(kindOf({ lookup: null }), "loading");
  assert.equal(S.deriveView(base({ lookup: null })).disabled, true);
});

// --- label invariants -------------------------------------------------------

const ALL_STATES = [
  base(),
  base({ lookup: { cached: true } }),
  base({ lookup: { cached: false, partial: true, sections: 2 } }),
  base({ lookup: null }),
  base({ errorKind: "no-token" }),
  base({ errorKind: "unreachable" }),
  base({ errorKind: "orphaned" }),
  base({ errorKind: "playback", error: "x" }),
  base({ playback: { ...IDLE_PB, hasVideo: false }, mountedAt: 0, nowMs: 10 }),
  base({ phase: "submitting" }),
  base({ phase: "starting" }),
  base({ phase: "stopping" }),
  base({ phase: "job", job: { status: "queued", job_id: "j" } }),
  base({ phase: "job", job: { status: "waiting", job_id: "j" } }),
  base({ phase: "job", job: { status: "running", job_id: "j", stage_states: { a: "running" } } }),
  base({ phase: "job", job: { status: "running", job_id: "j", stage_states: { download: "running" } } }),
  base({ phase: "job", job: { status: "prelude", job_id: "j" } }),
  base({ phase: "job", job: { status: "streaming", job_id: "j", sections_done: 3 } }),
  base({ job: { status: "failed", job_id: "j", error: "e" } }),
  base({ job: { status: "cancelled", job_id: "j" } }),
  base({ playback: { ...IDLE_PB, playing: true } }),
  base({ playback: { ...IDLE_PB, playing: true, buffering: true } }),
  base({ adShowing: true, playback: { ...IDLE_PB, playing: true } }),
];

test("no state introduces a fifth label", () => {
  const allowed = new Set(Object.values(S.LABELS));
  for (const st of ALL_STATES) {
    const v = S.deriveView(st);
    assert.ok(allowed.has(v.label), `${v.kind} -> ${v.label}`);
  }
});

test("only three labels ever appear on the primary path", () => {
  // Everything that is not an error or a configuration problem must read as
  // one of three Armenian words: start it, turn it off, or call it off.
  for (const st of ALL_STATES) {
    const v = S.deriveView(st);
    if (v.tone === "error" || v.kind === "no-token" || v.kind === "orphaned") continue;
    assert.ok(
      v.label === S.LABELS.TRANSLATE || v.label === S.LABELS.STOP
        || v.label === S.LABELS.CANCEL,
      `${v.kind} -> ${v.label}`);
  }
});

test("an in-flight job offers Չեղարկել, a finished dub playing offers Չթարգմանել", () => {
  // The reported problem: while translating, the button read `Չթարգմանել`
  // ("do not translate"), which describes a state the viewer is not in. The
  // fix is the word, not the action.
  const preparing = S.deriveView(base({
    phase: "job",
    job: { status: "running", job_id: "j", stage_states: { a: "running" } },
  }));
  assert.equal(S.formatLabel(preparing, S.STRINGS_HY), "Չեղարկել");

  const playing = S.deriveView(base({ playback: { playing: true, hasVideo: true } }));
  assert.equal(S.formatLabel(playing, S.STRINGS_HY), "Չթարգմանել");
});

test("a preparing job carries the progress the bar draws", () => {
  // The view always carried this and the DOM ignored it, which is why the
  // percentage only ever appeared as 12 px of subtitle text.
  const v = S.deriveView(base({
    phase: "job",
    job: {
      status: "running", job_id: "j",
      stage_states: { a: "done", b: "done", c: "running", d: "queued" },
    },
  }));
  assert.equal(typeof v.progress, "number");
  assert.ok(v.progress > 0 && v.progress < 1, `got ${v.progress}`);
});

test("cancel has both renderings, like every other label", () => {
  assert.equal(S.STRINGS_HY.labels.cancel, "Չեղարկել");
  assert.equal(S.STRINGS_EN.labels.cancel, "Cancel");
});

test("every state has both an Armenian and an English rendering", () => {
  for (const st of ALL_STATES) {
    const v = S.deriveView(st);
    for (const strings of [S.STRINGS_HY, S.STRINGS_EN]) {
      assert.equal(typeof S.formatLabel(v, strings), "string");
      const sub = S.formatSubtitle(v, strings);
      assert.equal(typeof sub, "string");
      assert.ok(!sub.includes("{"), `unsubstituted placeholder in ${v.kind}: ${sub}`);
    }
  }
});

test("Armenian progress carries a number and no stage name", () => {
  // The in-page surface is Armenian only: a viewer watching a cartoon should
  // not be reading "separate" or "tts".
  const v = S.deriveView(base({
    phase: "job",
    job: { status: "running", job_id: "j", stage_states: { a: "done", b: "running" } },
  }));
  const sub = S.formatSubtitle(v, S.STRINGS_HY);
  assert.match(sub, /50%/);
  assert.ok(!/[a-z]/.test(sub.replace("%", "")), `latin letters leaked: ${sub}`);
});

test("an opted-out cached video says so", () => {
  const v = S.deriveView(base({
    lookup: { cached: true }, playback: { ...IDLE_PB, disabled: true },
  }));
  assert.equal(v.optedOut, true);
  assert.notEqual(
    S.formatSubtitle(v, S.STRINGS_HY),
    S.formatSubtitle(S.deriveView(base({ lookup: { cached: true } })), S.STRINGS_HY));
});

test("a long error is trimmed", () => {
  const v = S.deriveView(base({ job: { status: "failed", job_id: "j", error: "x".repeat(400) } }));
  assert.ok(S.formatSubtitle(v, S.STRINGS_HY).length < 200);
});

// --- createSession ----------------------------------------------------------

function makeFake(opts = {}) {
  const calls = [];
  let clock = 1000;
  const store = { ...(opts.store || {}) };
  const state = {
    lookup: opts.lookup || { cached: false, cache_key: "k1" },
    submit: opts.submit || { job_id: "j1", cache_key: "k1", status: "queued" },
    job: opts.job || null,
    status: opts.status || { ok: true, playing: false, hasVideo: true },
  };
  const transport = {
    sw: async (msg) => {
      calls.push(msg);
      if (msg.type === "cache:lookup") return state.lookup;
      if (msg.type === "jobs:submit") return state.submit;
      if (msg.type === "jobs:get") return state.job;
      if (msg.type === "jobs:cancel") return { job_id: msg.job_id, status: "cancelled" };
      throw new Error("unexpected message " + msg.type);
    },
    player: {
      play: async (k) => { calls.push({ type: "player:play", k }); state.status = { ok: true, playing: true, hasVideo: true }; return { ok: true }; },
      playWatchAlong: async (k) => { calls.push({ type: "player:playWatchAlong", k }); state.status = { ok: true, playing: true, hasVideo: true }; return { ok: true }; },
      stop: async () => { calls.push({ type: "player:stop" }); state.status = { ok: true, playing: false, hasVideo: true }; return { ok: true }; },
      status: async () => state.status,
    },
    getUrl: () => state.url || "https://www.youtube.com/watch?v=abc",
    storage: {
      get: async (keys) => {
        const want = Array.isArray(keys) ? keys : [keys];
        const out = {};
        for (const k of want) if (k in store) out[k] = store[k];
        return out;
      },
      set: async (obj) => Object.assign(store, obj),
    },
    now: () => clock,
  };
  return {
    transport, calls, state, store,
    advance: (ms) => { clock += ms; },
    clock: () => clock,
    typesOf: (t) => calls.filter((c) => c.type === t),
  };
}

test("refresh looks the URL up and reads playback status", async () => {
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  assert.equal(f.typesOf("cache:lookup").length, 1);
  assert.equal(s.getState().phase, "idle");
  assert.equal(s.view().kind, "not-cached");
});

test("a missing token is classified from the message sw.js actually throws", async () => {
  const f = makeFake();
  f.transport.sw = async () => {
    throw new Error("No bearer token set. Open the Patmir extension options page…");
  };
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  assert.equal(s.getState().errorKind, "no-token");
  assert.equal(s.view().action, "options");
});

test("any other transport failure reads as unreachable", async () => {
  const f = makeFake();
  f.transport.sw = async () => { throw new Error("Failed to fetch"); };
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  assert.equal(s.getState().errorKind, "unreachable");
});

test("pressing the button twice submits once", async () => {
  // The seq guard. Without it a double click on a big obvious button submits
  // two jobs; the server dedupes by cache key, but the second reply would also
  // clobber the first's state.
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  await Promise.all([s.primary(), s.primary()]);
  assert.equal(f.typesOf("jobs:submit").length, 1);
});

test("a cached quality dub plays without submitting anything", async () => {
  const f = makeFake({ lookup: { cached: true, cache_key: "kc" } });
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  await s.primary();
  assert.equal(f.typesOf("jobs:submit").length, 0);
  assert.deepEqual(f.typesOf("player:play").map((c) => c.k), ["kc"]);
  assert.equal(s.view().kind, "playing");
});

test("watch-along starts playing as soon as the job is accepted", async () => {
  // It buffers behind the playhead by design; waiting for `done` would mean
  // waiting for the whole video.
  const f = makeFake({
    submit: { job_id: "j1", cache_key: "kw", status: "prelude" },
  });
  const s = S.createSession({ transport: f.transport, mode: "watch-along" });
  await s.refresh();
  await s.primary();
  assert.equal(f.typesOf("player:playWatchAlong").length, 1);
  assert.equal(f.typesOf("player:play").length, 0);
});

test("tick respects the poll interval", async () => {
  const f = makeFake({ job: { job_id: "j1", status: "running", cache_key: "k1", stage_states: {} } });
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  await s.primary();                       // submits, sets jobId
  const before = f.typesOf("jobs:get").length;
  f.advance(500);
  await s.tick();
  assert.equal(f.typesOf("jobs:get").length, before, "too soon");
  f.advance(S.JOB_POLL_MS);
  await s.tick();
  assert.equal(f.typesOf("jobs:get").length, before + 1);
});

test("autoPlayOnDone plays when the job finishes; without it, nothing starts", async () => {
  for (const auto of [true, false]) {
    const f = makeFake({
      job: { job_id: "j1", status: "done", cache_key: "kd", stage_states: {} },
    });
    const s = S.createSession({ transport: f.transport, autoPlayOnDone: auto });
    await s.refresh();
    await s.primary();
    f.advance(S.JOB_POLL_MS + 1);
    await s.tick();
    assert.equal(f.typesOf("player:play").length, auto ? 1 : 0, `auto=${auto}`);
    // Either way the state must stop offering "not dubbed".
    assert.notEqual(s.view().kind, "not-cached");
  }
});

test("stopping a running quality job cancels it server-side", async () => {
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  await s.primary();
  await s.stop();
  assert.equal(f.typesOf("player:stop").length, 1);
  assert.equal(f.typesOf("jobs:cancel").length, 1);
});

test("stopping watch-along sends stream:stop, not jobs:cancel", async () => {
  // Watch-along work is stopped through the stream endpoint. The content
  // script sends the same message once playback has attached; before that
  // (job still in prelude) it sends nothing, so the button read Չեղարկել and
  // stopped nothing on the server. The endpoint is idempotent.
  const f = makeFake({ submit: { job_id: "j1", cache_key: "kw", status: "streaming" } });
  const s = S.createSession({ transport: f.transport, mode: "watch-along" });
  await s.refresh();
  await s.primary();
  await s.stop();
  assert.equal(f.typesOf("player:stop").length, 1);
  assert.equal(f.typesOf("jobs:cancel").length, 0);
  const stops = f.typesOf("stream:stop");
  assert.equal(stops.length, 1);
  assert.equal(stops[0].cache_key, "kw");
});

test("stopping a finished watch-along sends nothing to the server", async () => {
  const f = makeFake({ submit: { job_id: "j1", cache_key: "kw", status: "done" } });
  const s = S.createSession({ transport: f.transport, mode: "watch-along" });
  await s.refresh();
  await s.primary();
  await s.stop();
  assert.equal(f.typesOf("stream:stop").length, 0);
});

test("a reply that arrives after the URL changed is discarded", async () => {
  // A YouTube SPA navigation between request and response: the answer belongs
  // to the previous video and must not become this page's state.
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  const orig = f.transport.sw;
  f.transport.sw = async (msg) => {
    const r = await orig(msg);
    f.state.url = "https://www.youtube.com/watch?v=OTHER";
    return r;
  };
  await s.primary();
  assert.equal(s.getState().jobId, null, "the stale submit must not be adopted");
});

test("switching mode drops the old job and re-looks-up", async () => {
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  await s.primary();
  assert.ok(s.getState().jobId);
  await s.setMode("watch-along");
  assert.equal(s.getState().jobId, null);
  assert.equal(s.getState().mode, "watch-along");
  assert.equal(f.store.patmirMode, "watch-along");
  assert.equal(f.typesOf("cache:lookup").length, 2);
});

test("a submitted job is remembered so another surface can adopt it", async () => {
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  await s.primary();
  assert.deepEqual(f.store[S.ACTIVE_JOBS_KEY], { k1: "j1" });
});

test("refresh adopts a job started by the other surface", async () => {
  const f = makeFake({
    store: { [S.ACTIVE_JOBS_KEY]: { k1: "j9" } },
    job: { job_id: "j9", status: "running", cache_key: "k1", stage_states: { a: "running" } },
  });
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  assert.equal(s.getState().jobId, "j9");
  assert.equal(s.view().kind, "running");
});

test("a finished job is forgotten rather than adopted forever", async () => {
  const f = makeFake({
    store: { [S.ACTIVE_JOBS_KEY]: { k1: "j9" } },
    job: { job_id: "j9", status: "done", cache_key: "k1" },
  });
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  assert.equal(s.getState().jobId, null);
  assert.deepEqual(f.store[S.ACTIVE_JOBS_KEY], {});
});

test("a failed playback surfaces as a retryable state, not a silent no-op", async () => {
  const f = makeFake({ lookup: { cached: true, cache_key: "kc" } });
  f.transport.player.play = async () => ({ ok: false, error: "decode failed" });
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  await s.primary();
  const v = s.view();
  assert.equal(v.kind, "playback-error");
  assert.equal(v.action, "start");
});

test("setAdShowing only reports a change when it changes", async () => {
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  assert.equal(s.setAdShowing(true), true);
  assert.equal(s.setAdShowing(true), false);
  assert.equal(s.setAdShowing(false), true);
});

test("onChange fires on state transitions and can be unsubscribed", async () => {
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  let n = 0;
  const off = s.onChange(() => { n += 1; });
  await s.refresh();
  assert.ok(n > 0);
  off();
  const seen = n;
  await s.refresh();
  assert.equal(n, seen);
});

test("a throwing listener does not break the others", async () => {
  const f = makeFake();
  const s = S.createSession({ transport: f.transport });
  let ok = 0;
  s.onChange(() => { throw new Error("bad listener"); });
  s.onChange(() => { ok += 1; });
  await s.refresh();
  assert.ok(ok > 0);
});

test("NULL_PLAYER lets the popup run the same machine off-YouTube", async () => {
  // On a non-YouTube page there is no content script, so the popup drives a
  // local session with no player. It must still offer to translate — reporting
  // `hasVideo: false` there would show "waiting for the player" forever, so
  // NULL_PLAYER reports the honest answer, which is "unknown".
  const f = makeFake();
  f.transport.player = S.NULL_PLAYER;
  const s = S.createSession({ transport: f.transport });
  await s.refresh();
  assert.equal(s.getState().phase, "idle");
  assert.equal(s.getState().playback.hasVideo, null);
  assert.equal(s.view().kind, "not-cached");
});


// --- global auto-start --------------------------------------------------
//
// `patmirTranslateEnabled` is ONE remembered choice for every video. Author:
// "should remember the last state (not for a single video), if enabled it
// should work for all videos, if disabled should not translate."
//
// The old behaviour was a per-video opt-out plus an auto-play that only ever
// started an ALREADY-cached dub, so a video that had never been dubbed always
// played in the original language.

const AUTO_VIEW = (action, disabled) => ({ action, disabled: !!disabled });

test("enabled plus a startable view auto-starts", () => {
  assert.equal(S.shouldAutoStart(AUTO_VIEW("start"), { enabled: true }), true);
});

test("an undubbed video auto-starts too — that is the point", () => {
  // `not-cached` derives action "start", so enabling translation submits work
  // rather than skipping the video.
  const v = S.deriveView(base({ lookup: { cached: false, partial: false } }));
  assert.equal(v.action, "start");
  assert.equal(S.shouldAutoStart(v, { enabled: true }), true);
});

test("disabled never auto-starts", () => {
  assert.equal(S.shouldAutoStart(AUTO_VIEW("start"), { enabled: false }), false);
  assert.equal(S.shouldAutoStart(AUTO_VIEW("start"), {}), false);
});

test("a video already playing is not started again", () => {
  assert.equal(S.shouldAutoStart(AUTO_VIEW("stop"), { enabled: true }), false);
});

test("a failed job is NOT auto-retried", () => {
  // A failure that re-submits itself is a loop, and the failure is usually the
  // reason the viewer needs to see the button.
  const v = S.deriveView(base({ job: { status: "failed", job_id: "j" } }));
  assert.equal(v.action, "retry");
  assert.equal(S.shouldAutoStart(v, { enabled: true }), false);
});

test("a disabled view is never auto-pressed", () => {
  // Loading, or no video yet: pressing would fail and look broken.
  assert.equal(S.shouldAutoStart(AUTO_VIEW("start", true), { enabled: true }), false);
});

test("the tried latch stops a second press", () => {
  // The surface latches before awaiting the round trip; without it a second
  // tick inside that window would submit the same video twice.
  assert.equal(S.shouldAutoStart(AUTO_VIEW("start"),
    { enabled: true, tried: true }), false);
});

test("a missing view does not throw", () => {
  assert.equal(S.shouldAutoStart(null, { enabled: true }), false);
  assert.equal(S.shouldAutoStart(undefined, { enabled: true }), false);
});


// --- remembered choice: migration and mode default ----------------------

test("an explicit translate flag always wins", () => {
  assert.equal(S.resolveTranslateEnabled({ patmirTranslateEnabled: true }), true);
  assert.equal(S.resolveTranslateEnabled({ patmirTranslateEnabled: false }), false);
});

test("an explicit off beats a legacy on", () => {
  assert.equal(S.resolveTranslateEnabled(
    { patmirTranslateEnabled: false, patmirAutoPlayEnabled: true }), false);
});

test("a legacy auto-play preference migrates once", () => {
  // patmirAutoPlayEnabled DEFAULTED ON, so a stored true means this viewer
  // already had automatic dubbing. Reading only the new key silently reset
  // them: the author reloaded, opened a video, and got the original language.
  assert.equal(S.resolveTranslateEnabled({ patmirAutoPlayEnabled: true }), true);
  assert.equal(S.resolveTranslateEnabled({ patmirAutoPlayEnabled: false }), false);
});

test("a fresh install is off", () => {
  // On means every video you open submits GPU work, so it is one press away
  // rather than the default.
  assert.equal(S.resolveTranslateEnabled({}), false);
  assert.equal(S.resolveTranslateEnabled(null), false);
});

test("watch-along is the default mode", () => {
  // It starts playing in about half a minute rather than after a full pass
  // over the video.
  assert.equal(S.resolveMode({}), S.WATCH_ALONG);
  assert.equal(S.resolveMode(null), S.WATCH_ALONG);
});

test("an explicit quality choice is kept", () => {
  assert.equal(S.resolveMode({ patmirMode: "quality" }), S.QUALITY);
});

test("an unrecognised stored mode falls back to watch-along", () => {
  assert.equal(S.resolveMode({ patmirMode: "nonsense" }), S.WATCH_ALONG);
});


// --- watch-along progress -----------------------------------------------
//
// A stream record has NO `stage_states`: it does not walk the pipeline once,
// it dubs the video in sections. `jobProgress` returned null on the empty
// stage map, so the in-page progress bar showed nothing in watch-along —
// invisible until watch-along became the default mode and the author noticed
// the green fill had gone.

test("a stream reports progress as the fraction of the video dubbed", () => {
  // The measured record: two sections of a 677 s video.
  const p = S.jobProgress({ dubbed_s: 76.2, duration: 677.161 });
  assert.ok(p > 0.11 && p < 0.12, `got ${p}`);
});

test("a stream with nothing dubbed yet reads zero, not null", () => {
  // Zero draws an empty bar; null hides it. At the start of a stream the bar
  // should be there and empty.
  assert.equal(S.jobProgress({ dubbed_s: 0, duration: 677.161 }), 0);
});

test("dubbed seconds past the duration clamp to full", () => {
  // Sections can overshoot the reported duration by a hair (VAD snapping).
  assert.equal(S.jobProgress({ dubbed_s: 700, duration: 677.161 }), 1);
});

test("a stream without a duration reports nothing rather than guessing", () => {
  assert.equal(S.jobProgress({ dubbed_s: 76.2 }), null);
  assert.equal(S.jobProgress({ dubbed_s: 76.2, duration: 0 }), null);
});

test("quality mode still measures by stage states", () => {
  // The stage path must not change: it is the one that already worked.
  assert.equal(S.jobProgress({
    stage_states: { a: "done", b: "running" }, dubbed_s: 5, duration: 100,
  }), 0.5);
});

test("a streaming view now carries a progress fraction", () => {
  const v = S.deriveView(base({
    phase: "job",
    job: { status: "streaming", job_id: "j", dubbed_s: 76.2, duration: 677.161,
           sections_done: 2 },
  }));
  assert.equal(v.kind, "streaming");
  assert.equal(typeof v.progress, "number");
  assert.ok(v.progress > 0);
});
