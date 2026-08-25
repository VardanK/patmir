# 2026-08-25 — Watch-along: why the first chunk was slow, and why it froze

**Status:** accepted, implemented (Tasks 1-7). Buffer policy deferred pending
re-measurement.

## Context

The author reported two things about dubbing a 1297 s YouTube video in
watch-along mode on the WSL2 + RTX 3090 box: the first chunk takes too long, and
playback periodically freezes. They asked for benchmarking to find the slowest
sections.

The benchmarking request turned out to be the first finding. **The live stream
path recorded no timing at all** — `sections.json` held `{index, start, end,
file}` and `StreamRecord.to_public()` exposed `status`/`sections_done`/
`window_end`/`duration`. Every per-section number in `performance.md` came from
`bench` re-reading `runs/` afterwards. And `bench --mode watch-along` bypassed
`StreamManager` and `StreamScheduler` entirely, walking sections with its own
forward `while` loop — so coverage, lookahead, seek priority, idling and the
abandon timer were absent from the benchmark meant to characterise them. It also
took a local file only: **the reported video could not be benchmarked as
submitted.**

So the session was reconstructed by hand from artifacts still on disk: the cache
manifest at `~/.patmir/cache/d5af5653d745ed66/` and ten job dirs under
`/home/vardan/patmir/runs/20260825-182*`. Preserved in
`agent-os/specs/2026-08-25-1931-watchalong-startup-and-stalls/measurements/`.

### What the session showed

**Time to first audio: 72.4 s** for a 40 s opening window (RTF 1.81), on top of
a blocking full download that reported no progress. Steady-state sections ran
22.0-38.3 s of wall for ~30 s windows — mean RTF ~0.87, with 0.4 s of idle
between them, i.e. strictly serial with no lookahead. A buffer growing at 13% of
playback cannot absorb one bad section.

The prelude also ran **three times** on the identical 0-40 s window (18:21:43,
18:24:05 discarded, 18:24:13), which absolute stage timestamps confirm were
independent full analyses rather than one resumed into the other.

## Findings and decisions

### A1 — the section-end snap cut mid-speech, and collapsed the section

`_snap_window_end` took `max(end <= target)` over the VAD intervals. One
section's intervals were `[[0.0, 6.3], [6.3, 8.1], [8.1, 38.0]]`, so it cut at
**8.1** — analyzing 38 s to emit 8.1 s (RTF 4.0) and discarding 8 of its 10 ASR
sentences. Two of the session's seven sections were truncated this way.

The intervals look contiguous because the separate stage runs Silero with
`speech_pad_ms=500`: every speech run is padded 500 ms per side, so runs
separated by less than ~1 s of silence come back **touching**. An interval end
that equals the next interval's start is therefore *mid-speech*, and the old
rule could not tell it from a real boundary.

**Decision: two rules.** A candidate must be followed by a real gap
(`_MIN_SILENCE_GAP_S = 0.05`), and must not fall below `_MIN_SNAP_FRACTION`
(0.5) of the target — because even a genuine early silence is the wrong cut when
the only later boundary sits past the target. When nothing qualifies, cut at the
target. That can land mid-utterance; the transcript trim drops the partial
sentence and the next section re-analyzes from the boundary. On continuous
speech there is no silence to find, and a rough cut beats emitting a fifth of a
section.

### A3 — a pause longer than 45 s killed the stream, permanently

`_default_idle` ended the run when the *playhead* had not updated within
`abandon_seconds`. The browser client stops POSTing the playhead while paused,
so a pause read as abandonment. Worse, `submit()` then returned the `done`
record, so the viewer was frozen at the buffer edge with no recovery short of a
server restart. `Playhead.paused` was set but never read, contradicting the
scheduler's own docstring.

**Decision: abandonment keys off the client still asking for something**
(`alive_fn`, fed by `/sections` polling and playhead POSTs), not off playhead
movement — that is the only available signal that separates "paused" from
"left". And `submit()` restarts a `done` stream whose cached manifest is not
`complete`. `alive_fn=None` keeps the old behaviour for bench and tests.

### C1 — a restart overwrote cached audio and corrupted the manifest

A resumed stream re-seeded `Coverage` from `sections.json` but restarted its
section index at 0 (`dub_prelude_window` always returns index 0).
`append_section` names the file `section_{index:03d}.wav` while deduping entries
by **start**, so it overwrote `section_001.wav` and kept both entries. The
author's cache ended with indices 1 and 2 each appearing twice with different
windows — **the wrong audio at 154.9 s and 190.3 s, live and on every rewatch.**

**Decision: make the shape unrepresentable.** `_index_seed` numbers past the
highest cached index, and `append_section` drops any surviving entry that reuses
the index it is about to write — whose file has just been overwritten anyway.
That also repairs an already-damaged manifest on the next append, so no
migration is needed.

### B1 — the TTS batch tail

A chunk costs `max(steps)`, and chunks were split greedily at `max_batch=8`. The
prelude's 9 sentences became `[8, 1]`: 24.3 s + **17.3 s**, the trailing chunk
being the single longest line paying for a whole chunk — 42% of the TTS stage
for one sentence.

**Decision: balance the chunks** (`_balanced_chunks`): the same number of chunks
and the same memory ceiling, but boundaries move earlier, so 9 items become
`[5, 4]`. Never worse than greedy, and identical when `max_batch` divides
evenly, so the common full-batch case cannot move. `max_batch` itself was left
alone — it is a VRAM bound on a card already peaking at 16.4 of 24 GB, and
raising it is a sweep that needs the GPU, not a code change.

**Caveat:** batched decode is not bitwise-invariant across shapes (documented in
ADR `2026-08-24-tts-sampling-reproducibility`), so changing chunk composition
can redraw unrelated lines. Per-row seeding is by content hash, not slot, so
this does not reintroduce the slot-dependence that ADR fixed.

### B2/B3 — the models nobody prewarms

The first prelude showed a **14.3 s gap between ASR finishing and translate
starting** that no stage owned, no progress event covered and no manifest
recorded. It did not recur on the second prelude, identifying it as a one-time
lazy load. `prewarm_stage_cache` walks `cfg.entries`, so it only ever builds the
nine STAGES — the speaker embedder and the age/gender profiler are neither.

Measured directly: **18.0 s total, of which the age/gender profiler is 15.2 s
and ECAPA only 2.9 s**; a second call costs 0.000 s. The cost is overwhelmingly
the profiler, not the embedder.

Separately, the stage cache is keyed `(stage, candidate_id)` and
`streaming_config()` overrides `separate` and `translate`, so prewarming with
the batch config left both as misses and the first stream of every server
lifetime built a Demucs model inside its prelude.

**Decision:** prewarm the streaming stage set and both non-stage models at
startup, best-effort, never fatal.

### B4 — per-section language re-detection

Qwen3-ASR probes for the language when `ctx.source_lang` is unset — 3.5-4.6 s on
**every** section (12-15% of section wall) to re-answer a question the prelude
had already answered about the same soundtrack.

**Decision:** `PreludeResult` carries `source_lang`; sections pass it into
`run_job`, which short-circuits the probe.

### Telemetry

**Decision:** one shared reader (`streaming/timing.py:section_timing`) used by
both the live path and bench, so they cannot report different numbers for the
same run. Each section's timing is persisted into `sections.json` — additively,
with `index`/`start`/`end` always taken from the caller so audio and timing
cannot disagree. The new `other_s` field is `wall_s - sum(stage_wall_s)`: the
work no stage owns, measured at ~1.6 s per section and 15.5 s on a cold
prelude, and previously invisible to every per-stage view.

`bench --mode watch-along` now drives the real `StreamScheduler` with a
synthetic viewer (`--playhead greedy|realtime`, the latter counting actual
stalls), accepts `--url`, emits the tts/translate traces it previously
hard-coded to `None`, and reports `time_to_first_audio_s`.

## Not done

**Buffer policy (A2/A5) is deliberately deferred.** A minimum pre-buffer, and
overlapping section N+1's analysis with section N's dub, both change the
arithmetic that Tasks 5-7 have just altered — and the accelerator lock is held
for a whole stream, so any concurrency idea has to be checked against it rather
than assumed available. Picking a pre-buffer size before re-measuring would be a
guess. The ECAPA embedding list also still grows without bound
(`registry.resolve` scans all of it per section), which makes per-section cost
drift upward on a long video.

**The 4-speaker cap (quality issue #18) is untouched** and unrelated.

## Consequences

Nothing here has been measured end to end on the 3090 yet — the fixes are
verified by 75 new unit tests pinned to the measured failures, plus a direct
timing of the prewarm. The before/after baseline is the first thing to capture,
using the bench that now measures the real path.

*Revisit:* a clip where the snap's 0.5 target fraction cuts a genuinely good
boundary; an ear-test finding the mid-utterance fallback cut worse than a short
section; a card large enough to make `max_batch` worth sweeping; or watch-along
gaining a second worker, which would make the whole-stream accelerator lock the
binding constraint.
