# 2026-08-25 — Cooperative job cancellation

**Status:** accepted, implemented.

## Context

There was no way to stop a batch job at all. Watch-along had
`POST /stream/<key>/stop`, but it only set the scheduler's stop event, so **no
further** sections were scheduled while the one in flight ran to completion
(~35 s). And `stream.py:_run` caught `asyncio.CancelledError` and recorded
`status="failed", error="cancelled"`, so a deliberate stop rendered as a red
failure.

The shared accelerator lock (ADR `2026-08-25-one-accelerator-at-a-time`) made
this urgent rather than merely missing: a job submitted during a watch-along now
sits in `waiting`, possibly for a feature-length video, with no way to give up
on it.

## The constraint that decides the design

**A Python thread cannot be killed.** Every heavy stage runs through
`asyncio.to_thread`, so `task.cancel()` only abandons the *await* — the worker
thread keeps running, on the GPU, **still holding the accelerator lock**. A
"cancelled" job would therefore block the next one indefinitely, which is worse
than not offering cancellation. So:

> Cancellation is cooperative, and is never implemented with `task.cancel()`.

This composes correctly with the lock for free: `async with` releases only when
the coroutine exits, which happens when the thread actually returns. The next job
cannot start before the cancelled one has really stopped.

## Decision

A `threading.Event` per job, held in a side dict on each manager
(`_cancels: dict[job_id, Event]`) and passed into `run_job(cancel=…)` onto
`JobContext.cancel`, with `ctx.cancelled()` / `ctx.raise_if_cancelled()` helpers
and a distinct `JobCancelled` exception.

The token lives on `JobContext` as its own field rather than in `ctx.extra`:
`extra` is the cross-stage *data* channel (`vad_intervals`, `diarization`,
`emotion_predictions`) and a control signal is not that. It lives in a side dict
rather than on `JobRecord` because `_persist_index` does `asdict(r)`, which cannot
serialize an `Event`.

### Two cases, deliberately different

| | Mechanism | Latency |
| --- | --- | --- |
| `queued` / `waiting` | mark `cancelled`; the worker skips it | immediate and exact — **never touches the accelerator** |
| `running` | set the token; abort at the next checkpoint | `cancelling` → `cancelled`, bounded below |

### Checkpoints

| Where | Worst case |
| --- | --- |
| Each stage boundary (`runner.py`, before the stage builds) | the longest stage — tts, ~77 s on CUDA |
| Each TTS decode chunk (the adapter's `on_chunk`) | **~8 s** (~7.7 sentences/chunk) |
| Each sentence (the adapter's serial path) | per sentence |

Stage boundaries alone would make Stop take over a minute, because TTS is ~55 %
of batch wall. The TTS hook costs nothing extra: the candidate **already** calls
`on_chunk` after every chunk for progress and timing, so raising from it aborts
the loop with **no change to the candidate**. Pinned by
`ChunkLoopCancellationTest` — including that an exception in that callback really
does propagate rather than being swallowed.

### Partial work is kept

A cancelled run leaves every completed stage's artifacts on disk and marks the
stage `cancelled` (not `failed`, and with no traceback). `--resume-from` already
reads absolute paths out of the sidecar JSONs, so a cancelled job is resumable for
free. For watch-along it isn't even a question — sections already written to the
cache are the free-rewatch feature working as designed.

### API

`POST /jobs/{job_id}/cancel`, one endpoint for both modes because `job_id` already
disambiguates (stream ids are `stream-`-prefixed) and the lookup mirrors
`jobs_get`. **Idempotent:** cancelling a finished, failed or already-cancelled job
is a 200 no-op, so a double-clicked Stop never errors; only an unknown `job_id` is
a 404. `POST /stream/<key>/stop` stays as the extension's entry point, and now
*also* trips the token so the in-flight section stops too.

New statuses `cancelling` and `cancelled` on both managers, rendered in
`popup.js`.

## Consequences

- A waiting job can be abandoned instantly, which is what the accelerator lock
  made necessary.
- Stop on a running job takes up to ~8 s inside TTS, longer inside a non-TTS
  stage (asr ~20 s is the next largest). Not instant, and cannot be made instant
  without checks in the AR decode's inner loop — rejected below.
- A deliberate stop no longer looks like a crash.
- **`_process` in `stream.py` had to stop swallowing it.** Its
  `except Exception` would have caught `JobCancelled`, logged "section failed",
  returned `None` and let the stream carry on — silently downgrading Stop to "one
  section didn't work". It now re-raises `JobCancelled` before the generic handler.
- The concurrent (`ThreadPoolExecutor`) TTS path has no per-sentence checkpoint:
  its futures are already submitted. It is unused by the Higgs candidates, which
  batch instead, and the stage boundary still covers it.
- Cancelling during a yt-dlp download does not stop the download; the subprocess
  is killable in principle (unlike a thread) but that is not wired up.

## Alternatives rejected

- **`task.cancel()`** — the obvious implementation, and wrong: the thread
  survives and keeps the accelerator lock. This is the whole reason the design is
  what it is.
- **Checks in the AR decode step loop** (sub-second). A branch in the hottest
  inner loop of a memory-bandwidth-bound decode, on the stage that dominates wall
  time, to improve Stop from ~8 s to instant. Not worth measuring against.
- **Stage boundaries only** — one line, but ~77 s to respond, which reads as
  broken.
- **Deleting the job dir on cancel.** Cleaner state, but throws away expensive GPU
  work and contradicts how watch-along caching already behaves.
- **A separate `/streams/{key}/cancel` endpoint.** Two endpoints for one concept;
  `job_id` already routes correctly.
- **Putting the token in `ctx.extra`.** Free (no signature changes) but conflates
  the data hand-off channel with control flow, and `extra` is documented in
  `pipeline.md` as the former.

## Verification

`patmir/tests/test_cancellation.py` (11 tests) and
`test_higgs_torch_batch.ChunkLoopCancellationTest` (3). Full suite **413 passed**.

The load-bearing one is
`test_cancelled_run_releases_the_lock_for_the_next_job`: it asserts a second job
runs to completion after the first is cancelled, and that the lock is not left
held. That is exactly the property a `task.cancel()` implementation would violate
while still passing every other test here.

*Revisit:* Stop feeling slow in practice (the next lever is ASR's batched loop,
not the decode inner loop); cancelling a download needing to kill yt-dlp; or a
resume-a-cancelled-job flow getting a UI, which would need the manifest's
`cancelled` stage state to be surfaced.
