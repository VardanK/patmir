# 2026-08-25 — One accelerator at a time: a lock shared by both job managers

**Status:** accepted, implemented.

## Context

A watch-along stream and a quality-mode dub submitted ~27 s apart both ran, and
the RTX 3090 ran out of memory. Three OOMs landed within 230 ms, from two
independent code paths:

```
JobManager._worker  → _run_one    → emotion2vec softmax    CUDA driver error: out of memory
StreamManager._run  → _run_blocking → Higgs qwen3 attention  CUDA error: out of memory
StreamManager._run  → run_prelude  → htdemucs .to(device)   CUDA error: out of memory
```

The third is the diagnostic one: `wav.unsqueeze(0).to(self.device)` moves a
handful of MB. Once the card is full, allocation size stops mattering, and every
later submission fails until the process restarts.

**This is not this box's documented false-OOM.** That one is the Windows commit
limit (`docs/setup/windows-wsl2-cuda.md`) and presents as a tiny request failing
with gigabytes free. Measured here at the time of failure:

| | |
| --- | --- |
| Commit limit | 51.84 GB (31.84 RAM + 20 pagefile) |
| Commit used | 43.34 GB → **8.5 GB headroom** |
| GPU | **24,105 MiB used of 24,576 — 222 MiB free** |

Commit was not binding and the pagefile was the already-corrected 20 GB. The
card was genuinely full, confirmed independently by the author in `nvtop`.

### Why it fit before and not now

`app.create_app` gives `JobManager` and `StreamManager` **one shared
`stage_cache`** — correct, and the reason a warm submission reports
`build_s == 0.0`. But each manager had its **own** lock: `JobManager` a queue
plus a single worker, `StreamManager` an `asyncio.Lock` used only for record
bookkeeping. Each serialized its own work. **Nothing serialized them against
each other.**

Resident weights are already ~15–16 GB — tts ~8.5, asr + aligner ~5.5,
diarize 0.9, plus emotion2vec and **two** Demucs models, because batch selects
`htdemucs_vocals` while watch-along selects `htdemucs_fast`. That matches the
16.42 GB peak in `performance.md` and fits one pipeline comfortably. Two
concurrent pipelines add a second set of activations on top, and 24 GB is gone.

The constraint was already known in one place and not generalized: TTS was
unified onto a single candidate id on 2026-08-23 precisely because "one 4B
instance serves both modes ... required on a 24 GB GPU". That reasoning applies
to the *pipelines*, not just the weights.

## Decision

One `asyncio.Lock` constructed in `create_app` and passed to **both** managers,
held around each manager's run body — `JobManager._worker` around
`to_thread(_run_one)`, `StreamManager._run` around the new `_run_locked`.

**Deliberately coarse: held for a whole run.** A batch job submitted during a
watch-along waits until the stream ends, which for a feature-length video is a
long time. That is the honest semantics when only one pipeline fits in VRAM —
the alternative is losing both to an OOM and wedging the server until restart.

A blocked job reports a new `waiting` status (`JobStatus` gained it;
`StreamStatus` is `str`), set only when the lock is actually contended, so the
uncontended path keeps its single `queued → running` transition. `popup.js`
renders it in both the quality and watch-along views, because its fallback would
otherwise have shown a waiting batch job as "Running… (0/0)".

`accelerator_lock=None` is the default and preserves the unserialized behavior,
which is what the existing tests construct.

## Consequences

- Two submissions can no longer OOM each other. The second waits.
- A batch job can wait a long time behind a stream, and it now says so rather
  than looking hung.
- **The lock is held across the yt-dlp download** inside
  `StreamManager._run_blocking`, so a URL submission blocks batch work while
  doing pure network I/O. Wasteful, not incorrect — the download is upstream of
  the accelerator work in the same function, and splitting it out is a bigger
  change than this fix warranted. Worth doing if it bites.
- Nothing frees VRAM without a process restart: `_free_accelerator_cache()` runs
  only at prewarm (`runner.py:88`) and on an actual stage build
  (`runner.py:226`). That gate is deliberate — reverting it cost a 1.710 → 3.018
  watch-along RTF regression (ADR `2026-08-23-warm-model-residency-and-prewarm`)
  — so a wedged server still needs a restart, and this decision does not change
  that.

## Alternatives rejected

- **Per-section locking** (acquire around each unit of GPU work rather than the
  whole run). Finer-grained and would let a batch job interleave between
  sections, but it needs the lock threaded through `streaming/section.py` and the
  scheduler, and a batch job grabbing it mid-stream would stall production for
  ~139 s on a 5-minute clip and drain the viewer's buffer. Worse for the mode
  that has a realtime deadline.
- **A `threading.Lock` acquired inside the sync bodies.** Would allow placing it
  precisely around GPU work in synchronous code, but it blocks an executor
  thread instead of suspending a coroutine, and it does not match the existing
  injection idiom (`stage_cache` is app-owned and passed to both managers). The
  `asyncio.Lock` mirrors that.
- **Rejecting the second submission** with HTTP 409 instead of queueing it.
  Simpler, but it throws away work the user asked for, and the extension would
  need to distinguish "busy" from "failed".
- **Making the two modes share one `separate` candidate** to cut the duplicate
  Demucs. Saves only ~0.16 GB — the two Demucs models are not the problem, the
  second activation set is — and it would cost watch-along the faster base model.

## Verification

`patmir/tests/test_accelerator_lock.py`, 4 tests, including a **negative
control** (`test_without_the_lock_they_do_overlap`) that asserts peak occupancy
of 2 with no lock. Without it a green suite would not prove the serialization
came from the lock rather than from an incidentally single-threaded executor.
Full suite 399 passed.

*Revisit:* a second GPU or a card big enough for two pipelines; the download
holding the lock becoming a real annoyance; or job cancellation landing, which
changes what a queued-and-waiting job should do when the user gives up on it.
