# The server that got slower the longer it ran: one CUDA graph pool per chunk

Date: 2026-09-07. Status: accepted — `cuda_graphs = false` on the cuda profile,
verified over a 10-run soak. Quality issue #34.

> **Three other fixes were attempted first and all three failed**, the last
> making the problem worse. They are kept in full below: each was a plausible
> reading of the evidence, each cost a round trip, and together they are the
> reason the fix that shipped is the blunt one.

## The report

Author, on a live watch-along: *"again the only client is stuck on waiting for
gpu."* Then, when a restart was offered: *"you fixed this multiple times and
yet it still returns, I need a fix, not a simple server restart. Cancelled
jobs seems to hang up and accumulate. Fix the root cause and not the
symptom."*

The recurrence was the finding. Two earlier ADRs
(`2026-08-25-cooperative-job-cancellation`,
`2026-09-04-download-inside-the-lock-and-waiting-cancel`) had each fixed a real
bug presenting as "stuck waiting for GPU", and each verification ended with the
same sentence: the running server has to be restarted once to clear it. Neither
was this bug. This one is underneath all of them.

## What the client was actually waiting for

Nothing was deadlocked, and the extension was telling the truth. A batch job
held the single accelerator lock and two records waited correctly behind it —
the reported watch-along stream and a second quality job. The lock, the
cancel path and the queue were all working as designed.

The job holding the lock was in `tts` and would not finish. Measured on the
running process:

| | |
| --- | --- |
| progress | 24 of 371 sentences in 36 minutes |
| rate | 78.3 s per sentence; **ETA 7.5 hours** |
| per chunk | 923.8 s, then 954.7 s, for 12 sentences each |
| WAVs on disk | 0 |

And it was not specific to that job. Reading the `tts` stage wall out of every
`manifest.json` on the box gives a clean break with no gradient:

```
09-05 08:22:18  section     11.5s  done       <- healthy
09-05 08:23:03  section   1949.4s  cancelled  <- and never recovers
09-07 14:59:41  section    924.7s  done
09-07 15:15:15  section   1893.3s  done        (~2x)
09-07 16:02:57  section  15321.2s  done        (~16x)
```

Everything after the break is **~950 s per 12-sentence chunk or an exact
multiple of it**; everything before is 3-13 s. One watch-along *section* — 36 s
of audio — spent **4.25 hours** in TTS, and the batch job took the lock nine
seconds after it finally returned.

### The author's hypothesis was reasonable and wrong, and measuring it mattered

*"Cancelled jobs seem to hang up and accumulate"* fits the evidence as stated:
the slow runs are disproportionately `cancelled`, and N abandoned decode
threads would explain an N-times slowdown while the GIL kept only one of them
runnable at a time. It was measured and rejected — per-thread CPU deltas over
8 s across all 40 threads of the process:

```
tid 17433   +684 ticks (86% of a core)  state=R
tid 17418   +  1 tick                   state=S   (the asyncio loop)
```

**Exactly one thread executes Python.** Nothing accumulated. The `cancelled`
correlation is the reverse of the causal direction it suggests: the cancel
checkpoint fires per decode chunk, so once a chunk costs ~950 s a cancel takes
up to ~950 s to be noticed — the slow chunks are why those runs read
`cancelled` at that duration, not the other way round.

## Cause

The decode step is captured as a CUDA graph, and `higgs_v3_torch.py:881` took
a fresh one per chunk:

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):        # <- no pool=
```

A capture with no `pool=` gets its **own private memory pool**, holding
everything allocated during capture — for this model a full 36-layer
`StaticCache` sized `B x (seq + max_cap)`. The class docstring stated the
cadence plainly ("Capture is per chunk") and the perf ADR of 2026-09-02 had
already noticed the cost and filed it as an optimisation left on the table:
*"Capture costs ~0.6 s/chunk, which reusing a capture across same-shape chunks
would mostly recover."* It was priced as wasted milliseconds. It was a leak.

Each private pool is a fresh **GPU virtual address reservation**, and on WSL2
that address space — not VRAM — is what a long-lived process exhausts first.
The kernel says so directly:

```
misc dxg: dxgk: dxgkio_reserve_gpu_va: Ioctl failed: -75      (EOVERFLOW)
```

**3,661 of them in under one second**, enough to flush every other message out
of the ring buffer, arriving continuously while the decode thread pegged one
core and the GPU sat at 2-6 % and 30 W.

That also explains an OOM in the same log that had never made sense:

> Tried to allocate **120.00 MiB**. GPU 0 has a total capacity of 24.00 GiB of
> which **6.62 GiB is free**. [...] 15.83 GiB is allocated by PyTorch

Six gigabytes free and a 120 MiB allocation fails. It is not memory that runs
out, it is address space, and once reservation fails CUDA falls back to paths
that run on the host: ~1500 ms/step against the 17.33 ms in `performance.md`,
an **86x** regression with the card essentially idle.

The attribution is not circumstantial. The failures stopped at **exactly zero**
the moment the server process exited, and resumed only under load from the
next one — three server lifetimes, each healthy for its first runs and
permanently degraded afterwards.

## This is a known, documented failure mode

Found only after three failed attempts, and it should have been the FIRST step.
NVIDIA's own CUDA-Graphs-for-PyTorch guide states both halves:

> "By default, PyTorch allocates a separate private memory pool for each CUDA
> graph to ensure memory addresses remain constant across replays."

> **"A graph's private pool is only eligible for `empty_cache()` after the
> graph is destroyed."**

Which is why `_free_accelerator_cache()` — `gc.collect()` plus
`torch.cuda.empty_cache()`, running in a `finally` after **every** stage, ~9x
per job — reclaimed none of it. PyTorch's **CUDA Graph Trees** exist precisely
because of this "known memory bloat when using CUDA Graphs".

Three further points from the same sources, each of which contradicts something
attempted below:

- **Sharing a pool carries a correctness constraint nobody here knew about:**
  *"when graphs share a memory pool, they must be replayed in the same order as
  captured. Out-of-order replay can cause memory corruption."* This code
  captures then replays immediately, so it is in-order and safe — by luck, not
  by design.
- **`expandable_segments` reserves *"up to 1 1/8x total GPU memory"* in virtual
  address space.** On WSL2, where address space is the scarce resource, that is
  a large reservation and a plausible aggravator rather than a mitigation.
- **The recommended remedy for varying shapes is "reducing the number of
  graphs and expanding capture range"** — i.e. drive captures toward ONE, not
  toward "fewer".

A matching WSL2 report (microsoft/WSL#40401) records that GPU passthrough
carries CUDA driver overhead **invisible to `torch.cuda.mem_get_info()`**, so
PyTorch plans against wrong free-memory figures. That is exactly the
"120 MiB fails with 6.62 GiB free" line above.

Sources: `docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/memory-issues.html`,
`docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html`,
`docs.pytorch.org/docs/main/generated/torch.cuda.graphs.CUDAGraph.html`,
`dev-discuss.pytorch.org/t/understanding-cudagraph-trees/1967`,
`github.com/microsoft/WSL/issues/40401`.

## Attempt 1 — one shared graph pool (FAILED)

`torch.cuda.graph_pool_handle()` created once and passed as `pool=` to every
capture, so captures share one arena instead of taking one reservation each.
Output unchanged.

**Result: degraded at run 4 of a 6-run soak.** The job never finished within
the 16-minute poll window, `dxgkio_reserve_gpu_va` failures resumed, GPU back
to 2-6 %.

```
run 1 done   tts= 27.92s  slow=0  vram=18025MiB  new_va_errors=no
run 2 done   tts= 23.31s  slow=0  vram=18163MiB  new_va_errors=no
run 3 done   tts= 22.85s  slow=0  vram=18981MiB  new_va_errors=no
run 4 DID NOT FINISH in 16 minutes  vram=19090MiB  new_va=YES
```

VRAM climbed monotonically — **+3.7 GB over the 15403 MiB prewarm baseline in
four jobs**, never recovered. Why it could not work, in hindsight: the pool is
shared but it still GROWS, because each new geometry allocates into it and a
graph pool is never returned to the general allocator. And retaining
`self._graph_holder` for reuse keeps one graph permanently alive, which by the
doc above makes its pool permanently unreclaimable — the change made that half
strictly worse.

## Attempt 2 — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (FAILED, REVERTED)

Shipped in the same change as attempt 1 with the note that it "needs measuring
rather than assuming", and then not measured — the error this ADR most wants to
record. The A/B: with it ON the soak degraded at run 3, with it OFF at run 4.
It is not the deciding factor, and per the NVIDIA guide its 1 1/8x address-space
reservation is a poor idea on precisely this platform. **Reverted.**

## Attempt 3 — bucketed "fixed" geometry (FAILED, AND MADE IT WORSE)

Round `(B, seq, max_cap)` onto a coarse lattice (`/4`, `/128`, `/256`) so
chunks share shapes and captures recur across jobs. Padding rows are copies of
a real row, never zeros — an all-zero mask makes softmax return NaN and would
poison the real rows' logits.

**Result: degraded at run 2 — sooner than either previous attempt.**

```
run 1 done   tts= 28.52s  cap=8 reuse=2  vram=19014MiB  new_va_errors=no
run 2 STALLED — tts start 22:09:17, no chunk by 22:12:29 (normal: 3-5 s)
```

Two reasons it backfired. The bucketing was too fine to matter — a real job's
8 distinct geometries became **5**, not 1, and reuse fired twice out of ten
chunks. And padding *raised the memory floor*: 19014 MiB after run 1 against
18025 for attempt 1 and 18069 for the unmodified code, roughly 1 GB less
headroom, so the same accumulation hit the ceiling a run or two earlier. Capture
count was traded for memory pressure at a bad exchange rate.

## What is actually kept from this work

Nothing that claims to fix the leak. What stands on its own merits:

- **The diagnosis**, which is solid and is what the next attempt builds on.
- **The watchdog**: `graph_captures`, `graph_reuses`, `slow_chunks` and
  `slowest_ms_per_step` in `tts/voice_cloning_summary.json`, plus a one-per-job
  stderr diagnosis above `_SLOW_STEP_MS` (200 ms/step, 4x the slowest healthy
  renderer). This state previously existed only in `dmesg`.
- **`reset_summary` covering the new counters** — missed on the first pass,
  which is the exact mistake that function's own docstring already recorded
  about `onset_bursts`; now pinned by `GraphCounterResetTest` over the whole
  counter set rather than a hand-listed few.
- **`GraphReuseEquivalenceTest`**, CUDA-gated, proving a reused capture is
  bitwise identical to a fresh one. Reuse is correct; it is simply not
  load-bearing.

## The fix: `cuda_graphs = false` (verified)

Ten consecutive runs of the reference clip through the live server, cuda
profile, nothing else touching the card:

```
run  1 done  tts= 60.46s aggRTF=0.349  vram=16494MiB  new_va=no
run  2 done  tts= 51.73s aggRTF=0.318  vram=16494MiB  new_va=no
run  3 done  tts= 52.57s aggRTF=0.321  vram=16494MiB  new_va=no
run  4 done  tts= 51.62s aggRTF=0.317  vram=16457MiB  new_va=no   <- shared pool died here
run  5 done  tts= 55.28s aggRTF=0.332  vram=16448MiB  new_va=no
run  6 done  tts= 51.38s aggRTF=0.314  vram=16451MiB  new_va=no
run  7 done  tts= 65.24s aggRTF=0.364  vram=16449MiB  new_va=no
run  8 done  tts= 53.68s aggRTF=0.324  vram=16448MiB  new_va=no
run  9 done  tts= 61.47s aggRTF=0.353  vram=16451MiB  new_va=no
run 10 done  tts= 48.96s aggRTF=0.309  vram=16451MiB  new_va=no
```

**10/10 finished, 0 `reserve_gpu_va` failures, VRAM drift 46 MiB** against
+3.7 GB for the shared pool. `cap=0` throughout — no captures are taken, so the
mechanism is absent rather than mitigated. `ms/step` held 58.7-61.7 with no
trend across the whole soak, which is the flat line that says nothing is
degrading. The last VA failure on the box was 22:12:59, from attempt 3;
nothing in the 50 minutes and 10 runs after.

This also **closes the diagnosis**: graph private pools were the cause, and not
cancellation, thread accumulation or VRAM exhaustion.

| | with graphs (baseline) | `cuda_graphs = false` |
| --- | --- | --- |
| tts wall | 23.84 s | **55.24 s mean** (48.96-65.24) |
| agg RTF | 0.2135 | **0.330 mean** (0.309-0.364) |
| decode step | 17.33 ms | **~60 ms** |
| survives a 10-run soak | no (died at run 2-4) | **yes** |

**The price is +55% on aggregate RTF** and it is accepted deliberately: a
pipeline at 0.33 is still comfortably faster than realtime, so watch-along
keeps its margin, and a server that stays correct for a whole evening beats one
that is 55% quicker until it wedges. `cuda_graphs` is in
`ui.cache._NON_OUTPUT_KEYS`, so the cache key is unchanged (`9f030fff` before
and after) and no cached dub regenerates.

`fixed_geometry` remains in the code, defaulted **off** and absent from
`patmir.toml`, so it contributes nothing to the cache key. It is kept only as
groundwork for the recovery path below.

## Open, and worth naming

1. **A contradiction in our own docs about `cuda_graphs`.** It sits in
   `_NON_OUTPUT_KEYS` on the 2026-09-02 claim that graph replay is bitwise
   identical to eager stepping, while `_capture_decode_graph`'s own docstring —
   written later — says they attend over different widths and so give "same
   distribution, different draw". Both cannot be true. If the docstring is
   right, dubs rendered from now on differ slightly from cached ones **under
   the same key**, which is the "one key holds either rendering" hazard the
   `max_batch` comment warns about. Settling it costs one clip rendered both
   ways and an md5 comparison.
2. **The 55% is recoverable.** ONE process-wide geometry — `B = max_batch`,
   `cap = 1024`, a single seq bucket — means a single capture serves every
   chunk of every job, so the pool stops growing after the first. That is what
   "reduce the number of graphs" actually asks for, and what attempt 3
   approximated far too loosely.

## The two candidates as they stood before the soak

1. **`cuda_graphs = false`.** *(Taken — see above.)* Removes the mechanism entirely: no captures, no
   pools, no churn. Costs the decode step 17.33 -> 45.55 ms, so tts ~24 -> ~62 s
   on the reference clip (agg RTF ~0.235 -> ~0.33). Also the decisive
   experiment: if this soaks flat, graphs are conclusively the cause.
2. **ONE geometry for the process** — `B = max_batch`, `cap = 1024`, a single
   seq bucket — so a single capture serves every chunk of every job and the
   pool stops growing after the first. This is what NVIDIA's "reduce the number
   of graphs" actually asks for, and what attempt 3 approximated far too
   loosely. Costs padding on every chunk; needs measuring against option 1.

*Revisit:* whichever option is taken must be proven by a soak that runs
**longer than the 3-4 jobs at which every attempt so far has failed**, since a
healthy first run has repeatedly meant nothing.
