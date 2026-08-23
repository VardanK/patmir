# 2026-08-23 — Warm-model residency across jobs + startup prewarm

**Status:** accepted. Implements the residency half of the 2026-08-23 perf work.

## Context

`run_job(stage_cache=)` landed on 2026-07-14 to kill the per-section model-rebuild
tax in watch-along, and it worked — the Reacher re-baseline measures the rebuild
tax at **2.1% of wall**, down from 11–13%. But the mechanism was barely wired in:

| path | `stage_cache` | lifetime |
| --- | --- | --- |
| CLI `dub` (`cli.py`) | none | process exits after one job |
| **UI batch jobs** (`jobs.py`) | **none** | every `/jobs` submission rebuilt all 9 stages |
| UI streaming (`stream.py`) | yes | **local in `_run_blocking`** — died with the stream |
| bench batch (`bench.py`) | none | `--repeat` "warm" was only OS/page-cache warm |

The local server is a single uvicorn process using in-process `asyncio.to_thread`
workers, so nothing about the process model forced reloads — the cache simply was
not hoisted. There was also no prewarm anywhere: the FastAPI startup hook only
started the job worker, so the first submission after boot paid full model load
*plus*, on MLX, the one-time ~240 s Metal-kernel JIT.

Separately, `_free_accelerator_cache()` (`gc.collect()` + `mps.empty_cache()` +
`mx.clear_cache()`) ran in the per-stage `finally` **even on a cache hit** — nine
times per section, discarding the allocator pool the next stage immediately
re-requested.

## Decision

1. **One process-wide warm cache**, created in `_create_app` and shared by
   `JobManager` and `StreamManager`. Models now survive across submissions and
   across streams, not just within one stream.
2. **Keyed `(stage, candidate_id)`, not stage name.** `streaming_config()`
   overrides some stages, so batch and watch-along can select different
   candidates for the same stage; under a name-only key one mode would silently
   receive the other's model. This is a correctness requirement, not a nicety.
3. **`prewarm_stage_cache()` on the FastAPI startup hook**, off the event loop so
   the server serves immediately. Best-effort per stage: a stage that cannot be
   built (missing key, absent model) is skipped with its error printed rather
   than taking startup down — the normal JIT path surfaces the same failure on
   first use. Opt out with `[pipeline.options] prewarm_stages = false`.
4. **`_free_accelerator_cache()` only when a stage was actually built.** On a hit
   there is no build scratch to reclaim.

## Consequences

Verified with the real model set (`runs/bench/verify_residency.py`):

```
PREWARM built 9 stages in 15.4s
RUN 1: wall 19.1s  stages_done=9   build_s all zero? YES
RUN 2: wall 16.8s  stages_done=9   build_s all zero? YES
```

Every stage reports `build_s == 0.0` on the **first** job after startup, which was
the whole point. (Prewarm itself took only 15.4 s here because the HF cache and
MLX kernels were already warm from a prior process; on a genuinely cold process
the ~240 s Metal-kernel JIT lands in prewarm instead of in the first user-visible
job — which is the intent.)

Costs and caveats:

- **The full resident set is now held up front — ~10 GB with the current models.**
  Fine on 32 GB; `prewarm_stages = false` exists for memory-constrained boxes or
  when the server is started just to browse finished runs.
- **The 2026-08-23 TTS unification is what makes this fit a discrete GPU.** With
  batch and watch-along on separate Higgs candidates, a shared cache would have
  held two 4B instances (8.6 GB of TTS alone) and would not fit 24 GB of VRAM.
  One candidate id means one instance serves both modes.
- **Skipping `_free_accelerator_cache()` on hits was tried and REVERTED the same
  day.** Watch-along regressed from 1.710 to **3.018** agg RTF with every stage
  ~65–72% slower — the signature of a uniform system-level stall, not a logic
  bug. Peak RSS read 10.4 GB with the gate vs 7.3 GB without it, and `vm_stat`
  showed heavy swapping. MLX's Metal buffers live in unified memory but sit
  largely outside RSS, so accumulating scratch is invisible to process
  accounting while still pushing the machine into swap — which is why RSS was a
  misleading instrument for this question. Reverting recovered to 2.011 (a 33%
  improvement) but not to the 1.710 baseline; that residual is unexplained and
  the confirming run was contaminated by machine sleep, so treat the causal link
  as directional rather than proven. **Do not re-introduce the gate without a
  memory instrument that actually sees Metal allocations.**
- **This laptop is not a usable benchmark host for 15-minute GPU-saturating
  runs.** It suspends mid-run (per-stage walls come from `time.time()` and so
  absorb suspended time, while `agg_rtf` comes from `perf_counter` and does
  not), carries pre-existing swap pressure, and throttles under sustained load.
  Re-measure perf changes of this size on the dedicated workstation.
- `bench.py` batch still passes no cache deliberately, so `--repeat` continues to
  measure the cold-build path.

*Revisit:* memory pressure from all-stages-resident on a smaller box, watch-along
peak memory regressing from the skipped cache-free, or a second candidate being
reintroduced for a stage that both modes use.
