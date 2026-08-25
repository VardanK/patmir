# Watch-along startup + stalls — shaping notes

## Scope

Two author-reported symptoms on the WSL2 + RTX 3090 box, dubbing
`https://www.youtube.com/watch?v=os-jXiYRihI` in watch-along mode from the
Chrome extension:

1. the first chunk takes too long
2. playback periodically freezes

The author also asked explicitly for **benchmarking to identify the slowest
sections** — which turned out to be a finding in its own right, because the live
streaming path records no timing at all and `bench --mode watch-along` does not
exercise the scheduler it is meant to measure.

In scope: instrument the live path, make the watch-along bench measure the real
thing, and fix the measured causes of both symptoms. Out of scope: the `yt_dlp`
partial-download ingest candidate (roadmap slice 3, still unimplemented), and
any model swap.

## Decisions

- **Instrument *and* fix, not one or the other.** Confirmed with the author.
  Telemetry lands first (Tasks 3-4) so every later fix has a before/after, and
  so the next regression is not another archaeology exercise.
- **Freeze before startup.** Confirmed with the author. The freeze has a
  structural cause (A1) whose fix also removes wasted work that hurts startup,
  so the ordering is not just a preference — it is the cheaper path.
- **Build on `origin/main`, not the old worktree HEAD.** The cancellation +
  accelerator-lock work (PR #4, `1dadb82`) is the code that produced the
  measured session. Verified byte-identical to the server checkout before
  planning continued.
- **Task 8 (buffer policy) is deliberately last and deliberately unspecified.**
  Its options depend on numbers Task 3 does not yet produce. Picking a
  pre-buffer size now would be a guess.
- **`best-of-k` TTS selection stays out.** Already declined by the author on
  cost (quality issue #19); nothing here reopens it.

## How the numbers were obtained

No new run was needed — the session was still on disk. Method, so it is
repeatable:

- `~/.patmir/cache/by-url.json` maps `yt:os-jXiYRihI` → cache key
  `d5af5653d745ed66`.
- `~/.patmir/cache/d5af5653d745ed66/sections.json` gives the produced windows and
  exposed the duplicate-index corruption (C1) directly.
- The server's `runs_root` is `/home/vardan/patmir/runs` (the server runs from
  the main checkout, not this worktree). Ten job dirs `20260825-1821*..1826*`
  cover the session: three `-prelude`, seven `-section`.
- Per-stage walls came from each dir's `manifest.json`
  (`started_at`/`finished_at`/`build_s`); inter-stage gaps were computed as
  `next.started_at - prev.finished_at`, which is how the unattributed 14.3 s
  (B2) surfaced — no stage owns it, so no per-stage view could have shown it.
- `progress.jsonl` per dir gave the intra-stage detail: the TTS batch shape
  (`batch of 8` then `batch of 1`), the per-section `asr detected en` cost, and
  the translate round-trip head.
- `separated.meta.json`'s `vad_intervals` gave the exact input that collapsed
  section `182332` to 8.1 s, which is now the unit-test fixture for Task 5.

Everything is preserved under `measurements/`.

## Context

- **Visuals:** none.
- **References:** see `references.md`.
- **Product alignment:** watch-along (D3) is the additive execution model from
  ADR `2026-06-13-near-realtime-d3-spike`; quality mode (D1) stays the default
  and is untouched by this spec. `agent-os/product/` was not consulted for
  scope — this is a defect + instrumentation spec against an existing shipped
  mode, not new product surface.
- **Standards:** `agent-os/standards/` does not exist in this repo, so there is
  no `standards.md`. The binding conventions are `CLAUDE.md`'s maintenance
  contract (which docs to update for which action) and the existing test
  patterns named in `references.md`.

## Corrections found while shaping

Recorded here because they change what a reader should trust:

- **`run_job`'s docstring contradicts its own code.** `runner.py:129-130` says
  "`_free_accelerator_cache` runs only when a stage was actually built — on a
  hit there is no build scratch to reclaim." The `finally` block at
  `runner.py:233-245` does the opposite, unconditionally, with a comment
  explaining that the build-only gate was tried on 2026-08-23 and **reverted**
  after watch-along regressed 1.71 → 3.02 agg RTF. `decisions.md`'s
  2026-08-23 entry repeats the stale claim ("now runs only on a build").
  The code is right; two docs are wrong.
- That revert's evidence is explicitly MLX-specific — `vm_stat`, Metal buffers
  in unified memory, RSS called out as a misleading instrument. None of it
  transfers to a discrete CUDA card, which is why re-testing the gate on the
  3090 stays on the Task 8 list rather than being ruled out by the ADR.
- `performance.md`'s headline "watch-along sections 1-9 = RTF 0.831, the first
  time watch-along cleared realtime on any machine" was measured on a run with
  no snap collapse, no restart and a different TTS batch shape. It is not wrong,
  but it does not describe what the author experienced.
