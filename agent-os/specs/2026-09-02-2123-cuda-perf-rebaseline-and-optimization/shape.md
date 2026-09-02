# CUDA pipeline performance re-baseline and optimization — Shaping Notes

Shaped 2026-09-02 with Fable; implementation assigned to Opus. The plan (`plan.md`) is written to be executed without the shaping conversation.

## Scope

Re-baseline the whole 9-stage pipeline on the WSL2 + RTX 3090 box now that the GPU power limit is removed (cooling fixed), measure every stage including the work no stage owns today, rank bottlenecks, work out the improvement options, and implement the top quality-neutral wins. Both execution modes (batch/quality and watch-along). Clip: `samples/sample_1.mkv` only (301.1 s — the clip every prior CUDA number used, so before/after is comparable).

## Decisions

- **Yesterday's runs are the "before".** `runs/bench/{baseline-batch,baseline-watch,probe-batch}.json` (2026-09-01) were run under the power limit (author-confirmed). They are the under-limit reference; everything is re-measured fresh at 350 W. `probe-batch` carries no record of what it probed and is treated as a duplicate baseline.
- **The old numbers were untrustworthy for two reasons, both fixed by this spec:** the power limit was never recorded anywhere in the repo, and the bench captured no GPU power/clock/temperature/throttle telemetry — so no past run can be distinguished from a throttled one. Telemetry becomes a permanent part of `patmir bench` before any new baseline is taken.
- **Measure before optimizing.** The bench gains real wall for batch (`other_s`), CPU time, per-repeat RSS, GPU telemetry, per-stage internal probes and batch-translate timings first (Task 2); the baseline (Task 3) is taken with those in place; only then are optimizations ranked (Task 4) and implemented (Tasks 5-9).
- **Quality gate in three classes**, applied per change: Class A must be byte-identical on the relevant artifacts; Class B is same semantics with ULP-level numeric drift and goes on the author's ear-test list with a one-line revert; Class C is behavioural and neutral-or-positive by construction with an explicit acceptance test. A Class A change that is not bitwise is a bug, not a trade.
- **VRAM budget up to ~21 GB** (peak today 16.4 GB, 7.9 GB free), leaving headroom for WSL2 driver and fragmentation.
- **The headline is the TTS decode step, and the design was reshaped by measurement.** Read-only micro-benchmarks during shaping showed the 47 ms/step is CPU launch overhead (a tiny 36-layer model costs 42.9 ms/step eagerly on this box; CUDA-graph replay 5.8 ms), so sync removal alone cannot win. Order: bitwise-safe changes first (sync removal, batched sampling filter with per-row multinomial — measured bitwise identical 40/40, vectorised state machine, static mask), then StaticCache + captured CUDA graph + GQA-native decode attention as one flagged Class B step. Realistic target ~2.2× on TTS, ~25 % off batch wall; not the 4-5× the bandwidth arithmetic suggested.
- **Deliberately not built:** model swaps; cross-stage concurrency in the runner (translate→TTS pipelining, emotion ‖ asr) — recorded as options with estimates; watch-along buffer policy unless the realtime-playhead baseline records stalls; best-of-k TTS selection (author declined 2026-08-24 on cost); the `yt_dlp` streaming ingest; `torch.compile` on the decode step (deferred behind the manual graph); batched codec decode (not bitwise); aligner drop in watch-along (per-word speaker attribution needs word timestamps). `cudnn.benchmark=True` is prohibited — it makes codec algorithm selection run-dependent and would break byte-identical output.
- **Bench reports are copied into this spec's `measurements/`** because `runs/` is gitignored and the 2026-08-24 CUDA reports the docs cite no longer exist on disk.
- Ear-test: one author listen at the end on the final `out.mkv`, with the Class B/C list attached, not per change.

## Context

- **Visuals:** None.
- **References:** see `references.md`.
- **Product alignment:** `agent-os/product/roadmap.md` Phase 3 — "Sustained pipeline throughput must exceed playback rate so the buffer never drains" and the hard rule "Mode B must not compromise Mode A quality"; `tech-stack.md` — "Track VRAM budget per stage so the full pipeline fits". No numeric performance targets exist in the product docs; this spec's targets (batch warm agg RTF toward ~0.28-0.30, 0 stalls at realtime playhead, time to first audio < 40 s) are engineering targets, and the measured numbers are reported even if they fall short.
- **Environment facts established during shaping:** `nvidia-smi` reports power limit 350 W (default/max), P8 idle, no active throttle reasons; the driver's since-boot counters show past SW power-cap (337 s) and SW thermal-slowdown (203 s) events, so throttling did occur on this boot before the limit was removed. `HF_HOME` is on ext4 (cold builds 154 → 22 s, never written up). torch 2.11.0+cu128, transformers 5.16.1; `pynvml` not installed. 16 CPUs, 15 GB RAM visible to WSL.

## Standards Applied

`agent-os/standards/index.yml` is empty — no project standards exist. The binding convention is `CLAUDE.md`'s maintenance contract (which docs to update for which action) and the auto-loaded architecture docs.
