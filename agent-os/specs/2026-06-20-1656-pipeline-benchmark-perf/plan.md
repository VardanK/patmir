# Pipeline benchmark + bottleneck localization + perf roadmap

## Context

Patmir's pipeline performance is tracked only by anecdote: per-stage durations land
in each run's `manifest.json`, and RTF numbers cited across the docs ("agg RTF
1.10", "~20s/60s", "0.25× realtime") come from throwaway `spike/d3_throughput/*`
scripts run once and discarded. There is **no repeatable way to measure where time
goes** — so every perf claim is unverifiable and regressions are invisible.

This spec builds a **reusable `patmir bench` command**, runs it across representative
clips in **both** execution modes (batch D1, watch-along D3), and produces a
**bottleneck report + prioritized perf roadmap**. Per the user's scope decision, this
spec **measures and plans only** — the actual optimizations ship in follow-up specs.

Strong prior hypotheses from code exploration (to be confirmed by measurement):

- **Batch (D1) is TTS-bound.** Higgs v3 q8 runs agg RTF ~1.10 with a *fresh KV
  cache per sentence* and the speaker ref re-embedded per sentence
  (`patmir/eval/candidates/tts/higgs_v3_mlx.py` AR loop ~L263–284, ref cache
  L177–195). Per-speaker prefix-KV caching is an explicitly-deferred follow-up.
- **Translate** is cloud (OpenRouter) and likely serial per sentence
  (`patmir/pipeline/stages/translate/adapter.py` ~L70–86) — concurrency is free latency.
- **ASR** Whisper large-v3 is ~6× slower than the registered `nemotron_mlx` alternate.
- **Watch-along (D3) rebuilds every stage model twice per section.**
  `process_section` calls `run_job` twice (`patmir/pipeline/streaming/section.py`
  L169 + L180), and the runner builds+frees stage models per call
  (`patmir/pipeline/runner.py` L105, L124–125). The "warm-model loop" is the
  documented open optimization.

## Existing infrastructure to reuse (do NOT rebuild)

- **Per-stage timing**: `manifest.json` `StageStatus.started_at/finished_at`
  (`patmir/pipeline/manifest.py` L18–24). Stage wall = `finished_at - started_at`.
  The CLI already formats these (`patmir/cli.py` L230–243) — reuse that pattern.
- **Source duration**: `_read_duration(job_dir)` reads `audio.meta.json`
  (`patmir/ui/jobs.py` ~L450). Denominator for RTF.
- **RTF convention**: `wall_s / output_audio_s` (or `/ source_s`), as in
  `spike/d3_throughput/tts_sweep.py` (`wav_duration()` helper, `run_config()`).
- **Per-candidate timing patterns**: `patmir/eval/tts_eval.py` (`timings.jsonl`,
  `perf_counter`) and `patmir/eval/translation_eval.py` (`latency_ms`).
- **CLI shape**: argparse subparsers in `patmir/cli.py` (L38); add a `bench`
  subparser + `cmd_bench` next to `cmd_dub`/`cmd_runs_*`.

---

## Task 1: Save spec documentation

Create `agent-os/specs/<YYYY-MM-DD-HHMM>-pipeline-benchmark-perf/` with:

- **plan.md** — this plan
- **shape.md** — scope (measure+plan only; both modes, batch first; reusable CLI),
  decisions, the prior hypotheses above, context
- **standards.md** — relevant standards (if `agent-os/standards/index.yml` has any
  on CLI/eval conventions); otherwise note "none applicable"
- **references.md** — the reuse pointers above (manifest timing, `_read_duration`,
  `tts_sweep.py`, eval harnesses, `cli.py` subparser pattern)

## Task 2: Build the `patmir bench` CLI command

New subcommand in `patmir/cli.py` (`bench` subparser + `cmd_bench`), backed by a new
module `patmir/bench/` (or `patmir/eval/bench.py`).

**Interface:**
```
patmir bench <clip> [--mode batch|watch-along] [--config patmir.toml]
                    [--repeat N] [--report runs/<id>/bench.json] [--label TEXT]
```

**Batch mode** — call `run_job(clip, cfg)`, then read the resulting `manifest.json`:
- per-stage wall (`finished_at - started_at`), RTF (`stage_wall / source_s`), and
  `% of total` for every stage in `STAGE_ORDER`
- aggregate RTF = `sum(stage_walls) / source_s` (>1.0 = slower than realtime)
- sentence count (from the transcript artifact) for normalizing TTS/translate

**Watch-along mode** — drive sections directly (not via the live UI) so each
section is timed: run `run_prelude` + `dub_prelude_window`, then loop
`process_section` over the clip. Time the **two `run_job` calls separately**
(analysis vs translate→tts→mux) to quantify the model-rebuild cost the docs flag.
Aggregate to per-section RTF + a mode-level RTF.

**Lightweight per-item instrumentation** (measurement infra, reusable — in scope):
- TTS adapter (`patmir/pipeline/stages/tts/adapter.py` `_synth`): emit
  `tts/timings.jsonl` rows `{idx, elapsed_s, audio_s, overflow, reseeds}` guarded by
  a `ctx.extra["bench"]` flag so normal runs are untouched. This surfaces
  per-sentence latency + Higgs overflow/reseed rate that manifest can't.
- Translate adapter (`.../translate/adapter.py`): emit `translate/timings.jsonl`
  `{idx, latency_ms}` under the same flag — exposes serial-vs-parallel headroom.

**Output:** a `bench.json` (machine-readable: stages, RTFs, %, per-item stats, env:
model ids from `patmir.toml`, clip, mode, repeat) + a pretty stdout table mirroring
the CLI summary style. `--repeat N` reports cold (first) vs warm (median) separately.

## Task 3: Run the benchmark across representative clips

Run `patmir bench` on the clips already in `samples/`, capturing `bench.json` each:

- `samples/masha-2-5min.mkv` — fast cartoon Russian (TTS- and ASR-stress; the
  recurring quality clip)
- `samples/Fallout-S02E01-5min.mkv` — slower Russian (different speech density)
- `samples/clip-en-test.mkv` — English source (cross-language sanity)

Each in **batch** and **watch-along** mode, `--repeat 2` (cold vs warm). Store the
`bench.json` artifacts under their run dirs. ~6 batch + ~6 watch-along runs total.

## Task 4: Localize bottlenecks (analysis writeup)

From the `bench.json` set, write `docs/architecture/performance.md` (new baseline doc):

- Per-stage RTF + % table per clip/mode, aggregate RTF per mode
- Confirm/refute the four prior hypotheses with numbers (TTS share of batch wall;
  translate serial latency; Whisper vs Nemotron delta; watch-along rebuild cost =
  delta between the two run_job calls vs a warm baseline)
- TTS per-sentence distribution + Higgs overflow/reseed rate from `timings.jsonl`
- Call out the single largest contributor per mode and the realtime gap for D3

## Task 5: Write the prioritized perf roadmap

Append a "Roadmap" section to `docs/architecture/performance.md` ranking candidate
optimizations by **measured ROI** (expected RTF win × confidence ÷ effort). Seed
candidates (final ranking driven by Task 4 data):

- **Watch-along warm-model loop** — build stages once, process sections in place;
  kills the twice-per-section rebuild (`section.py` L169/L180, `runner.py` L105).
- **TTS per-speaker prefix-KV caching** — reuse the ref/system prefix KV across a
  speaker's sentences (deferred follow-up; Higgs AR loop).
- **Parallel/batched translate** — concurrent cloud round-trips (3-cap aware).
- **ASR Nemotron for speed-tolerant clips** — flip `[pipeline].asr` per content.
- **MLX/MPS cache-clear tuning** — is `_free_accelerator_cache()` per-stage costing
  reload time in batch? (measure, don't assume).

Each entry: hypothesis, expected win, affected files, risk, and a one-line
"becomes its own spec" note. **No fixes implemented in this spec.**

## Task 6: Update knowledge base (maintenance contract)

- Add `docs/architecture/performance.md` to the on-demand tier and reference it from
  the relevant docs.
- One-line entry in `docs/architecture/decisions.md` only if a benchmarking
  *methodology* decision is locked (e.g. "RTF = sum(stage_wall)/source_s; `patmir
  bench` is the canonical harness"); otherwise skip (this spec doesn't swap models).
- Do **not** touch `current-models.md` / `quality-issues.md` (no model or quality
  change here).

---

## Verification

1. `patmir bench samples/masha-2-5min.mkv --mode batch --report /tmp/b.json` runs
   end-to-end and prints a per-stage table + aggregate RTF.
2. Cross-check: the stage walls in `bench.json` match
   `patmir runs show <job_id>` (same `manifest.json` source) within rounding.
3. `--mode watch-along` emits per-section timings with the two run_job calls timed
   separately (rebuild cost visible).
4. `tts/timings.jsonl` and `translate/timings.jsonl` exist for a bench run and are
   absent for a normal `patmir dub` run (flag-guarded).
5. `docs/architecture/performance.md` exists with a populated per-stage RTF table
   for ≥2 clips and a ranked roadmap.
