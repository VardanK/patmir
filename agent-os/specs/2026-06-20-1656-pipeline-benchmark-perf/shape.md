# Pipeline benchmark + perf — Shaping Notes

## Scope

Build a **repeatable benchmark** for the Patmir pipeline, run it across
representative clips in **both** execution modes (batch D1, watch-along D3),
**localize the bottlenecks** with real numbers, and write a **prioritized perf
roadmap**. This spec **measures and plans only** — the actual optimizations ship
in follow-up specs.

## Decisions (from shaping with the author)

- **Deliverable: measure + plan only.** No optimizations implemented here.
- **Target: both modes, batch first.**
- **Benchmark form: a reusable `patmir bench` CLI command** (not a throwaway
  `spike/` script), so every future perf change is measurable against a baseline.
- Benchmark tooling reuses the existing per-stage timing in `manifest.json`
  rather than re-instrumenting; new instrumentation is limited to (a) a `build_s`
  field on `StageStatus` to split model-load from compute, and (b) flag-guarded
  per-sentence latency traces in the TTS/translate adapters.

## Prior hypotheses (to confirm/refute with measurement)

1. **Batch is TTS-bound.** Higgs v3 q8 ~RTF 1.10; fresh KV cache per sentence +
   ref re-embedded per sentence (`patmir/eval/candidates/tts/higgs_v3_mlx.py`).
   Per-speaker prefix-KV caching is a deferred follow-up.
2. **Translate is serial.** Active candidate `openrouter_gemini_flash` uses the
   per-sentence path (no `translate_batch`) → N serial cloud round-trips.
3. **ASR Whisper large-v3 is ~6× slower** than the registered `nemotron_mlx`.
4. **Watch-along rebuilds every stage twice per section.** `process_section`
   calls `run_job` twice (`streaming/section.py` L169/L180); the runner builds +
   frees stage models per call (`runner.py` L105, L124–125). "Warm-model loop" is
   the documented open optimization.

Smoke run (7s `clip-en-test.mkv`, batch) already qualitatively confirms the
ranking — TTS 63.8% of wall, `build_s` shows diarize/emotion/tts load dominating
on short clips, translate serial ~1.4s/sentence. Full 5-min runs give the
representative RTFs.

## Context

- **Visuals:** None.
- **References:** see `references.md` (manifest timing, `_read_duration`,
  `spike/d3_throughput/tts_sweep.py`, eval harnesses, `cli.py` subparser pattern,
  `streaming/runner.py` `streaming_config`/`stream_sections`).
- **Product alignment:** Local-first, 32 GB Apple Silicon, MLX/MPS. Watch-along
  (D3) is the realtime-critical path; batch (D1) targets best-quality throughput.

## Standards Applied

`agent-os/standards/index.yml` is empty — no standards apply.
