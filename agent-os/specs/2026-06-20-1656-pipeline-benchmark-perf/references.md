# References for Pipeline benchmark + perf

## Timing infrastructure reused (not rebuilt)

### Per-stage wall-clock
- **Location:** `patmir/pipeline/manifest.py` — `StageStatus.started_at` /
  `finished_at` (+ the new `build_s`). `patmir/pipeline/runner.py` L99–125 sets
  them around each stage.
- **Relevance:** the benchmark reads stage wall straight from `manifest.json`;
  zero new per-stage code.
- **Borrowed:** the CLI's duration formatting (`cli.py` `_print_summary` L230–243).

### Source duration
- **Location:** `patmir/ui/jobs.py` `_read_duration` (reads `audio.meta.json`).
  `bench.py` reads `audio.meta.json` inline to avoid importing the UI package.
- **Relevance:** denominator for RTF.

### RTF + concurrency measurement patterns
- **Location:** `spike/d3_throughput/tts_sweep.py` (`wav_duration`, `run_config`),
  `demucs_slice.py`, `elevenlabs_timing.py`, `scribe_size_sweep.py`.
- **Relevance:** `bench.py` is the durable replacement for these throwaways;
  same `wall/audio` RTF math.

### Per-candidate timing
- **Location:** `patmir/eval/tts_eval.py` (`timings.jsonl`, `perf_counter`),
  `patmir/eval/translation_eval.py` (`latency_ms`).
- **Relevance:** the adapter per-sentence traces mirror these formats
  (`tts/timings.jsonl`, `translate/timings.jsonl`).

## Execution-path references

### Batch runner
- **Location:** `patmir/pipeline/runner.py` `run_job` L30–130; `_run_stage`
  L163; `config.py` `build_stage` L53.
- **Relevance:** `bench_batch` calls `run_job(..., extra={"bench": True})`.

### Watch-along path
- **Location:** `patmir/pipeline/streaming/runner.py` — `streaming_config()`
  (the D3 overrides: `htdemucs_mlx_fast` / `higgs_v3_mlx_q8_fast` /
  `openrouter_gemini_flash_batch`), `source_duration()`, `stream_sections()`.
  `streaming/prelude.py` `run_prelude`; `streaming/section.py`
  `dub_prelude_window` + `process_section` (the double `run_job`, L169/L180).
- **Relevance:** `bench_watch_along` reuses `streaming_config` + drives
  prelude → section 0 → sections, timing each and reading per-section manifests.

## TTS bottleneck internals
- **Location:** `patmir/eval/candidates/tts/higgs_v3_mlx.py` — AR loop L263–284,
  ref-encode cache L177–195, `summarize()` (overflow count) L322.
- **Relevance:** prefix-KV caching candidate optimization lives here;
  `bench.py` reads `summarize()`'s `tts/voice_cloning_summary.json` for overflows.
