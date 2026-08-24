# References for CUDA baseline + ASR segmentation

## Sentence grouping — the shared helper the fix should reuse

### `group_into_sentences`

- **Location:** `patmir/pipeline/stages/asr/_sentence.py:208`
- **Relevance:** this is the function `qwen3_asr.py` should be calling and is not.
  Signature `group_into_sentences(words, max_gap_s=0.6)`; splits on terminal
  punctuation **or** a gap over `max_gap_s` between consecutive words. The gap branch
  is what makes it work on a transcript with zero punctuation.
- **Key patterns:** the module also owns `filter_words_by_vad`, `is_non_lexical` and
  `word_in_target_script` — all three are already used by `qwen3_asr.py`, so the
  import is in place and only the grouping call is missing.

### Candidates that already group correctly (copy their shape)

- `patmir/pipeline/stages/asr/whisper_mlx.py` — 1 call site
- `patmir/pipeline/stages/asr/faster_whisper.py` — 2 call sites. Also the registered
  `faster_whisper_large_v3_fp16` fallback on the cuda profile, which makes it the
  same-box A/B for separating a segmentation problem from a transcription problem.
- `patmir/pipeline/stages/asr/elevenlabs_scribe.py`, `openrouter_stt.py` — 2 each

### The two candidates that don't

- `patmir/pipeline/stages/asr/qwen3_asr.py:355-388` — the torch/CUDA candidate; one
  `Sentence` per chunk. Chunking itself is `build_chunks` at `:140-188`, fed from
  `ctx.extra["vad_intervals"]`; the transformers version gate is `_check_transformers`
  at `:117-137` (`MIN_TRANSFORMERS=(5,13)`).
- `patmir/pipeline/stages/asr/qwen3_asr_mlx.py` — the Mac candidate, same defect.

### Speaker attribution — where a turn-boundary split may belong

- **Location:** `patmir/pipeline/stages/asr/speaker_tag.py` (`apply_speaker_tags`),
  called by the runner immediately after the ASR candidate returns.
- **Relevance:** it assigns one *dominant* speaker per sentence, which is what turns
  a multi-turn sentence into a multi-character voice. Being shared by every ASR
  candidate argues for putting a turn-boundary split here rather than in one
  candidate.
- **Input available:** `ctx.extra["diarization"]` carries the turns (108 on the
  reference run) — see `docs/architecture/pipeline.md` "Cross-stage plumbing".

## Benchmark harness

### `patmir/bench.py`

- **Relevance:** the asymmetry to fix. `bench_batch` (:195-218) calls `run_job` with
  no `stage_cache`; `bench_watch_along` (:247+, cache created at :266) does, and its
  comment says "bench must measure the warm path, not the legacy rebuild path".
- **Key patterns:** `BenchResult` (:52-68) is serialized with `asdict`, so new fields
  need defaults. `_tts_stats` (:123-148) ignores the `kind` field. Watch-along
  `agg_rtf` uses `dubbed_audio_s` (:296-298, 328) while batch uses source seconds.
  Dead import at :28.
- **CLI:** `patmir/cli.py:100-115` (parser), `:265-296` (`cmd_bench`), `:135-136`
  (`--profile` exported into the environment before dispatch).

### `patmir/pipeline/runner.py`

- **Relevance:** the cache contract the fix relies on. `run_job`'s stage loop at
  :175-200 — a `stage_cache` hit sets `st.build_s = 0.0` (:188), otherwise it times
  `cfg.build_stage(name)` (:190-192). `prewarm_stage_cache` at :55-89 is the
  server-startup path and is deliberately *not* what bench should use (it would zero
  repeat 0 and it swallows build failures at :84-86). `_free_accelerator_cache` at
  :289-308 runs in the per-stage `finally`, including on cache hits.
- **Manifest:** `patmir/pipeline/manifest.py:17-29` — `StageStatus.build_s` is
  documented as included in `[started_at, finished_at]` and broken out precisely so
  the benchmark can separate model-load from compute.

### `patmir/pipeline/stages/tts/adapter.py`

- **Relevance:** why bench's TTS percentiles are chunk-level. `:286-293` explains
  that inventing a per-sentence wall would multiply `total_s`/`tts_rtf` by the batch
  size; `:361-366` are per-sentence rows, `:373-386` the per-chunk rows with
  `"kind": "chunk"`. `:388` writes `voice_cloning_summary.json` unconditionally.
- Also `_standin` / `_standin_use` at `:94-95,192-223` — keyed by `speaker_id`, the
  one cross-repeat side effect of warm caching worth noting.

## Prior art for this harness

### `agent-os/specs/2026-06-20-1656-pipeline-benchmark-perf/`

- **Relevance:** the spec that built `patmir bench` and produced every Mac baseline
  this one is compared against. `results/reacher-{batch,watch}.json` are the stored
  reports that must still deserialize after `BenchResult` gains fields.

### `docs/setup/windows-wsl2-cuda.md` Phase 9 (line 200)

- **Relevance:** the checklist this spec is executing. Carries the baselines to beat
  (M1 batch 1.17 / watch-along 1.611), the ~0.35 / ~0.51 projection marked "measure
  it, do not assume it", the two first-run watch items (diarization return shape, TTS
  output), and the `max_batch` sweep with its memory arithmetic.

### `docs/architecture/performance.md`

- **Relevance:** the doc Task 10 updates, and the source of the RTF convention
  (`wall/source`, >1 = slower than realtime, :22-23) and the `caffeinate` warning
  (:342-346) about suspended runs yielding impossible per-stage rows.

## Test patterns to follow

- `patmir/tests/test_stage_cache.py` — the cache-contract test style for the new
  `test_bench.py`.
- `patmir/tests/test_qwen3_asr_torch.py` — where the segmentation tests belong.
- `patmir/tests/test_higgs_torch_batch.py:697`
  (`test_bench_timings_are_per_chunk_and_totals_stay_true`) — already covers the
  adapter emitting chunk rows; the new tests cover bench *reading* them.
- `patmir/tests/test_streaming_section.py` — import-site monkeypatching for the
  watch-along path.
- `patmir/tests/conftest.py` — scrubs `$PATMIR_PROFILE`, which matters because it is
  exported in this shell.
