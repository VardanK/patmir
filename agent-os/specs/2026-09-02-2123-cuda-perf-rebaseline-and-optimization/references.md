# References for CUDA pipeline performance re-baseline and optimization

## Harness and timing

### `patmir bench`
- **Location:** `patmir/bench.py` (`bench_batch` ~294-367, `bench_watch_along` ~443-638, `_tts_stats` 204-247, `_translate_stats` 250-264, `_resource_peaks` 168-201, `BenchResult` 79-103), CLI in `patmir/cli.py:106-158, 342-390`.
- **Relevance:** the only perf instrument; Task 2 extends it (GPU telemetry, real wall, CPU time, probes) and every later task is measured with it.
- **Key patterns:** one `stage_cache` shared across repeats makes repeat 0 cold and the rest warm; `warm` is measured (`build_total <= 0.05 s`), not declared; `notes` is the extensible bag for new metrics; the report JSON must stay back-compatible with the stored 2026-09-01 files (`test_bench.py` checks this).

### Shared watch-along timing reader
- **Location:** `patmir/pipeline/streaming/timing.py` (`section_timing`, `other_s` = wall − Σ stage walls).
- **Relevance:** the model for the batch `other_s` added in Task 2; bench and the live path must read one number.

### Measurement-folder pattern
- **Location:** `agent-os/specs/2026-08-25-1931-watchalong-startup-and-stalls/measurements/` (`README.md`, `RESULTS.md`, raw JSON/JSONL artifacts).
- **Relevance:** copy this shape for `measurements/` here — raw reports preserved in git, `RESULTS.md` as the authoritative write-up.

## Hot paths

### Higgs v3 torch TTS candidate
- **Location:** `patmir/eval/candidates/tts/higgs_v3_torch.py` — `_decode_chunk` 485-623, `_sample_codes` 181-215, `_balanced_chunks` 218-249, `_request_seed` 166-178, `synthesize_batch` 626-693, `build()` ~728.
- **Relevance:** 52-55 % of pipeline wall; Task 5.
- **Key patterns:** per-row `torch.Generator` seeded from a content hash (reproducibility); upstream's `_sampler_step` state machine is driven with a one-hot at temperature 0; `on_chunk` drives progress and the cancellation checkpoint; retries only re-decode failed rows.
- **Upstream remote code (read only):** `~/.cache/huggingface/modules/transformers_modules/_30f01593ee6a12efa586c92455afe4b76e45095d/1c9dab85523d953d/modeling_higgs_multimodal_qwen3.py` — `apply_delay_pattern` 46, `reverse_delay_pattern` 59, `_SamplerState` 112, `_sample` 120, `_sampler_step` 141-165, `_decode_codes` 265, `_build_prompt_ids` 285, `_prefill_embeds` 300.
- **transformers 5.16 internals (read only):** `cache_utils.py` (StaticCache / StaticLayer ~398-500), `masking_utils.py` (~205-300), `integrations/sdpa_attention.py` (27-130; `use_gqa_in_sdpa` + `repeat_kv` are why HF's path is slow with a padding mask).

### TTS equivalence tests
- **Location:** `patmir/tests/test_higgs_torch_batch.py` (`BatchEquivalenceTest` on a tiny CPU model, per-row seeding determinism, overflow/retry, early-finish isolation, adapter dispatch, `on_chunk`/cancellation).
- **Relevance:** the tests every Task 5 step must keep green; new classes (`BatchedFilterEquivalenceTest`, `SamplerStateVectorizationTest`, graph-vs-eager CUDA test, probe adapter test) join them.

### TTS adapter
- **Location:** `patmir/pipeline/stages/tts/adapter.py` (capability probing 42-83, three dispatch paths 297-344, `timings.jsonl` write ~397-400, cancellation in `on_chunk` 302-314).
- **Relevance:** where the bench-only probe is started/dumped (duck-typed like `summarize()`).

### Qwen3 ASR torch candidate
- **Location:** `patmir/pipeline/stages/asr/qwen3_asr.py` (`_transcribe_batch` 265-279, language probe 323-333, chunk loop 335-340, aligner 344-364).
- **Relevance:** Task 6 (cheap language probe) and the ASR probe in Task 2.

### OpenRouter translate candidates
- **Location:** `patmir/eval/candidates/translation/openrouter.py` (`translate_concurrency = 6` at 160, request loop 215-273, no retries 254-258), `openrouter_batch.py` (serial chunk walk 222-249), adapter `patmir/pipeline/stages/translate/adapter.py` (per-sentence pool 145-154, `translate_batch` 86-113 writes no timings).
- **Relevance:** Task 7 and Task 2 item 5.

### Runner
- **Location:** `patmir/pipeline/runner.py` (stage loop 171-230, `stage_cache_key` 43-53, `prewarm_stage_cache` 56-90, `_free_accelerator_cache` 233-245 with the MLX-regression comment at 236-244, ASR post-steps 395-431, emotion post-steps 341-369).
- **Relevance:** Task 8 (CUDA-only cache-free gating) and the runner-side probe spans in Task 2.

### Separate / emotion / mux / output candidates
- `patmir/pipeline/stages/separate/htdemucs.py` (Silero loaded per call at 206, stem writes 262-263) — Task 9; `speech_vocals` is read by `patmir/pipeline/stages/diarize/ref_picker.py:129` and `patmir/pipeline/stages/asr/gap_filler.py:113`, so it must stay.
- `patmir/pipeline/stages/emotion/emotion2vec.py` (per-turn `generate` 153-170) — Task 9.
- `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py` (per-sentence `pyrubberband` 280-289) and `patmir/pipeline/stages/output/mkv.py` (single ffmpeg with AAC re-encode 49-79) — probes only; documented as options.

### Watch-along
- `patmir/pipeline/streaming/section.py` (two `run_job` calls per section, `source_lang` pass-through at 239), `scheduler.py` (lookahead 120 s, `_next_start`), `runner.py:streaming_config` (MLX overrides displaced by `[pipeline.profiles.cuda.streaming]`), `patmir.toml:776-777` (the `htdemucs_fast` override Task 9 replaces).

## Config
- `patmir.toml` — `[pipeline.profiles.cuda]` 756-777; `higgs_v3_torch_bf16` 469-481 (`max_batch = 8`, "UNMEASURED" comment at 465); `qwen3_asr_1_7b` 1159-1169; `openrouter_gemini_flash` 90-104 and `_batch` 122-134; `htdemucs_vocals` 898-904; `[pipeline.options]` 779-814.

## Docs to update
- `docs/architecture/performance.md` (802 lines; CUDA batch baseline 84-127, watch-along 323-398, roadmap 604-778; cites report files that no longer exist).
- `docs/setup/windows-wsl2-cuda.md` (§ GPU power limit 485-509 — prescriptive only, never recorded as state; pagefile/commit 300-421; `HF_HOME` 183-199).
- `docs/architecture/current-models.md`, `decisions.md`, `quality-issues.md`, `runs.md`, `stages/tts/higgs-audio-v3.md`.
