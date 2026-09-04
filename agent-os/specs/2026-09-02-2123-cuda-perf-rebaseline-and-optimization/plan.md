# CUDA pipeline performance: re-baseline at full power, then optimize (quality-neutral)

Spec folder: `agent-os/specs/2026-09-02-2123-cuda-perf-rebaseline-and-optimization/`

> Planned with Fable; **implemented by Opus**. Every task below is written to be executed without this conversation's context: exact files, commands, acceptance criteria. Read `CLAUDE.md` and the four auto-loaded docs first; the maintenance contract in `CLAUDE.md` applies to every task that changes behaviour.

## Context

The dev box is a WSL2 + RTX 3090 workstation. Until now the card ran under a reduced power limit (the "80 % / 280 W" mitigation in `docs/setup/windows-wsl2-cuda.md:485-509`) because GDDR6X memory overheated. The cooling is fixed and the limit is removed — `nvidia-smi` now reports the default 350 W ceiling with no active throttle reason. Nothing in the repo records the limit's existence, so **every CUDA number in `docs/architecture/performance.md` was measured under it**, and the doc is additionally one bench generation stale: `runs/bench/{baseline-batch,baseline-watch,probe-batch}.json` from 2026-09-01 (still under the limit) are already faster than the doc (batch warm agg RTF 0.38-0.41 vs the doc's 0.461; watch-along 0.68 vs 1.46; cold builds 22-26 s vs 154 s, because `HF_HOME` moved to ext4 — also never written up). The 2026-08-24 report files the doc cites no longer exist on disk (`runs/` is gitignored).

The user wants: (1) a fresh, trustworthy baseline at full power, (2) every stage measured — including the work no stage owns today, (3) bottlenecks ranked, (4) the improvement options worked out, and (5) the top quality-neutral wins implemented, for **both** batch and watch-along, on `samples/sample_1.mkv` (301.1 s, the clip every prior CUDA number used). Quality is verified by artifact diff per change plus one author ear-test at the end. VRAM budget: up to ~21 GB (peak today 16.4 GB, 7.9 GB free).

### What the surveys established (facts to build on, not re-derive)

Warm batch, 2026-09-01, `sample_1`, per-stage share of wall (best repeat, 114 s, RTF 0.380):

| stage | wall s | share | notes |
|---|---:|---:|---|
| tts | 60.0 | 52.5 % | `higgs_v3_torch_bf16`, 10 chunks × ~8 rows, `tts_rtf` 0.28, 0 overflows |
| asr | 16.7 | 14.6 % | of which **4.6 s is a language probe** (extra `generate` on the longest chunk) |
| translate | 14.1 | 12.3 % | 80 cloud calls × ~1.0 s at concurrency 6; **no retries** |
| separate | 8.8 | 7.7 % | Silero VAD reloaded per call on CPU; 2 stem WAV writes mux no longer reads |
| output | 5.9 | 5.2 % | one ffmpeg: video copy + **AAC re-encode of the dub** |
| mux | 4.7 | 4.1 % | numpy loop + up to 2 `rubberband` subprocess spawns per sentence, serial |
| emotion | 2.4 | 2.1 % | ~108 funasr `generate` calls at batch 1 |
| diarize | 1.4 | 1.2 % | |
| ingest | 0.4 | 0.4 % | |

Watch-along, same day: agg RTF 0.682 (source-denominated 0.748), time to first audio 45.4 s, TTS 53 %, ASR 18 %, `other_s` ~2.0 s/section, sections 6-7 above RTF 1.0.

Intra-TTS breakdown found on disk (`runs/20260901-020038-sample_1/tts/bench_probe.json`, written by instrumentation **no longer in the tree**): forward 60.7 s over 1293 steps at B≈8 → **~47 ms/step**; sampling 6.7 s; prefill 3.3 s; codec 1.9 s. A bandwidth-bound 4B bf16 step on a 3090 is ~9-12 ms, so the decode step is 4-5× off the bound. Cause is visible in `patmir/eval/candidates/tts/higgs_v3_torch.py:561-613`: a Python loop over B rows every step, a `.cpu()` device sync per row per step (line 591), `torch.cat` on the mask every step (602), a fresh `cache_position` tensor every step (609), per-row prefill (505-515) and per-row codec decode (616-622). `max_batch=8` was never measured (`patmir.toml:465`).

Measurement gaps in `patmir bench` (`patmir/bench.py`): batch `total_wall_s` is a **sum of stage walls**, so inter-stage work (9× `_free_accelerator_cache()` in `runner.py:245`, manifest rewrites) is invisible; no GPU power/clock/temperature/throttle telemetry at all; no CPU time; `peak_rss_gb` is process-lifetime; the batched translator writes no `timings.jsonl`; no per-stage internal breakdown.

## Goals / non-goals

**Goals:** a fresh baseline with GPU telemetry; permanent intra-stage probes; a ranked options table with measured shares; implement the wins classed quality-neutral below; re-measure; update the docs per the maintenance contract.

**Non-goals (explicitly out of scope):** model swaps (TTS/ASR/diarize/translate candidates stay); cross-stage concurrency in the runner (translate→TTS pipelining, emotion‖asr) — documented as options with estimated gains, not built; watch-along buffer policy (pre-buffer, section pipelining) unless the realtime-playhead run records stalls; best-of-k TTS selection (declined by the author 2026-08-24); the `yt_dlp` streaming ingest candidate.

## Quality gate — three classes, applied per task

- **Class A — bitwise.** Scheduling, allocation, sync removal, vectorized filtering, caching a model, batching an inference whose per-item result is independent. Acceptance: the relevant artifacts are **byte-identical** to the pre-change run with the same inputs (TTS: `md5sum tts/*.wav`; emotion: `emotion/predictions.json` equal; separate: `vad_intervals` equal and stems within float noise; ASR: `transcript.json` equal). A Class A change that is not bitwise is a bug, not a trade.
- **Class B — same semantics, different numerics.** `max_batch` change (batching is documented as not bitwise-invariant across shapes), StaticCache/CUDA graphs, TF32. Acceptance: same sentence count, same speaker ids, 0 overflows, `tts_rtf` improves, and the change is listed for the author's ear-test. If the ear-test rejects it, revert that item only.
- **Class C — behavioural, quality-positive or neutral by construction.** Translate concurrency/retries (cloud output is already non-deterministic at temperature 0.2), cheaper language probe (must detect the same language), streaming `separate` → the vocals specialist batch already uses (strictly better vocals). Acceptance stated per task.

Each optimization task ends with a **warm A/B**: `patmir bench samples/sample_1.mkv --profile cuda --mode batch --repeat 2 --label <task> --report runs/bench/<task>.json`, compared against the Task 3 baseline (warm repeat only), plus the artifact diff for its class. Keep the Windows desktop quiet during benchmarks (Chrome video holds commit — `windows-wsl2-cuda.md:414-421`).

---

## Task 1: Save spec documentation

Create `agent-os/specs/2026-09-02-2123-cuda-perf-rebaseline-and-optimization/` with:

- `plan.md` — this plan.
- `shape.md` — scope (both modes, `sample_1` only, measure + implement top wins), decisions (quality classes above; VRAM ≤ ~21 GB; yesterday's runs are the under-limit "before"; Opus implements), context (no visuals; references below; product alignment: roadmap Phase 3 "sustained throughput must exceed playback rate", "Mode B must not compromise Mode A quality").
- `standards.md` — `agent-os/standards/index.yml` is empty; state that no project standards apply and that `CLAUDE.md`'s maintenance contract is the binding convention.
- `references.md` — `patmir/bench.py` (harness), `patmir/pipeline/streaming/timing.py` (shared timing reader), `patmir/eval/candidates/tts/higgs_v3_torch.py` (decode loop), `patmir/tests/test_higgs_torch_batch.py` (equivalence tests to extend), `agent-os/specs/2026-08-25-1931-watchalong-startup-and-stalls/measurements/` (the measurement-folder pattern to copy), `docs/architecture/performance.md` (structure to update).
- `measurements/README.md` — placeholder; **every bench JSON produced in this spec is copied here**, because `runs/` is gitignored and the 2026-08-24 CUDA reports were lost that way.

## Task 2: Close the measurement gaps in `patmir bench` (before optimizing anything)

Files: `patmir/bench.py`, `patmir/pipeline/runner.py`, `patmir/cli.py`, per-candidate files listed below, `patmir/tests/test_bench.py`.

1. **GPU telemetry sampler.** A daemon thread started per repeat that polls at 1 Hz and stops with the repeat. Prefer `pynvml` (add `nvidia-ml-py` to the `[cuda]` extra in `pyproject.toml`; under WSL2 it loads `/usr/lib/wsl/lib/libnvidia-ml.so.1`); fall back to `nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits` if import fails. Record into `notes["gpu"]`: `power_w {avg,max}`, `sm_mhz {avg,min}`, `temp_c {max}`, `util_pct {avg}`, `mem_used_mb {max}`, `throttle_reasons` (union of names seen, e.g. `sw_power_cap`, `sw_thermal`, `hw_slowdown`), `power_limit_w` (read once). This is what makes a throttled run distinguishable from a clean one — the reason the old numbers cannot be trusted.
2. **Real wall for batch.** Wrap `run_job` in `bench_batch` with `perf_counter`; publish `notes["real_wall_s"]` and `notes["other_s"] = real_wall - Σ stage walls` (the batch twin of watch-along's `other_s`). Keep `total_wall_s`/`agg_rtf` as they are for comparability, add `notes["agg_rtf_real"]`.
3. **CPU time and per-repeat RSS.** `resource.getrusage` deltas (`ru_utime`, `ru_stime`) per repeat → `notes["cpu_s"]`; sample `/proc/self/status VmRSS` from the telemetry thread → `notes["peak_rss_gb"]` becomes per-repeat.
4. **Intra-stage probes**, written only when `ctx.extra.get("bench")` is set, each to `<job>/<stage>/bench_probe.json`, each span bracketed by `torch.cuda.synchronize()` when on CUDA:
   - tts (`higgs_v3_torch.py`): `prefill_s, forward_s, sample_s, codec_s, steps, chunks, rows_per_chunk[]` — restore the instrumentation whose output already exists on disk.
   - asr (`patmir/pipeline/stages/asr/qwen3_asr.py`): `probe_s, generate_s, align_s, n_chunks, n_generate_batches, n_align_batches`; runner side: `tag_s` for `apply_speaker_tags`+`fill_gaps`+`apply_emotion_tags`+`backfill_ref_text`+`write_transcript` (`runner.py:395-431`).
   - separate (`patmir/pipeline/stages/separate/htdemucs.py`): `extract_s` (ffmpeg), `model_s`, `vad_load_s`, `vad_s`, `write_s`.
   - emotion (`patmir/pipeline/stages/emotion/emotion2vec.py`): `resample_s, infer_s, n_turns`; runner side `pick_refs_s`.
   - mux (`ffmpeg_pad_stretch.py`): `read_s, stretch_s, n_stretch_calls, write_s`.
   - output (`output/mkv.py`): `ffprobe_s, ffmpeg_s`.
   Bench collects them into `notes["probes"][stage]`, and `format_report` prints them under each stage row.
5. **Translate batch path timings.** `patmir/pipeline/stages/translate/adapter.py:86-113` (`translate_batch`) writes `translate/timings.jsonl` rows `{kind:"chunk", sentences, elapsed_s}` so watch-along translate is instrumented; `_translate_stats` handles both kinds like `_tts_stats` does.
6. **Tests:** extend `test_bench.py` — sampler produces the schema with a fake poller; `other_s` non-negative; report back-compat with the stored 2026-09-01 JSONs (no key removed); probe files merged when present and ignored when absent.

Acceptance: `patmir bench samples/sample_1.mkv --profile cuda --repeat 1` prints GPU telemetry, real wall, and per-stage probes; all tests pass (`.venv/bin/python -m pytest patmir/tests -q`).

## Task 3: Fresh baseline at 350 W

Commands (record `nvidia-smi -q -d POWER,PERFORMANCE` before and after into `measurements/`):

```
export PATMIR_PROFILE=cuda
.venv/bin/patmir bench samples/sample_1.mkv --mode batch --repeat 3 --label baseline-350w --report runs/bench/baseline-350w-batch.json
.venv/bin/patmir bench samples/sample_1.mkv --mode watch-along --prewarm --playhead realtime --label baseline-350w --report runs/bench/baseline-350w-watch.json
.venv/bin/patmir bench samples/sample_1.mkv --mode watch-along --prewarm --label baseline-350w-greedy --report runs/bench/baseline-350w-watch-greedy.json
```

`--playhead realtime` costs one video duration (5 min) and is the only run that **counts stalls**; it has never been run on this box. Copy all JSONs + the pre/post `nvidia-smi` dumps into the spec's `measurements/`. Keep the baseline job dirs (`runs/<job_id>`) — they are the artifact-diff reference for every later task; record their ids in `measurements/README.md`.

Deliverable: `measurements/RESULTS.md` section "Baseline at 350 W" with the per-stage table for both modes, the GPU telemetry (was the card ever power-capped or thermally throttled during the run? — the sampler answers this), and a **before/after vs the 2026-09-01 under-limit runs**. If the delta is within run-to-run noise (~±5 %), say so: the workload is bandwidth-bound and the doc's rationale for the limit predicted exactly that.

## Task 4: Bottleneck ranking and options table

Write `docs/architecture/performance.md` § "2026-09-02 CUDA re-baseline (350 W)" replacing the stale headline numbers (keep old tables as history, mark superseded), and a new § "Ranked optimization options (2026-09-02)" — one row per option: stage, mechanism, measured share of wall it attacks, expected saving, quality class (A/B/C), status (`this spec` / `option` / `rejected`). Seed it with the list below; Opus fills in the measured numbers from Task 3's probes and adjusts the ranking if the probes disagree with the 2026-09-01 picture.

Implemented in this spec (Tasks 5-9), in order of expected end-to-end gain:

| # | option | attacks | class |
|---|---|---|---|
| 1 | TTS decode loop: the 47 ms step is CPU launch overhead → StaticCache + captured CUDA graph + GQA-native decode attention, after bitwise-safe sync removal and batched sampling | ~55 % of wall; step ~52 → ~24 ms, batch wall ≈ −25 % | A then B |
| 2 | TTS `max_batch` sweep 8/12/16 after #1 (VRAM free 7.9 GB, KV ≈ 147 KB/token) | tail chunks | B |
| 3 | ASR language probe: cheap probe (short slice, few tokens), cached per job/stream | 4.6 s = 29 % of ASR | C |
| 4 | Translate: concurrency 6 → measured best of 8/12/16 + bounded retry with backoff | 12 % of wall, cloud-bound | C |
| 5 | Runner: on CUDA, `_free_accelerator_cache` only after a build (measure `other_s` first) | inter-stage overhead | A |
| 6 | Emotion: batch funasr `generate` over turns | 2 % of wall | A |
| 7 | Separate: cache Silero VAD in the candidate; skip unused stem writes if nothing reads them | ~1-2 s | A |
| 8 | Watch-along: `[pipeline.profiles.cuda.streaming].separate = "htdemucs_vocals"` | 6 % of watch wall + one fewer resident Demucs | C |

Options documented but not built (with estimate): translate→TTS pipelining (bounded by translate wall, ~12 %, needs a non-global TTS sort — conflicts with balanced chunking); emotion ‖ asr overlap (≤ 2 %); output AAC encode → faster encoder flags or encode in parallel with mux (≤ 5 %); mux parallel stretch (placement loop is sequential because `voice_cursor` ripples — only the read+first stretch is parallelizable); StaticCache + CUDA graphs on the TTS step (class B, large potential, compile-time per shape); aligner drop in watch-along (**rejected**: per-word speaker attribution, quality issue #17, needs word timestamps); TF32 (irrelevant for bf16 decode; record as measured no-op if tried).

## Task 5: TTS decode loop (the headline)

Files: `patmir/eval/candidates/tts/higgs_v3_torch.py` (decode loop 485-623, sampler 181-215, `build()` ~728, docstring 11-59), `patmir/tests/test_higgs_torch_batch.py`, `patmir/pipeline/stages/tts/adapter.py` (~300-400), `patmir/bench.py`, `patmir/cli.py`, `patmir.toml` (`higgs_v3_torch_bf16`, ~469-481), `docs/architecture/stages/tts/higgs-audio-v3.md`. Reference only, do not edit: upstream remote code `~/.cache/huggingface/modules/transformers_modules/_30f01593ee6a12efa586c92455afe4b76e45095d/1c9dab85523d953d/modeling_higgs_multimodal_qwen3.py` (state machine 111-165, `_decode_codes` 265-272); transformers 5.16 `cache_utils.py` (StaticCache/StaticLayer ~398-500), `masking_utils.py` (~205-300), `integrations/sdpa_attention.py` (27-130).

### The finding that sets the design

Read-only micro-benchmarks run on this box during planning (venv torch 2.11+cu128, transformers 5.16.1):

- A 36-layer `Qwen3Model` with **tiny** widths (GPU work negligible) costs **42.9 ms/step** at B=8 eagerly with DynamicCache + `torch.cat` mask — the exact shape of `_decode_chunk` — and 41.4 ms with a StaticCache. The real model's 47 ms/step is therefore **CPU launch overhead** (WSL2 makes launches worse), with the 4B model's GPU work hidden underneath. Removing our own per-row syncs cannot get below ~42 ms on its own.
- Manual `torch.cuda.CUDAGraph` capture of the StaticCache step works with this transformers version (its `is_tracing()` honours stream capture; the mask code has no `.item()`): tiny-model replay is **5.8 ms/step**.
- The real GPU floor: two real-dimension layers (2560 / 32 heads / 8 kv / 128 d / 9728) under graph replay cost 1.82 ms (L=800) to 2.32 ms (L=1500) → 33-42 ms for 36 layers **as HF's SDPA path stands**, because with any padding mask `use_gqa_in_sdpa` is False and `repeat_kv` materialises K/V 4× per layer per step (`sdpa_attention.py:27-40,98-100`). A GQA-native matmul attention for q_len=1 measured **0.17 / 0.13 ms** per layer vs 0.48 / 0.77 for `repeat_kv`+sdpa; `F.sdpa(enable_gqa=True)` with a mask fell to cuDNN and was worse (0.99 / 1.80).
- Batched top-p filtering + per-row `multinomial` is **bitwise identical** to today's per-row `_sample_codes` on CUDA: 40/40 trials incl. tie-heavy bf16-valued logits. Sampling loop today 4.1 ms/step; batched 1.26 ms.

Realistic target: forward ~20-23 ms/step (graph replay + GQA attention) + ~1.5-2 ms sampling/state ⇒ **~52 → ~24 ms/step (≈2.2×)**, TTS decode 67 → ~31 s, TTS stage ~73 → ~37 s, **batch wall about −25 %**. Not 4-5×: the eager-kernel ceiling is ~6 ms of small-kernel time plus ~11-12 ms of skinny-GEMM weight reads; `torch.compile` fusion is the lever after that (deferred, see 5g).

### Two classes of change, landed as two separate commits

- **Class A (bitwise):** 5.0-5.4 below. Every generator still does exactly one `torch.multinomial([N,V],1)` per step, same order, same probabilities. Gate: all 75-81 `tts/*.wav` md5 identical to baseline.
- **Class B (ULP-level logits change):** 5.5 — StaticCache attends over `max_cache_len` masked keys; the custom GQA attention changes reduction order (measured max |diff| 1.3e-3 vs `repeat_kv`+sdpa). RNG consumption is unchanged, but at temperature 0.8 a one-ulp logit change flips some draws, so **some lines redraw once** — the same non-invariance class the module docstring (50-59) and ADR 2026-08-24 already document for batch composition. Reproducibility (same set → same bytes) is preserved and tested. Flag it in the ADR and on the ear-test list.

### Steps

**5.0 Permanent intra-stage probe (first; everything after is measured with it).** Add a `_Probe` class next to `_Request`: `enabled`, `level` (`"chunk"` | `"step"`), accumulators `prefill_s, forward_s, sample_s, codec_s, capture_s, steps, chunks, live_row_steps, total_row_steps`, per-chunk `[{B, seq, max_cache_len, steps, wall_s}]`, `trace_path`. `section(name)` is a context manager that `torch.cuda.synchronize()`s before/after **only when enabled** (disabled = no sync, zero cost). `"chunk"` level times prefill / decode loop / codec once per chunk; `"step"` (opt-in, inflates totals) splits forward vs sample per step. `dump(path)` writes the six legacy keys (`prefill_s, forward_s, sample_s, codec_s, steps, chunks`) plus `ms_per_step_forward, ms_per_step_sample, live_row_fraction, capture_s, cuda_graphs, attn_impl, max_batch, level, chunk_details`. Trace option: `torch.profiler.profile` around **chunk 0 only** → `tts/trace_chunk0.json`. Adapter: next to the `timings.jsonl` write, `if ctx.extra.get("bench") and hasattr(self.inner, "probe")`: start with `level=ctx.extra.get("tts_probe","chunk")`, `trace_path` if `ctx.extra.get("tts_trace")`; dump to `tts/bench_probe.json` after synthesis (duck-typed like `summarize()`). `bench.py:_tts_stats` merges it as `out["probe"]`; `format_report` prints `tts probe: N steps, fwd X ms/step, sample Y ms/step, prefill P s, codec C s, live rows F%`; `bench_batch` passes `tts_probe`/`tts_trace` from new `--tts-probe {chunk,step}` / `--tts-trace` flags on the bench parser. Test: stub inner with a `probe` attribute → file exists under bench, absent without. (This also satisfies Task 2 item 4's tts probe.)

**5.1 Remove per-row per-step syncs; preallocate device outputs** (`higgs_v3_torch.py:556-597`). `out = torch.empty(B, max_cap, N, long, device)`, `length/done/overflowed/cap` as `[B]` device tensors; each step writes `out[:, step] = codes` for all rows; **one** `.cpu()` per chunk. Terminate via `done.all().item()` checked every `SYNC_EVERY = 4` steps (≤3 wasted steps per chunk) plus the `range(max cap)` hard stop. Dead rows keep being fed; attention is per-sequence (same argument as the current comment at 565-567). Class A. Existing `BatchEquivalenceTest` must pass.

**5.2 Batched top-k/top-p filter, per-row multinomial** (`181-215`, `574-578`). New `_filter_logits(logits_BNV, temperature, top_p, top_k) -> probs_BNV` running the identical op sequence on `[B,N,V]`; then `drawn = stack([probs[b].multinomial(1, generator=gens[b]) for b])` — draw for **every** row including done ones so live rows' generator streams are byte-for-byte today's. Keep `_sample_codes` (tests import it) delegating to `_filter_logits` on `unsqueeze(0)` so the paths cannot drift. Class A. New `BatchedFilterEquivalenceTest`: 50 seeds, CPU (+CUDA if available), fp32 and bf16-rounded logits, batched == per-row.

**5.3 Vectorise upstream's `_sampler_step` state machine** (`579-597`, reimplementing upstream 141-165 for a `[B]` state; import `BOC_ID`/`EOC_ID` from `self._up`). Upstream per row at 0-based step `s` (all rows start together so `delay_count == min(s, N)`): `s < N`: if `s+1 < N` set `codes[s+1:] = BOC`, no EOC check, row appended. `s >= N` with countdown active: `countdown -= 1`; `<= 0` → done **before** append. `s >= N`, countdown inactive, `codes[0] == EOC`: countdown = `N-2` (N=8 → 6), this row **is** appended, 5 more follow, the 7th step after detection retires it. After append: `len >= cap` → done + overflowed. Vectorised: `countdown = full([B], -1)`; per step `if s < N-1: codes[:, s+1:] = BOC` / `elif s >= N: has = countdown >= 0; countdown = where(has, countdown-1, countdown); newly_done = has & (countdown <= 0); eoc = ~has & (codes[:,0] == EOC); countdown = where(eoc, N-2, countdown)` (if `N <= 2`, `eoc` is `newly_done` directly); `first_done = newly_done & ~done; length = where(first_done, s, length); done |= newly_done; out[:, s] = codes; hit = ~done & (s+1 >= cap); overflowed |= hit; length = where(hit, s+1, length); done |= hit`. The one-hot round trip is dropped (identity). Class A. New `SamplerStateVectorizationTest`: 300 random `[T,N]` code streams with EOC injected at random steps (incl. during the delay ramp, where it must be ignored) and random caps, compared against a loop over upstream `_sampler_step` on one-hots (skip if the snapshot module is absent, like `BatchEquivalenceTest`); plus the existing stop-token / cap-overflow tests.

**5.4 Preallocate the attention mask; drop `cache_position`** (`602`, `609`). `Qwen3Model.forward` in 5.16 has no `cache_position` parameter (swallowed by `**kwargs`) and `StaticLayer.update` derives positions from its own `cumulative_length`, so the per-step tensor is a pure cost — delete it. `mask_full = zeros(B, L, bool)` once per chunk (bool skips transformers' `.to(bool)` copy), `mask_full[:, :seq]` from the left-pad mask, per step `mask_full[:, seq+s] = True`. Small saving on its own; a static-address mask is a hard requirement for 5.5. Class A.

**5.5 StaticCache + GQA-native decode attention + manual CUDA graph** (Class B; **not** `torch.compile`).
- (a) `cache = StaticCache(config=<Qwen3 text config, not the remote wrapper config>, max_cache_len=L)` with `L = seq + max(cap in chunk)`; prefill with `past_key_values=cache`. KV ≈ 147 KB/token → B=8, L=1500 ≈ 1.77 GB (same order as today's DynamicCache; the transient `repeat_kv` copies disappear with (b)).
- (b) Register once at construction a `_gqa_decode_attention(module, query, key, value, attention_mask, scaling=None, **kw)` via `transformers.AttentionInterface.register("patmir_gqa_sdpa", ...)` + `AttentionMaskInterface.register("patmir_gqa_sdpa", sdpa_mask)` and set `model.model.config._attn_implementation = "patmir_gqa_sdpa"`. For `query.shape[2] != 1 or attention_mask is None` delegate to HF's `sdpa_attention_forward` (prefill unchanged). For q_len=1: `s = matmul(q.view(B,KVH,G,D), k.transpose(-1,-2)).float() * scaling` → `[B,KVH,G,L]`; `masked_fill(~mask[:, :, 0].unsqueeze(1), -inf)`; `softmax(-1).to(v.dtype)`; `matmul(p, v).view(B,H,1,D).transpose(1,2)`, return `(out, None)` (HF expects `[B, q, H, D]`). Masks, RoPE, norms, MLP, cache stay HF's — this is not a second port. Knob `decode_attention = "gqa" | "sdpa"` (default `gqa`).
- (c) After prefill: static buffers `codes_buf [B,N] long`, `pos_buf [B,1] long`, `mask_full`; two eager warm-up steps on a side stream (required before capture); reset every `cache.layers[i].cumulative_length.fill_(seq)`, `pos_buf`, `mask_full[:, seq:]`; then capture `emb = audio_embedding(codes_buf).unsqueeze(1).to(dtype); h = model.model(inputs_embeds=emb, attention_mask=mask_full, position_ids=pos_buf, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]; logits_buf = audio_head(h).float(); pos_buf += 1` inside `torch.cuda.graph(g)`. Per step: `mask_full[:, seq+s] = True; g.replay()`; eager 5.2/5.3 on `logits_buf`; `codes_buf.copy_(codes)`; `out[:, s] = codes_buf`. Capture ≈ 0.15 s per chunk vs ≈ 4 s saved per 130-step chunk → capture per chunk, no shape bucketing in v1 (one `(B, L_bucket)` cache+graph reused via `cache.reset()` is optional v2). `try/except` around capture → eager StaticCache step with the same buffers (identical numerics). Knob `cuda_graphs = true`; auto-off on CPU (unit tests). `torch.multinomial` with per-row CUDA generators stays outside the graph on the current stream. Graph replay runs the very same kernels as the eager StaticCache step, so **graph vs eager-static must be bitwise identical** — new CUDA-only test (skip without CUDA) on the tiny model, `torch.equal` codes with `cuda_graphs` on/off. `BatchEquivalenceTest` at temperature 0 still passes on CPU.
- Why not `torch.compile(mode="reduce-overhead")` now: 1-3 min inductor compile per distinct `(B, L)` (every chunk differs, ~10/job; the FX cache helps only across processes), dynamic-shape care, and a second numerics drift. Feasible later (`model.model` is HF's `Qwen3Model`; the remote code's missing `_supports_static_cache` is irrelevant because we never call `generate`). Deferred as 5.7.

**5.6 `max_batch` sweep 8/12/16** — only meaningful after 5.5 (today the step is CPU-bound so B barely matters). Bench-only env override in `build()`: `max_batch = int(os.environ.get("PATMIR_TTS_MAX_BATCH", config.get("max_batch", 8)))`, documented as bench-only. Per value: `PATMIR_PROFILE=cuda PATMIR_TTS_MAX_BATCH=12 .venv/bin/patmir bench samples/sample_1.mkv --mode batch --repeat 2 --label mb12 --report runs/bench/mb12.json`. Decision on the warm repeat: minimise TTS `wall_s` subject to peak VRAM ≤ 21 GB, `overflows` not increasing, and `live_row_fraction` not collapsing (bigger chunks widen the cap spread). KV at L=1500: B=12 → 2.6 GB, B=16 → 3.5 GB. Expect diminishing returns past 12. Write the chosen value into `patmir.toml:481` with the sweep table in the comment and in `higgs-audio-v3.md`. Class B (documented non-invariance).

**5.7 Deferred (record as options in performance.md, do not build):** `torch.compile` fusion on the step (~15-25 % over eager kernels); compacting finished rows mid-chunk (invalidates the graph; a dead row costs ~1/8 of the attention share — revisit only if the probe's `live_row_fraction` < ~60 %).

**Rejected, with reasons for the options table:** batched prefill — already batched (505-515 only assembles embeddings; the forward at 533 is one call), no-op. Batched codec decode — right-padding rows puts non-zero activations where the single-row conv stack sees zero padding, so tails differ and cuDNN picks algorithms by shape: not bitwise for 1.9 s (2.7 % of TTS); optional later for equal-length rows only. TF32 — bf16 decode is unaffected; the codec's fp32 convs already run TF32 under cuDNN defaults. **`cudnn.benchmark=True` is harmful**: it makes codec algorithm selection run-dependent and would break byte-identical output — do not enable.

### Bitwise A/B driver (used by 5.1-5.5)

The resume path re-runs only TTS+mux from the stored `translated.json` and `diarize/refs.json`:

```
cp -r runs/<task-3 baseline job> runs/ab-baseline
cp -r runs/<task-3 baseline job> runs/ab-work
md5sum runs/ab-baseline/tts/*.wav > <scratch>/baseline.md5
PATMIR_PROFILE=cuda .venv/bin/patmir dub samples/sample_1.mkv --resume-job runs/ab-work --resume-from tts
md5sum runs/ab-work/tts/*.wav | diff - <scratch>/baseline.md5
```

First confirm a resume at HEAD reproduces the baseline md5s (the 2026-08-24 reproducibility fix says it must). After 5.1-5.4: all md5s identical, any diff is a bug. After 5.5: run the new code **twice** — the two runs identical (reproducibility kept); vs baseline report the fraction of WAVs that changed; also a temperature-0 A/B via a copied toml (expect near-total identity; any diff must trace to a near-tie argmax). Probe deltas (`ms_per_step_forward` ~47 → ~20-23) go into `performance.md`.

New ADR for this task: `docs/architecture/decisions/2026-09-02-tts-decode-cuda-graphs.md` (+ one line in `decisions.md`): the CPU-bound finding, manual graphs over compile, the Class B redraw and its evidence, the `cudnn.benchmark` prohibition. `test_config_profiles.RuntimeParityTest` must still pass (`EMOTION_TAGS` untouched).

## Task 6: ASR language probe

File: `patmir/pipeline/stages/asr/qwen3_asr.py:327-333`, `patmir/tests/test_qwen3_asr_torch.py`.

Today the probe is `self._transcribe_batch([clips[longest]], None)` — a **full transcription** of the longest chunk (up to 28 s) at `max_new_tokens=256`, whose text is then thrown away; only `probe[0]["language"]` is read. Measured 4.6 s of a 16 s stage. Change to a dedicated `_detect_language(clip)` that (a) truncates the probe clip to the first ~8 s of the longest chunk (still the most-speech chunk) and (b) passes a small `max_new_tokens` — Qwen3-ASR emits the language tag before the text, so read `processor.decode(..., return_format="parsed")` in transformers' `qwen3_asr` to find the minimum token count that still yields `language`, and pin it with a test. `ctx.source_lang` is already set after detection (`qwen3_asr.py:333`) and persisted to the manifest (`runner.py:212`), so a resumed job and the gap-filler do not re-probe; watch-along already passes the prelude's language (`section.py:239`). The server submits URL jobs with `source_lang=None` (`ui/jobs.py:51`), so this cost is real in production, not just in bench.

Class C acceptance: detected language identical on `sample_1` and on `samples/clip-en-test.mkv` (en); `transcript.json` byte-identical to baseline (the transcription path is unchanged once the language is known).

## Task 7: Translate concurrency and resilience

Files: `patmir/eval/candidates/translation/openrouter.py` (`translate_concurrency`, request loop), `patmir/pipeline/stages/translate/adapter.py`, `patmir.toml` `[[stages.translate.candidates]]` for `openrouter_gemini_flash` and `_batch`.

- `translate_concurrency` is a class attribute (`openrouter.py:160`, `= 6`) read by the adapter via `getattr` (`adapter.py:145`). Make it a constructor/config knob on both candidates (default stays 6); sweep 6/8/12/16 with `patmir bench --repeat 1` (translate wall only matters; read from the report). Pick the largest value that does not produce 429s or raise p90 latency; document.
- Add bounded retry with exponential backoff on 429/5xx/timeouts (3 attempts, 0.5/1/2 s) — today one failed call fails the stage. Per-sentence failure after retries: keep the current behaviour (raise) but log which sentence.
- Watch-along `_batch`: chunks are walked serially (`openrouter_batch.py:222-249`); run chunks concurrently under the same knob.

Class C acceptance: sentence count identical, no empty translations (the audience guardrail already counts them), translate wall reduced.

## Task 8: Runner — accelerator cache free on CUDA

File: `patmir/pipeline/runner.py:233-245`, `patmir/tests/test_accelerator_cache.py`.

Measure first with Task 2's `other_s` and a one-off timing of `_free_accelerator_cache` per stage. If the 9 calls cost ≥ 1 % of wall: on CUDA only, free after a stage **only if that stage was built this run or the next stage is tts** (largest resident); keep the unconditional free on MLX/MPS (the 1.71→3.02 regression was unified-memory, see the comment at `runner.py:236-244` — do not touch that branch). Class A; also assert `peak_vram_gb` does not rise by more than the freed cache (`max_memory_reserved`).

## Task 9: Small stages

- **Emotion** (`emotion2vec.py:153-170`): pass the list of turn slices to one `AutoModel.generate` call (batches of, say, 32) instead of one call per turn. Class A: `emotion/predictions.json` identical. If funasr's batched path changes scores in the last digit, treat as Class B and document.
- **Separate** (`htdemucs.py:206`, `262-263`): load Silero once in `build()` (lives in `stage_cache`) and, if it is faster on CUDA for whole-file masking, move it there (measure both). `speech_vocals.wav` **must stay** — `ref_picker.py:129` slices ref clips from it and `gap_filler.py:113` reads it. `non_speech_vocals.wav` has no reader except the resume loader (`runner.py:559`; the mux comment at `ffmpeg_pad_stretch.py:73-75` says it stopped using the split) — confirm with `grep -rn non_speech patmir/`, then skip that write unless a new `[pipeline.options] write_non_speech_stem = false` knob is set, keeping the `SeparatedAudio.non_speech_vocals` field `None` in that case (`stage.py:59` already allows it). Update `docs/architecture/runs.md` and `data-contracts.md`. Class A on `vad_intervals`, `vocals.wav`, `speech_vocals.wav`.
- **Watch-along separate** (`patmir.toml:776-777`): set `[pipeline.profiles.cuda.streaming].separate = "htdemucs_vocals"`. Class C, quality-positive: same vocals model batch uses; one fewer Demucs resident. Re-run the watch-along bench; expect section `separate` wall to drop and resident VRAM to fall. Update `current-models.md` (cuda profile table) and `pipeline.md`'s watch-along note.

## Task 10: Re-measure, compare, ear-test handoff

Re-run the three Task 3 commands with `--label after`. Produce `measurements/RESULTS.md` § "After": the per-stage table before/after for both modes, `time_to_first_audio_s`, stall count, GPU telemetry, peak VRAM. Produce the ear-test package: path to the final `runs/<job>/out.mkv` from the batch run, the list of Class B/C changes the author should listen for, and the exact revert for each (one config line or one commit).

## Task 11: Documentation (maintenance contract)

- `docs/architecture/performance.md` — new baseline sections (Task 4), roadmap rows for each option with status, remove the claim that `HF_HOME` is on 9p, note the report files now live in the spec's `measurements/`.
- `docs/architecture/current-models.md` — cuda profile rows: tts `max_batch` value + the decode-loop change; streaming `separate` swap; asr probe change; translate knobs.
- `docs/architecture/decisions.md` + new ADR `docs/architecture/decisions/2026-09-02-cuda-perf-rebaseline-and-tts-decode.md` — why the old numbers were untrustworthy (power limit unrecorded, no telemetry), the quality-class gate, what was accepted/rejected with numbers, revisit triggers (a bigger card; upstream gaining batched decode/StaticCache; ear-test rejecting a Class B item).
- `docs/architecture/stages/tts/higgs-audio-v3.md` — decode-loop internals + sweep table.
- `docs/setup/windows-wsl2-cuda.md` — § "GPU power limit": record that the card now runs at the default 350 W after the cooling fix, and that the bench records power/clock/throttle so a return of throttling is visible; § filesystem: `HF_HOME` is on ext4 (measured cold-build delta 154 → 22 s).
- `docs/architecture/quality-issues.md` — only if a Class B item is shipped pending ear-test: add a row `awaiting ear-confirm`.
- `docs/architecture/runs.md` — the new `<stage>/bench_probe.json` artifacts; the stem-write option if Task 9 changes it.

## Verification (end to end)

1. `.venv/bin/python -m pytest patmir/tests -q` green after every task.
2. Artifact diffs per class, recorded in `measurements/RESULTS.md` with the md5 lists.
3. Task 10's before/after tables; the target the plan is aiming for: batch warm agg RTF from ~0.38 toward ~0.28-0.30 (TTS step ≈ 2.2× faster, translate/ASR trims on top), watch-along steady-state well under 1.0 with **0 stalls** at realtime playhead, time to first audio below 40 s. Report the numbers actually measured, not the target, if they fall short.
4. Author ear-test of the final `out.mkv` (Class B/C list attached). Only after that flips any `awaiting ear-confirm` row.
