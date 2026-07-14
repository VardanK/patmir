# Pipeline performance — baseline + roadmap

Last updated: 2026-06-20 (spec `2026-06-20-1656-pipeline-benchmark-perf`)

> **Status: baseline measured.** Full matrix complete (masha + Fallout ×
> batch + watch-along, + English batch). All numbers below are measured via
> `patmir bench` on 32 GB Apple Silicon, models warm + `HF_HUB_OFFLINE=1`.
> This spec measures + plans only; the roadmap items are not yet implemented.

## How these numbers are produced

`patmir bench <clip> --mode batch|watch-along [--repeat N] [--report bench.json]`
(`patmir/bench.py`). Stage wall + model-build come from each run's
`manifest.json` (`StageStatus.started_at/finished_at/build_s`); per-sentence TTS
and translate latencies come from `tts/timings.jsonl` / `translate/timings.jsonl`
(emitted only under `--bench`); Higgs overflow counts from
`tts/voice_cloning_summary.json`.

**RTF convention:** `wall_seconds / source_seconds`. **>1.0 = slower than
realtime.** Per-stage `rtf` is that stage's share against the full source.

Hardware: 32 GB Apple Silicon, MLX/MPS. Active models per
`current-models.md` / `patmir.toml [pipeline]`.

---

## Batch (D1, quality mode)

> **Cold vs warm — read this first.** The first-ever batch run on a fresh
> process pays a large **one-time TTS warmup** (~240s on masha): Higgs' first use
> triggers a model fetch + MLX **Metal-kernel JIT compilation** that lands inside
> the TTS *stage* wall but **not** in per-sentence synth time. So the first run
> reads agg RTF **1.74** while two warm repeats read **0.89 / 0.92** with an
> *identical* per-sentence rate (`tts_rtf ~1.37`). **Warm steady-state is the
> real baseline; batch already clears realtime.** Numbers below are warm.

### masha-2-5min.mkv — Russian, fast cartoon dialogue (canonical clip)

`source 300.0s · wall 267.0s · agg RTF 0.89 (warm) · 37 sentences`
(cold first run: wall 521.8s · RTF 1.74 — TTS stage inflated by ~242s warmup)

| stage     | wall (s) | build (s) | rtf   | % of wall |
| --------- | -------: | --------: | ----: | --------: |
| ingest    |     0.34 |      0.00 | 0.001 |      0.1% |
| separate  |    27.54 |      0.10 | 0.092 |     10.3% |
| diarize   |     7.94 |      0.63 | 0.026 |      3.0% |
| emotion   |     4.58 |      2.07 | 0.015 |      1.7% |
| **asr**   |    78.48 |      0.00 | 0.262 |  **29.4%**|
| translate |    40.17 |      0.00 | 0.134 |     15.0% |
| **tts**   |    99.88 |      5.86 | 0.333 |  **37.4%**|
| mux       |     3.12 |      0.00 | 0.010 |      1.2% |
| output    |     4.94 |      0.00 | 0.016 |      1.9% |

- **TTS is the largest stage (37.4%)** but no longer "the" bottleneck once warm.
  Higgs q8 per-sentence: mean 2.54s, p90 3.35s, max 4.28s, `tts_rtf` 1.365,
  **0 overflows / 37**. Note `tts_rtf > 1` (synth is slower than realtime
  *per spoken second*) yet the TTS *stage* is only 0.33 RTF of the video —
  because only ~23% of the video is speech (68.9s of TTS audio over 300s).
- **ASR is #2 at 29.4%** (78.5s, rtf 0.262) — Whisper large-v3. Hypothesis #3:
  Nemotron is ~6× faster; the swap would cut the single biggest *fixed* chunk.
- **translate 15.0%, fully serial** — 37 round-trips, mean 1.08s each, 40s total.
  Hypothesis #2 confirmed: latency, not compute → concurrency ≈ free win.
- **separate 10.3%** (27.5s) — already vocals-only optimized (ADR 2026-06-19).
- diarize/emotion tiny in compute but carry **build_s** (model load) — irrelevant
  for one batch job, but this is exactly the per-section rebuild cost in D3.

### clip-en-test.mkv — English (cross-language sanity, 7s)

`source 7.2s · agg RTF 5.39 cold / 3.91 warm · 2 sentences`

Cross-language path works (English source → EA). RTF is dominated by **fixed
model-load + tiny sentence count** — 2 sentences can't amortize ~25s of per-stage
model builds. Confirms the fixed-cost floor; not a throughput data point.

### Fallout-S02E01-5min.mkv — Russian, dialogue-dense live action

`source 300.0s · wall 408–499s · agg RTF 1.36–1.66 (warm) · 48–60 sentences`
(two repeats found 48 vs 60 sentences — ASR segmentation variance; the 60-sentence
run below is the heavier/representative one.)

| stage     | wall (s) | rtf   | % of wall |
| --------- | -------: | ----: | --------: |
| ingest    |     0.33 | 0.001 |      0.1% |
| separate  |    53.70 | 0.179 |     10.8% |
| diarize   |    11.81 | 0.039 |      2.4% |
| emotion   |     4.89 | 0.016 |      1.0% |
| **asr**   |   145.83 | 0.486 |  **29.2%**|
| translate |    63.51 | 0.212 |     12.7% |
| **tts**   |   207.82 | 0.693 |  **41.7%**|
| mux       |     5.93 | 0.020 |      1.2% |
| output    |     5.04 | 0.017 |      1.0% |

- **Batch RTF is content-dependent.** Fallout is **dialogue-dense** — 166s of TTS
  audio over 300s (**~55% speech** vs masha's ~23%). TTS (207.8s) and ASR (145.8s)
  both scale with the amount of speech, so the same warm pipeline that clears
  realtime on masha (0.89) runs **1.4–1.7× slower than realtime** here.
- TTS per-sentence: mean 3.38s, max 8.37s, `tts_rtf` 1.223, **0 overflows / 60**.
- ASR 29.2% (145.8s) — Whisper large-v3 is the second-largest fixed cost and grows
  with audio length; the Nemotron swap (hyp. #3) matters most on dense content.
- translate 12.7% serial — 60 round-trips × ~1.06s.

**Takeaway:** "Is batch realtime?" depends on speech density — yes for sparse
content, no for dense. TTS + ASR are the two levers; both scale with speech.

---

## Watch-along (D3, streaming mode)

Overrides vs batch (`streaming_config`): `separate=htdemucs_mlx_fast`,
`tts=higgs_v3_mlx_q8_fast` (max_retries=1), `translate=openrouter_gemini_flash_batch`.
Each section runs `run_job` **twice** (analysis ingest→asr, then dub
translate→mux) — **since 2026-07-14 with a shared warm-model `stage_cache`**
(models built once per stream; see the re-measure below). The pre-2026-07-14
baselines below rebuilt every stage model on both calls.

### masha-2-5min.mkv — re-measure 2026-07-14 (warm-model loop + pad-dubbing + P0 fixes)

`source 300.0s · 12 sections · wall 471.9s · agg RTF 1.47` — **−0.5 to −0.8 vs
the 1.94–2.31 baseline below**, and this was a *cold* run (first in process;
the baseline numbers were warm). ADR
`2026-07-14-watchalong-voice-consistency-and-warm-loop`.

| stage     | wall (s) | % | vs baseline |
| --------- | -------: | --: | --- |
| tts       |   139.54 | 31.8% | 195.08 → 139.5 |
| asr       |   127.51 | 29.1% | 175.72 → 127.5 (fewer, longer sections) |
| separate  |    90.46 | 20.6% | 67.51 → 90.5 (cold-run JIT; re-measure warm) |
| emotion   |    34.41 |  7.8% | |
| diarize   |    21.56 |  4.9% | |
| translate |    20.06 |  4.6% | |
| ingest/mux|     5.08 |  1.1% | |

- **Rebuild tax collapsed: 15.9s = 3.4% of wall (was 80.2s = 13.1%)** — the
  `stage_cache` warm loop pays one build in section 0 and reuses after.
- **12 sections instead of 13; spans reach 36–38s** — sections now VAD-snap
  into the 8s pad instead of stopping at the 30s target (roadmap #5).
- Remaining gap to realtime ≈ 0.5 RTF, almost entirely ASR (0.40 RTF) + TTS
  (0.44 RTF) → the P2 levers (streaming ASR candidate; Higgs runtime work).

### masha-2-5min.mkv — pre-2026-07-14 baseline (historical)

### masha-2-5min.mkv

`source 300.0s · 13 sections · dubbed audio 314.7s · wall 610.8s · agg RTF 1.94`
(rep1 read 2.31, the difference is prelude variance: 22.6s vs 43.1s)

Aggregated stage wall across all 13 sections:

| stage     | wall (s) | % of wall | vs batch (masha) |
| --------- | -------: | --------: | ---------------- |
| **asr**   |   175.72 |  **34.2%**| per-section Whisper on padded slices — way up from batch's 29% *share* and far more absolute |
| **tts**   |   195.08 |  **38.0%**| Higgs q8 *fast* (max_retries=1) |
| separate  |    67.51 |     13.1% | per-section `htdemucs_mlx_fast` |
| emotion   |    26.63 |      5.2% | up from 1.7% batch — per-section rebuild + re-run |
| translate |    26.01 |      5.1% | batched (`..._flash_batch`) → cheaper than batch's serial |
| diarize   |    16.00 |      3.1% | |
| ingest/mux|     6.69 |      1.3% | |

- **Watch-along runs ~2× slower than realtime (RTF 1.94–2.31).** It does *not*
  clear realtime on its own — it leans on the pre-buffer (consistent with the
  current-models note). The realtime gap to close is ~1.0 RTF.
- **Model-rebuild tax = 80.2s, 13.1% of wall** (hypothesis #4 confirmed,
  quantified). Each of 13 sections runs `run_job` **twice**, rebuilding every
  stage model both times. A warm-model loop reclaims most of this.
- **ASR is the single largest cost (34.2%).** Watch-along does **not** override
  asr, so it runs **Whisper large-v3 per section** on a 38s padded slice — and the
  8s pad overlaps the next section, so that audio is **re-analyzed** every section
  (dubbed audio 314.7s > 300s source ≈ the overlap waste). Nemotron + not
  re-analyzing the pad are the two biggest D3 wins.
- **Per-section RTF 0.69–3.05** (mean ~2). Short spans are worst: section 2's span
  was 10.8s but analysis still cost 23.4s (RTF 2.85) — the fixed per-section
  analysis cost doesn't amortize when a VAD cut lands early. Section 0 (reuses
  prelude analysis) is fastest at 0.69.

### Fallout-S02E01-5min.mkv — dialogue-dense

`source 300.0s · 11 sections · dubbed audio 300.0s · wall 618.1s · agg RTF 2.06`
(repeat 1)

| stage     | wall (s) | % of wall |
| --------- | -------: | --------: |
| **tts**   |   277.69 |  **52.5%**|
| **asr**   |   119.95 |  **22.7%**|
| separate  |    54.99 |     10.4% |
| translate |    30.66 |      5.8% |
| emotion   |    25.07 |      4.7% |
| diarize   |    13.21 |      2.5% |
| ingest/mux|     7.67 |      1.4% |

- Same ~2× realtime as masha watch-along (RTF 2.06), but the mix shifts with
  density: dense dialogue pushes **TTS to 52.5%** (vs ASR-led on masha).
- **Model-rebuild tax 66.4s = 10.7% of wall** — consistent with masha's 13%.
- Confirms the universal pattern: **TTS + ASR are 75% of every watch-along run**;
  the rebuild tax is a flat ~11–13% on top.

---

## Bottleneck summary

Full matrix (all warm except the noted cold first-run):

| run | RTF | TTS | ASR | rebuild | note |
| --- | --: | --: | --: | ------: | ---- |
| masha batch (sparse, 23% speech)   | **0.89** | 37% | 29% | – | clears realtime |
| Fallout batch (dense, 55% speech)  | **1.36–1.66** | 42% | 29% | – | dense → >realtime |
| masha watch-along (2026-07-14, warm loop) | **1.47 (cold)** | 32% | 29% | 3.4% | roadmap #2/#5 landed |
| masha watch-along (pre-07-14)      | **1.94–2.31** | 38% | 34% | 13% | historical |
| Fallout watch-along (pre-07-14)    | **2.06** | 52% | 23% | 11% | re-measure pending |
| en batch (7s)                      | 3.9–5.4 | – | – | – | fixed-cost floor only |

**Findings:**

1. **TTS and ASR are the two universal bottlenecks** — together 55–75% of *every*
   run. Both scale with the amount of speech, so they dominate regardless of mode.
2. **Batch clears realtime only on sparse content.** masha (sparse) = 0.89;
   Fallout (dense) = 1.4–1.7. Speech density is the driver, not a fixed pipeline
   cost. The earlier "1.74" was a **cold first-run MLX warmup artifact**, not the
   steady state.
3. **Watch-along is ~2× realtime on both clips** and does not clear realtime
   on its own — it depends on the pre-buffer.
4. **Watch-along carries a flat ~11–13% model-rebuild tax** (`run_job` twice per
   section, fresh models each call) — pure overhead, fully reclaimable.
5. **Watch-along re-runs full ASR per section on padded slices**, and the pad
   overlaps the next section → that audio is analyzed twice (masha dubbed-audio
   314.7s > 300s source).
6. **Translate is serial but small** (5–15%); the batch candidate already exists
   and is used in watch-along — a near-free batch-mode win.
7. **Cold-start is real for one-shot batch:** first-ever Higgs use costs ~240s of
   model-fetch + MLX Metal-kernel JIT compilation (one-time per process).

**Realtime gaps to close:** batch-dense ≈ 0.4–0.7 RTF; watch-along ≈ 1.0 RTF.

---

## Roadmap

Ranked by measured ROI (expected RTF win × confidence ÷ effort). **No fixes are
implemented in this spec** — each becomes its own follow-up. Re-measure each with
`patmir bench` against the baselines above.

### 1. Batch translate → batched/concurrent  ·  ✅ LANDED 2026-07-14
(per-sentence loop now concurrent via `translate_concurrency=6` on the
OpenRouter candidate — byte-identical output, no ear-test needed)
- **Hypothesis:** batch translate is serial (40–63s, 12–15% of batch wall, mean
  ~1.05s/round-trip). The concurrent `openrouter_gemini_flash_batch` candidate
  *already exists* and is used by watch-along.
- **Expected win:** translate 40–63s → ~5–10s; batch RTF −0.10 to −0.18.
- **Files:** `patmir.toml [pipeline].translate`, or parallelize the per-sentence
  loop in `patmir/pipeline/stages/translate/adapter.py`.
- **Risk:** low — chunked translation already ear-validated for watch-along;
  confirm batch-mode quality parity. Cloud rate limits.
- **Becomes:** a small config/A-B spec.

### 2. Watch-along warm-model loop  ·  ✅ LANDED 2026-07-14
(`run_job(stage_cache=)`; rebuild tax 13.1% → 3.4%; masha agg RTF 1.94–2.31 →
1.47 cold — see re-measure above)
- **Hypothesis:** `process_section` calls `run_job` twice and rebuilds every stage
  model each call → measured **11–13% of watch-along wall** (66–80s / 5 min).
- **Expected win:** reclaim most of the rebuild tax; watch-along RTF −0.2 to −0.3.
- **Files:** `patmir/pipeline/streaming/section.py` (the two `run_job` calls),
  `patmir/pipeline/runner.py` (`build_stage` per stage). Needs a "build stages
  once, process sections in place" runner variant.
- **Risk:** medium — must preserve the resume/artifact contract and the unified-
  memory "one model resident at a time" property; watch the 32 GB ceiling if
  multiple stage models stay resident.
- **Becomes:** its own spec (runner refactor).

### 3. ASR speed in watch-along  ·  flipped to Qwen3 2026-07-14; speed edge ~3% in-pipeline
**Update 2026-07-15:** author revised 2026-07-14 — "both Whisper and Qwen
garble sometimes → pick the fastest" → `[pipeline].asr =
qwen3_asr_1_7b_mlx_8bit`. In-pipeline Reacher A/B (watch-along, warm loop):
qwen3 asr 118.7s / agg **2.10** vs whisper 136.3s / agg **2.16** — the
standalone 8× is eaten by the ForcedAligner pass (Whisper's word timestamps
are native). Qwen3 also chunks 3.5× coarser (29 vs 104 sentences / 5 min).
Recovery ideas if Qwen3 stays: split chunks at aligner word gaps; drop the
aligner in watch-along (sentence timing may suffice) → restores most of the
8×. Ear-test pending: `runs/20260715-reacher-asr-ab/`. Voxtral + Nemotron
remain rejected. See `stages/asr/qwen3-asr-mlx.md`.
- **Hypothesis:** watch-along does **not** override asr → runs Whisper large-v3
  per section (**23–34% of D3 wall**); `nemotron_mlx` is ~6× faster.
- **Expected win:** masha D3 ASR 175s → ~30s; watch-along RTF −0.4 to −0.5 — the
  single biggest D3 lever.
- **Files:** add `asr` to the `streaming_config` overrides in
  `patmir/pipeline/streaming/runner.py`.
- **Risk:** **high quality risk** — Nemotron garbles fast cartoon dialogue (the
  exact reason Whisper is the active batch ASR, 2026-06-19). Gate by content:
  safe for clear/slow speech, not for fast cartoon. Needs an ear-test per content
  class before defaulting.
- **Becomes:** a spec with an ear-test gate (possibly per-content ASR selection).

### 4. TTS per-speaker prefix-KV caching  ·  back to highest ceiling, highest effort
**Update 2026-07-14:** mlx-audio's native Higgs v3 runtime measured (the ADR
revisit trigger): native-q8 ≈ parity with our port (RTF 1.018 vs 1.052);
native-q4 = 0.744 (−29%) without collapse — but the **author ear-test rejected
native-q4 (worst of three) and ranked ours-q8 best**, so no runtime swap.
Prefix-KV caching on our port is again the only real TTS speed lever.
See `stages/tts/higgs-audio-v3.md`.
- **Hypothesis:** TTS is **37–52% of every run** — the largest single stage.
  Each sentence builds a *fresh* KV cache and re-embeds the speaker ref
  (`higgs_v3_mlx.py` `_generate_once` ~L256, `_encode_ref` ~L299). Reusing the
  ref+system prefix KV across a speaker's sentences should cut per-sentence steps.
- **Expected win:** uncertain magnitude (decode is bandwidth-bound); even 20–30%
  off `tts_rtf` (1.2–1.4) is RTF −0.1 to −0.2 batch and more in dense D3.
- **Files:** `patmir/eval/candidates/tts/higgs_v3_mlx.py` (AR loop + cache reuse).
- **Risk:** high effort, medium confidence — needs careful KV-prefix handling per
  speaker; verify no quality/voice drift. Already flagged as the deferred Higgs
  follow-up (ADR 2026-06-20).
- **Becomes:** its own spec (TTS-internal optimization).

### 5. Watch-along: stop re-analyzing the pad overlap  ·  ✅ LANDED 2026-07-14
(sections now VAD-snap into the pad up to `slice_len − END_GUARD_SECONDS`;
12 sections vs 13 on masha)
- **Hypothesis:** each section re-separates/re-ASRs an 8s pad that the next
  section also analyzes (dubbed audio 314.7s > 300s on masha ≈ overlap waste).
- **Expected win:** ~5–10% of D3 analysis (separate + asr); RTF −0.05 to −0.15.
- **Files:** `patmir/pipeline/streaming/section.py` (`DEFAULT_PAD_SECONDS`,
  slice/snap logic), `streaming/scheduler.py` coverage.
- **Risk:** low–medium — the pad exists so VAD can find the next silence; shrink
  carefully or cache the boundary analysis instead of recomputing.
- **Becomes:** folds into the warm-model-loop spec (#2) or a small standalone.

### Not recommended now
- **Cold-start MLX kernel pre-warm** (~240s first-run) — only affects the first
  run per process; batch is one-shot so it's a real one-time cost, but persisting
  the MLX compile cache is fiddly for a once-per-session hit. Revisit only if
  first-dub latency becomes a complaint.
- **separate** is already vocals-only optimized (ADR 2026-06-19); ~10% and not
  worth further work until TTS/ASR are addressed.

## Reproducing

**Default clip is Reacher (2026-07-14, author decision):** masha is a
cartoonish edge case (fast squeaky voices, singing, SFX) — bench and ear-test
on `samples/Reacher-S02E01-5min.mkv` (live-action RU dub, representative
content) first; keep masha as the cartoon stress test.

```
patmir bench samples/Reacher-S02E01-5min.mkv --mode batch       --repeat 2
patmir bench samples/Reacher-S02E01-5min.mkv --mode watch-along --repeat 2
patmir bench samples/masha-2-5min.mkv        --mode watch-along --repeat 1   # cartoon stress test
patmir bench samples/Fallout-S02E01-5min.mkv --mode watch-along --repeat 1
```

Run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` once models are cached — it
removes transient HF-metadata failures and network variance from timings.
Watch-along repeat-2 on a 5-min clip peaks at ~8 GB of scratch in `runs/`; keep
headroom or reclaim between runs.
