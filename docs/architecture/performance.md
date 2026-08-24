# Pipeline performance — baseline + roadmap

Last updated: 2026-08-24 (first RTX 3090 / CUDA baselines, both modes)

> **Status: two machines, and the Apple-Silicon rows need a caveat.** The
> **sample_1 (Reacher)** rows are the Apple-Silicon numbers — current active set
> (`qwen3_asr_1_7b_mlx_8bit` + `higgs_v3_mlx_q8`) on 32 GB M1 Pro, models warm +
> `HF_HUB_OFFLINE=1`. The **masha and Fallout rows are historical twice over**:
> measured on `whisper_large_v3_mlx_fp16`, and both clips have since been removed
> from disk, so they cannot be re-measured at all (`eval-clips.md`).
>
> Two things were found on 2026-08-24 that affect how every pre-2026-08-24 row
> reads. **Sentence counts before that date are artefacts:** both Qwen3 ASR
> candidates emitted one `Sentence` per ASR chunk, so "25 sentences per 5 min"
> meant 25 chunks; the fix took the same clip from 23 to 85 sentences with no
> change in word count, which moves every per-sentence TTS percentile. And
> **`patmir bench --mode batch` was not measuring the warm path** it claimed to —
> it passed no `stage_cache`, so each `--repeat` rebuilt every model. Invisible on
> MLX (`build_s` 0-3 s); 45 % of wall on torch.
>
> Stored reports:
> `agent-os/specs/2026-06-20-1656-pipeline-benchmark-perf/results/reacher-{batch,watch}.json`
> (Apple Silicon), `runs/bench/*.json` (CUDA, gitignored).

## How these numbers are produced

`patmir bench <clip> --mode batch|watch-along [--repeat N] [--no-warm-cache]
[--report bench.json]` (`patmir/bench.py`). Stage wall + model-build come from
each run's `manifest.json` (`StageStatus.started_at/finished_at/build_s`); TTS
and translate latencies come from `tts/timings.jsonl` /
`translate/timings.jsonl`, emitted when the run sets `ctx.extra["bench"]` —
which `patmir bench` does and `patmir dub` does not (there is no `--bench`
flag; this doc claimed one until 2026-08-24). Higgs overflow counts come from
`tts/voice_cloning_summary.json`.

**RTF convention:** `wall_seconds / source_seconds`. **>1.0 = slower than
realtime.** Per-stage `rtf` is that stage's share against the full source.
One exception, deliberate: watch-along's `agg_rtf` divides by **dubbed audio
seconds**, because every historical watch-along number was computed that way and
reflooring the field would leave a reader unable to tell which convention a row
used. Since 2026-08-24 the report also prints the source-second figure and names
the denominator (`notes.rtf_denominator`, `notes.agg_rtf_source`).

Hardware: 32 GB Apple Silicon, MLX/MPS. Active models per
`current-models.md` / `patmir.toml [pipeline]`.

---

## Batch (D1, quality mode)

### sample_1.mkv (Reacher) on CUDA / RTX 3090 — 2026-08-24 (FIRST CUDA BASELINE)

`source 301.11s · warm wall 138.86s · agg RTF 0.461 (warm, mean of 2) · 76-77 sentences`
(cold first repeat: wall 288.43s · RTF 0.958 · 153.97s of model builds = 53.4% of wall)

Hardware: RTX 3090 24 GB on WSL2, `PATMIR_PROFILE=cuda` (`htdemucs_vocals`,
`sortformer_nemo`, `qwen3_asr_1_7b`, `higgs_v3_torch_bf16`). `HF_HOME` on a **9p**
Windows mount (~360 MB/s warm, ~176 MB/s cold) — that is most of why cold builds cost
154s; see `docs/setup/windows-wsl2-cuda.md`. Report:
`runs/bench/sample_1-cuda-batch.json`.

| stage     | warm wall (s) | rtf   | % of wall | M1 Pro warm (s) |
| --------- | ------------: | ----: | --------: | --------------: |
| ingest    |          0.40 | 0.001 |      0.3% |            0.30 |
| separate  |          8.84 | 0.029 |      6.4% |           37.70 |
| diarize   |          1.53 | 0.005 |      1.1% |            7.50 |
| emotion   |          2.31 | 0.008 |      1.7% |            7.70 |
| asr       |         20.53 | 0.068 |     14.8% |           38.01 |
| translate |         18.82 | 0.062 |     13.6% |            6.80 |
| **tts**   |     **76.78** | 0.255 | **55.3%** |          244.45 |
| mux       |          4.59 | 0.015 |      3.3% |            3.80 |
| output    |          5.07 | 0.017 |      3.7% |            5.90 |

- **Warm RTF 0.461 vs the M1 Pro's 1.17 — 2.5x faster**, and within ~30% of the ~0.35
  projection in `windows-wsl2-cuda.md`. The projection scaled decode by memory bandwidth
  and held cloud translate fixed; translate is now 13.6% of a much smaller wall, which
  is most of the gap. So the bottleneck model was roughly right, not wrong.
- **`tts_rtf` is 0.309-0.347, down from 1.136 on MLX.** This is the important one: synth
  is now *faster* than realtime per spoken second, so the hard floor that kept batch
  above realtime on Apple Silicon is gone. **0 overflows across all three repeats.**
- **TTS is still the largest stage (55%)** but no longer the whole story — at 76.8s it is
  3.2x faster than MLX q8's 244.5s, from bf16 + batched decode (10 chunks over 77
  sentences, ~7.7 sentences/chunk).
- **Cold vs warm is 0.958 vs 0.461** — model builds alone are 53.4% of a cold run
  (asr 67.4s, tts 59.0s, diarize 15.6s). Most of that is 9p read latency, not compute.
  A one-shot `patmir dub` pays it; `--repeat` amortizes it.
- **Translate is now #2 at 13.6%** purely because everything local got faster. It is
  cloud round-trip latency (77 calls, mean ~1.2-1.5s, concurrent), and the only stage the
  GPU cannot help.
- **Sentence count 76-77**, after the 2026-08-24 segmentation fix. Pre-fix this clip read
  23 "sentences" and TTS chunk p90 was 44-55s; it is now 11.6-15.7s.
- Peak **16.42 GB VRAM** allocated (16.98 reserved), peak RSS 10.64 GB. Note this box
  needed its Windows pagefile raised before the warm path fit at all — CUDA under WSL2
  charges host *commit* ~1:1 per GB of VRAM. See the setup doc.


> **Cold vs warm — read this first.** The first-ever batch run on a fresh
> process pays a large **one-time TTS warmup** (~240s on masha): Higgs' first use
> triggers a model fetch + MLX **Metal-kernel JIT compilation** that lands inside
> the TTS *stage* wall but **not** in per-sentence synth time. So the first run
> reads agg RTF **1.74** while two warm repeats read **0.89 / 0.92** with an
> *identical* per-sentence rate (`tts_rtf ~1.37`). **Warm steady-state is the
> real baseline; batch already clears realtime.** Numbers below are warm.

### sample_1.mkv (Reacher) — dialogue-dense live action (canonical clip, 2026-08-23)

`source 301.1s · wall 352.2s · agg RTF 1.17 (warm) · 25 sentences`
(cold first run: wall 390.7s · RTF 1.298)

First Reacher baseline. Measured on the current active set — `qwen3_asr_1_7b_mlx_8bit` +
`higgs_v3_mlx_q8` — so it supersedes the masha/Fallout rows below, which were measured on
`whisper_large_v3_mlx_fp16`.

> **Two caveats added 2026-08-24, before comparing this row to the CUDA rows.**
> (1) This section was headed "English"; the file now on disk (`samples/sample_1.mkv`,
> same 301.1 s) has a single **Russian** audio track. Either the Mac had a different
> copy or the label was wrong — resolve it before reading anything into a
> stage-by-stage comparison, since ASR and TTS cost scale with what is spoken.
> (2) The 25-sentence figure below is an artefact, not a property of the content:
> the Qwen3 candidates emitted one `Sentence` per ASR chunk until 2026-08-24. On
> the same clip, correcting that took 23 sentences to 85 without changing the word
> count. The per-sentence TTS percentiles here therefore describe chunks-sized
> units and are not comparable to post-fix numbers.

| stage     | wall (s) | build (s) | rtf   | % of wall |
| --------- | -------: | --------: | ----: | --------: |
| ingest    |     0.30 |      0.00 | 0.001 |      0.1% |
| separate  |    37.70 |      0.10 | 0.125 |     10.7% |
| diarize   |     7.50 |      0.20 | 0.025 |      2.1% |
| emotion   |     7.70 |      2.00 | 0.026 |      2.2% |
| asr       |    38.01 |      0.80 | 0.126 |     10.8% |
| translate |     6.80 |      0.00 | 0.022 |      1.9% |
| **tts**   |   244.45 |      3.02 | 0.812 |  **69.4%**|
| mux       |     3.80 |      0.00 | 0.013 |      1.1% |
| output    |     5.90 |      0.00 | 0.020 |      1.7% |

- **TTS is no longer one bottleneck among several — it is the bottleneck, at 69.4%.**
  The share nearly doubled (37–42% on the Whisper-era clips) because the Qwen3 ASR flip
  removed ASR as a competitor, not because TTS got slower.
- **ASR collapsed to 10.8%** (38.0s) from Whisper's 29% — the single clearest win of the
  2026-07-14 flip, and larger in-pipeline than the ~3% the A/B predicted.
- **`tts_rtf` = 1.136** — synth is slower than realtime *per spoken second* (241.4s of
  compute for 212.4s of audio). Reacher is ~70% speech, so the stage lands at 0.81 RTF of
  the video. This is the hard floor: no software change on this machine makes batch clear
  realtime while `tts_rtf > 1`.
- **Per-sentence TTS variance is extreme:** median 6.35s but p90 21.4s and max 30.4s
  (mean 9.65s). Qwen3 chunks ~3.5× coarser than Whisper — 25 sentences per 5 minutes vs
  ~104 — so each sentence carries ~8.7s of audio and the long ones dominate. Total decode
  cost is roughly linear in output tokens, so splitting chunks would not cut total TTS
  time much, but it would flatten the tail — and a 30s sentence is a direct watch-along
  buffer risk. **0 overflows / 25.**
- **translate concurrency confirmed working:** 25 calls totalling 37.0s of serial latency
  complete in 6.8s of stage wall (5.4×). Roadmap #1 is delivering; translate is now 1.9%.

### sample_1.mkv (Reacher) — RTX 3090 / WSL2, `cuda` profile (2026-08-24)

`source 301.1s · wall 296.7s · agg RTF 0.985 (COLD, mean of 2) · 74-75 sentences`
Reports: `runs/bench/sample_1-cuda-batch-cold.json`. Hardware: RTX 3090 24 GB,
WSL2 on a 31.8 GB host, `HF_HOME` on a **9p** Windows mount.

> **These are COLD numbers and that is not a choice.** Every repeat rebuilds every
> stage model, so **44-46 % of the wall is model loading**. The warm path — which is
> what the Apple-Silicon rows measure and what `performance.md` has always called the
> baseline — could not be measured on this box: warm-caching all five stage models
> (6.48 GB) and then building Higgs (~8.5 GB) needs ~15 GB, and the machine's CUDA
> ceiling was ~16 GB of *Windows commit headroom*, not VRAM. See
> `docs/setup/windows-wsl2-cuda.md`. Subtracting the build tax puts the warm figure
> near **0.54 RTF**, but that is arithmetic, not a measurement.

| stage     | wall (s) | build (s) | rtf   | % of wall |
| --------- | -------: | --------: | ----: | --------: |
| ingest    |     0.39 |      0.00 | 0.001 |      0.1% |
| separate  |    10.92 |      1.87 | 0.036 |      3.7% |
| diarize   |    11.43 |      9.68 | 0.038 |      3.9% |
| emotion   |    11.11 |      8.89 | 0.037 |      3.7% |
| **asr**   |    77.77 |     58.09 | 0.258 |  **26.2%**|
| translate |    22.41 |      0.00 | 0.074 |      7.6% |
| **tts**   |   152.98 |     54.32 | 0.508 |  **51.6%**|
| mux       |     4.53 |      0.00 | 0.015 |      1.5% |
| output    |     5.15 |      0.00 | 0.017 |      1.7% |

- **Batch clears realtime on the 3090 even cold: 0.985 vs the M1 Pro's 1.17 warm.**
  Like-for-like the gap is much larger than that, since the M1 figure excludes model
  loading and this one does not.
- **TTS is 51.6 % and ASR 26.2 %** — the same two bottlenecks as every other run in
  this document, but note **`build_s` is over a third of each**: asr 58.1 s of 77.8 s,
  tts 54.3 s of 153.0 s. Model *loading*, not compute. The 9p `HF_HOME` contributes
  roughly half of that (~360 MB/s warm, ~176 MB/s cold, vs 3.4-4.0 GB/s on the ext4
  vhdx); the rest is safetensors parsing and host→device copy.
- **`tts_rtf` 0.386-0.426** — synth is ~2.5× faster than realtime *per spoken second*,
  against **1.136 on the M1 Pro at q8**. This is the clearest hardware win: bf16 on a
  3090 beats q8 on an M1 Pro by a wide margin, and TTS is no longer bandwidth-starved.
- **Sentence count 74-75, and that is the post-fix number.** Before the segmentation
  fix the same clip yielded 23. Per-sentence TTS percentiles here are **per chunk**
  (10 chunks, ~7.4 sentences each, batched decode): mean 8.6-9.6 s, p90 15.2-15.9 s.
  Not comparable to the Apple-Silicon per-sentence figures.
- **translate 7.6 % and rising with sentence count** — 74 round-trips at ~1.35-1.64 s
  each, 100-123 s of serial latency compressed into 17-28 s of wall. Concurrency is
  working, but finer sentences mean more calls, so this share grows as segmentation
  improves.
- **1 overflow in repeat 0, 0 in repeat 1** (of 74/75). First overflows seen on the
  torch candidate; `max_batch=8`, 10 batches.
- Peak **10.03 GB VRAM** allocated, peak **10.67 GB host RSS**.

**Not measured yet on this box:** watch-along, and the `max_batch` sweep. Both were
blocked on the commit ceiling above.

### masha-2-5min.mkv — Russian, fast cartoon dialogue (HISTORICAL — clip removed)

> **Not reproducible.** `samples/masha-2-5min.mkv` was retired 2026-08-24 (not a
> good translation candidate) and is no longer on disk. Measured on
> `whisper_large_v3_mlx_fp16`, which is also no longer the active ASR. Kept for
> the speech-density comparison only — masha is the ~23 %-speech end of the
> range. See `eval-clips.md`.

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

### Fallout-S02E01-5min.mkv — Russian, dialogue-dense live action (HISTORICAL — clip removed)

> **Not reproducible.** No longer on disk, and measured on
> `whisper_large_v3_mlx_fp16`. Kept for the density comparison (~55 % speech).
> See `eval-clips.md`.

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

### sample_1.mkv (Reacher) on CUDA / RTX 3090 — 2026-08-24 (FIRST CUDA BASELINE)

`source 301.11s · dubbed audio 300.97s · wall 427.37s · agg RTF 1.42 · 10 sections`
(M1 Pro on the same clip: 1.611. Report: `runs/bench/sample_1-cuda-watch.json`.)

| stage     | wall (s) | rtf   | % of wall |
| --------- | -------: | ----: | --------: |
| ingest    |     1.15 | 0.004 |      0.3% |
| separate  |    39.15 | 0.130 |      9.9% |
| diarize   |    14.00 | 0.047 |      3.6% |
| emotion   |    14.65 | 0.049 |      3.7% |
| **asr**   |   123.53 | 0.410 |  **31.4%**|
| translate |    28.58 | 0.095 |      7.3% |
| **tts**   |   168.74 | 0.561 |  **42.9%**|
| mux       |     3.82 | 0.013 |      1.0% |

**The aggregate badly understates this. Read the per-section rows.**

| sec | span (s) | wall (s) | build (s) | rtf |
| --: | -------: | -------: | --------: | --: |
| 0 | 33.6 | 74.91 | **158.53** | 2.230 |
| 1 | 35.8 | 17.63 | 0.0 | 0.492 |
| 2 | 35.7 | 26.22 | 0.0 | 0.734 |
| 3 | 35.1 | 15.29 | 0.0 | 0.436 |
| 4 | 31.1 | 32.72 | 0.0 | 1.052 |
| 5 | 36.5 | 20.72 | 0.0 | 0.568 |
| 6 | 30.4 | 30.40 | 0.0 | 1.000 |
| 7 | 21.8 | 34.46 | 0.0 | 1.581 |
| 8 | 33.6 | 38.38 | 0.0 | 1.142 |
| 9 | 7.4 | 6.33 | 0.0 | 0.859 |

- **Steady state clears realtime: sections 1-9 are 222.15s of wall for 267.37s of audio =
  RTF 0.831.** This is the first time watch-along has cleared realtime on any machine —
  Apple Silicon measured 1.47-2.31 and always depended on the pre-buffer to survive.
  The stream can now keep ahead of playback rather than racing it.
- **The whole 158.53s build tax is in section 0**; sections 1-9 read `build 0.0`. The
  warm `stage_cache` works in the streaming path, and unlike the Mac's flat 2-13% spread
  the cost is entirely front-loaded. What the 1.42 aggregate measures is therefore mostly
  *startup*: a 130.31s prelude plus section 0.
- **Per-section variance is wide (0.436-1.581)** and tracks speech density in the span,
  not section length — section 7 is the shortest span (21.8s) and the slowest (1.581),
  because a dense span carries more sentences to synthesize.
- **ASR is 31.4% here vs 14.8% in batch** — each section re-runs ASR on its own padded
  slice, and the ForcedAligner is a second 0.6B forward per chunk. Dropping the aligner
  in watch-along remains the obvious lever, worth ~5-6% of wall.
- Both RTF denominators are now reported (`notes.rtf_denominator`,
  `notes.agg_rtf_source`): 1.42 per dubbed audio, 1.419 per source. They agree here
  because the pad double-analysis fix landed in 2026-07; they were 4.9% apart before it.


Overrides vs batch (`streaming_config`): `separate=htdemucs_mlx_fast`,
`tts=higgs_v3_mlx_q8_fast` (max_retries=1), `translate=openrouter_gemini_flash_batch`.
Each section runs `run_job` **twice** (analysis ingest→asr, then dub
translate→mux) — **since 2026-07-14 with a shared warm-model `stage_cache`**
(models built once per stream; see the re-measure below). The pre-2026-07-14
baselines below rebuilt every stage model on both calls.

### sample_1.mkv (Reacher) — 2026-08-23 (current active set)

`source 301.1s · dubbed audio 301.0s · wall 514.6s · agg RTF 1.71 · 10 sections`

| stage     | wall (s) | rtf   | % of wall |
| --------- | -------: | ----: | --------: |
| ingest    |     1.80 | 0.006 |      0.4% |
| separate  |    54.90 | 0.183 |     11.6% |
| diarize   |    19.00 | 0.063 |      4.0% |
| emotion   |    16.00 | 0.053 |      3.4% |
| asr       |    82.15 | 0.273 |     17.4% |
| translate |    25.30 | 0.084 |      5.3% |
| **tts**   |   269.49 | 0.895 |  **57.0%**|
| mux       |     4.00 | 0.013 |      0.8% |

- **Rebuild tax is down to 2.1%** (10.8s of 514.6s), from the 11–13% measured pre-07-14.
  Roadmap #2 (warm-model loop) is confirmed landed and effective.
- **Dubbed audio 301.0s == source 301.1s**, so the pad is no longer analyzed twice.
  Roadmap #5 confirmed landed (masha previously read 314.7s dubbed over 300s source).
- **Section pipelining: big on paper, small on MLX.** The analysis half
  (ingest+separate+diarize+emotion+asr) is 0.578 RTF; the dub half (translate+tts+mux) is
  0.992 RTF, and they run strictly sequentially, so the two sum to 1.71. Overlapping them
  would cap at `max(0.578, 0.992)` ≈ 0.99 RTF — a 42% cut — **but only with real
  parallelism, which MLX does not provide.** The D3 spike measured this directly
  (`spike/d3_throughput/FINDINGS.md`): "thread-concurrency is a dead end — MLX serializes on
  one GPU stream; conc 2/4 leave wall-clock flat and just inflate per-sentence latency."
  Both halves are GPU-bound, so on this machine only the cloud-translate wait (25.3s, 5.3%)
  and the CPU-only ECAPA/remap work would actually overlap — call it **5–8%, not 42%**.
  Worth building on CUDA, where streams overlap for real; not worth the concurrency
  complexity (two workers, two-level coverage, seek-vs-in-flight reconciliation) for single
  digits on MLX.
- **ASR is 17.4% here vs 10.8% in batch** because each section re-runs ASR on its own
  padded slice. The ForcedAligner (a second 0.6B encode+forward per chunk) is the
  droppable part — worth ~5–6% of wall, not the ~29% it would have been under Whisper.

### masha-2-5min.mkv — re-measure 2026-07-14 (HISTORICAL — clip removed)

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

### masha-2-5min.mkv — pre-2026-07-14 baseline (HISTORICAL — clip removed)

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

### Fallout-S02E01-5min.mkv — dialogue-dense (HISTORICAL — clip removed)

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
| **sample_1 batch, CUDA/3090** (warm) | **0.461** | **55%** | **15%** | 0% | first CUDA baseline 2026-08-24; cold 0.958 with builds 53% of wall |
| **sample_1 watch-along, CUDA/3090** | **1.42** | 43% | 31% | 37% | **steady-state sections 1-9 = 0.831, clears realtime**; all build cost in section 0 |
| **Reacher batch, M1 Pro** (dense, ~70% speech, qwen3+higgs) | **1.17** | **69%** | **11%** | – | Apple Silicon reference |
| **Reacher watch-along** (qwen3+higgs) | **1.71** | 57% | 17% | 2.1% | current active set |
| masha batch (sparse, 23% speech)   | **0.89** | 37% | 29% | – | clears realtime |
| Fallout batch (dense, 55% speech)  | **1.36–1.66** | 42% | 29% | – | dense → >realtime |
| masha watch-along (2026-07-14, warm loop) | **1.47 (cold)** | 32% | 29% | 3.4% | roadmap #2/#5 landed |
| masha watch-along (pre-07-14)      | **1.94–2.31** | 38% | 34% | 13% | historical |
| Fallout watch-along (pre-07-14)    | **2.06** | 52% | 23% | 11% | re-measure pending |
| en batch (7s)                      | 3.9–5.4 | – | – | – | fixed-cost floor only |

**Findings:**

0. **Superseded by the 2026-08-23 Reacher baseline (findings 1/4/5 below are historical).**
   On the current active set, **TTS alone is 69% of batch wall and ASR is 11%** — they are
   no longer co-bottlenecks. The Qwen3 flip removed ASR as a competitor, which means
   essentially all remaining batch headroom is in TTS, and TTS is memory-bandwidth-bound
   (`tts_rtf` 1.136 at the ear-approved q8, and q4 was ear-rejected). The rebuild tax
   (finding 4) is down to 2.1% and the pad double-analysis (finding 5) is gone.

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

### Rejected: skipping the inter-stage accelerator-cache free on warm hits

Tried 2026-08-23 alongside the warm-model residency work, on the theory that a cache
hit has no build scratch to reclaim. **Reverted the same day.**

| watch-along run | agg RTF | peak RSS |
| --- | --: | --: |
| baseline | 1.710 | – |
| skipping the free on hits | **3.018** | 10.4 GB |
| reverted, run spanned a machine suspend | 2.011 (discard) | 7.3 GB |
| **reverted, clean run under `caffeinate`** | **1.611** | **6.1 GB** |

Every stage slowed ~65–72% — uniform across Demucs, Qwen3-ASR and Higgs, which is a
system-level stall rather than a bug in any one stage. MLX's Metal buffers live in
unified memory but sit largely **outside RSS**, so the accumulating scratch never showed
up in process accounting while still driving the machine into swap. RSS was the wrong
instrument for the question.

The clean re-measure came in at **1.611 — better than the 1.710 baseline** — which confirms
the gate as the sole cause and shows the same-day TTS unification (`max_retries` 2 vs 1)
costs nothing in the normal path. Peak RSS tracked the mechanism directly: 10.4 GB with the
gate, 6.1 GB without.

Two notes for whoever revisits this:

- **Always run benches under `caffeinate -i -m -s`.** An intermediate run suspended
  mid-bench and read 2.011; that number is an artifact and was discarded. Per-stage walls
  come from `time.time()` and absorb suspended time, while `agg_rtf` comes from
  `perf_counter` and does not — so a suspended run yields impossible per-stage rows (one
  showed `tts` 13562 s inside a 605 s run) beside a plausible-looking aggregate.
- Roughly 28 GB is swapped out during a *healthy* watch-along run on this host, so swap
  activity alone is not a red flag. It was the gate's extra footprint on top that tipped
  the machine into pathological swapping.

**Realtime gaps, 2026-08-24.** On the **RTX 3090 both modes are through it**: batch warm **0.461** (gap closed, 2.2x margin) and watch-along steady-state **0.831** (gap closed; the 1.42 aggregate is startup — a 130s prelude plus section 0's builds). `tts_rtf` fell from 1.136 on MLX to **0.309-0.347**, so synthesis is faster than realtime per spoken second and the floor that made this impossible on Apple Silicon is gone. What is left is not a throughput problem: **startup latency** (prelude + first-section builds, 37% of watch-along wall) and **cloud translate** (13.6% of batch wall, the one stage the GPU cannot touch). On the **M1 Pro** the 2026-08-23 gaps stand unchanged: batch 0.17, watch-along 0.71. No remaining MLX-side software lever closes either: section pipelining is 5–8% here (MLX serializes on one GPU stream), dropping the watch-along ForcedAligner is ~5–6%, and splitting ASR chunks flattens the TTS p90 tail without cutting total TTS time. The MLX optimization well is essentially dry — Neither gap closes on this machine while `tts_rtf > 1` — the Higgs q8 decode runs at ~154 GB/s effective, 77% of the M1 Pro's 200 GB/s peak, so it is bandwidth-saturated rather than inefficient.

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

### 3. ASR speed  ·  ✅ LANDED 2026-07-14 — far bigger than the A/B predicted

**2026-08-23 update:** the Reacher baseline shows ASR at **10.8% of batch wall (38.0s)**, down from Whisper's 29%. The standalone A/B predicted only ~3% in-pipeline; the real win was much larger. Cost: chunking is ~3.5× coarser (25 sentences per 5 min vs ~104), which pushes TTS per-sentence p90 to 21.4s and max to 30.4s. Splitting chunks at word gaps would flatten that tail — a watch-along buffer concern, not a total-throughput one.

**2026-08-23, later:** VAD-snapped chunking implemented on the **CUDA profile only**, as a side effect of rewriting the torch Qwen3-ASR candidate against transformers' native `qwen3_asr` (the `qwen-asr` package's exact `transformers==4.57.6` pin is unusable alongside Higgs). `qwen3_asr.build_chunks()` cuts on the separate stage's Silero VAD silences capped at `chunk_seconds=28` and runs the chunks batched through both the ASR and the aligner. It was expected to give finer sentences and so a flatter TTS tail.

**2026-08-24 — MEASURED, and the hypothesis was wrong in an instructive way.** VAD-snapped chunking gave **23 sentences per 5 min**, no better than the MLX path's 25. The reason is that chunking was never the mechanism: **both Qwen3 candidates emitted one `Sentence` per chunk** and never called `group_into_sentences`, which is what every other ASR candidate uses to split words on punctuation or word gaps. So "sentence" meant "up to 28 s of audio" no matter how the chunks were cut, and finer chunks would only ever have produced *slightly* finer 28-second buckets.

This was **not a TTS-tail nicety. It was a quality bug**, and it is why this entry was mis-filed as a performance follow-up for a month:

- 23 sentences against diarization's **108 turns** on the same clip. `apply_speaker_tags` assigns one *dominant* speaker per sentence, so a multi-turn span was synthesized in **one character's voice** — one measured 22.9 s "sentence" spanned 11 turns at 86 % dominance, just over the 0.85 threshold, swallowing another character's 1.5 s line.
- Ref clips inherited it: `ref_text` for a speaker read like several people talking.
- The 44–55 s TTS decode chunks were a symptom, not the disease.

Fixed in both candidates by calling the shared grouper, plus a `max_sentence_s` cap that splits at the widest interior gap (needed because `max_gap_s` has a floor — that 22.9 s run's widest gap was 0.32 s). Measured on `samples/sample_1.mkv`, identical 398 words:

| | before | after |
| --- | --: | --: |
| sentences | 23 | **85** |
| span mean / median | 9.67 s / 8.56 s | **1.94 s / 1.20 s** |
| spans > 8 s | 12 | **0** |
| spans > 15 s | 7 | **0** |

**Consequence for this document: every per-sentence TTS percentile measured before 2026-08-24 describes chunk-sized units**, on both machines, and is not comparable to anything measured after.

#### Original entry
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

### 4. TTS per-speaker prefix-KV caching  ·  ❌ DROPPED 2026-08-23

Measured ceiling was −14% of TTS time on the MLX port. Dropped because the CUDA move leaves that port behind for a PyTorch candidate on the 3090. Do not tune code the active machine no longer runs. Original measurement retained below.

**Correction (2026-08-23):** this entry originally justified the drop by saying the PyTorch path "supplies bf16, flash-attention and batched decode across sentences." Only bf16 was right. Upstream's port declares `_supports_sdpa` only, so flash-attention is refused; and its `generate_speech()` is batch-1, so batched decode had to be written (see #6), not inherited. The drop still stands — prefix-KV was an MLX-only optimization — but it was not free.

#### Original entry  ·  measured ceiling: −14% of TTS time
**Update 2026-07-14:** mlx-audio's native Higgs v3 runtime measured (the ADR
revisit trigger): native-q8 ≈ parity with our port (RTF 1.018 vs 1.052);
native-q4 = 0.744 (−29%) without collapse — but the **author ear-test rejected
native-q4 (worst of three) and ranked ours-q8 best**, so no runtime swap.

**Update 2026-07-15 — prefill/decode split measured** (8 sentences, real
Reacher ref, q8 warm; `runs/20260715-reacher-asr-ab/higgs_prefill_split.json`):
decode loop **73.1%**, prefill 21.5%, torch codec decode 5.0%. Direct
cached-prefix experiment (prefix KV built once, suffix-only prefill per
sentence): **saving = 14.3% of TTS time** (per-sentence prefill 0.55–0.65s →
0.13–0.34s; the 120-token speaker prefix dominates the ~150-token prompt).
So KV caching ≈ TTS RTF 0.99 → ~0.85 on Reacher, agg ~2.10 → ~1.95 — worth
doing (moderate effort, quality-neutral by construction) but NOT the realtime
unlock: **decode is the wall (73%)** and it's memory-bandwidth-bound with
quantization already at the ear-approved limit (q8). Realtime on dense
content = pre-buffer, or a decode-level idea (speculative/multi-token
decoding — research-grade).
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

### 6. TTS batched decode across sentences  ·  ✅ LANDED 2026-08-23, MEASURED 2026-08-24 (CUDA only)

The successor to #4, and a much larger lever on paper. The prefill/decode split measured under #4
is what makes the case: **decode is 73% of TTS time and is memory-bandwidth-bound**. At batch 1
every step reads all 4B weights to advance one sentence. Reading them once to advance B sentences is
close to free, so the ceiling here is roughly the batch size — where prefix-KV caching capped out at
−14%.

- **Where:** `patmir/eval/candidates/tts/higgs_v3_torch.py` (`synthesize_batch` /
  `_decode_chunk`), dispatched by `patmir/pipeline/stages/tts/adapter.py`. CUDA only — the MLX port
  is untouched, and MLX would not benefit the same way since it serializes on one GPU stream.
- **Knob:** `max_batch` on the candidate entry, default 8. Chosen from memory arithmetic, not
  measurement: ~8 GB of bf16 weights, and the KV cache is only ~144 KiB/token (GQA, 8 KV heads), so
  B=8 over a 1500-token context adds ~1.7 GiB on a 24 GB card. The ceiling should be compute.
- **Chunks are sorted by expected length.** A chunk costs `max(steps)`, not `mean(steps)`, so
  pairing a one-word line with a long sentence wastes the short row's slot. Sorting by the token cap
  (proportional to text length) keeps `sum/max` per chunk near the batch size — without it the win
  degrades badly on mixed-length dialogue.
- **Verified:** batch-vs-single decode equivalence, bitwise, on a tiny randomly-initialized model on
  CPU at temperature 0 — including a row retiring mid-batch on a smaller cap while its neighbours
  continue (the masking path). **Not verified:** anything about throughput.
- **Measuring it:** `tts/timings.jsonl` rows become per **chunk** under batching (`"kind": "chunk"`)
  because a per-sentence wall no longer exists. `total_s` and `tts_rtf` stay truthful; mean/median/
  p90/max now describe chunks. The per-sentence p90/max that flagged the watch-along buffer risk has
  no direct successor — the chunk distribution is the thing to watch instead.
- **Open follow-up:** a finished row still occupies its slot until the whole chunk drains. With
  length-sorted chunks the waste is bounded, but compacting the batch mid-flight would recover it.

### Not recommended now
- **Cold-start MLX kernel pre-warm** (~240s first-run) — only affects the first
  run per process; batch is one-shot so it's a real one-time cost, but persisting
  the MLX compile cache is fiddly for a once-per-session hit. Revisit only if
  first-dub latency becomes a complaint.
- **separate** is already vocals-only optimized (ADR 2026-06-19); ~10% and not
  worth further work until TTS/ASR are addressed.

## Reproducing

**Default clip is `samples/sample_1.mkv`** — the Reacher S02E01 5-minute clip
(live-action RU dub, representative content), renamed from
`Reacher-S02E01-5min.mkv`. It is the only 5-minute clip still on disk: masha was
retired 2026-08-24 (not a good translation candidate) and Fallout is gone. See
`eval-clips.md` for provenance and for why a measurement citing a removed clip
cannot be reproduced.

```
patmir bench samples/sample_1.mkv --mode batch       --repeat 3
patmir bench samples/sample_1.mkv --mode watch-along --repeat 2
```

`--repeat 3` in batch gives the cold number (repeat 0) and the warm baseline
(repeats 1-2) from one command — all repeats share one model cache. Add
`--no-warm-cache` for N independent cold samples, or if keeping every stage model
resident (~14.5 GB on the cuda profile) is tight on the box.

Run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` once models are cached — it
removes transient HF-metadata failures and network variance from timings.
Watch-along repeat-2 on a 5-min clip peaks at ~8 GB of scratch in `runs/`; keep
headroom or reclaim between runs.
