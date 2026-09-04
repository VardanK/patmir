# 2026-09-02 — Re-baselining the 3090 at full power, and what was actually slow

## Context

The GDDR6X cooling on the WSL2 + RTX 3090 workstation was fixed and the power
limit removed. That invalidated the numbers rather than just improving them,
because **the limit had never been recorded anywhere in the repo**. Every CUDA
figure in `performance.md` was measured on a card running the 80 % / 280 W
mitigation that `docs/setup/windows-wsl2-cuda.md` prescribes, and no bench
report captured power, clock, temperature or throttle state — so a throttled
run and a clean one were indistinguishable after the fact. The driver's
since-boot counters showed 337 s of SW power-cap and 203 s of SW thermal
slowdown, so the throttling was real.

The docs were also one measurement generation stale: `runs/bench/*.json` from
2026-09-01 were already faster than the tables, and the 2026-08-24 report files
the tables cite no longer existed, because `runs/` is gitignored.

## Decisions

### 1. A benchmark that cannot describe the machine is not reproducible

`patmir bench` now samples GPU telemetry at 1 Hz (NVML, falling back to
`nvidia-smi`, then to nothing) for the duration of each repeat.

The load-bearing detail is a distinction NVML does not make: it reports one
`sw_power_cap` bit for two very different situations. A 350 W 3090 sitting at
its **own stock ceiling** under a bandwidth-bound decode is healthy and
expected; a card **capped below stock** is the 280 W mitigation and makes a run
incomparable. Only the second, plus any thermal slowdown, sets `throttled`.
`GpuIdle` is not throttling at all and is published separately as `idle_pct`.

Conflating them would have reproduced the original failure in a new form: a
report that cries throttling on every healthy run teaches its reader to ignore
it.

### 2. Measure inside a stage, not just around it

Per-stage walls come from the manifest and say nothing about where a stage's
own time goes. The intra-TTS numbers that first showed this decode running far
off its bandwidth bound came from instrumentation that was reverted, leaving
`runs/*/tts/bench_probe.json` on disk with no code left to regenerate it.

`patmir/pipeline/probe.py` is that instrumentation, kept. Disabled it hands
back a shared no-op context manager and never touches torch; enabled it syncs
the device on both edges of a span, because a GPU span without that measures
enqueue. Spans and counters sit at the top level of the dump and non-additive
identities and derived ratios under `values`, so aggregating a stream's
sections cannot sum a `max_batch` into 88 — which the first watch-along report
did.

Batch also gained `real_wall_s` / `other_s` (the sum of stage walls had made
everything between stages invisible) and `cpu_s` **including child processes**,
since four stages spawn ffmpeg and the mux spawns `rubberband` twice per
sentence.

### 3. The power limit was never the bottleneck

| | warm agg RTF |
| --- | --- |
| 2026-09-01, capped | 0.393 |
| 2026-09-02, 350 W | **0.377** |

About 4 %, inside the run-to-run spread, and exactly what the setup doc
predicted for a memory-bandwidth-bound workload. Two of three repeats logged
**one second each** of thermal slowdown at 80-81 C — a transient that cannot
move a 113 s wall, but the first thermal number this repo has ever had, and
worth watching on a feature-length run.

### 4. The TTS decode step was CPU-bound, and three measurements were needed to know it

TTS is ~53 % of batch wall and its step measured 45.5 ms where the card's
bandwidth allows roughly 9-12. The tempting reading — a bandwidth problem — was
wrong. A 36-layer model with *negligible widths* costs 42.9 ms/step on this
box, so ~93 % of the step was the CPU issuing kernels.

That produced a deliberately-kept **negative result**: removing ~10,000
device-to-host syncs per chunk, batching the sampler's filtering, vectorizing
the delay/EOC state machine and preallocating the mask moved the step from
45.55 to 45.88 ms. Nothing. Those changes are kept anyway, because CUDA graph
capture requires exactly what they provide — static addresses, no host reads
inside the region, no `.item()` in the state machine.

**Ordering follows from the same measurement and is not obvious.** CPU launch
(~43 ms) and GPU execution (~33 ms) pipeline, so the step sits at the max of
the two. Swapping in a cheaper attention kernel *first* would therefore also
have measured ~0, lowering a floor that was not binding. The graph had to land
first to expose the GPU floor.

### 5. The graph is byte-identical, which was measured, not assumed

|  | ms/step | TTS stage | agg RTF |
| --- | ---: | ---: | ---: |
| eager | 45.55 | 60.55 s | 0.377 |
| + CUDA graph | 27.15 | 39.51 s | 0.309 |
| + gqa attention | **17.33** | **29.27 s** | **0.277** |

The plan classified the whole graph step as a numerics change needing an
ear-test. It is not. `StaticCache` is bitwise equal to `DynamicCache` even with
ragged left-padded prompts, and graph replay is bitwise equal to eager
stepping; all 80 WAVs of a resumed job match md5-for-md5. Both properties are
pinned by CUDA-gated tests, so a future torch or transformers that breaks them
fails a test rather than quietly re-rolling every line of every dub.

Sampling stays **outside** the captured region. Per-row generators must keep
advancing on the host; capturing `multinomial` would freeze its RNG offsets and
undo ADR `2026-08-24-tts-sampling-reproducibility`.

### 6. A knob that changes audio belongs in the config, not in a code default

`decode_attention = "gqa"` skips transformers' `repeat_kv`, which expands 8 KV
heads to 32 on every layer of every step because `use_gqa_in_sdpa` is False
whenever a mask is present (0.48 vs 0.17 ms per layer). Worth a further 10 % of
pipeline wall.

It changes the reduction order, so **all 80 lines re-roll** — durations move up
to 1.24 s, total speech within 1.4 %, reproducible run to run. Expected quality
is unchanged (same distribution, different draw), but issue #19 measured this
TTS's draws spreading 2.4-27.9 dB on identical input, so a re-roll re-spins a
known lottery.

It therefore lives in `patmir.toml`, where `ui.cache.config_hash` sees it
(2cc3dd87 -> 4a56ed45) and cached dubs regenerate once — the price the audience
guardrail also paid. A code default would have left one cache key able to hold
either rendering. `cuda_graphs`, being bitwise, is correctly a code default and
correctly absent from the key.

Author ear-test 2026-09-02 on `runs/20260902-230429-sample_1`: "sounds a little
better". Recorded as **neutral-to-positive, not an improvement** — one sample
cannot establish a direction inside that spread. What it establishes is that
the faster kernel is not worse.

### 7. The watch-along benchmark had never terminated

Taking the watch-along baseline hung: every section produced, then a 2 s idle
loop until killed. `bench_watch_along` passed neither `idle_fn` nor `alive_fn`,
so the scheduler fell back to the playhead's `updated_at` — which the synthetic
viewer refreshes on every call, precisely to keep the abandon timer fed. So the
default driver could never end, and every stored watch-along baseline had been
taken with `--forward-loop`. Production was never affected: the server passes
`alive_fn` keyed on the client still polling.

A benchmark is now over when nothing in the video is left to dub, asked of
*coverage* rather than of `_next_start`'s None — because under `--playhead
realtime` that None also means "buffered far ahead of the viewer", where
waiting is correct.

### 8. Three more stages, and two plan items dropped on measurement

**ASR language probe: 4.63 s -> 0.71 s.** It was a full transcription of the
longest chunk — up to 28 s at `max_new_tokens=256` — whose text was discarded;
only the language tag was read. The model emits `language <LANG><asr_text>...`,
so the tag precedes the transcription: 8 s of audio and a 24-token budget
suffice. The probe **validates** rather than assumes, because a truncated
output parses to `language: None` and would have fallen through to "en",
mislabelling a Russian soundtrack; a missing tag costs one full-length call
instead. This is a batch-mode win only — watch-along sections already receive
`source_lang` from the prelude, so only the prelude ever paid it.

**Translate concurrency 6 -> 12: 16.53 s -> 8.22 s**, per-call latency
unchanged at ~1197 ms so nothing was rate limited. Plus bounded retry on both
candidates: any non-200 previously raised out of the adapter's thread pool and
failed the whole stage, discarding every sentence already translated, for one
429.

**Emotion batched: 0.65 s -> 0.22 s on 24 real turns, 24/24 labels and
confidences identical.** The equivalence is structural, not lucky: funasr still
runs each clip at batch_size 1 internally, so no padding enters the pooled
utterance embedding. Padding would have shifted scores, which feed the TTS
emotion tag and therefore the audio — the stage would have got faster while
quietly re-labelling emotions.

**Two plan items dropped because the probes contradicted their estimates.**
Caching Silero and skipping the unread `non_speech_vocals` write were budgeted
at 1-2 s; they measure 0.03 s and 0.20 s. And gating the per-stage accelerator
free was budgeted against inter-stage overhead that turned out to be 1.9 s,
1.7 % of wall. None justifies a behaviour change.

**One plan item dropped because its reasoning did not survive checking.** The
plan expected pointing watch-along's `separate` at `htdemucs_vocals` to cut the
section wall, citing vocals-only's 3.6x. But that 3.6x was against the
four-model `htdemucs_ft` **bag**; streaming already runs `htdemucs_fast`, a
single network, and the specialist is also a single network — per second of
audio they measure the same (0.026 vs 0.028 s/s). The swap stays defensible on
quality and VRAM, which is a different argument needing its own ear-test.

### 9. Scheduling knobs must not move the cache key

`config_hash` hashed every key of a candidate entry, on the sound reasoning that
model ids, voices, dtypes and sampling knobs all change output. But
`concurrency`, `max_attempts`, `retry_backoff_s` and `cuda_graphs` cannot change
a single sample, and making them tunable means putting them in `patmir.toml` —
which would have re-dubbed a finished library for a change that alters nothing.
They are now excluded, with the bar stated in the code: it must be impossible
for the knob to change one sample. `decode_attention` is named there as the near
miss that stays in the key.

### 10. What translate costs watch-along, and an old decision that inverted

Watch-along translate is **44.2 s of a 153.9 s wall (29 %) — now the largest
stage in that mode**, and it barely moved when concurrency was raised. Measured
per-section, one call each: 1.75 / 1.89 / 2.23 / 2.96 / 3.62 / 4.06 s, then
**10.83 / 12.20 / 13.34**. Three of nine calls are 36 of the 44 s. All 43 API
responses were HTTP 200 first try, so this is neither the new retry logic nor
rate limiting but provider-side latency on the JSON-schema-constrained decode;
the per-sentence path over 78 calls has p90 1.46 s and max 1.63 s.

`streaming_config()` selects the batched translator because, when the
per-sentence path was **serial**, batching was faster. That is no longer true:
per-sentence runs at concurrency 12 and a section holds 4-16 sentences, so all
of them fit in one round-trip's wall against 2-13 s for one batched call —
an estimated 44 s -> ~15 s. Deliberately **not taken**: the batched candidate
exists so the model sees a whole scene plus the previous chunk's translations,
which is what keeps register and pronouns consistent. That is latency bought
with scene coherence and wants an ear verdict, not a performance commit.

**What was taken instead — §11 — is strictly better**: it removes the same cost
without touching the translation at all.

### 11. The section's phases want different resources, so stop serializing them

Translate is a network round-trip with the GPU idle. Every other phase is GPU
work. The sections ran serially not because the work is serial but because
nothing had separated "waiting on a network" from "using the accelerator".

    section k    [analyse][translate .......][speak]
    section k+1           [analyse]                 [translate ....][speak]
                          ^^^^^^^^ the GPU, busy during k's network wait

`SectionPipeline` (`streaming/pipeline.py`) drives the live server and the
scheduler-driven bench; `process_section` stays the serial path for
`stream_sections`, the bench forward-loop and the tests. Measured on the greedy
driver — the one that measures production — **agg RTF 0.511 -> 0.396, wall
153.9 -> 119.3 s (22 %)**, with 7 prefetch hits of 8 possible and 0 discarded.
Stage totals barely move: the same work happens with ~33 s of it hidden.

**The GPU is never used by two phases at once, and that is load-bearing.** The
TTS decode captures a CUDA graph (§5) and capture fails or corrupts if another
thread allocates on the device — so the optimization that made TTS 2.5x faster
is precisely what forbids the naive version of this one. One lock serializes
every GPU phase *and* the registry mutation identity resolution performs;
translate is the only phase outside it. That constraint is why
`process_section` had to be split into three phases at all, which the runner's
`resume_from`/`stop_after` range already permitted.

Prefetch is one section deep, so a seek discards at most one section's work,
and it refuses to trust stale analysis in four ways — each of them a way to
emit audio for the wrong span, the class of bug behind quality issues #22 and
#24:

* a different start discards it;
* a `max_end` clamp discards it, because the clamp changes where the section
  ends;
* a prefetch is never *started* with a clamp, since the next window's clamp
  comes from coverage the scheduler has not updated yet;
* `close()` waits for an in-flight analysis, because a torch forward cannot be
  interrupted and abandoning the thread would hand the GPU to the next job
  while we still held it.

Numbering stays with the caller: a resumed stream must number past the highest
cached index or it overwrites `section_NNN.wav` (#22).

The load-bearing test is a **live overlap detector** — it runs the phases with a
long translate to give the prefetch every chance to collide and asserts two GPU
phases are never in flight together. Reasoning about that was not good enough
for a constraint whose violation corrupts a capture.

`[pipeline.options] section_prefetch = false` restores the serial loop.

### 12. Two ways the watch-along numbers can mislead, both now written down

**A realtime wall is viewer-bound once production outpaces the lookahead.**
`DEFAULT_LOOKAHEAD_SECONDS` is 120 and the clip is 301 s, so the last section
cannot start until the playhead passes ~181 s however fast we get. Of the
pipelined realtime run's 174.0 s of section time, **73.3 s was idle waiting for
the viewer** (production 100.7 s, against serial's 145.7 s — a 31 % gain that
the wall figure hides). Its elapsed wall therefore went *up*, 145.7 -> 187.0 s,
and that is not a regression. **Read greedy for throughput and realtime for
stalls and first-audio latency.**

**`sections_wall_s` changed definition here**, from the sum of per-section
walls to real elapsed, because overlap makes the sum meaningless. Serial runs
agree on both, so pre-pipelining *greedy* rows stay comparable; realtime rows
do not, since elapsed includes idle.

A `pipeline_overlap_s` field was added and then removed rather than shipped:
bench times `process()`, which excludes an analysis already prefetched, so it
read **0.0 next to a 23 % improvement**. `prefetch_hits` is the honest signal.

### 13. Translate was also the overlap *window*, so moving it off the main thread made things worse

Section pipelining (§11) leaves the next obvious lever in plain sight: a
prefetch hit still pays for its own translate on the main thread, so why not
have the background worker translate the section it just analysed and hand it
over ready to speak? Implemented, tested (5 tests, including one asserting
translate never holds the GPU lock), measured — and **reverted, because it is
14 % slower**: agg RTF 0.396 -> **0.439**, wall 119.3 -> **132.2 s**, on a run
where the prefetch worked exactly as designed (7 of 8 sections arrived
pre-translated) and every phase was individually as fast or faster (mean
translate 5.52 -> 3.15 s).

The per-section walls say what happened. They **oscillate** — 4.73 s on one
section, 21.18 s on the next — while the phase sums stay flat, which is the
signature of an overlap that engages every other cycle instead of every cycle.

| | analyse | translate | speak | GPU (analyse+speak) | mean section wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| §11 pipeline | 4.48 | 5.52 | 3.83 | 8.31 | **12.21** |
| + translate prefetch | 4.65 | 3.15 | 3.86 | 8.51 | **13.55** |

The cause is the GPU lock, and it inverts the intuition that motivated the
change. There is exactly **one** phase in a section that does not need the GPU,
and the worker's analysis needs the GPU — so the *only* window in which a
prefetch can run is the window while the main thread is translating. Move
translate to the worker and the main thread goes from one GPU phase to
back-to-back GPU phases, the worker's analysis is pushed to the *end* of the
cycle, and the translate chained behind it lands after the next cycle has
already asked for the section. So the main thread waits for both, having saved
nothing:

```
§11:   main [translate 5.5 .......][speak 3.9]      cycle 9.4
       work [analyse 4.7 ....]                       <- fits in the translate window

+tr:   main                        [speak 3.9]      cycle 11.7
       work [blocked on GPU][analyse 4.7][translate 3.2]
                                    ^ no window, so the analysis starts late
```

Two things follow, and they are why this is written down rather than deleted:

- **The GPU is the floor, not the network.** analyse + speak is 8.5 s of
  serialized GPU work per section against 3.2-5.5 s of network, so the ceiling
  for any arrangement of these three phases is ~8.5 s/section — about 30 %
  below the 12.2 s the shipped pipeline achieves. Reaching it needs the main
  thread to **yield the GPU to a pending analysis** before speaking (or a
  second lookahead level), which trades ~4.6 s of per-section delivery latency
  for throughput. That is a buffer-policy decision, and with 0 stalls and
  production at 2.5x realtime there is nothing to spend it on today.
- **A phase's resource class is not a property of the phase alone.** §11
  reasoned about translate as "the phase that does not use the GPU", which is
  true and was the right basis for overlapping it. This attempt reasoned about
  it as "the phase that does not need to be on the critical path", which is
  false: on a single-lock pipeline, the non-GPU phase *is* the critical path's
  contribution to everyone else's progress.

Report: `measurements/pipelined-tr-greedy.json`. Nothing shipped; the
`section_prefetch` flag and its behaviour are unchanged.

## Measured outcome

**Batch, warm, `sample_1`, 3 repeats:**

| | before (capped) | 350 W baseline | after |
| --- | ---: | ---: | ---: |
| agg RTF | 0.393 | 0.377 | **0.2135** |
| wall | 118.3 s | 113.3 s | **64.2 s** |
| cold agg RTF | 0.482 | 0.499 | **0.339** |

**43 %.** Per-stage: tts 60.55 -> 22.47 s, asr 15.24 -> 11.77, translate
16.02 -> 8.21, emotion 1.94 -> 1.54. Unchanged: separate ~8.9, output 5.5
(all of it an AAC re-encode, still open), mux 4.1, diarize 1.3. Final-run GPU:
214.5 W avg / 349.1 W max of 350, 78 C max, nothing capped below stock, no
thermal slowdown. Peak VRAM 16.7 GB of 24.

**Watch-along, `--prewarm`, real scheduler.** Greedy is the throughput driver:

| | baseline | stage work | **+ pipelining** |
| --- | ---: | ---: | ---: |
| agg RTF | 0.623 | 0.511 | **0.396** |
| wall | 205.8 s | 153.9 s | **119.3 s** |
| first audio | 27.8 s | 23.0 s | **21.6 s** |
| worst section | 1.053 | 0.676 | **0.485** |

**42 % off watch-along**, matching batch. Realtime playhead, which is the stall
and latency check rather than a throughput measure:

| | baseline | stage work | **+ pipelining** |
| --- | ---: | ---: | ---: |
| **stalls** | **0** | **0** | **0** |
| first audio | 38.9 s | 22.2 s | **21.4 s** |
| production time | — | 145.7 s | **100.7 s** |

**No section exceeds realtime in either driver any more**, so the buffer grows
monotonically instead of being clawed back. The realtime rows are the ones that
matter: it is the only driver whose viewer advances with the wall clock and can
outrun production, and it had never been run on this box (greedy cannot stall by
construction, and neither can the `--forward-loop` driver every stored baseline
used).

**No section exceeds realtime in either driver any more** (worst 0.485 against
1.416 at the baseline), so the buffer grows monotonically instead of being
clawed back.

Remaining, in order of size: **graph capture** (5.96 s, 25 % of the TTS stage,
recoverable by reusing a capture across same-shape chunks); **`other_s`** in
watch-along (~20 s, the ffmpeg slice/trim + ECAPA + age/gender work no stage
owns — now a larger share since everything around it shrank); the **output
stage's AAC re-encode** (5.5 s, 8 % of batch wall); and **per-sentence
translate in watch-along** (§10), which is a coherence trade rather than a free
win now that §11 has removed the latency it was meant to fix.

## Revisit

- A bigger card, or a second one, changing what "one pipeline fits" means.
- Upstream transformers gaining a GQA-native masked decode path, which would
  make our kernel redundant; or breaking the bitwise properties in §5, which
  the CUDA-gated tests will catch.
- Graph capture reuse across same-shape chunks, worth the ~0.85 s/chunk it
  currently costs.
- The ~8.5 s/section GPU floor from §13, reachable only by yielding the GPU to
  a pending analysis (or a second lookahead level) at the cost of ~4.6 s of
  per-section delivery latency — worth revisiting if watch-along ever needs the
  headroom, and *not* by prefetching translate, which §13 measured.
- Prefetching the *dub* of section k+1, left alone deliberately: it competes
  for the GPU rather than filling an idle one.
- The 120 s lookahead becoming the binding constraint on how far ahead a
  viewer can be buffered, now that production is well clear of realtime.
- An ear-test that finds the gqa re-roll worse on other content, in which case
  `decode_attention = "sdpa"` reverts it and keeps the graph's 18 %.
- `max_batch` was swept later the same day (8/12/16; 12 set, 16 worse —
  see `current-models.md` and `measurements/RESULTS.md`). Splitting 8 from 12
  more finely needs repeats, which the sweep did not have.
