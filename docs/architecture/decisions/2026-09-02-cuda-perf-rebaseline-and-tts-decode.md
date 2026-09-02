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

## Measured outcome

- **Batch: agg RTF 0.377 -> 0.277 (26 %).** TTS 60.55 -> 29.27 s (2.1x), decode
  step 2.6x. Peak VRAM 16.5 GB of 24.
- **Watch-along at 350 W: 0.623 per dubbed second, 0 stalls at a realtime
  playhead** (the first stall measurement taken on this box), first audio 27.8 s
  with the prewarm the server does at boot.
- Translate is the **second-largest watch-along stage at 23 % of wall**, against
  14 % in batch, because watch-along forces the batched translator and
  `translate_batch` walks its chunks serially with no concurrency at all. Not
  yet addressed.

## Revisit

- A bigger card, or a second one, changing what "one pipeline fits" means.
- Upstream transformers gaining a GQA-native masked decode path, which would
  make our kernel redundant; or breaking the bitwise properties in §5, which
  the CUDA-gated tests will catch.
- Graph capture reuse across same-shape chunks, worth the ~0.6 s/chunk it
  currently costs.
- An ear-test that finds the gqa re-roll worse on other content, in which case
  `decode_attention = "sdpa"` reverts it and keeps the graph's 18 %.
- `max_batch` is still unswept, and only meaningful now that the step is no
  longer launch-bound.
