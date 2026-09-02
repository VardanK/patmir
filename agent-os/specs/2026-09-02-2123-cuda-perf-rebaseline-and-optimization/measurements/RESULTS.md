# Measured results

All runs: `samples/sample_1.mkv` (301.11 s), `PATMIR_PROFILE=cuda`, WSL2 + RTX 3090.
Raw reports in this folder. `runs/` is gitignored, which is how the 2026-08-24
CUDA reports were lost, so every report is copied here.

## Baseline at 350 W (2026-09-02)

`patmir bench samples/sample_1.mkv --mode batch --repeat 3 --label baseline-350w`
→ `baseline-350w-batch.json`, job ids `20260902-221552` (cold),
`20260902-221823` and `20260902-222021` (warm).

### Was the card actually free to run?

This is the question the whole re-baseline exists to answer, and before
2026-09-02 the bench could not answer it at all.

| | value |
| --- | --- |
| enforced power limit | **350.0 W** = the board's stock default |
| capped below stock | **no** — this is what the 280 W mitigation would have shown |
| at its stock 350 W ceiling | 16-17 of 111-150 samples per repeat (expected under load) |
| **thermal slowdown** | **1 sample of 150, and 1 of 111** — brief, at 73 C and 81 C max |
| GPU temp | 68.7-71.4 C avg, **80-81 C max** on the warm repeats |
| power draw | 218 W avg / 325 W max on the warm repeats |

So the power limit is genuinely gone and no run was power-capped below stock.
Two of three repeats did register **one second each of thermal slowdown** at
80-81 C core. That is a transient, not sustained throttling, and it is not
enough to move a 113 s wall — but it is not zero either, and it is the first
time this repo has had any thermal number at all. Worth an eye on a longer
run: a feature-length dub is 10-20x this clip's length at the same duty cycle.

(The first rendering of that line said "THERMAL SLOWDOWN in 150 samples",
printing the sample total where it meant the reason's count. Fixed, with a
test; the numbers above are the corrected ones.)

### Before / after the power limit

The 2026-09-01 runs in this folder were taken with the limit in place.

| | before (2026-09-01, capped) | after (2026-09-02, 350 W) |
| --- | --- | --- |
| warm agg RTF | 0.406, 0.380 (mean **0.393**) | 0.385, 0.368 (mean **0.377**) |
| warm wall | 122.3 s, 114.3 s | 115.9 s, 110.7 s |
| cold agg RTF | 0.482 | 0.499 |

**Removing the limit bought about 4 %**, which is inside the run-to-run spread
and roughly what the setup doc predicted: "a memory-bandwidth-bound workload
gives up very little to a reduced core power limit while shedding real heat."
The limit was never the thing holding this pipeline back. Per-stage warm means
move by under a second each way except TTS (63.7 → 60.6 s).

### Where the warm wall goes (mean of the two warm repeats, 113.3 s, RTF 0.377)

| stage | wall s | share | what the probe says inside it |
| --- | ---: | ---: | --- |
| tts | 60.6 | 53 % | decode 52.7-56.7 s, **45.4-45.7 ms/step**, prefill 3.7-4.4, codec 1.0-1.3 |
| translate | 16.0 | 14 % | 80 cloud calls, 1.14 s mean, no GPU work at all |
| asr | 15.2 | 13 % | **probe 4.4-4.6 s (29 %)**, generate 10.0, align 0.5, tagging 0.02 |
| separate | 8.4 | 7 % | demucs 5.4, VAD infer 1.3, stem writes 0.2, ffmpeg extract 0.36 |
| output | 5.4 | 5 % | **ffmpeg 5.3** — the AAC re-encode of the dub is the whole stage |
| mux | 4.1 | 4 % | **rubberband 3.3-3.6 s over 97-104 spawns**, reads 0.02, write 0.10 |
| emotion | 1.9 | 2 % | 101-104 per-turn forwards in 1.3-1.4 s, ref-picking 0.25 |
| diarize | 1.3 | 1 % | |
| ingest | 0.4 | 0 % | |

Beyond the stage table: `other_s` (real wall minus the sum of stage walls) is
**1.9 s, 1.7 % of wall** — nine accelerator-cache frees, nine manifest
rewrites, artifact recording. CPU time is 108-113 s of a 113-118 s wall, of
which **11.4 s is child processes** (ffmpeg + rubberband). Peak VRAM 16.41 GB
allocated / 16.98 reserved, 7.88 GB free. Peak RSS 3.9 GB on a warm repeat and
10.6 GB on the cold one, where the model loads happen.

### The three facts that change the plan

1. **The TTS decode step is 45.4-45.7 ms and the GPU is mostly idle inside
   it.** `util_pct` averages 45 % across a run whose largest stage is a dense
   4B decode, and the driver's own idle bit is set in 23-69 % of samples. Both
   corroborate the planning finding that the step is bound by CPU launch
   overhead, not by the 936 GB/s the card can read. This is the headline.
2. **The ASR language probe costs 4.4-4.6 s of a 15.2 s stage.** It is a full
   28-second transcription at `max_new_tokens=256` whose text is discarded;
   only `probe[0]["language"]` is read.
3. **Two plan items shrank on contact with data.** Inter-stage overhead is
   1.9 s, so gating the per-stage accelerator free (plan Task 8) has a ceiling
   under 2 % and drops down the ranking. And `live_row_fraction` is 0.64-0.67,
   above the ~0.60 threshold the plan set for compacting retired rows out of a
   decode batch, so that stays deferred — a third of row-steps are wasted, but
   not enough to justify invalidating a captured graph.

Sentence counts differ per repeat (77 / 81 / 80) with diarization turns at
100 / 107 / 111. That is the known Sortformer non-determinism recorded in
quality issue #16, not an effect of anything measured here, and it is why
per-stage numbers are quoted as ranges.

### A harness bug found by taking the baseline

The first attempt at the watch-along baseline **hung**. The greedy run produced
every section for the clip and then sat in a 2 s idle loop for five minutes
until it was killed.

`bench_watch_along` passed neither `idle_fn` nor `alive_fn` to
`StreamScheduler`, so `_default_idle` fell back to the playhead's `updated_at`
— which `BenchPlayhead` refreshes on every call, deliberately, to keep the
abandon timer fed. The abandon branch could therefore never fire, and once
coverage was complete `_next_start` returned None forever while the idle
predicate kept answering "keep waiting".

So **`patmir bench --mode watch-along` had never terminated under its own
default driver**, and that is why every stored watch-along baseline in this
repo was taken with `--forward-loop`. The production path was never affected:
the server passes `alive_fn` keyed on the client still polling, which is the
signal that separates a paused viewer from a departed one (quality issue #21).

Fixed in `bde9e2d` with 6 tests, including a negative control that asserts the
old predicate does idle on a fully-covered video. The bench now ends when
nothing in the video is left to dub, asking *coverage* rather than treating
`_next_start`'s None as the end — because under `--playhead realtime` that None
also means "buffered more than `lookahead_seconds` ahead of the viewer", where
the right move is to keep waiting.

Two reporting bugs were corrected alongside, both of which **overstated** a
finding rather than hiding one:

- the thermal line printed the sample TOTAL where it meant the reason's count,
  so a single 1-second transient rendered as "THERMAL SLOWDOWN in 150 samples";
- peak RSS was still labelled "process lifetime" after becoming a per-repeat
  sampled figure. Both the sampled peak and the process high-water mark are now
  published, since a 1 Hz sampler can miss an allocation spike and `ru_maxrss`
  cannot.

And one gap: the memory line sat behind the same two-repeat guard as the
cold/warm comparison, so `--repeat 1` reported no VRAM at all.

## Watch-along at 350 W (2026-09-02)

`--mode watch-along --prewarm`, scheduler driver, greedy playhead
→ `baseline-350w-watch-greedy.json`, prelude job `20260902-223546`.

| | value |
| --- | --- |
| aggregate RTF | **0.623** per dubbed second, **0.683** per source second |
| wall | 205.8 s for 330.1 s of dub over 10 sections |
| **time to first audio** | **27.8 s** (with prewarm; the prewarm itself cost 30.7 s before the timer) |
| stalls | 0 — but greedy cannot stall by construction; see the realtime run |
| section RTF | 0.43-1.05; **one section (idx 7) above realtime** |
| `other_s` | 20.95 s total, ~2.1 s per section (ffmpeg slice/trim, ECAPA, age-gender) |
| peak VRAM | 15.52 GB |

**Not comparable to the 2026-09-01 watch figure without care.** That run had no
`--prewarm` and used the `--forward-loop` driver (it had to: the scheduler
driver could not terminate, see above). So its 45.4 s to first audio and its
0.682/0.748 RTF describe a different configuration. What can be said is that
this configuration — the one the live server actually runs, prewarmed at boot —
reaches first audio in 27.8 s and produces at 0.62 of realtime.

### The finding that changes the ranking

**Translate is the second-largest stage in watch-along, at 46.6 s of a 205.8 s
wall (23 %)** — against 14 % in batch.

**Correction (2026-09-02, after checking the per-section traces).** My first
reading of this was wrong. I attributed it to `translate_batch` walking its
chunks serially with no `translate_concurrency`, and concluded that
parallelizing chunks was the second-best lever here. The traces say otherwise:
sections carry 3-16 sentences against `chunk_size = 20`, so **every section is
a single chunk and makes exactly one call**. There is no serial chunk walk to
parallelize. The cost is one structured, JSON-schema-constrained round-trip per
section at 1.9-3.8 s.

Two consequences. Chunk concurrency would buy nothing here, so it is not
implemented. And the remaining lever is **overlapping a section's translate
with the previous section's TTS** — cross-stage pipelining, which this spec
deliberately excludes. Note also that the serial walk is not a bug even where
it does run: each chunk's prompt carries the previous chunk's translated
`overlap` pairs, which is the cross-chunk coherence the batched candidate
exists to provide. Parallelizing it would trade translation quality for
latency, not win for free.

Translate is 23 % of watch-along wall mostly because everything around it got
faster.

Per-stage, watch-along: tts 81.4 s (40 %), translate 46.6 (23 %), asr 29.1
(14 %), separate 10.9 (5 %), the rest under 3 s each. ASR's language probe is
paid **once**, in the prelude (4.15 s total across 11 job dirs), which confirms
the `source_lang` pass-through into sections is working.

### A probe-aggregation bug in this report

`_aggregate_probes` sums every numeric field across the prelude and all
sections. That is right for spans and counters and wrong for identities and
derived ratios, so the greedy report shows `max_batch 88` (8 x 11 job dirs),
`ms_per_step_decode 476.83` and `live_row_fraction 7.03`. The underlying spans
are correct — `decode_s 82.33` over `steps 1866` is a true **44.1 ms/step**,
consistent with batch's 45.4-45.7 — and every per-section probe file is intact
on disk, so the aggregate is recoverable without re-running. Fixed below.

## After the optimizations (2026-09-02)

Batch, warm, `sample_1`, cuda profile. Each row is the change on top of the one
above it, measured separately so nothing is attributed to the wrong lever.

| step | ms/step | tts | asr | translate | wall | agg RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline at 350 W | 45.55 | 60.55 | 15.24 | 16.02 | 113.3 | 0.377 |
| decode sync/sampler/state-machine | 45.88 | 61.38 | — | — | — | 0.377 |
| **+ CUDA graph** | 27.15 | 39.51 | — | — | — | 0.309 |
| **+ gqa attention** | 17.33 | 29.27 | 15.97 | 15.99 | 83.4 | 0.277 |
| **+ ASR language probe** | — | 27.80 | **11.32** | 16.53 | 77.5 | 0.257 |
| **+ translate concurrency 12** | — | — | — | **8.22** | **70.5** | **0.234** |

**38 % off the pipeline.** The plan targeted 20-25 %.

### What each lever actually cost or bought

- **Sync removal, batched sampling, vectorized state machine: 0 ms.** Kept
  because CUDA graph capture requires exactly what they provide. Recorded in
  full rather than quietly folded into the next commit, because the reason it
  bought nothing is the useful part: a 36-layer model with negligible widths
  already costs 42.9 ms/step here, so the step was ~93 % kernel-launch cost and
  the syncs were never the constraint.
- **CUDA graph: 45.55 -> 27.15 ms/step, byte-identical.** The plan classified
  this as a numerics change needing an ear-test; measurement said otherwise
  (`StaticCache` ≡ `DynamicCache`, graph replay ≡ eager stepping, both verified
  bitwise with ragged left-padded prompts). It shipped with no quality question
  attached.
- **gqa attention: -> 17.33 ms/step, re-rolls every line.** Ear-tested and
  approved (`runs/20260902-230429-sample_1`), enabled in `patmir.toml` so the
  cache key moves (2cc3dd87 -> 4a56ed45) and cached dubs regenerate once.
- **ASR probe: 4.63 -> 0.71 s, 0 fallbacks.** It was transcribing up to 28 s at
  a 256-token budget to read a tag that precedes the transcription.
- **Translate concurrency 6 -> 12: 16.53 -> 8.22 s.** Mean per-call latency
  unchanged (1196.6 -> 1197.8 ms), so nothing was rate limited.

### Ordering was load-bearing, and not obvious

CPU launch (~43 ms) and GPU execution (~33 ms) **pipeline**, so the step sits at
the max of the two. Swapping the attention kernel first would therefore have
measured ~0, exactly as the sync removal did — it lowers a floor that is not
binding. The graph had to land first to expose the GPU floor. Any future work
here should check which of the two is binding before optimizing either.

### The largest remaining TTS inefficiency is now graph capture

At `max_batch = 8` capture costs **7.34 s of a 29.14 s TTS stage — 25 %**, or
~0.73 s per chunk, three times the original estimate. It still pays for itself
(it buys ~28 ms on each of ~1160 steps) but it is now the biggest single piece
of waste in the stage. Reusing a captured graph across chunks with matching
`(B, seq + max_cap)` would recover most of it; a larger `max_batch` reduces it
for free by cutting the chunk count.

### `max_batch` sweep (2026-09-02)

Only meaningful after the graph: while the step was launch-bound, batch width
barely mattered. Warm repeat each, `sample_1`, with the graph and gqa kernel.

| B | ms/step | tts wall | capture | chunks | steps | live rows | ovf | VRAM | agg RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 17.19 | 29.14 s | 7.34 s | 10 | 1160 | 0.647 | 0 | 16.50 GB | 0.234 |
| **12** | 18.11 | **23.84 s** | 5.96 s | 7 | 888 | 0.592 | 0 | 16.62 GB | **0.218** |
| 16 | 19.14 | 27.03 s | 5.31 s | 6 | 1032 | 0.468 | **1** | 17.38 GB | 0.228 |

**12 wins and 16 loses**, which is the informative part. Per-step cost rises
monotonically with B (more KV to read, more attention work), so every gain has
to come from needing fewer chunks. But **a chunk costs `max(steps)` over its
rows**, so widening the batch also widens the token-cap spread inside each
chunk. At 16 that dominates: total steps went *up* (888 -> 1032) despite one
fewer chunk, more than half of all row-steps were spent on already-retired rows
(live 0.468 vs 0.647 at B=8), and it produced the only TTS overflow measured
anywhere in this spec.

VRAM was never the binding constraint — 16.5-17.4 GB peak with ~7.5 GB free
throughout, against the 21 GB budget. The plan expected memory to decide this;
chunk shape decided it instead.

Caveat: one warm repeat per setting, and Sortformer non-determinism gave
79/81/80 sentences across the three, so step counts are not perfectly
comparable. The 12-vs-16 gap (23.84 vs 27.03 s) is far larger than that noise;
a reader wanting to split 8 from 12 more finely should repeat both.

`max_batch = 12` set. It changes the audio (batching is not bitwise-invariant
across shapes), so `config_hash` moves **4a56ed45 -> 3f421293**. Sequenced
deliberately *before* any re-dub of the library: the gqa change had already
moved the key, so folding this in now costs one regeneration instead of two.

Ear check available at `runs/20260902-233655-sample_1/out.mkv` (the mb12 warm run) against
`runs/20260902-230429-sample_1/out.mkv` (the approved gqa rendering at B=8).
