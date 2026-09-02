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
wall (23 %)** — against 14 % in batch. The reason is in the code, not the
network: watch-along forces `openrouter_gemini_flash_batch`, and
`translate_batch` walks its chunks **serially** with no `translate_concurrency`
at all, while the per-sentence path used by batch fires 6 at a time. So the
mode with the tighter latency budget uses the slower translator. That raises
plan Task 7's watch-along half from a nicety to the second-best lever here.

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
