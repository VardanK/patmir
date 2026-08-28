# 2026-08-28 — Overlap a section's dub with the next section's analysis; and what the decode step is actually bound by

**Status:** landed. **Goal not met** — measured 0.616 → 0.552 (1.12×) against a target
of 0.44. Two verifications outstanding: an ear verdict on `max_batch=12` (it redraws
every line's audio) and a `--playhead realtime` stall count (the pipeline worsens
per-section latency variance, which greedy-playhead benchmarking cannot see).

**Goal:** watch-along steady-state RTF 0.877 → ≤ 0.44 on `samples/sample_1.mkv`,
cuda profile (RTX 3090 / WSL2).

## The measurement that reordered the plan

The plan opened with the author's hypothesis — that the 3090 is an fp16 card and we
are leaving 2× on the table by running bf16, with q8 as the analogous win where fp8
would be. Both were tested and both are dead, for the same reason, and the reason is
worth stating once because it disposes of a whole family of ideas:

> For a weight-dominated GEMM `[M,K] @ [K,N]`, arithmetic intensity **equals the batch
> size M**. This card's ridge point is ~78 FLOP/byte. We run M=8. We are ~10× below the
> ridge, so the decode is memory-bound and peak TOPS does not enter the time equation
> at all.

Measured at the model's true MLP shape (`[8,2560]@[2560,9728]`), bf16 already achieves
**784 of 807 GB/s — 97% of roof** — and fp16 with fp16-accumulate is **0.93×, i.e.
slower**. A positive control at 8192³ confirms the fp16 datapath is real and worth
1.60× when compute-bound, and a batch sweep confirms the crossover lands exactly where
the roofline says (ratio ≈1.0 for B=8→64, 1.48× at B=128). fp8 is unavailable on SM 8.6
regardless. So: **not a dtype problem.** (`measurements/TASK2-RESULTS.md`.)

A correction to the record, since it was published in this spec: an earlier note here
claimed fp16 and bf16 "run at the same rate on GA102". That was wrong as stated —
PyTorch defaults fp16 matmul to fp32 accumulate, and the second Ampere datapath needs
`allow_fp16_accumulation=True`. The conclusion survives; the reasoning that reached it
did not.

## What the decode step is bound by — the attribution that was wrong

Task 2 split a 43.13 ms decode step (B=8, kv=300) into 9.06 ms of weight bandwidth,
4.73 ms of sampler, and a **29.34 ms residual it attributed to kernel launch overhead**.
That residual was inferred, not measured — it was the unexplained remainder, with
"~2800 launches at 10.5 µs" fitted to it afterwards.

Task 3 tested it the only way that settles it: a CUDA graph, which eliminates launch
overhead entirely.

| config | eager | graph | speedup |
| --- | --: | --: | --: |
| kv=300, static cap=1408 | 49.26 ms | 41.89 ms | **1.18×** |
| kv=1200, static cap=1408 | 50.46 ms | 42.69 ms | **1.18×** |

**1.18×, against the ~4× the attribution predicts.** With every launch gone the step
still needs ~42 ms to move ~9 GB — about **214 GB/s, 26% of roof**. The residual is not
dispatch overhead; the kernels themselves run far below roof at seq-1 shapes. Graph
capture is therefore dropped: 1.18× does not pay for bucketed static caches, pinned
input buffers and a re-capture on every shape change. `StaticCache` was independently
neutral (0.98× at kv=300, 1.06× at kv=1200) and is only involved because capture
demands static shapes.

The same fact seen from the other side: step time is **flat from B=1 to B=24**
(45–49 ms) while per-sentence cost falls linearly. The decode leaves most of the
machine idle at every batch size we can reach.

**Method note.** The first multi-arm graph probe **segfaulted inside `g.replay()`**.
Capture and replay were then verified sound in isolation at four levels — plain `mm`,
`sdpa` with a mask, full `model.model` with `StaticCache`, and after a DynamicCache run
with and without `empty_cache()` — 60 replays each, all clean. The crash is
contamination *between* configs in one process. Every GPU probe in this spec now runs
one config per child process, and future ones here should too rather than trust a
single process to survive repeated model reconfiguration.

## Decision 1 — `max_batch` 8 → 12 (measured, not extrapolated)

The microbenchmark says B=24 is 3.1× cheaper per sentence. The end-to-end measurement
says otherwise, and the end-to-end measurement governs. Driving the real
`HiggsV3Torch.synthesize_batch` over a real watch-along section (12 translated
sentences, real pinned refs):

| max_batch | TTS wall | chunk shape | peak GB | overflows |
| --: | --: | --: | --: | --: |
| 8 (shipped) | 24.16 s | 6+6 | 10.17 | 0 |
| **12** | **18.08 s** | 12 | 11.38 | 0 |
| 16 | 20.41 s | 12 | 11.38 | 0 |
| 24 | 20.44 s | 12 | 11.38 | 0 |

**1.34×, not 3.1×**: a section has only ~12 sentences, so at `max_batch=12` it is
already one chunk and the microbenchmark's remaining headroom has nothing to batch.
12, 16 and 24 produce the identical single chunk — the spread between them is
run-to-run variance (independently observed at 7–13% on repeated identical configs),
not an ordering. Cost is **+1.21 GB peak**.

**This changes every line's audio, and that is the one part of this work that needs an
ear check.** Per ADR `2026-08-24-tts-sampling-reproducibility` the decode is
reproducible but not batch-*invariant*: chunk membership changes the left-padded
prefill shape, and at temperature 0.8 a one-bit logit difference flips a multinomial
draw. Measured: 12 of 12 lines redrawn. Nothing about the change is directional — it is
a different draw from the same distribution, not a worse one — but the author has
ear-tested the current output and this replaces all of it.

Not taken from Task 3: removing the per-row `.cpu()` sync (~4% of a step whose real
problem is elsewhere, still bitwise-safe and still worth doing later), and vectorizing
the sampler — its attractive 0.17 ms figure was measured with ONE global `multinomial`,
which violates the per-row-generator requirement that ADR exists to enforce.

## Decision 2 — the scheduler pipelines analysis against dub

`StreamScheduler.run()` was strictly serial: `analyze(N)` then `dub(N)` then
`analyze(N+1)`. On the stored baseline the two halves are **77.20 s of analysis against
140.48 s of dub** over nine steady-state sections, so overlapping them makes the loop
cost `max(analyze, dub)` per section instead of their sum — and the dub dominates, so
the analysis becomes free.

`process_section` is split into `analyze_section` → `SectionAnalysis` → `dub_section`,
with `process_section` retained as their composition (the CLI, bench's serial control
and every existing test drive it). The scheduler pipelines only when handed both
halves; handed neither, it runs the old loop.

Three properties the split has to preserve, each pinned by test:

* **Order.** One dub worker, and section N's dub is emitted before N+1's is submitted.
  Two entries sharing an index is exactly the corruption quality issue #22 fixed.
* **Isolation.** `SectionAnalysis` carries registry **snapshots** taken on the scheduler
  thread. A running dub must not observe the next section's `_resolve_identities`
  half-applied — that would make its voice assignments depend on how far the next
  analysis had got. The two halves also use **disjoint stages** (separate/diarize/
  emotion/asr vs translate/tts/mux), so they share no instance out of the warm
  `stage_cache`.
* **Nothing dropped.** The in-flight dub is drained before idling and in `finally`.
  Without that, the last section of every stream is lost — it is already written to the
  cache, which is the free-rewatch store.

The load-bearing test is `test_dub_overlaps_next_analysis`, whose **negative control**
asserts the serial path does *not* overlap. Without that control the overlap test passes
just as well against a scheduler that never pipelined anything.

Two things fixed en route, both pre-existing:

* `dub_section` now takes the `cancel` token. The serial version never threaded one into
  the second `run_job`, so a Stop could not interrupt translate/tts/mux. Tolerable when
  the dub was the last thing in a section; with the dub now running under the next
  section's analysis, an uncancellable dub would hold the stream open past a Stop.
* **`patmir bench --mode watch-along` never terminated under the scheduler driver.**
  `BenchPlayhead` deliberately keeps `updated_at` fresh — a synthetic viewer never leaves
  — so the scheduler's abandon timer can never fire, and the loop spun at 2 s a tick
  forever once the last section landed, never writing the report. Every stored scheduler
  run used `--max-sections`, which stops through a different path, so a full-length
  scheduler run had never reached this state. This was hit on the first full run of this
  work and cost one benchmark.

### The bench had to change to be able to see this at all

`sections_wall_s` is the **sum of per-section walls**, and under the pipeline those
walls overlap — a section's analysis runs at the same wall-clock time as the previous
section's dub, so the sum counts that time twice and would report no improvement
however well the overlap worked. Bench now measures real elapsed seconds around
`sched.run()` and reports `sections_elapsed_s` and `pipeline_overlap_s` beside the sum,
which stays as the right denominator for per-stage attribution. `--serial-sections`
runs the pre-change behaviour as a same-code control, because every stored baseline was
produced by an older bench (forward-loop; none of them record a `driver`) and is not a
clean control for this.

## Results — and the goal is NOT met

Two arms, same code, same session, idle card, `samples/sample_1.mkv`, cuda profile,
scheduler driver, greedy playhead. Steady state = sections 1..9, excluding the prelude
and its dub, which are identical either way. Reports in
`agent-os/specs/2026-08-28-0202-watchalong-2x-cuda/measurements/`.

| | serial control | pipelined |
| --- | --: | --: |
| steady elapsed | 182.52 s | **158.48 s** |
| steady audio | 296.5 s | 287.3 s |
| **steady RTF** | **0.616** | **0.552** |
| sum of per-section walls | 182.52 s | 300.33 s |
| peak VRAM | 15.61 GB | 15.60 GB |
| time to first audio | 163.5 s | 165.9 s |

**0.616 → 0.552, i.e. 1.12×.** The goal was ≤ 0.44 and this does not reach it.

### Why the overlap buys 12% and not 45%

The per-section walls tell the story. Overlapped, the *same work* takes far longer per
section:

| | serial | pipelined |
| --- | --: | --: |
| asr (steady sum) | 31.08 s | 64.97 s |
| tts (steady sum) | 82.92 s | 113.61 s |
| analysis half (sum) | 47.13 s | 82.25 s |
| dub half (sum) | 115.17 s | 143.33 s |

ASR takes **2.09×** longer and TTS **1.37×** longer when they run concurrently. Both
halves are GPU-bound, and two threads issuing CUDA work on one device time-slice rather
than parallelize — the card is not idle waiting for them. So the overlap can only ever
recover the genuinely non-GPU work: the cloud translate call (28.58 s), the ffmpeg
slice/trim, ECAPA and the age/gender profile. Total work rose from 162.3 s to 225.6 s
while elapsed fell 24 s. It is not a memory problem — peak VRAM is unchanged at 15.6 GB
with 7.8 GB free.

This is the same fact Task 3 found from the other end: the decode already runs at ~20%
of bandwidth roof, so "the GPU has spare capacity" is true of *bandwidth* and false of
*scheduling*.

### The cost that is not in the RTF

Pipelining trades throughput for **per-section latency variance**, which is the thing
watch-along actually cares about at the buffer edge:

* serial: worst section 31.19 s of wall for 36.5 s of audio (0.85 RTF)
* pipelined: worst section 44.14 s of wall for 36.5 s of audio (**1.21 RTF** — the
  buffer shrinks during that section)

Aggregate production still outruns playback comfortably (0.55×), so a buffer a minute
deep absorbs this. But **it has not been measured against a real viewer**: both arms ran
`--playhead greedy`, which measures throughput and cannot stall by construction. Before
this is called done it wants a `--playhead realtime` run on both arms, which counts the
freezes a viewer would actually see. That is the one verification this ADR is missing.

### Where the rest of the 2× would have to come from

Steady-state work is now tts 82.92 s (45%), asr 31.08 s (17%), translate 28.58 s (16%),
separate 10.06 s (6%), unowned ~20 s (11%). Since concurrency does not help GPU-bound
stages, the remaining levers are all "make a stage do less work", not "run stages at the
same time": q8 weight-only on the 4B decode (halves the bytes the decode is bound by —
the one lever the roofline still endorses), and dropping the forced aligner in
watch-along.

### A stored-baseline caveat

`runs/bench/sample_1-cuda-watch-fixed.json` reads 0.877 steady-state, and this work's
serial control reads 0.616 — before the pipeline, and with only `max_batch` changed.
`max_batch` explains part of it (TTS 0.360 → 0.280 s per audio-second, 1.29×, matching
the isolated 1.34×) but **not ASR, which is 2.1× faster per audio-second than that
baseline for reasons nothing here changed.** The stored baselines also record no
`driver`, so they predate the scheduler-driven bench entirely. Treat 0.877 as
uncomparable and the same-code serial control as the only valid control — which is why
`--serial-sections` now exists.

## Revisit

* An ear verdict on `max_batch=12` rejecting the redrawn lines — revert is one number.
* A second GPU worker, or a card where two pipelines fit: the accelerator lock
  (`2026-08-25-one-accelerator-at-a-time`) serializes streams against batch jobs but
  not the two halves *within* a stream, which is what makes this legal today. A future
  change that widens the overlap to two dubs would breach the residency budget that
  lock exists to protect.
* A release making the batched decode bitwise-stable across shapes, which would remove
  the ear-check cost from every future `max_batch` change.
* Watch-along sections growing past ~12 sentences (a longer `section_seconds`, or
  denser dialogue), which would make `max_batch` above 12 start paying again.
