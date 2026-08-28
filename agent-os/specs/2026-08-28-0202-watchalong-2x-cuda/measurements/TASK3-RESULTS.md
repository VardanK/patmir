# Task 3 results — and the Task 2 attribution it overturned

All measured on an idle RTX 3090 (WSL2), bf16, `samples/sample_1.mkv`-derived data.
Every probe refuses to run above 2 GB of resident VRAM (`require_idle_card`), because
a second CUDA process drops HBM 807 → 295 GB/s and would make every number here a
third of the truth.

## What Task 2 claimed, and what was wrong with it

Task 2 attributed one decode step (B=8, kv=300, 43.13 ms) as:

| term | ms | share |
| --- | --: | --: |
| weight bandwidth (floor) | 9.06 | 21% |
| sampler loop (as shipped) | 4.73 | 11% |
| **residual — assumed kernel launch/dispatch** | **29.34** | **68%** |

The residual was *inferred*, not measured: it was whatever the other two did not
explain, and "~2800 launches at 10.5 µs" was a plausible story fitted to it. Task 3
tested it the only way that settles it — by removing launches entirely.

## 3(b) CUDA graph capture: the attribution was wrong

`probe-tts-cudagraph.py`, one process per config (see below):

| config | eager ms | graph ms | × floor | speedup |
| --- | --: | --: | --: | --: |
| kv=300 DynamicCache (ships today) | 47.38 | — | 5.2 | — |
| kv=1200 DynamicCache | 49.90 | — | 5.5 | — |
| kv=300 static cap=1408 | 49.26 | 41.89 | 4.6 | 1.18× |
| kv=1200 static cap=1408 | 50.46 | 42.69 | 4.7 | 1.18× |

**1.18×, not the ~4× the attribution predicts.** With every kernel launch eliminated
the step still needs ~42 ms to move ~9 GB — about **214 GB/s, 26% of the 807 GB/s
roof**. So the residual is not dispatch overhead; the kernels themselves are running
far below bandwidth roof at these shapes. `StaticCache` was already known to be
neutral-to-slightly-worse (0.98× at kv=300, 1.06× at kv=1200), and it is only used
here because capture requires static shapes.

Graph capture is therefore **not worth its complexity**: 1.18× on the forward, in
exchange for bucketed static caches, pinned input buffers, and a capture step that
has to be re-done whenever a shape changes.

### A process-isolation finding worth keeping

The first version of this probe ran every config in one process and **segfaulted
inside `g.replay()`**. Capture and replay were then verified sound in isolation at
four levels — plain `mm`, `sdpa` with a mask, full `model.model` with `StaticCache`,
and after a DynamicCache run with and without `empty_cache()` — all fine, 60 replays
each. So the crash is contamination *between* configs in one process, not a property
of CUDA graphs on this box. The probe now runs one config per child process. Any
future multi-arm GPU probe here should do the same rather than trust a single
process to stay clean across model reconfigurations.

## The real finding: the step is flat in batch size

`probe-tts-batch-scaling.py`, kv=300, one process per arm:

| B | step ms | per-sentence ms | GB moved | GB/s | % roof |
| --: | --: | --: | --: | --: | --: |
| 1 | 45.16 | 45.16 | 7.35 | 163 | 20% |
| 2 | 46.22 | 23.11 | 7.40 | 160 | 20% |
| 4 | 49.54 | 12.38 | 7.49 | 151 | 19% |
| 8 | 48.28 | **6.03** | 7.66 | 159 | 20% |
| 16 | 46.31 | 2.89 | 8.02 | 173 | 21% |
| 24 | 46.72 | 1.95 | 8.37 | 179 | 22% |
| 32 | 53.53 | 1.67 | 8.72 | 163 | 20% |

Step time is **flat from B=1 to B=24** while per-sentence cost falls linearly: B=24 is
3.10× cheaper per sentence than the shipped B=8. Achieved bandwidth stays at ~20% of
roof throughout, which is the same fact as the graph result seen from the other side —
the decode leaves most of the machine idle at every batch size we can reach.

(The `gemm GB/s` control column in that probe reads 37–106 GB/s against a standalone
`torch.mm` at the same shape measuring 782 GB/s. That control is mismeasured — the
model is faster than its own control, which is impossible — so it is **not** used for
any conclusion here. The `step_ms` column is the load-bearing one and it reproduces
the independent Task 2 figures at B=8 within 12%.)

## But a microbenchmark is not a plan

`probe-tts-maxbatch-e2e.py` drives the real `HiggsV3Torch.synthesize_batch` over a
real watch-along section (`runs/20260825-182659-section`, 12 translated sentences,
real pinned refs):

| max_batch | wall s | rtf | chunk shape | peak GB | overflows | lines changed |
| --: | --: | --: | --: | --: | --: | --: |
| 8 (ships today) | 24.16 | 0.592 | 6+6 | 10.17 | 0 | — |
| **12** | **18.08** | **0.439** | 12 | 11.38 | 0 | 12/12 |
| 16 | 20.41 | 0.496 | 12 | 11.38 | 0 | 12/12 |
| 24 | 20.44 | 0.496 | 12 | 11.38 | 0 | 12/12 |

**1.18–1.34× on the TTS stage, not 3.1×.** The microbenchmark's headroom is real but
unreachable here: a watch-along section only has ~12 sentences, so once `max_batch`
reaches 12 the section is one chunk and there is nothing left to batch. 12, 16 and 24
all produce the identical single chunk — the 18.08 / 20.41 / 20.44 spread across
those three is run-to-run variance (independently observed at 7–13% on repeated
identical configs), not a real ordering.

Cost: **+1.21 GB peak**, 0 overflows at every arm.

**All 12 lines change their audio.** That is expected and is not a regression: per ADR
`2026-08-24-tts-sampling-reproducibility` the decode is reproducible but *not*
batch-invariant, because batched matmuls and the left-padded prefill are not
bitwise-stable across shapes and at temperature 0.8 a one-bit logit difference flips a
multinomial draw. Changing `max_batch` changes which sentences share a chunk, so every
line is redrawn. **This needs an ear check before it ships**, and it is the only part
of Task 3 that does.

## Consequences for the plan

* **3(b) graph capture — dropped.** 1.18×, high complexity, and it disproves its own
  premise.
* **3(a) removing the per-row `.cpu()` — deprioritised.** It was sized at 1.79 ms of a
  43.13 ms step from the sampler measurement, i.e. ~4%, and the sampler is 11% of a
  step whose real problem is elsewhere. Still bitwise-safe and still worth doing, but
  it is not a lever.
* **3(c) sampler vectorization — not worth it.** Ceiling 4.56 ms/step (10.6%), and the
  0.17 ms figure that made it look attractive was measured with ONE global
  `multinomial`, which violates the per-row-generator requirement of ADR
  `2026-08-24-tts-sampling-reproducibility`. Keeping per-row streams costs most of the
  saving.
* **`max_batch` 8 → 12 — the one thing worth taking from Task 3.** ~1.3× on TTS,
  +1.21 GB, needs an ear check.
* **The 2× has to come from Task 4**, the scheduler pipeline, which changes no numerics
  at all.
