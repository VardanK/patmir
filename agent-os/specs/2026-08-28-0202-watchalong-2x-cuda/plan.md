# Watch-along 2× on WSL2 + RTX 3090

## Context

Find the bottleneck in watch-along (near-realtime) dubbing on the WSL2 + RTX 3090 box and cut
wall time 2×. The stated hypothesis was that we are leaving speed on the table by running bf16
where the 3090 prefers fp16, and q8 where fp8 would do. A prior agent had produced a lever list
that rejected fp16 outright.

**Three findings, all measured on the box, and they reorder everything:**

1. **The fp16 hardware claim is real.** The 3090 has a second tensor-core datapath — fp16 with
   **fp16 accumulate** — that PyTorch does not use by default. On the exact GEMM shape our TTS
   decode runs, it is **1.46× faster** than bf16. Both the prior agent and my own first
   benchmark "disproved" this by measuring the default fp32-accumulate path, where fp16 and bf16
   are genuinely identical.
2. **It does not survive contact with the real models.** On Qwen3-ASR with token count controlled
   and transcripts byte-identical, fp16+accumulate is worth **1.06×** against a 1.3% noise floor.
   Reason: ASR's decode runs at **37.3 ms/step against a 4.21 ms/step bandwidth floor — 8.8× off,
   so ~89% of the stage is not arithmetic at all.** The GEMM is ~10% of the wall, so a 1.46× GEMM
   buys 6%. Amdahl, exactly.
3. **So the bottleneck is per-step host overhead, not precision** — which is precisely the class
   the prior list dismissed as "single-digit-%". That, plus the fact that the scheduler is
   strictly serial, is where a 2× actually lives.

**fp8 is impossible** on this card: `torch._scaled_mm` requires compute capability ≥ 8.9, the 3090
is 8.6. **q8 takes the fp8 slot** — it is the Ampere-appropriate way to buy the same thing (fewer
bytes per weight) and it is the best quality-validated option we have, since q8 is already the
active ear-approved TTS on the Mac path.

**fp16 and q8 are independent levers on different terms of the same step, and they stack:** fp16
raises *achieved* bandwidth (527 → 769 GB/s, a kernel-efficiency win), q8 halves the *bytes*
(8 GB → 4 GB, lowering the floor itself). Neither replaces the other. And both are gated on the
same unmeasured number — how far the 3090's decode step sits above its 9.91 ms bandwidth floor.
**If decode is at the roof, both pay off; if it is overhead-dominated, neither pays until the
overhead is removed.** That single measurement is Task 2.

---

## Measurements taken for this plan (RTX 3090, SM 8.6, torch 2.11.0+cu128, WSL2)

All figures below are from an **idle card** (< 1.4 GB VRAM in use), re-measured after an earlier
pass turned out to have run with the patmir server resident. That accident is itself a finding:
with 21.8 GB held, HBM read as **295 GB/s** instead of 807, kernel launch **44.7 µs** instead of
10.5, and `_int_mm` **24 TOPS** instead of 59 — while **every dtype ratio survived** (fp16+accum
1.41× vs 1.46×). Ratios are robust to contention; absolute roofline figures are not, and it is the
absolute per-step floor that decides whether q8 and fp16 are worth anything. `probe-hardware.py`
now refuses to run above 2 GB of resident VRAM rather than emit a third of the real bandwidth. The
same caution applies to comparing against `runs/bench/*.json`. Scripts and raw output:
`measurements/`.

### Isolated GEMM — the accumulate finding

Interleaved A/B/A, 5 rounds × 300 iters, tight min/max. Shape is the Qwen3-4B MLP projection at
the decode batch:

```
decode GEMM [8,4096] @ [4096,14336]    weight 117 MB    HBM roof 0.147 ms
  bf16 default                 0.223 ms    527 GB/s    1.00x
  bf16 + reduced_reduction     0.216 ms    544 GB/s    1.03x   <- no bf16 route to the win
  fp16 default (fp32 accum)    0.220 ms    534 GB/s    1.01x   <- the trap: reads as "no gain"
  fp16 + allow_fp16_accum      0.153 ms    769 GB/s    1.46x   <- 95% of the card's roof
```

Mechanism is **not** the datasheet's FLOPS line: at B=8 this GEMM runs at 16 flops/byte and
FLOP-bound would need ~79, so the 142-vs-71 TFLOPS figure is not what binds. What binds is the
accumulator — fp32 accumulators double the register/shared-memory budget and cut occupancy on a
thin GEMM, so that kernel reaches only 527-544 GB/s of the card's 807 while the fp16-accumulate
kernel reaches 769. On genuinely compute-bound shapes (B≥512) it does show as FLOPs: 1.55-1.62×.
Requires **both** `dtype="float16"` and `allow_fp16_accumulation=True`; either alone gives nothing.

### Real Qwen3-ASR-1.7B — controlled, and it refutes the projection

8 × 28 s chunks (matching `chunk_seconds=28` / `batch_size=8`), 382 prompt tokens, **62 generated
tokens in every arm**, driven through the candidate's own `apply_transcription_request`:

```
bf16           2309.7 ms   1.000x
fp16_f16acc    2176.8 ms   1.061x
bf16_again     2218.7 ms   1.041x   <- A/B/A control: noise floor 4.1%
transcripts byte-identical across all three arms
```

**37.3 ms/step measured against a 4.21 ms/step weight-read floor (1.7B × 2 B / 807 GB/s) = 8.8×
off the floor, ~89% overhead.** At 10.5 µs/launch that implies ~3500 kernel launches per step.
The 1.061× reproduced *exactly* on both a contended and an idle card, so the effect is real and
small; the contended run read 41.9 ms/step / 9.8× / 90%, i.e. the same conclusion.

### Real Higgs 4B, full chunk — too noisy to conclude anything

```
bf16          7288.5 ms      fp16_f16acc  8166.6 ms (0.89x)      bf16_again  7973.1 ms (0.91x)
```

**The bf16 control varied 9.4% against itself, so this probe cannot resolve a dtype effect** and
the apparent 0.89× for fp16 is not a result. Reported so it is not mistaken for one. Two things it
*did* establish: bf16 output is md5-identical run to run (the 2026-08-24 sampling-reproducibility
fix works), and fp16 changes the audio (expected — perturbed logits redraw at temperature 0.8).
A clean per-step measurement is Task 2.

### q8 (int8 weight-only) — the fp8 substitute, and the best-validated lever

**A correction to my own earlier reasoning.** I dismissed int8 by citing `torch._int_mm` at 59-60 of
the datasheet's 284 TOPS. That figure describes **W8A8** (int8×int8→int32 compute). **q8
weight-only is W8A16**: weights are stored int8 and dequantized in-register, with the matmul still
in bf16/fp16, so it never touches that kernel. What it buys is **halving the bytes read** — exactly
the right lever for a bandwidth-bound decode, and the *only* one here that lowers the floor rather
than chasing it.

**Quality precedent is already in this repo, on this exact model.** `higgs_v3_mlx_q8` is the
**active, ear-approved batch TTS on Apple Silicon**, chosen over bf16 with "no quality need" (ADR
`2026-06-20-higgs-v3-tts-mlx-q8`). So q8's quality risk is better understood than fp16's — though
that is a reason to sequence it earlier, not a reason to drop fp16, which buys a different term.
**q4 is out** — that ADR records it as "rejected: runaway/collapse", and a later native-q4 A/B was
ear-rejected as worst of three.

Recomputing that ADR's own per-step numbers as achieved bandwidth:

```
M1 Pro torch-bf16   69 ms/step  ->  116 GB/s achieved
M1 Pro MLX-bf16     48 ms/step  ->  167 GB/s achieved     1.44x over torch, pure kernel efficiency
M1 Pro MLX-q8       28 ms/step  ->  143 GB/s achieved     1.71x over MLX-bf16 (theoretical max 2.00x)
```

Two things fall out. First, **q8 delivered 1.71× of a theoretical 2× on this model** when decode
was at the bandwidth roof — that is the number to expect. Second, MLX-bf16 achieved 167 GB/s where
torch-bf16 achieved 116 on the same silicon: **the same 1.44× kernel-efficiency gap I measured on
the 3090 between bf16 (527 GB/s) and fp16-accumulate (769 GB/s)**. The pattern recurs, which is
mild independent support for the accumulate finding. (Doc-drift note: that ADR calls 116 GB/s "this
Mac's ceiling"; it was torch's *achieved* rate, and MLX beat it by 44%.)

**The payoff is 1.16× or 2.0×, decided by one number.** Everything turns on where the 3090's
decode step sits relative to its 9.91 ms bf16 floor (8 GB / 807 GB/s):

| if the 3090's decode step is… | bandwidth share | q8 gain on decode |
| --- | --: | --: |
| at the roof (~9.9 ms), as the Mac was | ~100% | **~2.0×** (1.71× realized, per the Mac) |
| 3.6× above it (~36 ms), as my uncontrolled probe hints | ~28% | **~1.16×** |

**Task 2 measures exactly this**, which is why it gates. Counterintuitive consequence worth stating:
the "obvious" bandwidth fix pays almost nothing until the *overhead* is removed — and after
overhead removal it becomes the largest remaining lever. Order matters.

**One benefit is order-independent: VRAM.** q8 takes the TTS backbone from ~8.5 GB to ~4.5 GB.
Given peak 16.5 GB of 24 GB, the `dxgk` allocation failures I hit, and that Task 4 (pipelining) and
Task 6 (`max_batch` 16-24, ~3.4 GB of KV) both need headroom, **q8 unblocks the other two levers**.
That may be its biggest contribution regardless of the timing outcome.

**Implementation:** torchao `int8_weight_only()` via `quantize_()` on the loaded module — verified
`torchao==0.18.0` resolves cleanly against torch 2.11 as a **single package with no dependency
churn**, which matters given the `transformers` pin conflict that once broke this stage. Avoid
bitsandbytes `load_in_8bit`: LLM.int8()'s outlier decomposition is slower than fp16 for inference.
Optional stacking: q8 weights dequantized to **fp16 with fp16 accumulate** compounds both levers,
at compounded numerical risk — measure separately, do not ship together untested.

### Supporting probes

| probe | result | bearing |
| --- | --- | --- |
| `torch._scaled_mm` (fp8) | `RuntimeError: requires compute capability >= 9.0 or 8.9` | fp8 impossible on SM 8.6 |
| `torch._int_mm`, n=4096 / 8192 | **59.2 / 60.0 TOPS** (datasheet 284 dense) | rules out **W8A8** compute — *not* q8 weight-only, see below |
| 8192³ fp16+f16acc vs bf16 | 102.0 vs 63.3 TFLOP/s | the 2× datapath is real (1.61× realized) |
| HBM read+write (`copy_`, 1 GB) | **807 GB/s** (86% of the 936 spec) | 4B weight read = 9.91 ms/step floor |
| matvec `[B,4096]@[4096,4096]` | B=1 / 8 / 32 all **45 µs** | batching free; `max_batch=8` leaves room |
| tiny kernel launch | **10.5 µs** | ~2-3× a native-Linux box — WSL2 tax |
| `tensor.cpu()` D2H sync | **29.8 µs** | decode loop does one *per row per step* |
| `torch.compile(reduce-overhead)` / `torch.cuda.graph` | both work | overhead levers available |
| same Higgs model loaded 3× in one session | **46.6 / 17.9 / 52.5 s** | 2.9× spread from 9p page cache alone |
| `dmesg` under two resident 4B models | `dxgk: dxgkio_create_allocation: Ioctl failed: -75` | WSL2 commit limit is real; do not assume headroom |

---

## Review of the prior lever list

### Right, and kept

- **L1 (max_batch).** `max_batch = 8` at `patmir.toml:481`, and `patmir.toml:453-460` says outright
  it was "chosen to be safely conservative … rather than measured". Their VRAM + Windows-commit
  sweep gate is the right gate — and the `dxgk` failures above vindicate it.
- **L1.6.** Batched decode is not bitwise-invariant across chunk shapes (ADR
  `2026-08-24-tts-sampling-reproducibility`), so changing `max_batch` redraws lines and **needs an
  ear-test**. Confirmed incidentally: bf16 reproduces md5-identically, so any audio change is
  attributable.
- **L2.1 / L2.2 / L2.4.** No fp16 checkpoint exists (it is a load-time cast); the codec reached via
  `get_audio_codec()` (`higgs_v3_torch.py:457`) inherits the load dtype while only the input wav is
  forced to fp32 (`:467`); and fp16 needs an ear-test plus an overflow check. All correct.
- **L3.1/L3.2's conclusion** that ASR dtype is ~worthless — **correct, and now measured at 1.06×**,
  though for a different reason than stated (not "small model, short sequences" but "90% of the
  stage is not GEMM").
- **L5.2.** Verified: the accelerator lock is held around the whole stream run body
  (`ui/stream.py:378`), so it serializes streams against jobs and does **not** block concurrency
  within a stream.
- **L5.3.** Unbounded ECAPA growth is a real find.
- **R1 / R2 / R4.** Accurately cite the record.

### Wrong

1. **B0.5 — "Python runtime is NOT a viable lever … kernel launches are a single-digit-% overhead
   on bandwidth-bound decode."** This is the list's load-bearing claim, it was asserted rather than
   measured, and it is now **refuted by measurement: ASR is ~89% non-GEMM.** Mechanisms present in
   our code, none of them mentioned:
   - `higgs_v3_torch.py:561-608` runs a Python `for b in range(B)` **per step** containing
     `rows[b].append(codes_N.cpu())` — a **D2H sync per row per step** (29.8 µs each) — plus
     `_sample_codes`, `onehot.zero_()`, `scatter_`, `_sampler_step`: ~15 launches/row.
   - transformers 5.15's `DynamicLayer.update` does `torch.cat` on K and V **every layer every
     step** (`cache_utils.py:144-145`) — a full realloc-and-copy of the growing cache, **growing
     with sequence length**, which is the long-narration regime where TTS is worst. `StaticCache`
     is present (`cache_utils.py:1822`).
   - The only prefill/decode split in the repo (73.1/21.5/5.0, cited under L4) is **MLX, batch-1,
     M1 Pro**. Nothing has measured the CUDA split.
   - **R3 ("Rust/PyPy/Mojo") is a strawman.** Nobody needs a faster interpreter; the ask is fewer
     host↔device round trips.

2. **L5.1 — "pipelining does NOT cut steady RTF; do not credit it against the 2× goal."** Wrong.
   RTF is wall/audio; overlap changes wall and leaves compute and the denominator alone. From
   `runs/bench/sample_1-cuda-watch-fixed.json`, sections 1-9: `analysis_wall_s` sums to **77.20 s**,
   `dub_wall_s` to **140.48 s**, against **234.59 s** of wall. A two-stage pipeline is bounded by
   the slower stage → **0.877 → ~0.58, no model touched.** "Total compute unchanged" implies "wall
   unchanged" only if nothing idles — and **translate is 17% of steady-state wall spent waiting on
   OpenRouter with the GPU parked.**

3. **L2.3 — right premise, wrong conclusion.** Their premise (GA102 fp16 tensor rate ≈ 2× bf16) is
   substantially **correct**; I measured 102.0 vs 63.3 TFLOP/s. Their error was inferring
   "bandwidth-bound ⇒ dtype irrelevant" without noticing accumulate mode changes *achieved
   bandwidth*. **I made the same error and produced a benchmark that appeared to confirm it.**
   Recorded because the error shape is the reusable lesson: "bandwidth-bound" describes a kernel,
   not a dtype — and separately, a microbenchmark win must be divided by the stage's Amdahl share
   before it becomes a plan.

4. **fp8's impossibility is absent**, though it was asked about directly. L2.1 notes fp8
   checkpoints exist on HF without saying they cannot run natively on SM 8.6.

5. **B0.4's Amdahl framing is half-complete.** TTS is 61-84% per section on the narration clip
   (RESULTS.md) but **41%** of steady-state wall on sample_1 (96.15 s of 234.59 s; ASR 26%,
   translate 17%). At 41%, TTS → zero is only 1.7×. "Non-TTS cannot deliver 2×" **and** "TTS cannot
   either" are both true on dialogue-dense content.

6. **L4 quotes a dropped item's pre-drop text.** Roadmap #4 is `❌ DROPPED 2026-08-23`
   (`performance.md:685`); −14% is an MLX batch-1 number and the torch path already amortizes
   prefill across B rows.

7. **L1.3's formula is inverted.** Weight bytes per *step* are 8 GB regardless of B; the per-*
   sentence* cost goes as 8 GB/B. Their ~1.7× ceiling lands plausibly for a different reason: chunk
   cost is `max(steps)`, so merging two cap-sorted chunks saves the *shorter* chunk's max. Expect
   nearer **20-25%** of section wall than L1.8's 30-40%.

### Missing entirely

8. **`HF_HOME` is on a 9p Windows mount** (`/mnt/e/models_cache`, `df -T` → `E:\ 9p`), holding
   8.7 GB of Higgs and 3.9 GB of Qwen, with **921 GB free on ext4 `/`**. Prewarm is **147.1 s**
   = 33.6% of the watch-along wall, and I measured a **2.9× spread** loading the same model three
   times in one session. `docs/setup/windows-wsl2-cuda.md:194` says keep it inside WSL. Hits
   time-to-first-audio, **not** steady state, so it does not count toward the 2×.

9. **Watch-along starves every batch dimension, and finding 2 explains the magnitude.** Same models,
   same box: ASR RTF **0.225** in watch-along vs **0.068** in batch (3.3×); translate **0.150** vs
   **0.062** (2.4×); TTS **0.360** vs **0.255** (1.4×). If ~89% of ASR's cost is per-step overhead
   amortized across the batch, then batch 1-2 instead of 8 should cost ~3-4× — and the measured
   ratio is 3.3×. **The two independent numbers agree**, so this is a structural fix (batch chunks
   across sections / lookahead), not a dtype one.

---

## Plan

Target: **watch-along steady-state RTF 0.877 → ≤ 0.44** on `samples/sample_1.mkv`, cuda profile,
via `patmir bench --mode watch-along`. Reported but not gated: the narration clip
(`os-jXiYRihI`, agg 1.005) and time-to-first-audio (52.6 s).

Rule for every task: re-measure against `runs/bench/sample_1-cuda-watch-fixed.json`, and never
claim a win the harness did not print.

### Task 1 — Save spec documentation

`agent-os/specs/<YYYY-MM-DD-HHMM>-watchalong-2x-cuda/`: `plan.md`, `shape.md` (this review — keep
both the prior list's error and my own, since the error shape is the transferable lesson),
`references.md`, and `measurements/hardware-probes.md` with the probe scripts and raw output, so
the accumulate finding and the ASR Amdahl result are reproducible rather than quoted.

### Task 2 — Measure the Higgs decode step split  ·  ✅ DONE 2026-08-28

Results in `measurements/TASK2-RESULTS.md`. **7.31 GB decode-active weights → 9.06 ms/step floor;
measured 43.13 ms/step at kv=300 = 4.8× the floor, 79% overhead.** Attribution: bandwidth 21%,
sampler 11%, **launch/dispatch 68% (~2800 launches/step)**.

Three consequences, all measured:
- **CUDA graphs / `torch.compile` is the primary lever** (68% of the step), not a conditional one.
- **`StaticCache` is refuted** — 0.98× at kv=300, 1.06× at kv=1200. My `DynamicCache`-`torch.cat`
  hypothesis was wrong; dropped from Task 3.
- **fp16 is refuted and my 1.46× was a shape artifact.** At the true MLP shape
  `[8,2560]@[2560,9728]`, bf16 already achieves **782 of 807 GB/s** and fp16+accumulate is
  **0.93× (slower)**. The 1.46× came from `[8,4096]@[4096,14336]`, a shape this model does not
  have. fp16 is dropped from Tasks 8 and 9. This also retroactively explains ASR's 1.061% and the
  full-chunk 0.89% as real-but-neutral rather than noise.
- **q8's sequencing is confirmed quantitatively**: it saves 4.53 ms either way, which is 1.12× at
  today's 43 ms step but 1.68× once graphs bring the step to ~11 ms.

#### Original entry

The one number the full-chunk probe could not resolve. Instrument `_decode_chunk` for per-step
wall at fixed KV length (no sampler, no termination variance), and attribute it across: weight
bandwidth (floor 9.91 ms/step), `DynamicCache` cat, kernel launches, sampler loop. Sweep KV length
300/600/1200 to isolate the cache-growth term. Ablations as measurements, not commitments:
sampler bypassed; `StaticCache`; CUDA-graph replay of the forward.

**Run it with nothing else on the card** — two resident 4B models produced `dxgk` allocation
failures and a SIGBUS during my probing.

### Task 3 — Kill per-step overhead in the TTS decode loop  ·  ✅ DONE 2026-08-28 (premise refuted)

**Outcome: the Task 2 attribution this task was built on was wrong, and the task's primary lever
died with it.** Full numbers in `measurements/TASK3-RESULTS.md`; ADR
`docs/architecture/decisions/2026-08-28-watchalong-section-pipeline.md`.

- **CUDA graph capture: 1.18×, not ~4×.** With every kernel launch eliminated the step still needs
  ~42 ms to move ~9 GB — 214 GB/s, **26% of the 807 GB/s roof**. So the 29.34 ms residual is NOT
  dispatch overhead; the kernels run below roof at seq-1 shapes. Dropped: 1.18× does not pay for
  bucketed static caches, pinned buffers and re-capture on every shape change.
- **The step is flat in batch size** (45-49 ms from B=1 to B=24) while per-sentence cost falls
  linearly. That is the same fact from the other side, and it is where the win actually is —
  which makes Task 6 part of this task rather than a later one.
- **`max_batch` 8 → 12 shipped.** End-to-end on a real section: TTS 24.16 → 18.08 s (**1.34×**),
  +1.21 GB peak, 0 overflows. NOT 3.1× — a section holds ~12 sentences, so at 12 it is already one
  chunk and 16/24 are indistinguishable. **Redraws every line's audio** (12/12 measured); needs an
  ear check, revert is one number.
- Per-row `.cpu()` removal: **not done**, deprioritised — ~4% of a step whose real problem is
  elsewhere. Still bitwise-safe and still worth doing.
- Sampler vectorization: **not worth it** — its 0.17 ms figure used one global `multinomial`,
  violating the per-row-generator requirement.
- Process isolation is now mandatory for GPU probes here: the first multi-arm graph probe
  segfaulted in `g.replay()` from cross-arm contamination, while capture and replay verified clean
  in isolation at four levels.

#### Original entry

Justified by finding 2 and sized by Task 2:
- Drop the per-row `codes_N.cpu()` — codes stay on device, one D2H at chunk end. **1.79 ms of a
  43.13 ms step (4.2%), trivially equivalence-preserving; land this first.**
- Vectorize the sampler across the batch. Ceiling is 4.56 ms (10.6%), but the measured 0.17 ms
  used one global `multinomial`, which **violates** ADR `2026-08-24-tts-sampling-reproducibility`'s
  per-row-generator requirement. A shipped version must keep per-row streams (batch the top-k/top-p
  masking and the one-hot; draw per row, or pre-draw per-row uniforms and do inverse-CDF batched),
  so expect less than 4.56 ms.
- **`torch.compile` / CUDA-graph the decode step — the primary lever, 68% of the step.** Both
  verified working on this box. Target: collapse ~2800 launches/step into a replay.
- **NOT `StaticCache`** — measured 0.98× at kv=300 (Task 2). Refuted, do not re-chase.

Equivalence is testable: `test_higgs_torch_batch.py` already pins batch-vs-single bitwise at
temperature 0, and bf16 output is md5-reproducible, so any audio change is attributable.

### Task 4 — De-serialize the stream  ·  ✅ DONE 2026-08-28

`process_section` split into `analyze_section` -> `SectionAnalysis` -> `dub_section`;
`StreamScheduler` gained `_run_pipelined`, which runs section N's dub under section N+1's
analysis on a single worker. `process_section` kept as their composition. Confirmed on the first
full run: **8 of 8 sections started their analysis before the previous section's dub ended.**

Fixed en route, both pre-existing: `dub_section` now takes the `cancel` token (the serial version
never threaded one into the second `run_job`, so Stop could not interrupt translate/tts/mux), and
**`patmir bench --mode watch-along` never terminated under the scheduler driver** — `BenchPlayhead`
keeps the abandon timer fed forever, so the loop spun at 2 s a tick once the video was covered and
never wrote its report. Bench also had to stop reporting `sum(per-section wall)` as the wall, which
double-counts exactly what the pipeline overlaps; it now measures real elapsed and gains
`--serial-sections` as a same-code control.

8 tests in `test_streaming_pipeline.py`, including a **negative control** asserting the serial path
does not overlap — without it the overlap test passes against a scheduler that never pipelined.

#### Original entry

`StreamScheduler.run()` (`streaming/scheduler.py:152`) is a strictly serial
`while: process_fn(...)` loop. Run analysis of section N+1 under the dub of N, and overlap the
cloud translate with GPU work. Constraints: the accelerator lock is per-run-body
(`ui/stream.py:378`) so intra-stream concurrency is permitted; the coverage/index invariants from
quality issues #22/#23/#24 must hold — in particular no two emitted sections may overlap in span,
already property-tested. **Watch VRAM**: two pipelines in flight is what tripped `dxgk`, so
overlap analysis (small models) with dub (the 4B), not two dubs. Fold in the ECAPA list fix.

Expected on its own: **0.877 → ~0.58** — the largest single lever, and quality-neutral.

### Task 5 — Fix watch-along batch starvation (finding 9)

ASR runs at batch 1-2 on a 36 s section against `batch_size=8`. With Task 4's lookahead there are
multiple sections in flight, so chunks can be batched across them. Same argument for translate
(`chunk_size=20` against ~8-12 sentences). Expected to recover a large part of ASR's 3.3× penalty
— worth more than any dtype change on this stage.

### Task 6 — `max_batch` sweep

Sweep 8 → 16 → 24 on both modes. Reject any B leaving < 2 GB VRAM or < 2 GB Windows commit. Read
pass counts from `tts/timings.jsonl` (`"kind": "chunk"`), overflows from
`tts/voice_cloning_summary.json`. **Ear-test the winner** (L1.6). Note the mechanism has changed
with finding 2: raising B now also amortizes the per-step *overhead* over more rows, not just the
weight read — so this may beat the 20-25% estimate. If the modes want different values, add a
per-mode override then, not before.

### Task 7 — `HF_HOME` to ext4

Copy, verify, keep the 9p copy until a clean run. Expect prewarm 147 s → well under 60 s.
**Time-to-first-audio only; not counted toward the 2×.**

### Task 8 — q8 on the TTS decode (was: q8 + fp16; fp16 dropped by Task 2)

**Task 2 reduced this to a single arm.** fp16 is refuted at the model's true shape — bf16 already
achieves **782 of 807 GB/s** on `[8,2560]@[2560,9728]` and fp16+accumulate measures **0.93×** — so
there is no second term to stack and nothing to A/B against q8.

**q8** via torchao `int8_weight_only` (`torchao==0.18.0`, single package, no dependency churn).
Halves the measured 9.06 ms/step bandwidth floor to 4.53. Lowest quality risk of anything in this
plan: q8 is the active ear-approved TTS on the Mac for this exact model, chosen over bf16 with "no
quality need" (ADR `2026-06-20-higgs-v3-tts-mlx-q8`). q4 excluded — ear-rejected twice for collapse.

- **Sequence after Task 3.** Measured: 4.53 ms is **1.12×** of today's 43.13 ms step but **1.68×**
  of the ~11 ms step graphs should produce. Same change, 6× the payoff, purely from ordering.
- **The VRAM half is order-independent** — ~8.5 → ~4.5 GB, which unblocks Task 4's concurrency and
  Task 6's larger `max_batch`. Worth landing for the headroom even if the timing win disappoints.
- Because the decode matmuls are now *measured* at the roof (782/807 GB/s), q8 should approach its
  full 2× on the bandwidth term rather than the Mac's 1.71× — the Mac sat further off its own roof.

Gates: **ear-test** (weights change ⇒ prosody redraws, per ADR
`2026-08-24-tts-sampling-reproducibility`) and **overflow count 0-1** in
`tts/voice_cloning_summary.json`, since that ADR showed termination is sampling-sensitive. bf16
output is md5-reproducible, so every audio delta is attributable.

### Task 9 — Reserve

- ~~fp16 + accumulate on ASR~~ — **dropped after Task 2.** The 1.061× was real but the mechanism
  is now known to be shape-luck, not a datapath win at our shapes, and at the true Higgs shape the
  same setting measures 0.93×. Not worth a config divergence for ~1.5% of wall on one stage.
- Drop the ForcedAligner in watch-along (~5-6% of wall, already flagged in `performance.md`).

### Not doing

- **fp8 anything** — `torch._scaled_mm` refuses on SM 8.6. Silicon, not config. q8 (Task 8) is the
  substitute.
- **q4** — ear-rejected twice on this model (collapse/runaway, then worst-of-three).
- **W8A8 int8 compute** — `torch._int_mm` measured 60 of 284 TOPS. Task 8's q8 is W8A16 and does not
  use that path.
- **fp16 + fp16-accumulate anywhere** — refuted by Task 2 at the model's true shape (0.93×, i.e.
  slower). bf16 already achieves 782 of 807 GB/s there. The 1.46× that motivated it was measured on
  a shape this model does not have.
- **`StaticCache`** — measured 0.98× at kv=300 (Task 2).
- **bf16 tuning** — `allow_bf16_reduced_precision_reduction` measured 1.03×.
- **Prefix-KV caching** — dropped 2026-08-23; −14% is MLX batch-1 and torch already amortizes
  prefill across the batch.
- **Best-of-k TTS** — author declined on cost 2026-08-24.

### Amdahl check (sample_1 steady state: 234.59 s wall / 267.37 s audio, RTF 0.877)

| after | analysis / dub (s) | wall | RTF |
| --- | --- | --: | --: |
| baseline | 77.2 / 140.5 | 234.6 | 0.877 |
| + Task 4 (pipeline, bound = dub + drain) | — | ~155 | **~0.58** |
| + Tasks 3/6 taking TTS 96.2 → ~60 (dub → ~104) | 77.2 / 104 | ~115 | **~0.43** |
| + Task 5 (ASR 60.3 → ~30, analysis → ~47) | 47 / 104 | ~110 | **~0.41** |
| + Task 8 q8 on the remaining decode (TTS 60 → ~40) | 47 / 84 | ~90 | **~0.34** |

The 2× is cleared **before** Task 8, by de-serialization and overhead removal. Task 8 (q8, then
fp16) is what takes it past the target and provides the margin the narration clip needs — on that
clip (TTS 69%) Tasks 3-6 alone project only ~1.8×, so q8's contribution is what closes the gap
there. It also has to be sequenced last: until Task 3 removes the overhead, q8's own payoff is
~1.16× rather than ~2× (see the q8 table above).

### Verification

1. `patmir bench samples/sample_1.mkv --mode watch-along --repeat 2 --profile cuda` after each
   task; diff per-section `stage_wall_s`, `other_s`, `build_s` against
   `runs/bench/sample_1-cuda-watch-fixed.json`.
2. `patmir bench samples/sample_1.mkv --mode batch --repeat 3` — Tasks 3/6/7/8 are shared code, so
   batch must not regress.
3. `pytest` — `test_higgs_torch_batch.py` (batch-vs-single bitwise at temp 0),
   `test_qwen3_asr_torch.py`, `test_streaming_section.py`, and the scheduler property test that no
   two emitted sections overlap.
4. Audio equivalence for the quality-neutral tasks: md5 the `tts/*.wav` set before and after
   Tasks 3/4/5/7 — bf16 is reproducible, so any difference is a real behaviour change to explain.
5. One live-server run (`python -m patmir.cli ui --profile cuda`) on the narration clip:
   time-to-first-audio, per-section RTF, stall count via `--playhead realtime`.
6. Ear-test gates: Task 6's `max_batch` winner, and anything in Task 8. Nothing else changes output.
7. Docs, per the maintenance contract: `performance.md` (re-measure + roadmap statuses),
   `current-models.md` if a candidate knob changes, and **an ADR recording the dtype investigation**
   — the fp16-accumulate datapath is real and worth 1.46× on a GEMM but only 1.06× on the ASR
   stage; fp8 is impossible on SM 8.6, with q8 (W8A16) as its substitute; and torch's W8A8
   `_int_mm` at 60 of 284 TOPS does not bear on q8 either way. Written specifically so neither the
   positive nor the negative half is re-derived from a default-path benchmark again, and so the
   W8A8/W8A16 distinction is not re-conflated. Also correct the doc-drift in
   `2026-06-20-higgs-v3-tts-mlx-q8`, which calls 116 GB/s "this Mac's ceiling" when it was torch's
   achieved rate and MLX beat it by 44%.
