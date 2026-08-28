# Watch-along 2× on WSL2 + RTX 3090 — shaping notes

## Scope

Cut watch-along (near-realtime) dubbing wall time 2× on the WSL2 + RTX 3090 box:
steady-state RTF **0.877 → ≤ 0.44** on `samples/sample_1.mkv`, cuda profile, measured with
`patmir bench --mode watch-along`. Reported but not gated: the narration clip
(`os-jXiYRihI`, agg 1.005) and time-to-first-audio (52.6 s).

Origin: the author asked whether benchmark data existed for this box, where the bottleneck was,
and specifically whether we were leaving speed on the table by running **bf16 where the 3090
prefers fp16**, and **q8 where fp8 would do**.

## Decisions

- **fp8 is closed by silicon.** `torch._scaled_mm` requires compute capability ≥ 8.9; the 3090 is
  8.6. **q8 (W8A16) takes that slot.**
- **fp16 and q8 are independent levers on different terms of the same step, and they stack.**
  fp16+accumulate raises *achieved* bandwidth (527 → 769 GB/s); q8 halves the *bytes*
  (8 GB → 4 GB, lowering the floor itself). Neither replaces the other.
- **Both are gated on one unmeasured number** — how far the Higgs 4B decode step sits above its
  9.91 ms bandwidth floor. At the roof both pay; overhead-dominated, neither pays until the
  overhead is gone. That measurement is Task 2, and it sequences Task 8 last.
- **The 2× is carried by de-serialization and overhead removal, not by any dtype change.**
  Task 8 provides margin and is what closes the gap on narration content.
- **q4 is excluded** — ear-rejected twice on this model (collapse/runaway; then worst-of-three).
- **Probes require an idle card.** Enforced in code, not by convention.

## Context

- **Visuals:** none.
- **Benchmark data:** `runs/bench/` (6 reports, 2026-08-24) — batch warm 0.461, watch-along agg
  1.456 / steady-state 0.877. Plus
  `agent-os/specs/2026-08-25-1931-watchalong-startup-and-stalls/measurements/RESULTS.md` (live
  server, narration clip, agg 1.005).
- **New measurements:** `measurements/` — all numbers with the scripts that produced them.
- **Product alignment:** watch-along clearing realtime with buffer margin is the D3 goal
  (`decisions/2026-06-13-near-realtime-d3-spike.md`). Local-first is unaffected: every lever here
  is local, and translate stays the one cloud stage.

## Two wrong turns, recorded because the error shape is the transferable lesson

A prior investigation produced a lever list that **rejected fp16 as "≈ 0 gain"** and dismissed
host-side overhead as "single-digit-%". Reviewing it, I reproduced the *first* error myself.

1. **"fp16 is a no-op on Ampere" — wrong, and my own benchmark appeared to confirm it.** PyTorch
   defaults fp16 matmul to **fp32 accumulate**, where fp16 and bf16 genuinely are identical
   (measured 1.01×). The 3090's second datapath needs `allow_fp16_accumulation = True`, and then
   the same GEMM is **1.46×**. Lesson: *"bandwidth-bound" describes a kernel, not a dtype* — the
   accumulate mode changed achieved bandwidth from 527 to 769 GB/s without changing a single byte
   moved. Corroboration from this repo's own history: on the M1 Pro, MLX-bf16 achieved 167 GB/s
   where torch-bf16 achieved 116 on identical silicon — the same 1.44× kernel-efficiency gap.
2. **Then I over-projected from the microbenchmark.** A 1.46× GEMM buys **1.061×** on the real ASR
   stage, because ~89% of that stage is not GEMM. Lesson: *a microbenchmark win must be divided by
   the stage's Amdahl share before it becomes a plan.*
3. **The prior list's B0.5 was the load-bearing claim and it was asserted, not measured.** It is
   now refuted: ASR runs **8.8× above its bandwidth floor**. Mechanisms in our own code that it
   never named — a `codes_N.cpu()` **D2H sync per row per step** (29.8 µs each) in
   `higgs_v3_torch.py:561-608`, and transformers 5.15's `DynamicLayer.update` doing `torch.cat` on
   K and V **every layer every step** (`cache_utils.py:144-145`), a cost that grows with sequence
   length. Its R3 ("Rust/PyPy/Mojo") is a strawman: nobody needs a faster interpreter, the ask is
   fewer host↔device round trips.
4. **Its L5.1 — "pipelining does not cut steady RTF" — is wrong.** RTF is wall/audio; overlap
   changes wall and leaves compute and the denominator alone. Sections 1-9 of
   `sample_1-cuda-watch-fixed.json`: `analysis_wall_s` 77.20 s vs `dub_wall_s` 140.48 s against
   234.59 s of wall, so a two-stage pipeline is bounded by the slower stage → ~0.58.
5. **I mis-dismissed int8 too**, citing `_int_mm` at 60 of 284 TOPS. That is **W8A8** compute; q8
   is **W8A16** and never touches that kernel.

A methodological corollary found by accident: running the probe with the server resident depressed
HBM to 295 GB/s, launch to 44.7 µs and `_int_mm` to 24 TOPS while **every dtype ratio survived**.
Ratios are robust to contention; absolute roofline figures are not — and it is the absolute floor
that decides q8's and fp16's value. Hence the in-code guard.

## Standards applied

`agent-os/standards/index.yml` carries no performance or measurement standard, so none apply
directly. The binding conventions came from `CLAUDE.md` instead:

- **`decisions.md` first.** Checked before proposing anything: it is why prefix-KV caching (dropped
  2026-08-23), q4, and best-of-k TTS are in "not doing" rather than re-investigated.
- **Maintenance contract.** This spec must update `performance.md` (re-measure + roadmap statuses),
  `current-models.md` (any candidate knob change), `quality-issues.md` (if an ear-test regresses),
  and add an ADR for the dtype investigation — both halves, so neither is re-derived from a
  default-path benchmark again.
- **`patmir bench` is the canonical harness** (`performance.md`); the warm run is the baseline.
  Never claim a win the harness did not print.
