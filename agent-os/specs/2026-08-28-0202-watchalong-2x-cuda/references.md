# References — watch-along 2× on WSL2 + RTX 3090

## Baselines to beat

### `runs/bench/sample_1-cuda-watch-fixed.json`
- **Relevance:** the gate. Steady-state (sections 1-9) is 234.59 s of wall for 267.37 s of
  audio = **RTF 0.877**.
- **Key fields:** per-section `stage_wall_s`, `analysis_wall_s` / `dub_wall_s` (the split that
  sizes pipelining: **77.20 s vs 140.48 s**), `other_s`, `build_s`, and `notes.peak_vram_gb`.
- **Steady-state stage shares:** tts 96.15 s (41%), asr 60.25 s (26%), translate 40.08 s (17%,
  cloud round-trips), separate 10.85 s (5%), unowned 16.94 s (7%).

### `agent-os/specs/2026-08-25-1931-watchalong-startup-and-stalls/measurements/RESULTS.md`
- **Relevance:** the *other* content profile, driven through the live server rather than bench.
  Continuous narration: agg RTF 1.005, TTS **61-84%** of each section against 41% on sample_1.
- **Why it matters:** the TTS share is content-dependent, so a TTS-only plan reaches 2× on one
  clip and not the other. Also the source of the 147.1 s prewarm figure.

### `runs/bench/sample_1-cuda-batch.json`
- **Relevance:** the efficiency yardstick for finding 9. Same models, same box: ASR RTF **0.068**
  in batch vs **0.225** in watch-along (3.3×); translate 0.062 vs 0.150; tts 0.255 vs 0.360.

## Code to modify

### `patmir/eval/candidates/tts/higgs_v3_torch.py`
- **`_decode_chunk`, ~L485-624** — owns the decode loop. The per-step Python `for b in range(B)`
  with `rows[b].append(codes_N.cpu())` (a D2H sync per row per step) is Task 3's target;
  `out.past_key_values` is a `DynamicCache`, hence the `torch.cat` growth.
- **`_balanced_chunks`, L218** — already correct; a chunk costs `max(steps)`, so merging
  cap-sorted chunks saves the *shorter* chunk's max, not half the work. This is why the
  `max_batch` estimate is 20-25%, not 50%.
- **`build()`, L728** — `dtype` and `max_batch` entry points (Tasks 6 and 8).
- **`_encode_ref`, L457/467** — the codec comes from `self.model.get_audio_codec()` and therefore
  **inherits the load dtype**; only the input wav is forced to fp32. Must be pinned to fp32
  independently before any fp16 flip (the prior list's L2.2, and it was right).
- **Reuse, do not re-derive:** `up._SamplerState` / `up._sampler_step` / `reverse_delay_pattern`
  stay upstream's; `_request_seed`'s per-row generators are what make audio deltas attributable.

### `patmir/pipeline/streaming/scheduler.py`
- **`run()`, L152** — strictly serial `while: process_fn(start, max_end)`. Task 4.
- **`_next_start()`, L115** — already has `lookahead_seconds`; the buffer policy exists, the
  concurrency does not.
- **Invariants to preserve:** quality issues #22/#23/#24 — no two emitted sections overlapping in
  span (property-tested), index numbering past the highest cached index, `Coverage` not inventing
  coverage.

### `patmir/pipeline/streaming/section.py`
- **`process_section`, L209** — calls `run_job` twice (analysis `stop_after="asr"`, then dub
  `resume_from="translate"`). The two halves are what Task 4 overlaps across sections.
- **`_remap_refs` / `locked_refs`** — must keep working under concurrency (quality issue #12).

### `patmir/pipeline/stages/asr/qwen3_asr.py`
- **`_transcribe_batch`, L265** and **`_align_batch`, L281** — both batched at `batch_size=8`, but
  a 36 s section yields only 1-2 chunks, so watch-along runs them at batch 1-2. Task 5.
- **`dtype` / `aligner_dtype`** plumbing at L115-121, L205 — the Task 9 two-liner.

### `patmir/ui/stream.py` / `patmir/ui/app.py`
- **`app.py:71`** builds the one `accelerator_lock`; **`stream.py:378`** holds it around the whole
  stream run body. Verified this serializes *streams against jobs* and does **not** block
  concurrency within a stream — so Task 4 is permitted, but VRAM is the real constraint.

## Prior decisions that bound this work

- **`decisions/2026-06-20-higgs-v3-tts-mlx-q8.md`** — the q8 precedent, and the numbers to expect:
  per-step torch-bf16 69 → MLX-bf16 48 → **MLX-q8 28 ms**, i.e. q8 realized **1.71× of a
  theoretical 2×**. q8 chosen over bf16 with "no quality need"; **q4 rejected: runaway/collapse.**
  Contains doc drift to fix: it calls 116 GB/s "this Mac's ceiling" when that was torch's achieved
  rate and MLX beat it by 44%.
- **`decisions/2026-08-24-tts-sampling-reproducibility.md`** — any weight or numerics change
  redraws prosody, so every Task 8 arm needs an ear-test; and termination is sampling-sensitive, so
  overflow count must stay 0-1. Also why bf16 md5-reproducibility is a useful test oracle.
- **`decisions/2026-08-25-one-accelerator-at-a-time.md`** — one pipeline fits on 24 GB, two do not.
  Bounds Task 4 to overlapping *analysis* (small models) with *dub* (the 4B), not two dubs.
- **`performance.md` roadmap #4** — prefix-KV caching, `❌ DROPPED 2026-08-23`. Its −14% is an MLX
  batch-1 figure; do not carry it forward.
- **`performance.md` "Not recommended now"** — best-of-k TTS, declined by the author on cost.
- **`docs/setup/windows-wsl2-cuda.md:194`** — keep `HF_HOME` inside the WSL filesystem. It is
  currently on a 9p mount (Task 7). L300+ documents the commit-limit false-OOM that this session
  reproduced as `dxgk: dxgkio_create_allocation: Ioctl failed: -75`.
