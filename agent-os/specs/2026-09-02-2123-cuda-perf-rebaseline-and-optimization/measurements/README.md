# Measurements

Raw bench reports and telemetry for this spec. `runs/` is gitignored — the 2026-08-24 CUDA reports that `performance.md` cites were lost that way — so **every** report produced here is copied into this folder.

## Before (under the GPU power limit, 2026-09-01)

Copied from `runs/bench/` at shaping time; produced by a previous session with the power limit still in place (author-confirmed).

| file | mode | what |
|---|---|---|
| `before-baseline-batch.json` | batch | 3 repeats, cold + 2 warm; warm agg RTF 0.406 / 0.380 |
| `before-baseline-watch.json` | watch-along | greedy playhead, 10 sections, agg RTF 0.682, time to first audio 45.4 s |
| `before-probe-batch.json` | batch | 2 repeats; label "probe" but config identical to baseline; treat as a duplicate baseline |
| `before-tts-bench_probe.json` | batch | intra-TTS breakdown from `runs/20260901-020038-sample_1/tts/` (instrumentation no longer in the tree): forward 60.7 s / 1293 steps ≈ 47 ms/step |
| `before-asr-bench_probe.json` | batch | intra-ASR breakdown, same job: probe 4.58 s, generate 10.53 s, align 0.51 s |
| `nvidia-smi-shaping-2026-09-02.txt` | — | power/clock/throttle state at shaping time (350 W limit, no active throttle, since-boot counters show past power-cap and thermal events) |

## After (350 W) — filled in by Tasks 3 and 10

- `baseline-350w-batch.json`, `baseline-350w-watch.json`, `baseline-350w-watch-greedy.json` — Task 3
- `nvidia-smi-before-*.txt`, `nvidia-smi-after-*.txt` — Task 3
- per-task A/B reports (`<task>.json`) and md5 lists — Tasks 5-9
- `after-*.json` — Task 10
- Baseline job ids (artifact-diff reference): _record here in Task 3_

`RESULTS.md` is the authoritative write-up.
