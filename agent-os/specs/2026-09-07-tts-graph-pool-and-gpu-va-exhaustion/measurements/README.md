# Measurements — TTS graph pool / WSL2 GPU address-space exhaustion (2026-09-07)

Raw soak output behind ADR `2026-09-07-tts-graph-pool-and-gpu-va-exhaustion.md`
and quality issue #34. Each line is one full pipeline run of
`samples/sample_1.mkv` (301.1 s) through the live server on the cuda profile,
with the run dir deleted afterwards. Prewarm baseline is **15403 MiB**.

| file | config | outcome |
| --- | --- | --- |
| `soak-shared-pool-expandable-CONTAMINATED.txt` | shared pool + `expandable_segments:True` | degraded at run 3. **Do not cite:** a CUDA test was run on the same card mid-soak, so the card was shared. Kept because the contamination is itself the lesson. |
| `soak-shared-pool-noexpandable.txt` | shared pool, no allocator override | degraded at run 4; clean, nothing else touched the GPU |
| `soak-fixed-geometry.txt` | + bucketed geometry, no allocator override | degraded at **run 2** — worse |

`new_va_errors` is whether any `dxgkio_reserve_gpu_va` failure appeared in
`dmesg` after the server started, i.e. the WSL2 GPU address space failing.

## The failure signature, for whoever picks this up

A degraded run is not a crash. TTS simply stops emitting chunk progress: the
stage starts, and the first chunk that would normally land in 3-5 s has not
landed minutes later. The GPU reads 0-6 % utilisation at ~30 W while one host
thread holds ~85 % of a core, and `dmesg | grep reserve_gpu_va` shows failures
arriving continuously. Only restarting the server clears it.

## Method notes, including one that invalidated a run

- **Do not touch the GPU during a soak.** The first soak was invalidated by a
  CUDA-gated unit test run on the same card while jobs were in flight.
- A healthy first run means nothing here — every attempt so far survived 1-3
  runs before failing, so a verdict needs a soak well past 4.
- Wall-clock reading of "stuck" is unreliable: inter-run overhead includes
  `rm -rf` of ~1 GB of WAVs. Judge from `progress.jsonl` timestamps, not from
  how long the harness has been quiet.
