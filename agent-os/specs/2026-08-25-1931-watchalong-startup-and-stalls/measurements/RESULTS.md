# Measured results — after the fixes (2026-08-25 21:41, RTX 3090 / WSL2)

Same video (`os-jXiYRihI`), same box, same `cuda` profile, **driven through the
live server** (not bench), so this measures the real path including the
scheduler, the cache writes and the prewarm.

Server started with `python -m patmir.cli ui --port 8765 --profile cuda` from the
worktree. **This matters:** the `.venv/bin/patmir` console script resolves
`patmir` through an editable `.pth` pointing at `/home/vardan/patmir`, so it runs
the MAIN checkout regardless of cwd. Verified: the console script imports
`/home/vardan/patmir/patmir/__init__.py` (0 occurrences of `_balanced_chunks`),
`-m` from the worktree imports the worktree (2 occurrences).

## Startup

| | before (2026-08-25 18:21) | after | |
| --- | --- | --- | --- |
| time to first audio | **72.4 s** | **52.6 s** | **-27%** |
| prelude (analysis) | ~10 s + **14.3 s unexplained gap** | 16.2 s, no gap | gap eliminated |
| opening dub | 46.8 s (tts 41.6) | 36.4 s (tts 30.5) | |

The 14.3 s gap is gone because the two non-stage models now load at boot:

```
[prewarm] 9 stages in 147.1s: ... separate 2.1s ... asr 62.5s ... tts 58.4s
[prewarm] 2 watch-along stage(s) in 1.3s: separate 1.3s, translate 0.0s
[prewarm] 2 watch-along model(s) in 7.5s: embed:ecapa 1.0s, speaker_profile 6.5s
```

The middle line is B3 confirmed live: watch-along's `separate` built as a
**second** cache entry, proving the streaming candidate set really was a miss.

## Sections

| idx | window | len | wall | rtf | other_s | slowest stages |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0-40.0 | 40.0 | 36.4 | 0.909 | 0.0 | tts 30.53, asr 5.97, translate 3.83 |
| 1 | 40.0-76.5 | 36.5 | 42.2 | 1.157 | 3.1 | tts 28.33, asr 3.83, translate 3.52 |
| 2 | 76.5-113.0 | 36.5 | 32.2 | 0.883 | 3.1 | tts 19.57, asr 3.53, translate 3.16 |
| 3 | 113.0-149.5 | 36.5 | 39.4 | 1.078 | 3.1 | tts 26.03, translate 3.99, asr 3.33 |

**A1 fixed on real data.** Every section is a full 36.5 s. Before, this same
video produced an 8.1 s section from 38 s of analysis (RTF 4.0) and a 13.6 s
one — it is continuous narration, so there is never a qualifying silence and the
snap now cuts at target instead of collapsing.

**C1 fixed on real data.** Indices `[0, 1, 2, 3]` and files
`section_000..003.wav` are all unique; the manifest that previously held two
windows per file is clean.

**B1 confirmed on real data** — batch shapes are balanced, including the exact
measured pathology:

| sentences | before (greedy) | after |
| --- | --- | --- |
| 9 | `[8, 1]` = 24.3 s + **17.3 s** | **`[5, 4]`** |
| 11 | `[8, 3]` | `[6, 5]` |
| 12 | `[8, 4]` = 10.5 + 12.2 | `[6, 6]` |

**`other_s` = 3.1 s per section** — the ffmpeg slice/trim, ECAPA embed and
age-gender profile that no stage owns. Previously unrecorded anywhere.

## The honest caveat

**Aggregate RTF is 1.005 — production still does not clear realtime.** Sections
range 0.883-1.157. The buffer does not grow, so sustained viewing can still
outrun production; what changed is that the catastrophic case (RTF 4.0 from a
collapsed section) is gone. TTS remains 60-72% of every section.

`buffer_ahead_s` reads 76.5 / 113.0 / 149.5 because the playhead was left at 0 —
correct arithmetic for a stationary viewer, but not a viewing simulation. A
realtime-playhead run (`--playhead realtime`) would count actual stalls; it costs
one video-duration of wall time and was not run.

Two runs are not a controlled A/B: the before-run covered different parts of the
video (154.9-334.1 s after a seek) than this one (0-149.5 s). The structural
results (no collapse, unique indices, balanced batches, no unexplained gap) do
not depend on that; the exact RTF deltas do.

## A regression this exercise caused, and fixed

The first attempt **failed with CUDA OOM** in the emotion stage during the
prelude — 250 MiB refused with **7.1 GB free on the card**, the commit-limit
false-OOM from `docs/setup/windows-wsl2-cuda.md`.

Cause was mine: `speaker_profile`'s wav2vec2-large loaded on **CUDA**, and
prewarming it at boot made it resident *before* the prelude's emotion stage,
where it had previously loaded after ASR. Fixed by defaulting it to CPU — the
same reasoning `embeddings.py` already documents for ECAPA (a handful of short
ref clips per speaker; the accelerator buys almost nothing while the model sits
on the card for the whole server lifetime). `PATMIR_PROFILE_DEVICE` overrides.

Resident VRAM after prewarm: **17491 MiB -> 15900 MiB**, and the profiler's own
prewarm got *faster* (14.0 s -> 6.5 s, no GPU transfer).
