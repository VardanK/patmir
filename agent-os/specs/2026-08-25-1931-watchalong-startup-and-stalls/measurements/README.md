# Session measurements — `os-jXiYRihI`, 2026-08-25 18:21:43 → 18:27:37

Reconstructed from artifacts left by the author's own watch-along session. No
new run was made. Source of each file is below; the extraction is reproducible
by re-running the snippet in `shape.md` against
`/home/vardan/patmir/runs/20260825-182*`.

**Cache key:** `d5af5653d745ed66` · **source:** 1297.0 s YouTube video, audio
downloaded as an 18.0 MB webm · **profile:** cuda · **mode:** watch-along

## Files

| file | source | why it is kept |
| --- | --- | --- |
| `stage-timings.json` | 10 job dirs' `manifest.json` | per-stage walls, build_s, inter-stage gaps, `untracked_s` |
| `sections.json.corrupted` | `~/.patmir/cache/d5af5653d745ed66/` | the C1 duplicate-index damage, as produced |
| `progress-182143-prelude.jsonl` | first prelude | the 14.3 s gap and the TTS `batch of 8` + `batch of 1` |
| `progress-182332-section-collapsed.jsonl` | the RTF-4.0 section | 10 ASR sentences in, 2 to translate |
| `separated-182332-collapsed.meta.json` | same section | the VAD intervals that collapsed it — Task 5's test fixture |

## Timeline

| wall clock | job dir | window produced | wall | untracked |
| --- | --- | --- | --- | --- |
| 18:21:43→18:22:56 | `182143-prelude` | 0–40.0 | 72.3 s | **15.5 s** |
| 18:22:56→18:23:31 | `182256-section` | 154.9–190.3 | 35.3 s | 1.6 s |
| 18:23:32→18:24:04 | `182332-section` | 190.3–**198.4** | 32.8 s | 1.7 s |
| 18:24:05→18:24:13 | `182405-prelude` | 0–40.0, **discarded** | 8.1 s | 0.9 s |
| 18:24:13→18:24:56 | `182413-prelude` | 0–40.0 | 42.2 s | 1.6 s |
| 18:24:56→18:25:34 | `182456-section` | 198.5–229.8 | 37.5 s | 1.6 s |
| 18:25:34→18:25:56 | `182534-section` | 229.8–243.4 | 22.0 s | 1.6 s |
| 18:25:56→18:26:26 | `182556-section` | 243.4–272.5 | 30.0 s | 1.6 s |
| 18:26:27→18:26:59 | `182627-section` | 272.5–302.4 | 31.9 s | 1.6 s |
| 18:26:59→18:27:37 | `182659-section` | 302.4–334.1 | 38.3 s | 1.7 s |

**Time to first audio: 72.4 s** for a 40 s window (RTF 1.81).

## What the numbers pinned

**The prelude ran three times.** Absolute stage timestamps prove `182405` and
`182413` are two independent full analyses 8 s apart, not one resumed into the
other — which is what the healthy 18:21 pair did (`dub_prelude_window` resumed
into `182143` at `resume_from="translate"`, so `182143` holds both halves).
About 50 s of GPU work re-done, and the restart reset the section index.

**`untracked_s` is real and constant.** ~1.6 s on every section (ffmpeg
slice/trim + ECAPA embedding + age-gender profiling), ~5 % of section wall, and
recorded nowhere. The first prelude's 15.5 s is that 1.6 s plus a one-time
**14.3 s** model load — it does not recur on `182413`, which is how it was
identified as a lazy load rather than per-run work.

**The collapsed section.** `182332`'s VAD intervals were
`[[0.0, 6.3], [6.3, 8.1], [8.1, 38.0]]`. With `target = 38.0 − 1.5 = 36.5`, the
only ends at or below target are 6.3 and 8.1, so the section cut at **8.1** —
32.8 s of work for 8.1 s of dub, 8 of 10 ASR sentences discarded. This is the
Task 5 fixture.

**TTS batch shape.** Prelude: 9 sentences as `batch of 8` (24.3 s) + `batch of 1`
(**17.3 s**). A chunk costs `max(steps)`, so the remainder batch is nearly as
expensive as a full one. `182659`: 12 sentences as 8 (10.5 s) + 4 (**12.2 s**) —
the smaller batch cost *more*, because length-sorting groups the long lines.

**Per-section fixed costs.** `asr detected en` 3.5–4.6 s every section, though
the prelude already knows the language. Translate shows a 3.4–4.2 s gap before
`sentence 1/N` while the sentences themselves finish in <0.1 s.

**One speaker.** Diarization reported `1 speakers` on the prelude and on every
section — so the 40 s cast-establishment prelude and the ECAPA registry bought
almost nothing on this content.

## Caveat

This is **n=1**, one video, one session, one machine. Per-section walls vary
22.0–38.3 s on the same 38 s of analyzed audio, so single-section figures should
be read as samples. The structural findings (the snap collapse, the index reset,
the batch tail, the unprewarmed lazy loads) are mechanisms visible in the code
and do not depend on the sample size; the *magnitudes* do.
