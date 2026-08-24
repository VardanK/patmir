# Evaluation clips

Which media files the benchmarks and ear-tests actually run against. Every
measurement in `performance.md` names one of these; a measurement citing a clip
in the **Removed** section below is historical and **cannot be reproduced**.

`samples/` and `runs/` are both gitignored, so nothing here is in version
control — this file is the only record that a given clip existed and what it was.

## Current

| File | Source | Duration | Audio | Notes |
| --- | --- | --- | --- | --- |
| `samples/sample_1.mkv` | Reacher S02E01, first ~5 min | **301.113 s** | **Russian** (single track; the video stream is tagged `eng`) | **The canonical clip.** Dialogue-dense live action, ~60 % speech (180.8 s over 301 s), 4 speakers / ~110 diarization turns. 29 MB. |
| `samples/clip-en-test.mkv` | — | 7.183 s | English | Cross-language sanity check only. Too short to amortize model builds, so its RTF measures the fixed-cost floor and nothing else. Never quote it as throughput. 96 KB. |

### On the name

The canonical clip was called `Reacher-S02E01-5min.mkv` until it was renamed to
`sample_1.mkv`; only its orphaned `:Zone.Identifier` stream still carries the old
name. Docs written before 2026-08-24 use the old filename in runnable commands
and those commands fail as written. The 301.1 s figure quoted throughout
`performance.md` is this file.

### On the language

The Apple-Silicon baselines in `performance.md` describe the Reacher rows as
"English, dialogue-dense live action". The file on disk has exactly one audio
track and it is **Russian**. Either the Mac had a different copy or the label was
wrong; the CUDA baselines measured Russian. Worth resolving before comparing the
two sets stage-by-stage, since ASR and TTS cost both scale with what is actually
spoken.

## Removed

| File | Was | Why it's gone |
| --- | --- | --- |
| `samples/masha-2-5min.mkv` | 300 s, Russian, fast cartoon dialogue, ~23 % speech, 4 characters | **Deliberately retired 2026-08-24** — not a good translation candidate. It was the canonical clip through Phase 2 and appears throughout the ADRs and closed specs; those references are history and stay as written. Its rows in `performance.md` are marked historical. |
| `samples/Fallout-S02E01-5min.mkv` | 300 s, Russian, dialogue-dense live action, ~55 % speech | No longer on disk. Its `performance.md` rows are historical. |
| `samples/clip-en-2min.mkv` | English | Referenced in older specs; never present on this machine. |

## Choosing a clip for a measurement

Speech density, not length, is what drives pipeline cost — TTS and ASR both scale
with how much is actually spoken, so the same warm pipeline can clear realtime on
sparse content and miss it on dense. The historical spread: masha ~23 % speech
(batch RTF 0.89), Fallout ~55 % (1.36–1.66), sample_1/Reacher ~60 %. Quote the
density alongside any RTF, or the number is not comparable.

## Run directories referenced by the docs

Several `runs/<job>/` paths cited as "ear-test pending" no longer exist —
`runs/` is scratch and gets cleared. Among them:
`20260715-reacher-asr-ab`, `20260526-183305-masha-2-5min`,
`20260620-higgs-fallout-test`, `20260823-001311-Reacher-S02E01-5min`. Treat a
dangling run pointer as "this was checked once and the artifacts are gone", not
as something still on disk to inspect.
