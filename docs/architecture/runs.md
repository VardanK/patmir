# `runs/<job>/` artifact map

Every job creates a fresh directory under the runs root, named `<YYYYMMDD-HHMMSS>-<source-stem>/` (suffixed `-2`, `-3` on a same-second collision). Below is what each file is and how to use it for debugging.

The runs root defaults to `runs/` next to `patmir.toml` and is set by `[storage].runs_root` — see the "Storage roots" section of `pipeline.md` and ADR `decisions/2026-08-01-configurable-storage-roots.md`. Paths below are written `runs/<job>/` for brevity.

## Top-level files

| File                  | Stage    | Purpose                                                                                  |
| --------------------- | -------- | ---------------------------------------------------------------------------------------- |
| `manifest.json`       | runner   | Per-stage status (`pending` / `running` / `done` / `failed`), timestamps, `build_s`, artifact paths, error tracebacks. The first thing to read when something failed. Note `stages` is a **dict keyed by stage name**, not a list. |
| `progress.jsonl`      | runner   | One row per `ctx.progress(...)` call — `{stage, pct, msg, ts}`. The FastAPI UI tails this for live progress. |
| `audio.wav`           | ingest   | 16 kHz mono PCM extracted from the source video.                                         |
| `audio.meta.json`     | ingest   | Sample rate, duration, source video path (also in manifest).                             |
| `full_audio.wav`      | separate | Demucs input — usually the same content as `audio.wav` but at Demucs's native SR.        |
| `vocals.wav`          | separate | **Demucs vocal stem (full).** What ASR + diarize consume. Contains dialogue + laughs + screams + grunts. |
| `accompaniment.wav`   | separate | Music + sound-effects bed (drums + bass + other stems summed). What mux mixes under the AR voice. |
| `speech_vocals.wav`   | separate | VAD-masked vocals with 20 ms cosine fades. What `ref_picker` slices ref clips from. Cleaner than `vocals.wav`. |
| `non_speech_vocals.wav` | separate | Inverse VAD mask — atmospheric vocals only. Mixed back into the bed at mux time so cartoon screams/laughs survive. |
| `separated.meta.json` | separate | Sample rate, channels, VAD intervals list, paths.                                        |
| `transcript.json`     | asr      | Word-level ASR output: `language`, full `words` list (each with `start`/`end`/`text`), and grouped `sentences` (each with `speaker_id` + Phase-2.x `emotion`). |
| `sentences.json`      | asr      | Same sentences as in `transcript.json` but flat (legacy convenience).                    |
| `translated.json`     | translate | `TranslatedTranscript` — source + target text per sentence + timing + speaker_id (+ emotion). |
| `synthesized.json`    | tts      | Per-sentence WAV paths, durations, source timings, target text.                          |
| `tts/voice_cloning_summary.json` | tts | Per-speaker voice-cloning decisions for IVC candidates (ElevenLabs): `cloned` / `fallback_default` / `no_ref_provided`, with the voice_id, ref audio path, and line count. Written only when the candidate exposes `summarize()`. Open this file first when investigating "wrong character" reports. |
| `dub.wav`             | mux      | Single concatenated Armenian voice track (resampled to source's SR; per-sentence afade in/out applied). |
| `dub.meta.json`       | mux      | Duration, sample rate, options used (stretch_cap, music_bed_gain, voice_gain_db).        |
| `mux_placements.json` | mux      | Where each synthesized line was actually placed — the record to read when a line sounds early, late, rippled by `voice_no_overlap`, or compressed by `voice_max_compress`. |
| `out.mkv`             | output   | Final MKV — Armenian default audio, original retained as secondary, Armenian SRT as default subtitle track. |
| `out.hy.srt`          | output   | Standalone subtitle file (also muxed into `out.mkv`).                                    |
| `<stage>/bench_probe.json` | any | **Benchmark only** (2026-09-02). Intra-stage timing: where a stage's own wall went, which the manifest cannot say. Written by any stage with spans, plus the runner's own post-stage work (`pick_refs_s` under emotion, `tag_s` under asr). Spans and counters sit at the top level and ADD across a stream's sections; non-additive identities and derived ratios live under `values` — summing those is what once reported a `max_batch` of 88. Absent unless the job ran under `patmir bench`, where the probe syncs the device on each span edge; disabled it is a no-op. See `patmir/pipeline/probe.py`. |
| `tts/trace_tts.json`  | tts      | **Benchmark only, opt-in** (`patmir bench --tts-trace`). A `torch.profiler` chrome trace of the FIRST decode chunk. One chunk only: a whole-job trace runs to hundreds of MB and every chunk runs the same loop. |

## Published deliverables (`[storage].library_root`)

When `library_root` is configured, the runner copies three files out of the job dir after the `output` stage:

```
library_root/<job_id>/
├── out.mkv       # final container (Armenian default audio)
├── out.hy.srt    # Armenian subtitles
└── dub.wav       # Armenian audio alone
```

Copy, not move — everything above stays in the job dir, so `--resume-from` and the server's cache materialize still work. Files are written as `<name>.partial` and renamed inside the destination dir, so a `.partial` left behind means an interrupted transfer (safe to delete; rerun to republish). A publish failure does not fail the run: check `progress.jsonl` for a `publish failed: …` row and `manifest.json`'s `published_dir` (null when publishing was skipped or failed).

**Quality/batch mode only.** Watch-along sections stop at `mux` and never run the `output` stage, so they publish nothing — their audio lives in `cache_root/<key>/section_NNN.wav` instead.

## Subdirectories

```
runs/<job>/
├── diarize/
│   ├── turns.json            # Speaker turns + overlap intervals + `merged_speakers` (duplicate-slot aliases, or null)
│   ├── vocals_16k_mono.wav   # CUDA profile only: the mono 16 kHz stem sortformer_nemo requires
│   ├── refs.json             # SpeakerRef per (speaker[, emotion]) — paths + source windows + ref_text
│   └── refs/
│       └── <speaker>/        # Phase 2.x: one subdir per speaker
│           └── <emotion>.wav # Reference clip used by TTS for that (speaker, emotion) pair
├── emotion/                  # Phase 2.x
│   └── segments.json         # Per-turn emotion predictions (start, end, emotion, confidence)
└── tts/
    ├── 0000.wav              # One file per synthesized sentence
    ├── 0001.wav
    ├── ...
    └── voice_cloning_summary.json  # Per-speaker IVC decisions (ElevenLabs); absent for non-IVC candidates
```

## How to debug a failed run

1. Open `manifest.json`. The first stage with `state: "failed"` (or `state: "done"` with an `error` field, which happens for some recoverable errors) shows the traceback. Check its `error` field.
2. If the failed stage produced *any* output, the per-stage files above are still on disk — read them to see what the previous stage handed off.
3. Re-run from that stage: `patmir dub <video> --resume-job runs/<job> --resume-from <stage>`. The runner loads prior artifacts from disk and skips earlier stages.
4. To force a clean rerun, delete the failed stage's artifacts and resume from it. `--resume-from` is `STAGE_ORDER`-aware; deleting `transcript.json` and resuming from `asr` is enough to make ASR run again.

## How to inspect ref-clip selection

Open `diarize/refs.json`. Each entry has `source_start` / `source_end` — open `vocals.wav` (or `speech_vocals.wav`) at those timestamps to hear what the ref clip's source looked like. Listen to `diarize/refs/<speaker>/<emotion>.wav` for the actual sliced clip (post-trim, post-fade-guard).

If a character sounds wrong in `dub.wav`, the most likely cause is:

- Wrong speaker bucketed (diarizer error) — check `diarize/turns.json`. Read `merged_speakers` first: a non-null value means two of the
  diarizer's output slots were folded into one actor (quality issue #16), so the `speakers` list is shorter than the raw model output.
- A character speaking someone else's line — check `sentences.json` against `turns.json`. Each sentence is one same-speaker run of words
  (`apply_speaker_tags`, ADR `2026-08-24-speaker-identity-word-level-attribution`), so a sentence spanning two turns means the attribution
  deferred those words and inherited a neighbour's label.
- Right speaker, wrong emotion bin (so a calm-voice ref was used for an angry line) — check `emotion/segments.json` against `transcript.json`.
- Ref clip itself is poor (background bleed, non-lexical vocalization) — open the WAV and listen.

## Run-list CLI

`patmir runs ls` — table of all run dirs with stage status glyphs.
`patmir runs show <job_id>` — full manifest summary including speaker count + (Phase 2.x) per-speaker emotion-bin coverage.
