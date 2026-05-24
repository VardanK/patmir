# Phase 1 dogfood — definition of done

This file is the human-in-the-loop verification step that closes Phase 1. Fill
in the sections below as you run through it; the result feeds Phase 2 priorities
and the GPU re-eval target list.

## Pre-flight (one-time)

System binaries:

```bash
brew install ffmpeg rubberband
```

Python deps:

```bash
uv pip install -e ".[local,ui]"
```

## Smoke test (~2-minute clip)

Pick a short English-source clip you have on disk (e.g. a 2-minute clip cut out
of a cartoon).

```bash
patmir dub path/to/clip-en-2min.mkv
```

Then:

1. Open `runs/<ts>-<stem>/out.mkv` in mpv or VLC. Confirm:
   - [ ] Armenian audio plays by default.
   - [ ] Original audio is selectable as an alternate track.
   - [ ] An Armenian subtitle track is present and shows up by default.
2. Open `http://127.0.0.1:8765` (`patmir ui`) and:
   - [ ] The run appears in the list with all six stages green.
   - [ ] The run-detail page lists `transcript.json`, `translated.json`,
         per-sentence TTS WAVs, `dub.wav`, `out.mkv`, `out.hy.srt`.
3. Stage isolation:
   - Delete `runs/<ts>-<stem>/translated.json`.
   - Run `patmir dub <video> --resume-job runs/<ts>-<stem> --resume-from translate`.
   - [ ] Translate + TTS + mux + output re-run; ingest + ASR are skipped.

## RU-source smoke test (~2-minute clip)

```bash
patmir dub path/to/clip-ru-2min.mkv
```

- [ ] `manifest.json` shows `source_lang: "ru"`.
- [ ] Armenian audio still produced, even if narration quality is lower than EN.

## Stretch-cap sanity

Pick a sentence where Piper's Armenian audio is ~1.6× the source window
(common with long English idioms).

- [ ] The corresponding sample in `dub.wav` is stretched by exactly the cap
      (1.2× when `stretch_cap = 0.20`), then overruns the window. Adjacent
      sentences slide later by that overrun.

## Full dogfood — 20-minute cartoon

Pick a 20-minute English-source cartoon episode the son already wants to watch.

```bash
patmir dub path/to/episode.mkv
```

Watch the resulting `out.mkv` with the son.

**Pass = son watches the full episode in Armenian audio without asking to
switch back.**

Capture, sentence-by-sentence where possible:

### Failure modes observed

- **Out-of-sync sentences:** (timestamps in the source video where AR audio
  was noticeably misaligned)
- **Words Piper mispronounced:** (proper nouns, loanwords, numbers — likely
  candidates)
- **Sentences NLLB-600M mangled:** (idioms expected to fail; note any
  that surprised you positively or negatively)
- **Subtitle / audio mismatches:** (translation drift between SRT and TTS
  output if any — they share the same source `translated.json`, so any
  mismatch is a bug, not a divergence)

### Quantitative notes

- Total wall-clock pipeline time on the 20-minute episode:
- Approximate per-stage breakdown (from the manifest):
  - ingest: 
  - asr: 
  - translate: 
  - tts: 
  - mux: 
  - output: 

### Phase 2 / GPU re-eval priorities surfaced

- (write the 3–5 biggest issues here; these feed `agent-os/product/roadmap.md`
  Phase 2 ordering and the GPU re-eval target list)

## Verdict

- [ ] Phase 1 passes — son watched end-to-end.
- [ ] Phase 1 partial — usable for some content; specific failure modes above
      block other content.
- [ ] Phase 1 fails — pipeline is not yet watchable; specific blockers above
      must be fixed before re-dogfooding.
