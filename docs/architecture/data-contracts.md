# Data contracts

All inter-stage typed dataclasses live in `patmir/pipeline/stage.py`. This page mirrors them for quick reference. If the two ever disagree, `stage.py` wins and this file is stale.

## Stage I/O dataclasses

### `AudioTrack`

Produced by `ingest`. The 16 kHz mono PCM WAV extracted from the source video.

| Field         | Type    | Meaning                              |
| ------------- | ------- | ------------------------------------ |
| `path`        | `Path`  | Absolute path to the WAV             |
| `sample_rate` | `int`   | Always 16000 in current pipeline     |
| `duration`    | `float` | Seconds (from `ffprobe`)             |

### `SeparatedAudio`

Produced by `separate`. Demucs output + Silero VAD masking.

| Field               | Type                              | Meaning                                                                                  |
| ------------------- | --------------------------------- | ---------------------------------------------------------------------------------------- |
| `accompaniment`     | `Path`                            | Music + sound-effects bed (Demucs drums + bass + other stems summed)                     |
| `vocals`            | `Path`                            | Full Demucs vocal stem (dialogue + laughs + screams + grunts) — what ASR + diarize consume |
| `sample_rate`       | `int`                             | Demucs native (44.1 kHz typically)                                                       |
| `channels`          | `int`                             | Demucs native (stereo typically)                                                         |
| `speech_vocals`     | `Path \| None`                    | VAD-masked vocals with 20 ms cosine fades — what `ref_picker` slices from               |
| `non_speech_vocals` | `Path \| None`                    | Inverse VAD mask — mixed back into the bed so atmospheric vocals survive                 |
| `vad_intervals`     | `list[tuple[float, float]] \| None` | Speech windows from Silero VAD; ASR uses as a post-filter                              |

### `SpeakerTurn`

Produced by `diarize`. One contiguous span attributed to one speaker.

| Field        | Type    | Meaning                  |
| ------------ | ------- | ------------------------ |
| `start`      | `float` | Seconds                  |
| `end`        | `float` | Seconds                  |
| `speaker_id` | `str`   | pyannote's `SPEAKER_00` etc. |

Phase 2.x will add `emotion: Emotion | None = None`.

### `OverlapInterval`

Produced by `diarize`. A time window where ≥2 speakers are active simultaneously (from `pyannote.Annotation.get_overlap()`).

| Field   | Type    | Meaning |
| ------- | ------- | ------- |
| `start` | `float` | Seconds |
| `end`   | `float` | Seconds |

### `SpeakerRef`

Produced by `ref_picker` (runner step after `emotion`). A reference clip + transcript used to clone one speaker's voice via OmniVoice.

| Field          | Type    | Meaning                                                              |
| -------------- | ------- | -------------------------------------------------------------------- |
| `speaker_id`   | `str`   | Owning speaker                                                       |
| `ref_audio`    | `Path`  | Sliced WAV, 16 kHz mono, trimmed 100 ms inward from the source turn  |
| `ref_text`     | `str`   | Empty at ref-pick time; backfilled by runner after ASR runs          |
| `source_start` | `float` | Seconds; where in the source vocal stem the ref window starts        |
| `source_end`   | `float` | Seconds                                                              |

### `Diarization`

Returned by `DiarizeStage.diarize(...)`. Composite output of the diarize stage + the runner's post-emotion `ref_picker` pass.

| Field         | Type                              | Meaning                                                                                                       |
| ------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `turns`       | `list[SpeakerTurn]`               | All speaker turns, VAD-filtered                                                                               |
| `overlaps`    | `list[OverlapInterval]`           | Overlap regions                                                                                               |
| `refs`        | `dict[str, SpeakerRef]`           | Phase 2 shape — keyed by speaker only. Phase 2.x changes this to `dict[(speaker, emotion_bin), SpeakerRef]`. |
| `sample_rate` | `int`                             | Vocal stem's sample rate                                                                                      |

### `Word`

A single word with timing. Produced by ASR.

| Field   | Type    | Meaning             |
| ------- | ------- | ------------------- |
| `text`  | `str`   | Surface form        |
| `start` | `float` | Seconds             |
| `end`   | `float` | Seconds             |

### `Sentence`

Words grouped into a sentence. Produced by ASR via `to_sentences(max_gap_s=0.6)` (split on `.!?` or on inter-word gaps > 0.6 s).

| Field         | Type           | Meaning                                                                                          |
| ------------- | -------------- | ------------------------------------------------------------------------------------------------ |
| `idx`         | `int`          | Position in the sentence list (preserved across speaker-split)                                   |
| `text`        | `str`          | Reconstructed text                                                                               |
| `start`       | `float`        | First word's start (seconds)                                                                     |
| `end`         | `float`        | Last word's end (seconds)                                                                        |
| `words`       | `list[Word]`   | Underlying word objects                                                                          |
| `speaker_id`  | `str \| None`  | Tagged by `apply_speaker_tags` (≥85% dominant overlap)                                          |

Phase 2.x will add `emotion: Emotion | None = None`.

### `Transcript`

Full ASR output.

| Field       | Type            | Meaning                            |
| ----------- | --------------- | ---------------------------------- |
| `language`  | `str`           | ISO 639-1 — e.g. `"en"`, `"ru"`    |
| `words`     | `list[Word]`    | Flat word stream                   |
| `sentences` | `list[Sentence]`| Grouped + tagged                   |

### `TranslatedSentence`

One sentence translated to the target language.

| Field         | Type            | Meaning                            |
| ------------- | --------------- | ---------------------------------- |
| `idx`         | `int`           | Same as the source `Sentence.idx`  |
| `source_text` | `str`           | Original                           |
| `target_text` | `str`           | Eastern Armenian                   |
| `start`       | `float`         | Inherited timing                   |
| `end`         | `float`         | Inherited timing                   |
| `speaker_id`  | `str \| None`   | Passed through from `Sentence`     |

Phase 2.x will add `emotion: Emotion | None = None`.

### `TranslatedTranscript`

| Field         | Type                       | Meaning                          |
| ------------- | -------------------------- | -------------------------------- |
| `source_lang` | `str`                      | From `Transcript.language`       |
| `target_lang` | `str`                      | Always `"hy"` in Phase 1+        |
| `sentences`   | `list[TranslatedSentence]` |                                  |

### `SynthesizedSentence`

| Field          | Type    | Meaning                                        |
| -------------- | ------- | ---------------------------------------------- |
| `idx`          | `int`   |                                                |
| `wav_path`     | `Path`  | `tts/<idx>.wav`                                |
| `wav_duration` | `float` | Seconds — from WAV header, no decode           |
| `source_start` | `float` | From the source sentence timing                |
| `source_end`   | `float` |                                                |
| `target_text`  | `str`   | Synthesized text (for SRT generation downstream) |

### `SynthesizedAudio`

| Field         | Type                          | Meaning              |
| ------------- | ----------------------------- | -------------------- |
| `sentences`   | `list[SynthesizedSentence]`   |                      |
| `sample_rate` | `int`                         | TTS-native           |

### `MuxedAudio`

| Field         | Type   | Meaning                                  |
| ------------- | ------ | ---------------------------------------- |
| `path`        | `Path` | `dub.wav` — single concatenated AR track |
| `sample_rate` | `int`  | Source's sample rate (resampled on write) |

### `OutputArtifacts`

| Field        | Type           | Meaning                  |
| ------------ | -------------- | ------------------------ |
| `video`      | `Path`         | `out.mkv`                |
| `subtitles`  | `Path \| None` | `out.hy.srt`             |

## Config contracts

### `StoragePaths` (`patmir/paths.py`)

Resolved filesystem roots, built from the `[storage]` table and carried on
`PipelineConfig.storage`. Every field is absolute; relative values in the TOML resolve
against the config file's directory, and `~` is expanded. Unknown keys under `[storage]`
raise. ADR: `decisions/2026-08-01-configurable-storage-roots.md`.

| Field / property | Type           | Default                | Meaning                                                       |
| ---------------- | -------------- | ---------------------- | ------------------------------------------------------------- |
| `runs_root`      | `Path`         | `<config_dir>/runs`    | Working dir — one subdir per job, all intermediates            |
| `cache_root`     | `Path`         | `~/.patmir/cache`      | Finished dubs keyed by (source, config); what the extension plays |
| `downloads_root` | `Path`         | `~/.patmir/downloads`  | yt-dlp downloads for URL submissions                          |
| `state_root`     | `Path`         | `~/.patmir`            | Local server's bearer token + job index                       |
| `library_root`   | `Path \| None` | `None`                 | When set, finished deliverables are copied to `library_root/<job_id>/`; `None` = don't publish |
| `jobs_index`     | `Path` (prop)  | `state_root/jobs.json` | Derived                                                       |
| `secret_path`    | `Path` (prop)  | `state_root/secret`    | Derived                                                       |

`PipelineConfig` gained a `storage: StoragePaths` field alongside `entries` and `options`.
It is **excluded from the cache key** — `ui.cache.config_hash` hashes only `entries` — so
relocating a root never orphans an existing dub.

`Manifest` gained `published_dir: str | None` — the destination the deliverables were
copied to, or `None` when publishing was skipped or failed.

## Stage protocols

`@runtime_checkable Protocol` — see `patmir/pipeline/stage.py:159+`.

| Protocol         | Method                                                                              |
| ---------------- | ----------------------------------------------------------------------------------- |
| `IngestStage`    | `extract(video_path, ctx) -> AudioTrack`                                            |
| `SeparateStage`  | `separate(source_video, ctx) -> SeparatedAudio`                                     |
| `DiarizeStage`   | `diarize(separated, ctx) -> Diarization`                                            |
| `EmotionStage`   | (Phase 2.x) `detect(separated, diarization, ctx) -> list[EmotionPrediction]`        |
| `ASRStage`       | `transcribe(audio, ctx) -> Transcript`                                              |
| `TranslateStage` | `translate(transcript, ctx) -> TranslatedTranscript`                                |
| `TTSStage`       | `synthesize(translated, ctx) -> SynthesizedAudio`                                   |
| `MuxStage`       | `mux(synth, source, separated, ctx) -> MuxedAudio`                                  |
| `OutputStage`    | `write(muxed, source_video, translated, ctx) -> OutputArtifacts`                    |

## Phase 2.x additions (in flight)

The emotion-transfer spec adds:

| Name                  | Shape                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------- |
| `Emotion`             | `Literal["neutral", "happy", "sad", "angry", "fearful"]`                              |
| `EmotionPrediction`   | `start: float, end: float, emotion: Emotion, confidence: float`                       |
| `Sentence.emotion`    | `Emotion \| None`                                                                     |
| `TranslatedSentence.emotion` | `Emotion \| None`                                                              |
| `SpeakerTurn.emotion` | `Emotion \| None`                                                                     |
| `Diarization.refs`    | Re-keyed to `dict[(speaker_id, emotion), SpeakerRef]` (or `"spk__emotion"` string key for JSON portability) |
| `STAGE_ORDER`         | Grows from 8 → 9 entries: insert `"emotion"` between `"diarize"` and `"asr"`          |
