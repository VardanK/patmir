# Pipeline overview

Patmir is a 9-stage pipeline. Each stage is a `Protocol` in `patmir/pipeline/stage.py`; one concrete candidate per stage runs at a time, selected by id in `patmir.toml [pipeline]`. Stages are swappable — model upgrades are config changes, not code changes.

## Stage order

```
ingest → separate → diarize → emotion → asr → translate → tts → mux → output
```

Order is defined in `stage.py:STAGE_ORDER`. Resume-from-stage is supported via `patmir dub <video> --resume-from <stage>` (loads prior artifacts from disk).

## Data flow (ASCII)

```
video file
    │
    ▼
┌─────────┐
│ ingest  │  ffmpeg → 16 kHz mono PCM
└────┬────┘
     │ AudioTrack
     ▼
┌──────────┐
│ separate │  Demucs htdemucs_ft → vocals + accompaniment
│          │  Silero VAD → speech_vocals + non_speech_vocals + vad_intervals
└────┬─────┘
     │ SeparatedAudio
     ▼
┌─────────┐
│ diarize │  pyannote 3.1 on vocals
│         │  → SpeakerTurn list + OverlapInterval list
│         │  (writes diarize/turns.json; ref-WAVs deferred to post-emotion)
└────┬────┘
     │ Diarization
     ▼
┌─────────┐
│ emotion │  emotion2vec on vocals per turn
│         │  → list[EmotionPrediction]    ─→ ctx.extra["emotion_predictions"]
│         │  Runner then calls ref_picker.pick_refs(turns, overlaps, predictions, ...)
│         │  to populate Diarization.refs as dict[(speaker_id, emotion_bin), SpeakerRef]
└────┬────┘
     │
     ▼
┌─────┐
│ asr │   mlx-whisper large-v3 on vocals (with VAD post-filter)
│     │   → Transcript with word timestamps + sentence grouping
│     │   Runner then:
│     │     1. apply_speaker_tags(transcript, diarization)
│     │     2. apply_emotion_tags(transcript, predictions)
│     │     3. _backfill_ref_text(diarization.refs, transcript)
│     │   Each Sentence carries speaker_id + emotion after this point.
└──┬──┘
   │ Transcript
   ▼
┌───────────┐
│ translate │  NLLB-3.3B (current default) or Qwen3-Instruct-MLX (emotion-aware)
│           │  → TranslatedTranscript (Eastern Armenian)
└─────┬─────┘
      │
      ▼
┌─────┐
│ tts │  OmniVoice MLX, per-sentence ref clip:
│     │    refs[(speaker, emotion)] → refs[(speaker, "neutral")] → global default
│     │  guidance_scale set per emotion bin
│     │  → per-sentence WAVs in tts/*.wav
└──┬──┘
   │ SynthesizedAudio
   ▼
┌─────┐
│ mux │  ffmpeg pad+stretch (±20% cap, pitch-preserving)
│     │  + 50 ms afade in/out on each synthesized line (voice-leak guard)
│     │  + accompaniment mixed at music_bed_gain (default 0.6)
│     │  → single dub.wav
└──┬──┘
   │ MuxedAudio
   ▼
┌────────┐
│ output │  ffmpeg → MKV (Armenian default audio, original retained, hy SRT)
└────────┘
```

## Cross-stage plumbing via `ctx.extra`

`JobContext.extra` is a mutable `dict[str, Any]` carried through every stage. The runner uses it for any data that needs to flow *between non-adjacent stages* without bloating the typed dataclass contracts:

| Key                          | Set by         | Consumed by         | Purpose                                      |
| ---------------------------- | -------------- | ------------------- | -------------------------------------------- |
| `vad_intervals`              | runner (post-separate) | asr         | Filter ASR words to VAD windows              |
| `diarization`                | runner (post-diarize)  | asr, tts    | Speaker turns + per-`(speaker, emotion)` refs |
| `emotion_predictions`        | runner (post-emotion)  | asr (tag), ref_picker | Per-turn emotion labels              |

Pattern: a stage's typed return value is in `state[stage_name]`. `ctx.extra` is for cross-stage hand-offs the typed contracts don't carry.

## Post-stage runner steps

Two operations run inside the runner, *not* as their own stages, because they bridge stages without independent value:

1. **`pick_refs`** — after `emotion` returns, runner calls `pick_refs(turns, overlaps, predictions, ...)` to slice per-`(speaker, emotion_bin)` reference WAVs. The result is written back into `state["diarize"].refs`. Runs in the runner so `diarize` doesn't need to know about emotion.
2. **`apply_speaker_tags` + `apply_emotion_tags` + `_backfill_ref_text`** — fire immediately after the ASR candidate returns its `Transcript`. They mutate the transcript (attach `speaker_id` + `emotion`) and backfill `SpeakerRef.ref_text` (which couldn't be filled at diarize time because ASR hadn't run yet).

## Audio artifacts written per job

See `docs/architecture/runs.md` for the full file map.

## Stage protocols

See `patmir/pipeline/stage.py` for the typed Protocols. See `docs/architecture/data-contracts.md` for every dataclass field with explanations.
