# Pipeline overview

Patmir is a 9-stage pipeline. Each stage is a `Protocol` in `patmir/pipeline/stage.py`; one concrete candidate per stage runs at a time, selected by id in `patmir.toml [pipeline]`. Stages are swappable — model upgrades are config changes, not code changes.

## Stage order

```
ingest → separate → diarize → emotion → asr → translate → tts → mux → output
```

Order is defined in `stage.py:STAGE_ORDER`. Resume-from-stage is supported via `patmir dub <video> --resume-from <stage>` (loads prior artifacts from disk).

## Ingest sources

Two entry points feed the same pipeline:

- **Local file path** (CLI: `patmir dub <video>`, or `patmir submit <video>` to the local server) — the `ffmpeg` ingest candidate reads PCM from the file directly.
- **URL** (Chrome extension or `patmir submit --url <URL>`) — the local FastAPI server (`patmir ui`) pre-downloads with yt-dlp to `~/.patmir/downloads/<cache_key>/source.<ext>` and then runs the same `ffmpeg` ingest candidate on the downloaded file. A dedicated `yt_dlp` ingest candidate is on the roadmap (slice 3 of the D1 spec) so the URL doesn't have to be fully downloaded before the pipeline starts.

Successful runs are materialized into a global cache at `~/.patmir/cache/<key>/dub.wav` keyed by source + pipeline-config hash; subsequent submissions of the same URL/file under the same config return instantly. See `decisions/2026-05-27-low-friction-translation-d1.md`.

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

## Execution models (batch vs watch-along)

The same 9 stages run under two execution models, selected per submission by a
`mode` field (folded into the cache key, so the two never collide):

- **Quality mode (D1, default)** — `run_job` runs all stages strictly in order
  over the whole video and caches one final `dub.wav`. Best quality, one-time
  wait, instant rewatch. The runner loop is `runner.py` (sequential).
- **Watch-along mode (D3, additive)** — the dub is produced in VAD-snapped
  sections and streamed behind a short buffer while the viewer watches; each
  section is written into the cache as it lands (free rewatch). See ADR
  `decisions/2026-06-13-near-realtime-d3-spike.md`. Active TTS is the
  ElevenLabs `eleven_v3` preset-voice pool (`elevenlabs_pool`, no IVC cloning),
  which clears realtime; OmniVoice with a low `num_steps` is the offline fallback.

Watch-along control flow (`patmir/pipeline/streaming/` + `patmir/ui/stream.py`):

```
run_prelude(first ~60s)            establish the cast: profile speakers
   │                               (gender/age), assign pool voices, store
   │                               ECAPA embeddings → populated SpeakerRegistry
   ▼
dub_prelude_window()               reuse the prelude analysis → opening dub
   │                               (section 0, no second Demucs/Scribe pass)
   ▼
StreamScheduler.run()              playhead-driven loop:
   │  ├ Coverage tracks dubbed [start,end) intervals
   │  ├ next section = nearest undubbed point AT/AHEAD of the playhead
   │  │   (seek → jump there; paused → keep filling forward; else backfill)
   │  └ process_section(start) → analyze→ECAPA identity-resolve against the
   │      registry→remap+trim transcript→translate→tts→mux → section chunk
   ▼
append_section() → cache           each chunk written to ~/.patmir/cache/<key>/
                                    section_NNN.wav + sections.json as it lands
```

Cross-section speaker identity is ECAPA-TDNN cosine matching (threshold 0.12,
multi-embedding per speaker) so a character keeps one voice across sections and
across seeks. The browser client polls `/stream/<key>/sections`, fetches each
section WAV, and schedules it on the Web Audio clock at its global `start`
offset, pausing the `<video>` to buffer when the playhead outruns production.

NOTE (perf follow-up): `process_section` calls `run_job` twice per section
(analysis, then translate→tts→mux), rebuilding stage models each call. A
warm-model loop (build stages once, process sections in place) is an open
optimization, not a correctness issue.

## Audio artifacts written per job

See `docs/architecture/runs.md` for the full file map.

## Stage protocols

See `patmir/pipeline/stage.py` for the typed Protocols. See `docs/architecture/data-contracts.md` for every dataclass field with explanations.
