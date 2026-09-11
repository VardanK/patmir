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
- **URL** (Chrome extension or `patmir submit --url <URL>`) — the local FastAPI server (`patmir ui`) pre-downloads with yt-dlp to `<downloads_root>/<cache_key>/source.<ext>` and then runs the same `ffmpeg` ingest candidate on the downloaded file. A dedicated `yt_dlp` ingest candidate is on the roadmap (slice 3 of the D1 spec) so the URL doesn't have to be fully downloaded before the pipeline starts.

Successful runs are materialized into a global cache at `<cache_root>/<key>/dub.wav` keyed by source + pipeline-config hash; subsequent submissions of the same URL/file under the same config return instantly. See `decisions/2026-05-27-low-friction-translation-d1.md`.

## Storage roots

Every path above is a **default**, not a constant. All roots come from the `[storage]` table in `patmir.toml`, resolved by `patmir/paths.py` into a `StoragePaths` carried on `PipelineConfig.storage`. Relative paths resolve against the config file's directory (not the CWD). ADR: `decisions/2026-08-01-configurable-storage-roots.md`.

| Root | Default | Holds |
| --- | --- | --- |
| `runs_root` | `<config_dir>/runs` | working dir, one subdir per job, all intermediates |
| `cache_root` | `~/.patmir/cache` | finished dubs keyed by (source, config); what the extension plays |
| `downloads_root` | `~/.patmir/downloads` | yt-dlp downloads for URL submissions |
| `state_root` | `~/.patmir` | local server's bearer token + job index |
| `library_root` | unset | when set, each finished run's deliverables are copied here |

Storage does **not** contribute to the cache key, so relocating a root never orphans an existing dub. Keep `runs_root` and `state_root` on local disk — `runs/` is the live working dir (appended-and-seek-tailed progress log, non-atomic per-stage manifest rewrites, in-place ffmpeg/soundfile writes); `library_root` and `cache_root` are whole-file copies and are fine on a network share.

## Publish (post-`output` runner step)

When `[storage].library_root` is set, the runner copies the finished deliverables out of the job dir after the `output` stage returns:

```
library_root/<job_id>/out.mkv      final container (Armenian default audio)
library_root/<job_id>/out.hy.srt   Armenian subtitles
library_root/<job_id>/dub.wav      Armenian audio alone
```

Copy, not move — the job dir has to survive for `--resume-from` (which reads absolute paths out of the sidecar JSONs) and for `cache.materialize`. Each file is written as `.partial` and renamed inside the destination dir, so an interrupted transfer never leaves a truncated container behind. A publish failure is reported to the progress log and swallowed: the dub is already complete in the job dir. The destination lands in `manifest.published_dir`, surfaced by `patmir runs show`. See `patmir/pipeline/publish.py`.

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

### `JobContext.cancel` is NOT in `extra`

The cooperative cancel token is its own field (`cancel: threading.Event | None`),
with `ctx.cancelled()` and `ctx.raise_if_cancelled()` helpers, because `extra` is
the *data* hand-off channel above and a control signal is not that.

Cancellation has to be cooperative: the heavy stages run in a worker thread via
`asyncio.to_thread`, and a Python thread cannot be killed — `task.cancel()` only
abandons the await while the thread runs on, holding the GPU **and** the shared
accelerator lock, so the next job would queue behind a job the user believes is
dead. Checkpoints, and their cost:

| Checkpoint | Where | Worst-case latency |
| --- | --- | --- |
| Each stage boundary | `runner.py`, before the stage builds | bounded by the longest stage (tts, ~77 s on CUDA) |
| Each TTS decode chunk | the adapter's `on_chunk`, which the candidate already calls | **~8 s** (~7.7 sentences/chunk) |
| Each sentence | the adapter's serial (non-batched) path | per sentence |
| The yt-dlp download | `JobManager._download_url`: before it starts, per playlist entry (`match_filter`), per chunk (progress hook) | one chunk |

The download row exists because the download runs **inside** the accelerator
lock and, until 2026-09-04, was the one place the token was never read: a
YouTube Mix URL (`&list=RD...`) had yt-dlp walking the whole mix for 6+ minutes
with the GPU idle and every later submission `waiting` behind it. `noplaylist`
is set for the same reason — the cache key is per video, so the download must
be too. ADR `decisions/2026-09-04-download-inside-the-lock-and-waiting-cancel.md`.

Raising from `on_chunk` needs no cooperation from the TTS candidate, because it
already invokes that callback after every chunk. `None` (the CLI, bench, tests)
disables every checkpoint. A cancelled run leaves completed stages' artifacts on
disk and marks the stage `cancelled` rather than `failed`, so the job stays
resumable with `--resume-from`. ADR:
`decisions/2026-08-25-cooperative-job-cancellation.md`.

### The audience level rides `ctx.options`, not `extra` and not `entries`

The age register the translator is steered toward (`child` / `teenager` /
`adult`) is a *configuration* scalar, so it lives on `PipelineOptions` and
reaches stages through `ctx.options` — the channel that already exists for
exactly that. It is **not** in `ctx.extra` (the cross-stage data hand-off above)
and **not** in `cfg.entries` (which selects models).

That last exclusion is load-bearing rather than tidy. `stage_cache_key` is
`(stage, candidate_id)` and the cache is process-lifetime and shared between
`JobManager` and `StreamManager`, so a candidate that carried the level in its
constructor would be handed to the next job at a different level; and
`prewarm_stage_cache` builds translate at server boot, before any submission
exists. So the server derives a per-job config replacing **options only**
(`_cfg_for`), and the adapter forwards the level as a per-call `audience=` kwarg
discovered by signature probe — the same mechanism `emotion` uses. One warm
candidate serves every level, and watch-along needed no signature changes at all
because `run_prelude`, `dub_prelude_window` and `process_section` already take
`cfg`. ADR: `decisions/2026-08-28-audience-age-guardrail.md`.

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
  `decisions/2026-06-13-near-realtime-d3-spike.md`. On CUDA all four model
  stages are the **same candidates as batch** since 2026-09-03 (separate was
  the last holdout, `htdemucs_fast` -> `htdemucs_vocals`); only `translate`
  differs, using the batched candidate. Active TTS is the
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
append_section() → cache           each chunk written to <cache_root>/<key>/
                                    section_NNN.wav + sections.json as it lands
```

Cross-section speaker identity is ECAPA-TDNN cosine matching (threshold 0.12,
multi-embedding per speaker) so a character keeps one voice across sections and
across seeks. The browser client polls `/stream/<key>/sections`, fetches each
section WAV, and schedules it on the Web Audio clock at its global `start`
offset, pausing the `<video>` to buffer when the playhead outruns production.

### Section boundaries

Each section analyzes `[start, start + 30 + 8]` and cuts at a VAD silence inside
the pad. A cut candidate must be an interval end **followed by a real gap**: the
separate stage runs Silero with `speech_pad_ms=500`, so speech runs separated by
under ~1 s of silence come back *touching*, and an end equal to the next start
is mid-speech rather than a boundary. A candidate below half the target is also
rejected — an early silence is the wrong cut when the only later boundary sits
past the target, because the whole padded slice is analyzed either way. With no
qualifying candidate the section cuts at the target, accepting a mid-utterance
boundary (the transcript trim drops the partial sentence; the next section
re-analyzes from there). ADR
`2026-08-25-watchalong-startup-and-stall-causes`.

### Pause, abandonment and restart

A paused viewer is **not** an absent one. The scheduler's abandon timer keys off
the client still asking for something (`/sections` polls and playhead POSTs),
not off the playhead moving — the client stops POSTing the playhead while
paused, so the older rule ended the stream after 45 s of pause and `submit()`
then returned that finished record, leaving playback stuck with no recovery. A
`done` stream whose cached manifest is not `complete` now restarts and resumes
from the sections already cached, numbering **past** the highest cached index so
a restart cannot overwrite `section_NNN.wav`.

### Per-section telemetry

Every section's timing is written into its `sections.json` entry — per-stage
walls, `build_s`, `rtf`, `buffer_ahead_s`, and `other_s` (`wall_s` minus the sum
of stage walls: the ffmpeg slice/trim, ECAPA embed and age-gender profile that
no stage owns, ~1.6 s per section). The fields are additive and optional, and
`index`/`start`/`end` always come from the caller, so audio and timing cannot
disagree. `patmir/pipeline/streaming/timing.py` is the single reader, shared
with `patmir bench`, so the live path and the benchmark cannot drift.

Startup models that `prewarm_stage_cache` cannot reach — the speaker embedder
and the age/gender profiler, which are not stages — are prewarmed at server boot
alongside the **streaming** stage set, whose `separate` and `translate`
candidates differ from batch and so were previously cache misses on the first
stream of every server lifetime.

### Sections are pipelined (2026-09-02)

`process_section` runs its phases back to back and is still the simple path
(`stream_sections`, the bench forward-loop, the tests). The **live server and
the scheduler-driven bench** instead go through `SectionPipeline`
(`streaming/pipeline.py`), which overlaps the one thing in a section that does
not use the GPU with the one thing that does:

```
section k    [analyse][translate .......][speak]
section k+1           [analyse]                 [translate ....][speak]
                      ^^^^^^^^ the GPU, busy during k's network wait
```

Translate was **44.2 s of a 153.9 s watch-along wall (29 %)** — one structured
cloud round-trip per section, measured between 1.75 s and 13.34 s, with the GPU
idle throughout. Analysis is the GPU work that fits in that gap. Measured:
**agg RTF 0.511 -> 0.396, wall 153.9 -> 119.3 s**, 7 prefetch hits of 8
possible, 0 discarded.

**The GPU is never used by two phases at once**, and that is a hard requirement
rather than tidiness: the TTS decode captures a CUDA graph, and capture fails
or corrupts if another thread allocates on the device. One lock inside the
driver serializes every GPU phase (`analyze_section`, `speak_section`) plus the
registry mutation that identity resolution performs; `translate_section` is the
only phase that runs outside it. This is why `process_section` was split into
three phases in the first place — the runner's `resume_from`/`stop_after` range
already allowed it.

Prefetch is one section deep, so a seek throws away at most one section's work,
and it refuses to trust stale analysis in four ways: a different start
discards it, a `max_end` clamp discards it (the clamp changes where the section
ends), a prefetch is never *started* with a clamp (the next window's clamp comes
from coverage the scheduler has not updated yet), and `close()` waits for an
in-flight analysis because a torch forward cannot be interrupted and abandoning
the thread would hand the GPU to the next job while we still held it. Section
numbering stays with the caller, since a resumed stream must number past the
highest cached index (quality issue #22).

`[pipeline.options] section_prefetch = false` restores the fully serial loop.

## Audio artifacts written per job

See `docs/architecture/runs.md` for the full file map.

## Stage protocols

See `patmir/pipeline/stage.py` for the typed Protocols. See `docs/architecture/data-contracts.md` for every dataclass field with explanations.
