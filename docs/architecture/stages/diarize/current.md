# Active diarize config

```toml
[pipeline]
diarize = "sortformer_mlx"

[[stages.diarize.candidates]]
id                = "sortformer_mlx"
module            = "patmir.pipeline.stages.diarize.sortformer_mlx"
repo_id           = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32"
chunk_duration    = 5.0
merge_gap_seconds = 0.5
```

**Local** (2026-06-15) — NVIDIA Sortformer 4-speaker streaming diarization on
MLX via `mlx-audio` (`mlx_audio.vad`). Replaced cloud Scribe v2 after ElevenLabs
credits were exhausted (ADR `2026-06-15-local-asr-diarize-via-mlx-audio`). See
`sortformer-mlx.md` for the model file.

## How it produces the Diarization

- `mlx_audio.vad.load(repo_id).generate_stream(vocals, chunk_duration=5.0)`
  yields chunk results, each with `.segments`
  (`DiarizationSegment(.start, .end, .speaker:int)`).
- Speaker ints normalized to `SPEAKER_NN`; adjacent same-speaker segments with a
  gap ≤ `merge_gap_seconds` merged into one `SpeakerTurn`.
- VAD post-filter drops turns with zero intersection with the separate stage's
  Silero intervals.
- **Overlap intervals re-derived** from regions where ≥2 distinct speakers are
  active — restores the overlap output Scribe dropped (quality issue #5).
- `refs={}` on return; the runner's `pick_refs` fills per-`(speaker, emotion)`
  refs after the emotion stage (unchanged).

## Constraints

- **Capped at 4 speakers** (model architecture). Content with >4 distinct
  speakers will under-cluster — flip to `pyannote_3_1` (≤8) if that bites.
- End-to-end neural model: no clustering hyperparameters, no HF token / license
  gate (unlike pyannote).

## Fallbacks (registered, disabled by selector)

- `pyannote_3_1` (`pyannote/speaker-diarization-community-1`) — local fallback,
  ≤8 speakers, but merges short cartoon turns (the reason Scribe was adopted in
  the first place). Needs `HF_TOKEN` + license acceptance on the gated repos.
  Flip `[pipeline].diarize = "pyannote_3_1"` to revert.
- `elevenlabs_scribe_v2` (cloud, needs credits) remains registered.
