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
# duplicate_containment  = 0.4   # >1.0 disables the duplicate-slot merge
# duplicate_min_seconds  = 1.0
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
- **Duplicate speaker slots merged** (2026-08-24) — the model sometimes puts ONE
  actor on TWO of its output slots. `merge_duplicate_speakers` folds two ids
  whose aggregate `shared / min(total)` reaches `duplicate_containment` (0.4)
  over at least `duplicate_min_seconds` (1.0 s) of shared audio; union-find, so
  three slots collapse transitively, the most-speech id survives (ties by lowest),
  and relabelled turns are re-merged because two slots of one speaker become
  self-overlapping turns that `derive_overlaps` assumes cannot happen. Runs
  **before** the VAD filter and overlap derivation, so the phantom overlaps the
  duplicate creates never reach `pick_refs`. Set `duplicate_containment` above
  1.0 to disable. **The model is not deterministic** — four runs of one file gave
  98/106/110/111 turns and put the duplicated pair's containment at 0.705, 0.594
  and twice above 0.6, while genuinely-distinct pairs topped out at 0.259; 0.4
  sits in that gap, and a first cut at 0.6 fired on some runs and not others.
  Quality issue #16, ADR
  `2026-08-24-speaker-identity-word-level-attribution`.
- **Overlap intervals re-derived** from regions where ≥2 distinct speakers are
  active — restores the overlap output Scribe dropped (quality issue #5).
- `refs={}` on return; the runner's `pick_refs` fills per-`(speaker, emotion)`
  refs after the emotion stage (unchanged).

## Constraints

- **Capped at 4 speakers** (model architecture). Content with >4 distinct
  speakers will under-cluster — flip to `pyannote_3_1` (≤8) if that bites. This
  **does** bite on real content: `samples/sample_1.mkv` has at least six speaking
  characters and its squad scene runs entirely on two ids while the other two sit
  idle, with question and answer landing on the same id and one 7.16 s "turn"
  carrying a four-way exchange. Distinct actors then share one pinned ref clip
  and so one voice, which no downstream fix can undo. Quality issue #18.
- Two of a clip's "4 speakers" can be one person — see the duplicate-slot merge
  above. Check `diarize/turns.json`'s `merged_speakers` field before trusting the
  speaker count.
- End-to-end neural model: no clustering hyperparameters, no HF token / license
  gate (unlike pyannote).

## Fallbacks (registered, disabled by selector)

- `pyannote_3_1` (`pyannote/speaker-diarization-community-1`) — local fallback,
  ≤8 speakers, but merges short cartoon turns (the reason Scribe was adopted in
  the first place). Needs `HF_TOKEN` + license acceptance on the gated repos.
  Flip `[pipeline].diarize = "pyannote_3_1"` to revert.
- `elevenlabs_scribe_v2` (cloud, needs credits) remains registered.
