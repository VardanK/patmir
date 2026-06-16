# sortformer-mlx (diarize)

- **Repo / version:** `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32`
- **Library:** `mlx-audio` (`mlx_audio.vad`) — `github.com/Blaizzy/mlx-audio`
- **Architecture:** NVIDIA Sortformer v2.1 — end-to-end streaming speaker
  diarization with AOSC compression (≤4 speakers)
- **Apple Silicon (MLX):** ✅ first-class. fp32 weights.
- **Languages:** language-agnostic (acoustic)
- **License / gating:** non-gated HF download (no HF token / license-accept flow,
  unlike pyannote)
- **Integration surface:** `patmir/pipeline/stages/diarize/sortformer_mlx.py`
- **Status in pipeline:** **active** (2026-06-15) — replaced cloud
  `elevenlabs_scribe_v2`. ADR: `decisions/2026-06-15-local-asr-diarize-via-mlx-audio.md`.
- **Last evaluated:** 2026-06-15 (probe + 90s end-to-end run).

## Output contract

`generate_stream(path, chunk_duration=5.0)` yields chunk results with
`.segments`, each a `DiarizationSegment(.start, .end, .speaker:int)`. The
candidate:
- normalizes `speaker` ints → `SPEAKER_NN`,
- merges adjacent same-speaker segments (gap ≤ `merge_gap_seconds`) into
  `SpeakerTurn`s,
- VAD-post-filters against `separated.vad_intervals`,
- re-derives `OverlapInterval`s (≥2 distinct speakers active),
- returns `Diarization(turns, overlaps, refs={}, sample_rate)` and writes
  `diarize/turns.json`.

## Our experience

- Probe (45s): fast (<0.05× realtime), clean per-speaker segments, fragmented
  into many short same-speaker pieces (hence the merge step).
- End-to-end neural diarization resolves short cartoon turns better than
  pyannote — the failure mode that originally pushed us to cloud Scribe
  (`2026-05-27-scribe-v2-for-asr-and-diarize.md`).
- Restores overlap detection that Scribe lacked.

## Knobs (`[[stages.diarize.candidates]]`)

| Key | Default | Effect |
| --- | --- | --- |
| `repo_id` | — | HF repo |
| `chunk_duration` | `5.0` | streaming chunk size (seconds) |
| `merge_gap_seconds` | `0.5` | merge adjacent same-speaker turns within this gap |

## Revisit

- >4 distinct speakers, or turn merging issues → `[pipeline].diarize = "pyannote_3_1"` (≤8 speakers).
