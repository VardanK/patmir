# Stage: diarize

Assigns "who spoke when" labels (`SpeakerTurn`s) + detects overlap regions. Required for per-character voice cloning and per-`(speaker, emotion)` reference clip selection.

## I/O

- **Input:** [`SeparatedAudio`](../../data-contracts.md#separatedaudio)
- **Output:** [`Diarization`](../../data-contracts.md#diarization)

Note: ref-clip slicing (`ref_picker.pick_refs`) **runs in the runner after the `emotion` stage**, not inside this stage. Diarize writes `turns.json` + overlaps and returns `Diarization` with empty `refs`.

## Protocol

```python
class DiarizeStage(Protocol):
    def diarize(self, separated: SeparatedAudio, ctx: JobContext) -> Diarization: ...
```

## Candidates

- **[sortformer-mlx](sortformer-mlx.md)** — active (2026-06-15). Local NVIDIA Sortformer via mlx-audio; ≤4 speakers; re-derives overlaps.
- **[pyannote-3-1](pyannote-3-1.md)** — local fallback. License-gated; ≤8 speakers.
- `elevenlabs_scribe_v2` — cloud fallback (needs ElevenLabs credits; shared with asr).

## See also

- ADR: [Local ASR + diarize via mlx-audio](../../decisions/2026-06-15-local-asr-diarize-via-mlx-audio.md)
- ADR: [Scribe v2 for ASR + diarize](../../decisions/2026-05-27-scribe-v2-for-asr-and-diarize.md) (superseded for diarize)
- ADR: [pyannote 3.1 for diarization](../../decisions/2026-05-26-pyannote-3-1-for-diarization.md)
- [current.md](current.md)
