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

- **[pyannote-3-1](pyannote-3-1.md)** — active. License-gated.

## See also

- ADR: [pyannote 3.1 for diarization](../../decisions/2026-05-26-pyannote-3-1-for-diarization.md)
- [current.md](current.md)
