# Stage: emotion

Detects per-segment speech emotion from the source vocal stem. New in Phase 2.x. Sits between `diarize` and `asr`.

## I/O

- **Input:** [`SeparatedAudio`](../../data-contracts.md#separatedaudio) + [`Diarization`](../../data-contracts.md#diarization) (uses turn boundaries as inference windows)
- **Output:** `list[EmotionPrediction]` (`{start, end, emotion, confidence}` per turn window)

## Protocol

```python
class EmotionStage(Protocol):
    def detect(self, separated: SeparatedAudio, diarization: Diarization,
               ctx: JobContext) -> list[EmotionPrediction]: ...
```

## Candidates

- **[emotion2vec-plus-base](emotion2vec-plus-base.md)** — active.

## See also

- ADR: [Emotion routed through ref-clip binning](../../decisions/2026-05-26-emotion-routed-through-ref-clip-binning.md)
- Pipeline: emotion predictions flow into both `ref_picker` (chooses per-`(speaker, emotion)` ref clips) and `apply_emotion_tags` (tags sentences post-ASR).
