"""Post-ASR pass: tag each Sentence with an `emotion` bin.

Mirrors `speaker_tag.apply_speaker_tags` but for emotion. Called from the
runner inside the `"asr"` branch right after `apply_speaker_tags` runs —
once sentences carry `speaker_id`, this pass overlays the emotion bin from
the per-turn `EmotionPrediction` list emitted by the `emotion` stage.

Algorithm per sentence:

  1. Compute overlap-duration with each emotion prediction window.
  2. The bin with the largest accumulated overlap wins. If no prediction
     intersects (rare — a sentence entirely outside the diarizer's turn
     windows), default to `"neutral"`.

No sentence splits are performed here — emotion transitions inside a
single sentence are rare in practice, and even when they happen the
sentence has already been split (or not) by the speaker-tag pass, which
correlates strongly with emotion changes. Keeping this pass non-mutating
on sentence boundaries also avoids re-indexing the speaker-tagged list.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from patmir.pipeline.stage import (
    Emotion,
    EmotionPrediction,
    Sentence,
    Transcript,
)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _dominant_emotion(sent: Sentence, predictions: list[EmotionPrediction]) -> Emotion:
    weight: dict[Emotion, float] = defaultdict(float)
    for p in predictions:
        ov = _overlap(sent.start, sent.end, p.start, p.end)
        if ov > 0:
            weight[p.emotion] += ov
    if not weight:
        return "neutral"
    return max(weight, key=lambda e: weight[e])


def apply_emotion_tags(
    transcript: Transcript,
    predictions: list[EmotionPrediction],
) -> Transcript:
    """Return a new Transcript with `emotion` populated on each sentence.

    Pure overlay — does not re-index or split sentences.
    """
    out = [replace(s, emotion=_dominant_emotion(s, predictions)) for s in transcript.sentences]
    return Transcript(language=transcript.language, words=transcript.words, sentences=out)
