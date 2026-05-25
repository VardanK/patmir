"""Shared sentence grouping + VAD post-filtering for ASR stages.

Word-level timestamps come out of every ASR backend at different fidelity;
this groups them into sentences for downstream stages. Splits on terminal
punctuation OR a gap > `max_gap_s` between consecutive words.

`filter_words_by_vad` drops words whose times don't overlap any speech
interval from Silero VAD — the post-filter alternative to hard-masking the
audio before ASR. Lets ASR see full audio (no soft-onset clipping) while
still dropping hallucinations from pure non-speech-vocal regions.
"""
from __future__ import annotations

from patmir.pipeline.stage import Sentence, Word


def filter_words_by_vad(
    words: list[Word],
    intervals: list[tuple[float, float]] | None,
    *,
    margin_s: float = 0.10,
) -> list[Word]:
    """Drop words that don't overlap any VAD speech interval.

    `margin_s` widens each interval at both ends to absorb sub-window slop —
    word timestamps from ASR and VAD-window boundaries don't perfectly agree.
    """
    if not intervals or not words:
        return list(words)
    expanded = [(max(0.0, s - margin_s), e + margin_s) for s, e in intervals]
    kept: list[Word] = []
    for w in words:
        for s, e in expanded:
            if w.end >= s and w.start <= e:
                kept.append(w)
                break
    return kept


def group_into_sentences(words: list[Word], max_gap_s: float = 0.6) -> list[Sentence]:
    sentences: list[Sentence] = []
    bucket: list[Word] = []
    idx = 0

    def flush() -> None:
        nonlocal idx, bucket
        if not bucket:
            return
        text = " ".join(w.text for w in bucket).strip()
        sentences.append(Sentence(
            idx=idx,
            text=text,
            start=bucket[0].start,
            end=bucket[-1].end,
            words=list(bucket),
        ))
        idx += 1
        bucket = []

    for i, w in enumerate(words):
        bucket.append(w)
        ends_sentence = w.text.endswith((".", "!", "?", "。", "！", "？"))
        gap_next = words[i + 1].start - w.end if i + 1 < len(words) else 0.0
        if ends_sentence or gap_next > max_gap_s:
            flush()
    flush()
    return sentences
