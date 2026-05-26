"""Shared sentence grouping + VAD post-filtering for ASR stages.

Word-level timestamps come out of every ASR backend at different fidelity;
this groups them into sentences for downstream stages. Splits on terminal
punctuation OR a gap > `max_gap_s` between consecutive words.

`filter_words_by_vad` drops words whose times don't overlap any speech
interval from Silero VAD — the post-filter alternative to hard-masking the
audio before ASR. Lets ASR see full audio (no soft-onset clipping) while
still dropping hallucinations from pure non-speech-vocal regions.

`is_non_lexical_repetition` detects onomatopoeic sentences like 'Ха-ха-ха!'
or 'Ням-ням-ням' so they can be dropped before translate/TTS — the original
laughter/eating sounds already survive in the music bed via htdemucs.
"""
from __future__ import annotations

import re
from collections import Counter

from patmir.pipeline.stage import Sentence, Word


def is_non_lexical_repetition(text: str) -> bool:
    """True for onomatopoeic repetition like 'Ха-ха-ха!', 'Ня-ня-ня',
    'Ням-ням-ням-ням!', 'Ах-ха-ха!'. The threshold is 60% of tokens
    repeating, which catches both pure repetition (3/3 'ха') and minor
    variants (2/3 'ха' + 1 'ах') while keeping real short utterances
    like 'Да нет' or 'что-то' (50% repetition).
    """
    cleaned = re.sub(r"[!?.,…—]+", " ", text).strip()
    tokens = [t.lower() for t in re.split(r"[\s\-]+", cleaned) if t]
    if len(tokens) < 2:
        return False
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(tokens) >= 0.6


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
