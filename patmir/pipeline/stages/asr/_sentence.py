"""Shared sentence grouping + VAD post-filtering for ASR stages.

Word-level timestamps come out of every ASR backend at different fidelity;
this groups them into sentences for downstream stages. Splits on terminal
punctuation OR a gap > `max_gap_s` between consecutive words.

`filter_words_by_vad` drops words whose times don't overlap any speech
interval from Silero VAD — the post-filter alternative to hard-masking the
audio before ASR. Lets ASR see full audio (no soft-onset clipping) while
still dropping hallucinations from pure non-speech-vocal regions.

`is_non_lexical(text)` is the umbrella non-lexical detector. It returns True for:
  - multi-token onomatopoeic repetition like 'Ха-ха-ха!' (`is_non_lexical_repetition`)
  - single-token interjections in a known list like 'Ха!', 'Хм?', 'Свист!', 'Wow!'
  - multi-token sentences where every unique token is non-lexical
    (e.g. '-фью -фью цыпа цыпа' — bird chirps that Whisper transliterated)

The original laughter / animal sounds survive in the music bed via htdemucs's
non-speech-vocals stem, so dropping these sentences doesn't lose anything.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from patmir.pipeline.stage import Sentence, Transcript, Word


def write_transcript(job_dir: Path, transcript: Transcript) -> None:
    """Shared transcript.json + sentences.json writer for all ASR candidates
    and for the runner's post-speaker-tagging rewrite. Includes `speaker_id`
    on each sentence (will be `None` until the diarize/speaker_tag pass runs).
    """
    payload = {
        "language": transcript.language,
        "words": [asdict(w) for w in transcript.words],
        "sentences": [
            {
                "idx": s.idx,
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "speaker_id": s.speaker_id,
                "emotion": s.emotion,
                "words": [asdict(w) for w in s.words],
            }
            for s in transcript.sentences
        ],
    }
    (job_dir / "transcript.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    sentences_only = [
        {
            "idx": s.idx, "text": s.text, "start": s.start, "end": s.end,
            "speaker_id": s.speaker_id, "emotion": s.emotion,
        }
        for s in transcript.sentences
    ]
    (job_dir / "sentences.json").write_text(json.dumps(sentences_only, indent=2, ensure_ascii=False))


NON_LEXICAL_SINGLE_TOKENS: frozenset[str] = frozenset({
    # Russian laughter / breath / sighs
    "ха", "хи", "хо", "хы", "хэ", "хех",
    "ах", "ох", "ой", "ух", "эх", "их", "ой-ёй", "ой-ой", "ох-ох",
    "ого", "ага", "ого-го", "ну-ну", "ой-ой-ой",
    "а", "о", "у", "и", "э", "ы",
    "ммм", "мм", "мммм", "хм", "хмм", "угу", "ага", "ага-ага",
    "эй", "ну", "ня", "хух", "бу", "бр", "бр-р", "бр-р-р",
    "тьфу", "пф", "фф", "фью", "фьюх", "цыпа", "цып",
    "тс", "тсс", "ш", "ш-ш", "ши", "чу", "кш", "апчхи", "чих",
    # Sound names whisper sometimes outputs in place of the sound itself
    "свист", "хрюк", "хрюкс", "плюх", "бах", "бабах", "бац", "клац",
    # English equivalents
    "ha", "hi", "ho", "ah", "oh", "uh", "eh", "ew", "ugh",
    "hm", "mm", "mmm", "um", "huh", "wow", "hey",
    "boo", "woah", "yum", "shh", "psst", "argh", "ouch",
    "tsk", "phew", "whew", "yikes",
})


def _tokenize_for_lexical_check(text: str) -> list[str]:
    cleaned = re.sub(r"[!?.,…—]+", " ", text).strip()
    return [t.lower() for t in re.split(r"[\s\-]+", cleaned) if t]


def is_non_lexical_repetition(text: str) -> bool:
    """True for onomatopoeic repetition like 'Ха-ха-ха!', 'Ня-ня-ня',
    'Ням-ням-ням-ням!', 'Ах-ха-ха!'. The threshold is 60% of tokens
    repeating, which catches both pure repetition (3/3 'ха') and minor
    variants (2/3 'ха' + 1 'ах') while keeping real short utterances
    like 'Да нет' or 'что-то' (50% repetition).
    """
    tokens = _tokenize_for_lexical_check(text)
    if len(tokens) < 2:
        return False
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(tokens) >= 0.6


def is_non_lexical_interjection(text: str) -> bool:
    """True when the sentence is a single-token interjection from the known
    non-lexical set — 'Ха!', 'Хм?', 'Свист!', 'Wow!'. Whisper sometimes
    transcribes a real whistle or sniff as the corresponding word, and the
    multi-token repetition heuristic misses these because they're one token.
    """
    tokens = _tokenize_for_lexical_check(text)
    if len(tokens) != 1:
        return False
    return tokens[0] in NON_LEXICAL_SINGLE_TOKENS


def is_non_lexical(text: str) -> bool:
    """Umbrella check used by ASR / speaker-tag / ref-picker.

    Returns True if `text` matches any of:
      (a) `is_non_lexical_repetition` — multi-token repetition.
      (b) `is_non_lexical_interjection` — single token in the known set.
      (c) every unique token in the sentence belongs to the known set
          (e.g. '-фью -фью цыпа цыпа' — bird chirps Whisper transliterated).
    """
    if is_non_lexical_repetition(text):
        return True
    if is_non_lexical_interjection(text):
        return True
    tokens = _tokenize_for_lexical_check(text)
    if tokens and all(t in NON_LEXICAL_SINGLE_TOKENS for t in tokens):
        return True
    return False


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
