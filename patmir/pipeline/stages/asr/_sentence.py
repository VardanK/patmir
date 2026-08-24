"""Shared sentence grouping + VAD post-filtering for ASR stages.

Word-level timestamps come out of every ASR backend at different fidelity;
this groups them into sentences for downstream stages. Splits on terminal
punctuation OR a gap > `max_gap_s` between consecutive words, and optionally
caps sentence duration via `max_sentence_s` (splitting at the widest interior
gap) for backends that emit no punctuation.

`filter_words_by_vad` drops words whose times don't overlap any speech
interval from Silero VAD — the post-filter alternative to hard-masking the
audio before ASR. Lets ASR see full audio (no soft-onset clipping) while
still dropping hallucinations from pure non-speech-vocal regions.

`is_non_lexical(text)` is the umbrella non-lexical detector. It returns True for:
  - multi-token onomatopoeic repetition like 'Ха-ха-ха!' (`is_non_lexical_repetition`)
  - single-token interjections in a known list like 'Ха!', 'Хм?', 'Свист!', 'Wow!'
  - multi-token sentences where every unique token is non-lexical
    (e.g. '-фью -фью цыпа цыпа' — bird chirps that Whisper transliterated)

`word_in_target_script(text, lang)` rejects words with no character in the
target language's script — the Qwen3-ASR CJK-hallucination guard.

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


def is_non_lexical_letter_run(text: str) -> bool:
    """True when the sentence is a single token consisting of one letter
    (or one base+repeat pair) repeated many times — Whisper sometimes
    transcribes long screams or sighs as ``'Аааааааааааааа'`` (a single
    Cyrillic letter `а` repeated 14 times, no whitespace), which falls
    through both the multi-token-repetition AND the known-interjection
    checks. Such giant pseudo-words break downstream TTS (ElevenLabs
    returns malformed audio when asked to synth them).

    Threshold: one base letter (Counter most-common) covering ≥70% of
    a single token whose length is ≥4. Catches ``'Ааааа'``, ``'хххх'``,
    ``'Ииииии'``, and the 223-char ``'Աաաա...'`` we hit on masha-2-5min.
    """
    tokens = _tokenize_for_lexical_check(text)
    if len(tokens) != 1:
        return False
    tok = tokens[0]
    if len(tok) < 4:
        return False
    char_counts = Counter(tok)
    most_common_share = char_counts.most_common(1)[0][1] / len(tok)
    return most_common_share >= 0.70


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
    if is_non_lexical_letter_run(text):
        return True
    tokens = _tokenize_for_lexical_check(text)
    if tokens and all(t in NON_LEXICAL_SINGLE_TOKENS for t in tokens):
        return True
    return False


def word_in_target_script(text: str, language: str) -> bool:
    """True if `text` contains at least one character in the script the
    target language uses.

    Qwen3-ASR routinely hallucinates CJK characters and emoji on non-speech
    vocals (cartoon sound effects, screams) even when transcribing Russian or
    English, and those would translate to nonsense if let through. Shared by
    the MLX and transformers Qwen3-ASR candidates so the two cannot drift.
    """
    if not text:
        return False
    if language == "ru":
        return any("\u0400" <= ch <= "\u04ff" for ch in text)
    if language == "en":
        return any("a" <= ch.lower() <= "z" for ch in text)
    if language == "hy":
        return any("\u0530" <= ch <= "\u058f" for ch in text)
    # Unknown target language — be permissive, just reject pure CJK.
    return not all("\u4e00" <= ch <= "\u9fff" or ch.isspace() for ch in text)


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


def _split_longest_gap(run: list[Word], max_sentence_s: float) -> list[list[Word]]:
    """Recursively halve `run` at its widest interior word gap until every piece
    is at most `max_sentence_s` long (or cannot be split further).

    Needed because `max_gap_s` alone has a floor: some stretches of dialogue
    genuinely contain no pause above the threshold. Measured case — a 22.9 s
    run of 56 words whose widest interior gap was 0.32 s, spanning 11
    diarization turns. It was assigned to one speaker (86% dominance, just over
    the 0.85 threshold) and so synthesized in one voice, swallowing another
    character's 1.5 s line, and handed the TTS a 22.9 s decode chunk.

    Splitting at the widest gap rather than the midpoint keeps the cut on the
    most plausible utterance boundary available.
    """
    if len(run) < 2 or (run[-1].end - run[0].start) <= max_sentence_s:
        return [run]
    # Widest gap between consecutive words; ties take the earliest.
    best_i, best_gap = None, -1.0
    for i in range(len(run) - 1):
        gap = run[i + 1].start - run[i].end
        if gap > best_gap:
            best_i, best_gap = i, gap
    if best_i is None:
        return [run]
    left, right = run[: best_i + 1], run[best_i + 1 :]
    if not left or not right:
        return [run]
    return _split_longest_gap(left, max_sentence_s) + _split_longest_gap(right, max_sentence_s)


def group_into_sentences(
    words: list[Word],
    max_gap_s: float = 0.6,
    max_sentence_s: float | None = None,
) -> list[Sentence]:
    """Group `words` into sentences on terminal punctuation or a gap over
    `max_gap_s`.

    `max_sentence_s`, when set, additionally caps how long any one sentence may
    be — see `_split_longest_gap`. It defaults to None so the backends that were
    already producing short sentences (whisper, faster-whisper, scribe,
    openrouter) keep their exact previous behaviour; the Qwen3 candidates pass a
    value because their ASR emits no terminal punctuation at all, leaving
    `max_gap_s` as the only other signal.
    """
    runs: list[list[Word]] = []
    bucket: list[Word] = []

    for i, w in enumerate(words):
        bucket.append(w)
        ends_sentence = w.text.endswith((".", "!", "?", "。", "！", "？"))
        gap_next = words[i + 1].start - w.end if i + 1 < len(words) else 0.0
        if ends_sentence or gap_next > max_gap_s:
            runs.append(bucket)
            bucket = []
    if bucket:
        runs.append(bucket)

    if max_sentence_s is not None and max_sentence_s > 0:
        runs = [piece for run in runs for piece in _split_longest_gap(run, max_sentence_s)]

    sentences: list[Sentence] = []
    for run in runs:
        if not run:
            continue
        sentences.append(Sentence(
            idx=len(sentences),
            text=" ".join(w.text for w in run).strip(),
            start=run[0].start,
            end=run[-1].end,
            words=list(run),
        ))
    return sentences
