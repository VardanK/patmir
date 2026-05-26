"""Post-ASR pass: tag each Sentence with a speaker_id; split sentences
that cross a speaker boundary into per-speaker fragments when feasible.

Lives next to ASR because it operates on `Transcript`. Called from the
runner inside the `"asr"` branch right after `stage_obj.transcribe(...)`
returns — keeps every ASR candidate (`whisper_mlx`, `faster_whisper`,
`qwen3_asr`) untouched.

Algorithm per sentence:

  1. Compute overlap-duration with each speaker turn.
  2. If one speaker covers >= DOMINANT_THRESHOLD of the sentence's duration,
     tag the sentence with that speaker. Done.
  3. Otherwise pick the speaker-change boundary inside the sentence: the
     mid-point between the dominant-speaker's last-second and the next
     speaker's first-second within the sentence. Split words at the first
     word whose `start` >= boundary. If both halves contain >= 1 word,
     emit two re-indexed sentences; else fall back to dominant.

Also exposes `backfill_ref_text(diarization, transcript)` — fills in
`SpeakerRef.ref_text` for each speaker by joining the ASR words whose
midpoint falls inside the reference window.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from patmir.pipeline.stage import (
    Diarization,
    Sentence,
    SpeakerTurn,
    Transcript,
    Word,
)
from patmir.pipeline.stages.asr._sentence import is_non_lexical

DOMINANT_THRESHOLD = 0.85
MIN_HALF_WORDS = 1


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _dominant_speaker(sent: Sentence, turns: list[SpeakerTurn]) -> tuple[str | None, float, dict[str, float]]:
    """Return (best_speaker_id, best_overlap_seconds, per_speaker_overlap_seconds)."""
    per_speaker: dict[str, float] = defaultdict(float)
    for t in turns:
        ov = _overlap(sent.start, sent.end, t.start, t.end)
        if ov > 0:
            per_speaker[t.speaker_id] += ov
    if not per_speaker:
        return None, 0.0, {}
    best_spk = max(per_speaker.items(), key=lambda kv: kv[1])
    return best_spk[0], best_spk[1], dict(per_speaker)


def _boundary_time(sent: Sentence, turns: list[SpeakerTurn], dominant: str) -> float | None:
    """For a mixed-speaker sentence, return the time at which the speaker first
    changes away from `dominant`, or `None` if there is no clean change inside
    the sentence."""
    in_window = [
        t for t in turns
        if t.end > sent.start and t.start < sent.end
    ]
    in_window.sort(key=lambda t: t.start)
    # Find the first turn inside the sentence whose speaker != dominant.
    for t in in_window:
        if t.speaker_id == dominant:
            continue
        change = max(t.start, sent.start)
        if change > sent.start and change < sent.end:
            return change
    return None


def _split_words(words: list[Word], boundary: float) -> tuple[list[Word], list[Word]]:
    left: list[Word] = []
    right: list[Word] = []
    for w in words:
        if w.start < boundary:
            left.append(w)
        else:
            right.append(w)
    return left, right


def apply_speaker_tags(transcript: Transcript, diarization: Diarization) -> Transcript:
    """Return a new Transcript whose sentences carry `speaker_id` and whose
    mixed-speaker sentences are split where feasible. Sentences are re-indexed
    contiguously from 0.

    Non-lexical content is dropped here so it never reaches translate/TTS,
    regardless of which ASR backend produced it:
      - sentences whose whole text is non-lexical are skipped;
      - candidate splits where either half is non-lexical fall back to
        dominant-speaker assignment of the (still-lexical) parent sentence
        rather than emitting a non-lexical fragment.
    """
    out: list[Sentence] = []
    next_idx = 0
    for sent in transcript.sentences:
        # Centralized non-lexical drop. whisper_mlx already filters at the
        # segment level, but other ASR backends (faster_whisper, qwen3_*)
        # don't — keep this check here so the rule holds regardless of backend.
        if is_non_lexical(sent.text):
            continue

        dur = max(1e-6, sent.end - sent.start)
        best, best_ov, per = _dominant_speaker(sent, diarization.turns)

        if best is None:
            out.append(replace(sent, idx=next_idx, speaker_id=None))
            next_idx += 1
            continue

        if best_ov / dur >= DOMINANT_THRESHOLD:
            out.append(replace(sent, idx=next_idx, speaker_id=best))
            next_idx += 1
            continue

        boundary = _boundary_time(sent, diarization.turns, best)
        if boundary is None:
            out.append(replace(sent, idx=next_idx, speaker_id=best))
            next_idx += 1
            continue

        left_words, right_words = _split_words(sent.words, boundary)
        if len(left_words) < MIN_HALF_WORDS or len(right_words) < MIN_HALF_WORDS:
            out.append(replace(sent, idx=next_idx, speaker_id=best))
            next_idx += 1
            continue

        left_text = " ".join(w.text for w in left_words)
        right_text = " ".join(w.text for w in right_words)
        # If splitting would orphan a non-lexical fragment (e.g. cut off a
        # "Хм?" or "-ха" off the end of a real sentence), fall back to
        # dominant-speaker assignment of the whole sentence — better to keep
        # the lexical content intact and absorb the interjection than emit
        # a junk sentence for translate/TTS.
        if is_non_lexical(left_text) or is_non_lexical(right_text):
            out.append(replace(sent, idx=next_idx, speaker_id=best))
            next_idx += 1
            continue

        # Re-derive each half's speaker by max-overlap on the half's own time range.
        def _spk_for(word_run: list[Word]) -> str:
            w_start = word_run[0].start
            w_end = word_run[-1].end
            pseudo = Sentence(idx=0, text="", start=w_start, end=w_end, words=word_run)
            half_best, _, _ = _dominant_speaker(pseudo, diarization.turns)
            return half_best or best

        left_spk = _spk_for(left_words)
        right_spk = _spk_for(right_words)
        out.append(Sentence(
            idx=next_idx,
            text=left_text,
            start=left_words[0].start,
            end=left_words[-1].end,
            words=left_words,
            speaker_id=left_spk,
        ))
        next_idx += 1
        out.append(Sentence(
            idx=next_idx,
            text=right_text,
            start=right_words[0].start,
            end=right_words[-1].end,
            words=right_words,
            speaker_id=right_spk,
        ))
        next_idx += 1

    return Transcript(language=transcript.language, words=transcript.words, sentences=out)


def backfill_ref_text(diarization: Diarization, transcript: Transcript) -> None:
    """Fill each SpeakerRef.ref_text in place by joining ASR words whose
    midpoint falls inside the reference window. Mutates `diarization.refs`.

    Speakers whose ref window turns out to contain only non-lexical content
    (laughter, bird chirps, single-word interjections) are removed from
    `diarization.refs` entirely — the TTS adapter then falls back to the
    global ref voice for that speaker's sentences. A SPEAKER_03 character
    who only laughs in the source shouldn't seed their cloned voice with a
    "Ха-ха-ха!" reference.
    """
    drop: list[str] = []
    for spk, ref in diarization.refs.items():
        words = [
            w for w in transcript.words
            if ref.source_start <= 0.5 * (w.start + w.end) < ref.source_end
        ]
        ref.ref_text = " ".join(w.text for w in words).strip()
        if not ref.ref_text or is_non_lexical(ref.ref_text):
            drop.append(spk)
    for spk in drop:
        del diarization.refs[spk]
