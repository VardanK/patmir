"""Diarization-driven second-pass ASR.

Whisper's first pass over the whole vocal stem can lose real dialogue when
it sits inside a long sustained-noise region (e.g. 70 s of a character
laughing). The decoder anchors on the dominant content and the minority
speech gets transcribed as more laughter — or skipped entirely.

This module recovers those misses by running a *targeted* second pass on
slices defined by pyannote turns that the first pass left without lexical
content. The slice comes from `speech_vocals` (Demucs vocals masked by
Silero VAD with cosine fades), so the re-ASR sees clean speech without the
surrounding-noise anchoring problem.

Wiring: called by the runner immediately after `apply_speaker_tags`, before
the second `apply_speaker_tags` pass that assigns speakers to the newly
recovered sentences.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from patmir.pipeline.stage import (
    Diarization,
    SeparatedAudio,
    Sentence,
    SpeakerTurn,
    Transcript,
    Word,
)
from patmir.pipeline.stages.asr._sentence import (
    NON_LEXICAL_SINGLE_TOKENS,
    is_non_lexical,
)

# Min turn duration to bother re-ASR'ing. Shorter than this and Whisper
# usually doesn't produce useful words anyway (and noise floor dominates).
DEFAULT_MIN_TURN_SECONDS = 1.5

# Word margin to pad each side of the slice — softens cosine-fade edges
# from the VAD mask so Whisper sees a bit of context.
SLICE_PAD_SEC = 0.15

# When two recovered sentences within DEDUP_WINDOW_SEC have identical text,
# keep only the first. Overlapping diarization turns commonly produce
# duplicate re-ASR results on a single utterance.
DEDUP_WINDOW_SEC = 30.0

# Whisper hallucinates a small set of recurring credit-screen phrases when
# fed pure noise (laughter, applause, music). They come from the training
# data — popular YouTube subtitlers' credits leaked into the captions.
# Substring-matched case-insensitively against the recovered sentence text.
WHISPER_HALLUCINATION_SUBSTRINGS: tuple[str, ...] = (
    # Russian YouTube subtitle credits
    "субтитры сделал",
    "субтитры создал",
    "субтитры подогнал",
    "корректор субтитров",
    "субтитры от",
    "редактор субтитров",
    "dimatorzok",
    "субтитры",  # last-resort — any sentence that's a credit-screen mention
    # English Whisper credit hallucinations
    "subtitles by",
    "subscribe to",
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "amara.org",
    # Korean / generic
    "字幕志愿者",
)


def _looks_like_whisper_hallucination(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in WHISPER_HALLUCINATION_SUBSTRINGS)


def fill_gaps(
    transcript: Transcript,
    diarization: Diarization,
    separated: SeparatedAudio,
    *,
    asr_fn: Callable[[Path, str | None], list[Word]],
    min_turn_seconds: float = DEFAULT_MIN_TURN_SECONDS,
    progress: Callable[[str], None] | None = None,
) -> Transcript:
    """Re-ASR diarization turns the first pass missed; merge into the transcript.

    Args:
      transcript: output of the first ASR pass (already tagged with speaker_ids
        by `apply_speaker_tags`).
      diarization: pyannote turns + overlaps (used to find the misses).
      separated: provides `speech_vocals` — the cleanest available stem for
        slice-based re-ASR.
      asr_fn: callable `(wav_path, language) -> list[Word]`. The runner
        provides this by binding to the active ASR stage's `transcribe_slice`.
      min_turn_seconds: skip turns shorter than this (too noisy to recover).

    Returns: a new Transcript with recovered words and sentences merged in,
      sorted by time, re-indexed contiguously.
    """
    missed = _find_missed_turns(transcript, diarization, min_turn_seconds)
    if not missed:
        return transcript

    speech_vocals_path = separated.speech_vocals or separated.vocals
    if speech_vocals_path is None or not speech_vocals_path.exists():
        return transcript

    audio, sr = sf.read(str(speech_vocals_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    total_samples = len(mono)

    recovered_words: list[Word] = list(transcript.words)
    recovered_sentences: list[Sentence] = list(transcript.sentences)

    for turn in missed:
        pad_start = max(0.0, turn.start - SLICE_PAD_SEC)
        pad_end = turn.end + SLICE_PAD_SEC
        i0 = max(0, int(pad_start * sr))
        i1 = min(total_samples, int(pad_end * sr))
        if i1 - i0 < int(0.5 * sr):
            continue

        slice_mono = mono[i0:i1]
        # Write to a temp WAV that the underlying ASR can open. We could
        # accept numpy arrays directly, but file-path is the lowest common
        # denominator across mlx_whisper / faster-whisper / Qwen3-ASR.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, slice_mono, sr, subtype="PCM_16")
            tmp_path = Path(tmp.name)
        try:
            slice_words = asr_fn(tmp_path, transcript.language)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not slice_words:
            continue

        # Adjust timestamps from slice-local back to source-global.
        offset = pad_start
        for w in slice_words:
            w.start = float(w.start + offset)
            w.end = float(w.end + offset)

        # Filter the new words by the original turn bounds (drop the pad).
        in_turn = [
            w for w in slice_words
            if turn.start - 0.05 <= 0.5 * (w.start + w.end) < turn.end + 0.05
        ]
        if not in_turn:
            continue

        sent_text = " ".join(w.text for w in in_turn).strip()
        if not sent_text or is_non_lexical(sent_text):
            continue
        if _looks_like_whisper_hallucination(sent_text):
            # Whisper credit-screen hallucination on a noisy slice. Drop it
            # and don't add its words to the transcript either.
            if progress is not None:
                progress(f"drop hallucination '{sent_text[:50]}' @ {turn.start:.1f}s")
            continue

        if progress is not None:
            progress(f"recovered '{sent_text[:50]}' @ {turn.start:.1f}s")

        recovered_words.extend(in_turn)
        recovered_sentences.append(Sentence(
            idx=0,  # re-indexed below
            text=sent_text,
            start=in_turn[0].start,
            end=in_turn[-1].end,
            words=in_turn,
            speaker_id=None,  # apply_speaker_tags will fill this in
        ))

    # Sort, dedup near-duplicate consecutive recoveries, re-index.
    recovered_words.sort(key=lambda w: w.start)
    recovered_sentences.sort(key=lambda s: s.start)
    recovered_sentences = _dedup_consecutive(recovered_sentences)
    for i, s in enumerate(recovered_sentences):
        s.idx = i

    return Transcript(
        language=transcript.language,
        words=recovered_words,
        sentences=recovered_sentences,
    )


def _dedup_consecutive(sentences: list[Sentence]) -> list[Sentence]:
    """Drop sentences whose text matches a sentence within `DEDUP_WINDOW_SEC`
    earlier in the timeline. Overlapping diarization turns commonly cause
    re-ASR to produce the same utterance multiple times.
    """
    if not sentences:
        return sentences
    out: list[Sentence] = []
    for s in sentences:
        norm = s.text.strip().lower()
        is_dup = False
        for prior in reversed(out):
            if s.start - prior.start > DEDUP_WINDOW_SEC:
                break
            if prior.text.strip().lower() == norm:
                is_dup = True
                break
        if not is_dup:
            out.append(s)
    return out


def _find_missed_turns(
    transcript: Transcript,
    diarization: Diarization,
    min_turn_seconds: float,
) -> list[SpeakerTurn]:
    """Pick turns with no real lexical words from the existing transcript.

    'Real' = not in `NON_LEXICAL_SINGLE_TOKENS`. A turn with only laughter
    transcribed counts as missed because Whisper's first pass effectively
    skipped over real dialogue inside it.
    """
    out: list[SpeakerTurn] = []
    for turn in diarization.turns:
        if turn.end - turn.start < min_turn_seconds:
            continue
        in_turn = [
            w for w in transcript.words
            if turn.start <= 0.5 * (w.start + w.end) < turn.end
        ]
        real = [
            w for w in in_turn
            if w.text.strip(".,!?-").lower() not in NON_LEXICAL_SINGLE_TOKENS
        ]
        if not real:
            out.append(turn)
    return out
