"""Tests for `backfill_ref_text`.

The contract changed on 2026-05-27: backfill must never delete refs. The
adapter's per-candidate `ref_text_optional` flag now owns the "is this ref
usable" decision. These tests guard that change.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from patmir.pipeline.stage import (
    Diarization,
    SpeakerRef,
    SpeakerTurn,
    Transcript,
    Word,
)
from patmir.pipeline.stages.asr.speaker_tag import backfill_ref_text


def _make_diar(refs: dict) -> Diarization:
    return Diarization(turns=[], overlaps=[], refs=refs, sample_rate=16000)


def _make_transcript(words: list[Word]) -> Transcript:
    return Transcript(language="ru", words=words, sentences=[])


class BackfillRefTextTests(unittest.TestCase):
    def test_fills_text_when_words_inside_window(self):
        ref = SpeakerRef(
            speaker_id="SPEAKER_00",
            ref_audio=Path("/tmp/x.wav"),
            ref_text="",
            source_start=1.0,
            source_end=3.0,
        )
        diar = _make_diar({("SPEAKER_00", "happy"): ref})
        transcript = _make_transcript([
            Word(text="hello", start=1.2, end=1.6),
            Word(text="world", start=1.7, end=2.1),
            Word(text="far", start=5.0, end=5.4),  # outside the window
        ])

        backfill_ref_text(diar, transcript)

        self.assertEqual(ref.ref_text, "hello world")
        self.assertIn(("SPEAKER_00", "happy"), diar.refs)

    def test_keeps_ref_when_no_words_overlap(self):
        """Pre-fix behavior would delete this entry; post-fix keeps it so
        candidates with `ref_text_optional = True` (ElevenLabs) can still
        IVC-clone from the audio alone."""
        ref = SpeakerRef(
            speaker_id="SPEAKER_00",
            ref_audio=Path("/tmp/x.wav"),
            ref_text="",
            source_start=10.0,
            source_end=12.0,
        )
        diar = _make_diar({("SPEAKER_00", "sad"): ref})
        transcript = _make_transcript([
            Word(text="bye", start=0.5, end=0.9),
        ])

        backfill_ref_text(diar, transcript)

        self.assertEqual(ref.ref_text, "")
        self.assertIn(("SPEAKER_00", "sad"), diar.refs)

    def test_keeps_ref_when_text_is_non_lexical(self):
        """Refs whose window lands on laughter / screams used to be deleted.
        The current contract keeps them; gating is the adapter's job."""
        ref = SpeakerRef(
            speaker_id="SPEAKER_03",
            ref_audio=Path("/tmp/x.wav"),
            ref_text="",
            source_start=2.0,
            source_end=4.0,
        )
        diar = _make_diar({("SPEAKER_03", "happy"): ref})
        transcript = _make_transcript([
            Word(text="-ха", start=2.2, end=2.4),
            Word(text="-ха", start=2.5, end=2.7),
            Word(text="-ха!", start=2.8, end=3.0),
        ])

        backfill_ref_text(diar, transcript)

        self.assertIn(("SPEAKER_03", "happy"), diar.refs)
        # Text is populated even though it's non-lexical.
        self.assertTrue(ref.ref_text)

    def test_uses_midpoint_for_window_membership(self):
        """A word that starts before but extends past midpoint should still
        be excluded if its midpoint falls outside."""
        ref = SpeakerRef(
            speaker_id="SPK",
            ref_audio=Path("/tmp/x.wav"),
            ref_text="",
            source_start=5.0,
            source_end=6.0,
        )
        diar = _make_diar({("SPK", "neutral"): ref})
        transcript = _make_transcript([
            Word(text="early", start=4.5, end=4.9),     # midpoint 4.7 → out
            Word(text="straddle", start=4.7, end=5.5),  # midpoint 5.1 → in
            Word(text="cross", start=5.9, end=6.5),     # midpoint 6.2 → out
        ])

        backfill_ref_text(diar, transcript)
        self.assertEqual(ref.ref_text, "straddle")


if __name__ == "__main__":
    unittest.main()
