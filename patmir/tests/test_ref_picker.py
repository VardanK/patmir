"""Tests for `pick_refs`.

The picker is the upstream producer of `(speaker, emotion) → SpeakerRef`.
We test:

- `_dominant_bin` chooses the emotion with the largest overlap weight.
- `tag_turns_with_emotion` populates `turn.emotion` for every turn that has
  predictions overlapping it.
- `pick_refs` itself: cell selection, min-duration filtering (with the 100 ms
  edge trim applied), overlap rejection, and longest-window tie-breaking.

Requires `numpy` + `soundfile` (already pipeline-runtime deps) to write a
synthetic source-audio file for the picker to slice from.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from patmir.pipeline.stage import EmotionPrediction, OverlapInterval, SpeakerTurn
from patmir.pipeline.stages.diarize.ref_picker import (
    _dominant_bin,
    pick_refs,
    tag_turns_with_emotion,
)


def _make_synthetic_audio(duration_s: float, sr: int = 16_000) -> np.ndarray:
    n = int(duration_s * sr)
    # A simple sine wave so the WAV is non-zero and resampling-stable.
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class DominantBinTests(unittest.TestCase):
    def test_picks_largest_overlap(self):
        turn = SpeakerTurn(start=0.0, end=10.0, speaker_id="SPK")
        preds = [
            EmotionPrediction(start=0.0, end=3.0, emotion="happy", confidence=0.9),
            EmotionPrediction(start=3.0, end=10.0, emotion="sad", confidence=0.8),
        ]
        self.assertEqual(_dominant_bin(turn, preds), "sad")

    def test_defaults_to_neutral_when_no_overlap(self):
        turn = SpeakerTurn(start=100.0, end=101.0, speaker_id="SPK")
        preds = [EmotionPrediction(start=0.0, end=2.0, emotion="happy", confidence=0.9)]
        self.assertEqual(_dominant_bin(turn, preds), "neutral")


class TagTurnsTests(unittest.TestCase):
    def test_populates_emotion_in_place(self):
        turns = [
            SpeakerTurn(start=0.0, end=2.0, speaker_id="A"),
            SpeakerTurn(start=2.0, end=4.0, speaker_id="B"),
        ]
        preds = [
            EmotionPrediction(start=0.0, end=2.0, emotion="happy", confidence=0.9),
            EmotionPrediction(start=2.0, end=4.0, emotion="angry", confidence=0.8),
        ]
        result = tag_turns_with_emotion(turns, preds)
        self.assertIs(result, turns)
        self.assertEqual(turns[0].emotion, "happy")
        self.assertEqual(turns[1].emotion, "angry")


class PickRefsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

        # 30 s synthetic mono audio at 16 kHz.
        self.audio = _make_synthetic_audio(30.0)
        self.audio_path = self.tmp_path / "speech_vocals.wav"
        sf.write(str(self.audio_path), self.audio, 16_000, subtype="PCM_16")
        self.refs_dir = self.tmp_path / "refs"

    def test_picks_one_ref_per_speaker_emotion_cell(self):
        turns = [
            # SPEAKER_00 happy: 2.0-5.0 (3.0 s before trim → 2.8 s after trim)
            SpeakerTurn(start=2.0, end=5.0, speaker_id="SPEAKER_00"),
            # SPEAKER_00 sad: 10.0-13.0
            SpeakerTurn(start=10.0, end=13.0, speaker_id="SPEAKER_00"),
            # SPEAKER_01 happy: 20.0-23.0
            SpeakerTurn(start=20.0, end=23.0, speaker_id="SPEAKER_01"),
        ]
        preds = [
            EmotionPrediction(start=2.0, end=5.0, emotion="happy", confidence=0.9),
            EmotionPrediction(start=10.0, end=13.0, emotion="sad", confidence=0.9),
            EmotionPrediction(start=20.0, end=23.0, emotion="happy", confidence=0.9),
        ]
        refs = pick_refs(
            turns=turns,
            overlaps=[],
            predictions=preds,
            speech_vocals=self.audio_path,
            refs_dir=self.refs_dir,
            min_ref_seconds=1.5,
        )
        self.assertEqual(set(refs.keys()), {
            ("SPEAKER_00", "happy"),
            ("SPEAKER_00", "sad"),
            ("SPEAKER_01", "happy"),
        })
        # Sanity: WAV files actually got written.
        for ref in refs.values():
            self.assertTrue(ref.ref_audio.exists())

    def test_rejects_turns_below_min_duration(self):
        # 1.0 s minus 0.2 s of edge trim = 0.8 s effective. Below 1.5.
        turns = [SpeakerTurn(start=2.0, end=3.0, speaker_id="SPEAKER_00")]
        preds = [EmotionPrediction(start=2.0, end=3.0, emotion="happy", confidence=0.9)]
        refs = pick_refs(
            turns=turns,
            overlaps=[],
            predictions=preds,
            speech_vocals=self.audio_path,
            refs_dir=self.refs_dir,
            min_ref_seconds=1.5,
        )
        self.assertEqual(refs, {})

    def test_rejects_turns_intersecting_overlaps(self):
        turns = [SpeakerTurn(start=2.0, end=6.0, speaker_id="SPEAKER_00")]
        preds = [EmotionPrediction(start=2.0, end=6.0, emotion="happy", confidence=0.9)]
        overlaps = [OverlapInterval(start=3.0, end=4.0)]
        refs = pick_refs(
            turns=turns,
            overlaps=overlaps,
            predictions=preds,
            speech_vocals=self.audio_path,
            refs_dir=self.refs_dir,
            min_ref_seconds=1.5,
        )
        self.assertEqual(refs, {})

    def test_picks_longest_eligible_window(self):
        turns = [
            SpeakerTurn(start=2.0, end=4.0, speaker_id="A"),     # 2.0 s
            SpeakerTurn(start=10.0, end=15.0, speaker_id="A"),   # 5.0 s ← winner
            SpeakerTurn(start=20.0, end=23.0, speaker_id="A"),   # 3.0 s
        ]
        preds = [
            EmotionPrediction(start=2.0, end=4.0, emotion="happy", confidence=0.9),
            EmotionPrediction(start=10.0, end=15.0, emotion="happy", confidence=0.9),
            EmotionPrediction(start=20.0, end=23.0, emotion="happy", confidence=0.9),
        ]
        refs = pick_refs(
            turns=turns,
            overlaps=[],
            predictions=preds,
            speech_vocals=self.audio_path,
            refs_dir=self.refs_dir,
            min_ref_seconds=1.5,
        )
        self.assertEqual(set(refs.keys()), {("A", "happy")})
        ref = refs[("A", "happy")]
        self.assertAlmostEqual(ref.source_start, 10.1, places=2)
        self.assertAlmostEqual(ref.source_end, 14.9, places=2)

    def _write_audio_with_quiet_band(self, quiet: tuple[float, float]) -> Path:
        """30 s sine at -23 dBFS with one band attenuated to ~-63 dBFS."""
        audio = self.audio.copy()
        sr = 16_000
        audio[int(quiet[0] * sr):int(quiet[1] * sr)] *= 0.01
        path = self.tmp_path / "quiet_band.wav"
        sf.write(str(path), audio, sr, subtype="PCM_16")
        return path

    def test_quiet_window_loses_to_louder_shorter_one(self):
        # The longest window (10-15s) sits in the quiet band → the loud 3 s
        # window at 20-23s wins despite being shorter.
        audio_path = self._write_audio_with_quiet_band((9.0, 16.0))
        turns = [
            SpeakerTurn(start=10.0, end=15.0, speaker_id="A"),
            SpeakerTurn(start=20.0, end=23.0, speaker_id="A"),
        ]
        preds = [
            EmotionPrediction(start=10.0, end=15.0, emotion="happy", confidence=0.9),
            EmotionPrediction(start=20.0, end=23.0, emotion="happy", confidence=0.9),
        ]
        refs = pick_refs(
            turns=turns, overlaps=[], predictions=preds,
            speech_vocals=audio_path, refs_dir=self.refs_dir,
            min_ref_seconds=1.5, min_rms_db=-33.0,
        )
        self.assertAlmostEqual(refs[("A", "happy")].source_start, 20.1, places=2)

    def test_all_quiet_still_keeps_longest(self):
        # Every window below the floor → gate relaxes, longest wins (a quiet
        # ref beats no ref).
        audio_path = self._write_audio_with_quiet_band((0.0, 30.0))
        turns = [
            SpeakerTurn(start=10.0, end=15.0, speaker_id="A"),
            SpeakerTurn(start=20.0, end=23.0, speaker_id="A"),
        ]
        preds = [
            EmotionPrediction(start=10.0, end=15.0, emotion="happy", confidence=0.9),
            EmotionPrediction(start=20.0, end=23.0, emotion="happy", confidence=0.9),
        ]
        refs = pick_refs(
            turns=turns, overlaps=[], predictions=preds,
            speech_vocals=audio_path, refs_dir=self.refs_dir,
            min_ref_seconds=1.5, min_rms_db=-33.0,
        )
        self.assertAlmostEqual(refs[("A", "happy")].source_start, 10.1, places=2)


if __name__ == "__main__":
    unittest.main()
