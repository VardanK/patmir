"""Tests for the VAD section-end snap (spec 2026-08-25-1931, finding A1).

The snap decides where a watch-along section ends. It got this wrong in a way
that made the buffer unable to grow, which is the structural cause of the
freezes the author reported.

Every fixture below is either a real measurement from
`runs/20260825-182332-section` or a case constructed to pin one rule. The
headline one is `test_the_measured_collapse`: those exact intervals produced
**8.1 s of dub from 38 s of analyzed audio** (RTF 4.0), discarding 8 of the
section's 10 ASR sentences.

Why the intervals look contiguous: the separate stage runs Silero with
`speech_pad_ms=500`, padding every speech run by 500 ms per side, so two runs
separated by less than ~1 s of silence come back touching. An interval end that
equals the next interval's start is therefore mid-speech, not a boundary — the
old `max(end <= target)` could not tell the difference.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patmir.pipeline.streaming.prelude import _snap_window_end
from patmir.pipeline.streaming.section import (
    DEFAULT_PAD_SECONDS,
    DEFAULT_SECTION_SECONDS,
    END_GUARD_SECONDS,
)

# What process_section actually asks for: 30 + 8 pad, less the 1.5 s end guard.
SECTION_TARGET = DEFAULT_SECTION_SECONDS + DEFAULT_PAD_SECONDS - END_GUARD_SECONDS


def _snap(intervals, target=SECTION_TARGET, **kw) -> float:
    with tempfile.TemporaryDirectory() as td:
        jd = Path(td)
        (jd / "separated.meta.json").write_text(
            json.dumps({"vad_intervals": intervals}))
        return _snap_window_end(jd, target, **kw)


class MeasuredCollapseTest(unittest.TestCase):
    def test_the_measured_collapse(self):
        """runs/20260825-182332-section, verbatim. Was 8.1; must not be."""
        got = _snap([[0.0, 6.3], [6.3, 8.1], [8.1, 38.0]])
        self.assertNotAlmostEqual(got, 8.1, places=3)
        self.assertEqual(got, SECTION_TARGET)

    def test_the_collapse_emitted_less_than_a_quarter_of_the_slice(self):
        """Guards the property, not just the number: a section must not emit a
        small fraction of what it analyzed."""
        got = _snap([[0.0, 6.3], [6.3, 8.1], [8.1, 38.0]])
        self.assertGreater(got / SECTION_TARGET, 0.5)


class SilenceGapRuleTest(unittest.TestCase):
    """Rule 1: a cut point must be followed by real silence."""

    def test_touching_intervals_are_not_cut_points(self):
        # 8.1 touches the next run's start -> mid-speech, must be rejected even
        # though it is the largest end at or below target.
        self.assertEqual(_snap([[0.0, 8.1], [8.1, 38.0]]), SECTION_TARGET)

    def test_a_real_gap_makes_a_cut_point(self):
        # Same shape, but a genuine silence 8.1 -> 12.0.
        self.assertEqual(_snap([[0.0, 8.1], [12.0, 20.0], [24.0, 38.0]]), 20.0)

    def test_the_last_interval_end_is_always_a_candidate(self):
        """Nothing follows it, so the slice edge is the silence."""
        self.assertEqual(_snap([[0.0, 25.0]]), 25.0)

    def test_prefers_the_latest_qualifying_silence(self):
        self.assertEqual(_snap([[0.0, 18.0], [20.0, 30.0], [32.0, 38.0]]), 30.0)

    def test_a_gap_below_the_epsilon_is_float_noise_not_silence(self):
        self.assertEqual(_snap([[0.0, 8.1], [8.10001, 38.0]]), SECTION_TARGET)


class CollapseFloorRuleTest(unittest.TestCase):
    """Rule 2: even a genuine silence is the wrong cut if it guts the section."""

    def test_an_early_genuine_silence_does_not_collapse_the_section(self):
        # Real silence at 8.1, but the only later boundary is past target.
        self.assertEqual(_snap([[0.0, 8.1], [12.0, 38.0]]), SECTION_TARGET)

    def test_a_late_enough_silence_is_still_honoured(self):
        # 25.0 is above the floor (36.5 * 0.5 = 18.25), so it wins.
        self.assertEqual(_snap([[0.0, 25.0], [27.0, 38.0]]), 25.0)

    def test_the_floor_is_configurable(self):
        """A caller that would rather take any real silence can say so."""
        self.assertEqual(
            _snap([[0.0, 8.1], [12.0, 38.0]], min_fraction=0.0), 8.1)


class DegradesSafelyTest(unittest.TestCase):
    def test_missing_meta_falls_back_to_target(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(_snap_window_end(Path(td), 36.5), 36.5)

    def test_corrupt_meta_falls_back_to_target(self):
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td)
            (jd / "separated.meta.json").write_text("{not json")
            self.assertEqual(_snap_window_end(jd, 36.5), 36.5)

    def test_empty_intervals_fall_back_to_target(self):
        self.assertEqual(_snap([]), SECTION_TARGET)

    def test_unsorted_intervals_are_handled(self):
        """Order is not part of the meta contract; the gap rule depends on it."""
        self.assertEqual(_snap([[24.0, 38.0], [0.0, 8.1], [12.0, 20.0]]), 20.0)

    def test_never_returns_more_than_target(self):
        """Cutting past the guard would take audio the slice may have truncated
        mid-utterance (END_GUARD_SECONDS exists for exactly that)."""
        for ivs in ([[0.0, 38.0]], [[0.0, 20.0], [22.0, 38.0]], []):
            self.assertLessEqual(_snap(ivs), SECTION_TARGET)


class PreludeWindowTest(unittest.TestCase):
    """The prelude snaps its own window; a collapse there is worse, because the
    prelude establishes the cast for the entire stream."""

    def test_prelude_window_does_not_collapse_on_continuous_speech(self):
        got = _snap([[0.0, 6.3], [6.3, 8.1], [8.1, 40.0]], target=40.0)
        self.assertEqual(got, 40.0)

    def test_prelude_honours_an_interior_silence_when_speech_runs_past_target(self):
        # Real silence 33.0 -> 35.0; the final run ends at 40.0, past this
        # target, so the interior silence is the right cut.
        self.assertEqual(_snap([[0.0, 33.0], [35.0, 40.0]], target=36.0), 33.0)

    def test_the_latest_qualifying_candidate_wins_even_at_the_slice_edge(self):
        """Pins existing behaviour rather than changing it.

        The prelude calls the snap with target == window_seconds and NO end
        guard (unlike `process_section`, which subtracts END_GUARD_SECONDS), so
        a speech run reaching the slice edge yields a candidate exactly at the
        target and wins. That cut can land mid-utterance. It is pre-existing,
        unchanged by the A1 fix, and untouched here deliberately: the prelude's
        boundary was not implicated in the measured collapse, and giving it a
        guard would shorten every opening window without evidence that it
        helps.
        """
        self.assertEqual(_snap([[0.0, 33.0], [35.0, 40.0]], target=40.0), 40.0)


if __name__ == "__main__":
    unittest.main()
