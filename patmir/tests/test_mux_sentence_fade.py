"""The per-sentence fade-in must not ramp over a word's attack.

Quality issue #30, from the author's report that "the very first word starts
with noise" across five or more videos. The 50 ms raised-cosine fade-in
(issue #4's voice-leak guard) was written for lines that begin with silence,
and most do — median speech onset over 2961 measured synthesized lines is
0.062 s. But the **first line of a job starts at sample 0 in 74% of jobs**
against 26% for every other line, so on that line the full ramp landed on the
first consonant. It is audible there and nowhere else because nothing precedes
it to mask it.

The fade-in is now bounded by the leading quiet the clip actually has, with a
4 ms anti-click floor. The fade-out is deliberately unchanged.
"""
from __future__ import annotations

import unittest

import numpy as np

from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import (
    _CLICK_GUARD_SAMPLES,
    _leading_quiet_samples,
)

SR = 24000


def _tone(n, amp=0.5, f=200.0):
    # Cosine, not sine: a sine is zero at sample 0, which would put the
    # measured onset one sample late and make these assertions about the
    # fixture rather than the code.
    t = np.arange(n, dtype="float32") / SR
    return (amp * np.cos(2 * np.pi * f * t)).astype("float32")


class LeadingQuietTest(unittest.TestCase):
    def test_a_clip_starting_immediately_reports_no_lead(self):
        self.assertEqual(_leading_quiet_samples(_tone(SR)), 0)

    def test_leading_silence_is_measured(self):
        clip = np.concatenate([np.zeros(1200, dtype="float32"), _tone(SR)])
        self.assertEqual(_leading_quiet_samples(clip), 1200)

    def test_the_threshold_is_relative_to_the_clip_peak(self):
        # A quiet line must not be mistaken for one that starts with silence:
        # scaling the whole clip cannot change where its speech begins.
        clip = np.concatenate([np.zeros(600, dtype="float32"), _tone(SR)])
        self.assertEqual(_leading_quiet_samples(clip),
                         _leading_quiet_samples(clip * 0.02))

    def test_a_silent_clip_reports_its_whole_length(self):
        # Degenerate but reachable — the TTS silent-WAV fallback. Must not
        # divide by a zero peak or index an empty array.
        self.assertEqual(_leading_quiet_samples(np.zeros(500, dtype="float32")), 500)

    def test_an_empty_clip_does_not_raise(self):
        self.assertEqual(_leading_quiet_samples(np.zeros(0, dtype="float32")), 0)

    def test_a_low_level_ramp_before_speech_counts_as_quiet(self):
        # -40 dB of room tone ahead of the attack is lead, not speech.
        clip = np.concatenate([_tone(800, amp=0.005), _tone(SR)])
        self.assertGreater(_leading_quiet_samples(clip), 700)


class ClickGuardTest(unittest.TestCase):
    def test_four_milliseconds(self):
        self.assertEqual(_CLICK_GUARD_SAMPLES(SR), int(0.004 * SR))

    def test_never_zero_even_at_an_absurd_rate(self):
        # A zero-length fade would reintroduce the step discontinuity the
        # fade exists to remove.
        self.assertGreaterEqual(_CLICK_GUARD_SAMPLES(100), 1)


class FadeChoiceTest(unittest.TestCase):
    """The arithmetic the mux applies, pinned at the three cases that matter."""

    def _fade_in_len(self, clip, sentence_fade):
        lead = _leading_quiet_samples(clip)
        return int(min(sentence_fade, max(_CLICK_GUARD_SAMPLES(SR), lead)))

    def test_an_abrupt_clip_gets_only_the_click_guard(self):
        # The reported case: attack preserved, discontinuity still removed.
        n = self._fade_in_len(_tone(SR), int(0.05 * SR))
        self.assertEqual(n, _CLICK_GUARD_SAMPLES(SR))

    def test_a_clip_with_ample_silence_keeps_the_full_fade(self):
        clip = np.concatenate([np.zeros(int(0.08 * SR), dtype="float32"), _tone(SR)])
        self.assertEqual(self._fade_in_len(clip, int(0.05 * SR)), int(0.05 * SR))

    def test_a_clip_with_partial_silence_fades_over_exactly_that(self):
        lead = int(0.02 * SR)
        clip = np.concatenate([np.zeros(lead, dtype="float32"), _tone(SR)])
        self.assertEqual(self._fade_in_len(clip, int(0.05 * SR)), lead)

    def test_the_fade_never_exceeds_the_configured_length(self):
        clip = np.concatenate([np.zeros(int(0.5 * SR), dtype="float32"), _tone(SR)])
        self.assertEqual(self._fade_in_len(clip, int(0.05 * SR)), int(0.05 * SR))


class AppliedFadeTest(unittest.TestCase):
    def test_the_attack_survives_but_the_first_sample_does_not_step(self):
        """Both halves of the point, on one clip.

        Before this change a 50 ms ramp cut the first 50 ms of an abrupt clip
        to a median 39% of its energy. Now the attack is intact past 4 ms,
        while sample 0 is still driven to zero so there is no step.
        """
        sentence_fade = int(0.05 * SR)
        clip = _tone(SR)
        lead = _leading_quiet_samples(clip)
        fin_n = int(min(sentence_fade, max(_CLICK_GUARD_SAMPLES(SR), lead)))
        fin = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fin_n, dtype="float32")))
        faded = clip.copy()
        faded[:fin_n] *= fin

        self.assertAlmostEqual(float(faded[0]), 0.0, places=6)
        # Energy over the first 50 ms is essentially preserved...
        kept = float((faded[:sentence_fade] ** 2).sum()
                     / (clip[:sentence_fade] ** 2).sum())
        self.assertGreater(kept, 0.9)
        # ...whereas the old full-length ramp would have removed most of it.
        old = clip.copy()
        old_fin = 0.5 * (1 - np.cos(np.linspace(0, np.pi, sentence_fade,
                                                dtype="float32")))
        old[:sentence_fade] *= old_fin
        old_kept = float((old[:sentence_fade] ** 2).sum()
                         / (clip[:sentence_fade] ** 2).sum())
        self.assertLess(old_kept, 0.6)


if __name__ == "__main__":
    unittest.main()
