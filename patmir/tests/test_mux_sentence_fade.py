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


class OnsetBurstTrimTest(unittest.TestCase):
    """A loud pitch-free impulse at the very start of a clip is cut, not faded.

    Quality issue #30, second and correct diagnosis. The author's description
    was "a muffled gun shot" before the first word. At 20 ms resolution the
    final mix jumped 31 dB into a frame with ZERO energy below 300 Hz, louder
    than the source's own speech peak; the raw clip carried a 45 ms pitch-free
    head at 40-76% of its peak before the voice began.

    A fade cannot fix that — it scales the burst's amplitude and leaves its
    spectrum, and the burst is longer than the fade anyway (85 ms at p90 over
    affected clips). So it is cut.

    Measured over 3079 real lines: this fires on 33% of first lines and 2% of
    others, median 5 ms, max 75 ms, never reaching the 120 ms cap.
    """

    def _burst(self, ms, amp=0.8):
        n = int(ms / 1000 * SR)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n).astype("float32") * amp
        # High-passed: a decoder attack transient carries no pitch.
        return np.diff(np.concatenate([[0.0], x])).astype("float32")

    def test_a_burst_before_speech_is_removed(self):
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _trim_onset_burst
        clip = np.concatenate([self._burst(45), _tone(SR, amp=0.5, f=150.0)])
        out = _trim_onset_burst(clip, SR)
        removed = (len(clip) - len(out)) / SR
        self.assertGreater(removed, 0.030)
        self.assertLess(removed, 0.060)

    def test_a_tonal_onset_is_left_alone(self):
        # A clip that starts on a vowel has pitch in its first frame.
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _trim_onset_burst
        clip = _tone(SR, amp=0.5, f=150.0)
        self.assertEqual(len(_trim_onset_burst(clip, SR)), len(clip))

    def test_a_clip_that_starts_quietly_is_left_alone(self):
        # The common case — median first-frame amplitude over non-first lines
        # is 0.005 of peak. Nothing to cut.
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _trim_onset_burst
        clip = np.concatenate([np.zeros(int(0.06 * SR), dtype="float32"),
                               _tone(SR, amp=0.5, f=150.0)])
        self.assertEqual(len(_trim_onset_burst(clip, SR)), len(clip))

    def test_a_quiet_voiceless_fricative_onset_survives(self):
        """The safety case this detector exists to respect.

        A word-initial /s/ or /ʃ/ is pitch-free too, so only the *loudness*
        of the first frame relative to the clip's own peak separates it from
        a decoder impulse. A fricative sits well below the following vowel.
        """
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _trim_onset_burst
        clip = np.concatenate([self._burst(40, amp=0.05),          # quiet /s/
                               _tone(SR, amp=0.8, f=150.0)])       # loud vowel
        self.assertEqual(len(_trim_onset_burst(clip, SR)), len(clip))

    def test_the_cut_never_exceeds_the_cap(self):
        # A clip that is nothing but noise must not be trimmed away.
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import (
            _BURST_MAX_SECONDS, _trim_onset_burst,
        )
        clip = self._burst(600)
        removed = (len(clip) - len(_trim_onset_burst(clip, SR))) / SR
        self.assertLessEqual(removed, _BURST_MAX_SECONDS + 1e-6)

    def test_the_cut_stops_when_the_burst_drops_into_silence(self):
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _trim_onset_burst
        clip = np.concatenate([self._burst(20),
                               np.zeros(int(0.05 * SR), dtype="float32"),
                               _tone(SR, amp=0.5, f=150.0)])
        removed = (len(clip) - len(_trim_onset_burst(clip, SR))) / SR
        self.assertLess(removed, 0.035)

    def test_degenerate_clips_are_returned_unchanged(self):
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _trim_onset_burst
        for clip in (np.zeros(0, dtype="float32"),
                     np.zeros(200, dtype="float32"),
                     self._burst(2)):
            self.assertEqual(len(_trim_onset_burst(clip, SR)), len(clip))

    def test_the_real_reported_clip(self):
        """The measured case, if its artifacts are still on disk."""
        import os
        import soundfile as sf
        p = "runs/ab-fade/tts/0000.wav"
        if not os.path.exists(p):
            self.skipTest("reference run not on disk")
        from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _trim_onset_burst
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        removed = (len(x) - len(_trim_onset_burst(x, sr))) / sr
        self.assertAlmostEqual(removed, 0.045, places=2)
