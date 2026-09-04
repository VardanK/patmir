"""dubbed_s: the number the watch-along progress bar reads."""
import unittest
from patmir.ui.stream import _dubbed_seconds


class DubbedSecondsTest(unittest.TestCase):
    def test_spans_are_summed_not_counted(self):
        # Sections differ in length and the prelude window is its own size, so
        # `sections_done` cannot make a fraction.
        prior = {"sections": [
            {"index": 0, "start": 0.0, "end": 39.99},
            {"index": 1, "start": 39.99, "end": 76.2},
        ]}
        self.assertAlmostEqual(_dubbed_seconds(prior), 76.2, places=3)

    def test_a_malformed_entry_is_skipped_not_raised(self):
        # This feeds a progress bar; a bar is never worth failing a resume for.
        prior = {"sections": [{"start": 0}, {"index": 1, "start": 0.0, "end": 10.0}]}
        self.assertEqual(_dubbed_seconds(prior), 10.0)

    def test_nothing_cached_is_zero(self):
        self.assertEqual(_dubbed_seconds(None), 0.0)
        self.assertEqual(_dubbed_seconds({}), 0.0)
        self.assertEqual(_dubbed_seconds({"sections": []}), 0.0)

    def test_a_reversed_span_contributes_nothing(self):
        self.assertEqual(_dubbed_seconds(
            {"sections": [{"index": 0, "start": 50.0, "end": 10.0}]}), 0.0)
