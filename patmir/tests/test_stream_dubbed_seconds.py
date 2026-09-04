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


class PublicRecordTest(unittest.TestCase):
    def test_dubbed_s_is_in_the_public_record(self):
        # The extension reads `rec.dubbed_s` off `GET /jobs/{id}`, which is
        # `to_public()`. The field was maintained on the record but never
        # emitted, so the client always saw `undefined` and hid the bar —
        # the exact state the field was added to fix. Both halves had tests;
        # neither crossed this seam.
        import time
        from patmir.ui.stream import StreamRecord
        rec = StreamRecord(
            job_id="j", cache_key="k", source_id="s", source_url=None,
            source_path="/x.mkv", source_lang="ru", status="running",
            submitted_at=time.time(), duration=677.0,
        )
        rec.dubbed_s = 76.2
        pub = rec.to_public()
        self.assertEqual(pub["dubbed_s"], 76.2)
        self.assertEqual(pub["duration"], 677.0)
        # The shape the client's `jobProgress` requires: both numbers present.
        self.assertIsInstance(pub["dubbed_s"], float)
