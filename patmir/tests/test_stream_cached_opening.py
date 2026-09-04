"""A restart must not re-dub an opening window it already has.

Quality issue #31. `submit()` resumes a `done` stream whose manifest is not
`complete` (issue #21), and that path re-ran BOTH the prelude analysis and
`dub_prelude_window` unconditionally. The analysis has to re-run — the
`SpeakerRegistry` it builds is not persisted and every later section resolves
identity against it — but the opening *audio* is already on disk.

The cost is not just wasted GPU: `scheduler.run()` does not start until the
opening dub returns, so nothing NEW is produced for its duration and the
viewer sits frozen at the edge of the audio they already have. Measured on
cache key 8534a9320e5492c2 (a 3471 s video): two prelude runs 53 s apart with
exactly one section cached, section 0 costing 22.4 s of that, and the author
reported the video "stuck at 35 sec" — inside the 0-39.99 s already cached.
"""
from __future__ import annotations

import unittest

from patmir.ui.stream import _cached_opening, _index_seed


class CachedOpeningTest(unittest.TestCase):
    def test_no_prior_manifest_means_dub_it(self):
        self.assertIsNone(_cached_opening(None, 40.0))
        self.assertIsNone(_cached_opening({}, 40.0))
        self.assertIsNone(_cached_opening({"sections": []}, 40.0))

    def test_the_real_manifest_from_the_reported_stream(self):
        prior = {"sections": [
            {"index": 0, "start": 0.0, "end": 39.99, "file": "section_000.wav"},
        ]}
        got = _cached_opening(prior, 39.99)
        self.assertIsNotNone(got)
        self.assertEqual(got["index"], 0)

    def test_a_window_that_does_not_reach_the_end_is_not_enough(self):
        # Half an opening would leave a hole the scheduler fills from the
        # playhead, not from 0 — so it must be re-dubbed.
        prior = {"sections": [{"index": 0, "start": 0.0, "end": 12.0}]}
        self.assertIsNone(_cached_opening(prior, 40.0))

    def test_a_section_that_does_not_start_at_the_beginning_is_not_it(self):
        prior = {"sections": [{"index": 3, "start": 120.0, "end": 160.0}]}
        self.assertIsNone(_cached_opening(prior, 40.0))

    def test_it_is_matched_on_span_not_on_index_zero(self):
        """A manifest repaired by `append_section`'s index dedup (issue #22)
        can hold the opening window under any index."""
        prior = {"sections": [
            {"index": 7, "start": 0.0, "end": 40.2, "file": "section_007.wav"},
        ]}
        got = _cached_opening(prior, 40.0)
        self.assertEqual(got["index"], 7)

    def test_the_tolerance_absorbs_a_snap_a_hair_short(self):
        # VAD snapping moves the boundary; 39.99 against a 40.0 window is the
        # same window.
        prior = {"sections": [{"index": 0, "start": 0.0, "end": 39.2}]}
        self.assertIsNotNone(_cached_opening(prior, 40.0))

    def test_a_late_start_is_not_the_opening_even_inside_the_end_tolerance(self):
        # Coverage is seeded from the cached span, so accepting [0.9, 40) would
        # leave [0, 0.9) uncovered and the scheduler would dub a sub-second
        # section at 0. The end tolerance must not apply to the start.
        prior = {"sections": [{"index": 0, "start": 0.9, "end": 40.0}]}
        self.assertIsNone(_cached_opening(prior, 40.0))
        # Float slack at the start is fine.
        prior = {"sections": [{"index": 0, "start": 0.004, "end": 40.0}]}
        self.assertIsNotNone(_cached_opening(prior, 40.0))

    def test_a_malformed_entry_is_skipped_not_raised(self):
        prior = {"sections": [
            {"index": "x", "start": 0.0, "end": 40.0},
            {"start": 0.0},
            {"index": 0, "start": 0.0, "end": 40.0},
        ]}
        self.assertIsNotNone(_cached_opening(prior, 40.0))

    def test_all_malformed_means_dub_it(self):
        prior = {"sections": [{"index": "x"}, {"nonsense": True}]}
        self.assertIsNone(_cached_opening(prior, 40.0))


class IndexSeedSurvivesTheSkipTest(unittest.TestCase):
    """The skip must not undo issue #22's fix.

    `_index_seed` returns the highest index already cached; `SectionPipeline`
    pre-increments (`self._index += 1` before use), so the first NEW section
    lands past everything on disk. What matters here is that feeding it the
    *cached* opening's index instead of `opening.index` keeps that property —
    a restart that reused an index would overwrite `section_NNN.wav` while the
    manifest kept both entries.
    """

    def test_seed_is_the_highest_cached_index(self):
        prior = {"sections": [{"index": 0}, {"index": 1}, {"index": 2}]}
        self.assertEqual(_index_seed(prior, 0), 2)

    def test_the_cached_opening_index_does_not_lower_the_seed(self):
        prior = {"sections": [
            {"index": 7, "start": 0.0, "end": 40.0},
            {"index": 8, "start": 40.0, "end": 76.0},
        ]}
        opening = _cached_opening(prior, 40.0)
        self.assertEqual(opening["index"], 7)
        # 8 is on disk, so the next new section must be 9 — the seed is 8 and
        # the pipeline pre-increments.
        self.assertEqual(_index_seed(prior, int(opening["index"])), 8)

    def test_a_lone_cached_opening_still_seeds_from_it(self):
        prior = {"sections": [{"index": 7, "start": 0.0, "end": 40.0}]}
        opening = _cached_opening(prior, 40.0)
        self.assertEqual(_index_seed(prior, int(opening["index"])), 7)


if __name__ == "__main__":
    unittest.main()
