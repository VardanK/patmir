"""Tests for mux no-overlap ("ripple") voice placement.

ElevenLabs (and any TTS) lines are often longer than the source dialogue slot;
the old mux placed every line at its source time and let it overrun, so a fast
speaker's lines piled on top of each other. `_placement_start` ripples a line
to just after the previous one when it would overlap — and, crucially, resets
to source time at a speech gap so drift stays local to a dense burst.
"""
from __future__ import annotations

import unittest

from patmir.pipeline.stages.mux.ffmpeg_pad_stretch import _placement_start


class PlacementTests(unittest.TestCase):
    def test_no_overlap_off_uses_source_time(self):
        # Legacy behavior: always place at source time (overrun allowed).
        self.assertEqual(_placement_start(100, voice_cursor=500, gap=10, no_overlap=False), 100)

    def test_ripples_when_would_overlap(self):
        # Line's source start (100) is before the previous line's end (500) →
        # push to cursor + gap.
        self.assertEqual(_placement_start(100, voice_cursor=500, gap=10, no_overlap=True), 510)

    def test_no_ripple_when_gap_in_source(self):
        # A real pause: source start (900) is past the cursor (500) → keep it,
        # so drift from the previous burst resets here.
        self.assertEqual(_placement_start(900, voice_cursor=500, gap=10, no_overlap=True), 900)

    def test_burst_accumulates_then_resets(self):
        # Simulate a 3-line burst (each 200 samples, sources 1 apart) then a gap.
        cursor = 0
        placed = []
        # dense burst: sources 0, 50, 100 (slots 50) but audio 200 each
        for src in (0, 50, 100):
            start = _placement_start(src, cursor, gap=0, no_overlap=True)
            cursor = start + 200
            placed.append(start)
        self.assertEqual(placed, [0, 200, 400])     # rippled, no overlap
        # a pause at source 1000 (well past cursor 600) resets to source time
        start = _placement_start(1000, cursor, gap=0, no_overlap=True)
        self.assertEqual(start, 1000)


if __name__ == "__main__":
    unittest.main()
