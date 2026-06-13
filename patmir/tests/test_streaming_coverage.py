"""Tests for the live streaming scheduler's bookkeeping (Slice 3 #10).

`Coverage` (merged covered intervals) and `StreamScheduler` (playhead-driven
section ordering) are the pure brains of watch-along — they decide *which*
section to dub next without touching audio, models, or the network. We drive
the scheduler with fake `process_fn`/`playhead_fn` to assert the priority
policy: section at the playhead first, backfill the rest, terminate when the
whole video is covered, and re-prioritize after a seek.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from patmir.pipeline.streaming.coverage import Coverage
from patmir.pipeline.streaming.scheduler import Playhead, StreamScheduler


@dataclass
class FakeSection:
    index: int
    start: float
    end: float
    dub_wav: str = "x.wav"


class CoverageTests(unittest.TestCase):
    def test_merges_touching_intervals(self):
        c = Coverage()
        c.add(0, 30)
        c.add(30, 60)            # contiguous → one run
        self.assertEqual(c.intervals(), [(0, 60)])

    def test_merges_within_epsilon(self):
        c = Coverage()
        c.add(0, 30)
        c.add(30.3, 60)          # 0.3s slop < epsilon → merged
        self.assertEqual(c.intervals(), [(0, 60)])

    def test_keeps_disjoint_intervals(self):
        c = Coverage()
        c.add(0, 30)
        c.add(100, 130)          # real gap → two runs
        self.assertEqual(c.intervals(), [(0, 30), (100, 130)])

    def test_out_of_order_add_merges(self):
        c = Coverage()
        c.add(100, 130)
        c.add(0, 30)
        c.add(28, 102)           # bridges both → one run
        self.assertEqual(c.intervals(), [(0, 130)])

    def test_first_uncovered_in_gap_returns_point(self):
        c = Coverage()
        c.add(0, 30)
        self.assertEqual(c.first_uncovered(after=50), 50)   # 50 is in a gap

    def test_first_uncovered_inside_run_returns_run_end(self):
        c = Coverage()
        c.add(0, 30)
        self.assertEqual(c.first_uncovered(after=10), 30)   # fill forward of the run

    def test_first_uncovered_none_when_all_covered(self):
        c = Coverage()
        c.add(0, 100)
        self.assertIsNone(c.first_uncovered(after=0, until=100))

    def test_is_complete_within_tail_tolerance(self):
        c = Coverage()
        c.add(0, 119.5)
        self.assertTrue(c.is_complete(120.0))               # within 1s tail
        c2 = Coverage()
        c2.add(0, 100)
        self.assertFalse(c2.is_complete(120.0))


class SchedulerTests(unittest.TestCase):
    def _run(self, *, duration, prelude_end, playhead, section_len=30.0):
        """Drive the scheduler with a fake processor that always advances a
        fixed section length; returns the ordered list of starts it chose."""
        starts: list[float] = []

        def process_fn(start):
            starts.append(round(start, 3))
            return FakeSection(index=len(starts), start=start, end=min(start + section_len, duration))

        sched = StreamScheduler(
            duration=duration, prelude_end=prelude_end,
            process_fn=process_fn,
            playhead_fn=playhead,
            section_seconds=section_len,
        )
        sched.run()
        return starts, sched

    def test_forward_fill_from_prelude_end(self):
        # Playhead pinned at 0; prelude covered [0,60]. Should fill 60→duration.
        ph = Playhead(t=0.0)
        starts, sched = self._run(duration=180, prelude_end=60, playhead=lambda: ph)
        self.assertEqual(starts[0], 60)            # starts right after prelude
        self.assertTrue(sched.coverage.is_complete(180))
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))  # monotonic

    def test_seek_reprioritizes_to_playhead(self):
        # Viewer seeks far ahead into virgin territory before the forward fill
        # gets there → the next processed section must be at the playhead.
        state = {"t": 0.0}
        ph_obj = Playhead()

        def playhead():
            ph_obj.t = state["t"]
            return ph_obj

        starts: list[float] = []

        def process_fn(start):
            starts.append(round(start, 1))
            # After the first post-prelude section, simulate a seek to 120.
            if len(starts) == 1:
                state["t"] = 120.0
            return FakeSection(index=len(starts), start=start, end=min(start + 30, 200))

        sched = StreamScheduler(
            duration=200, prelude_end=60, process_fn=process_fn,
            playhead_fn=playhead, section_seconds=30,
        )
        sched.run()
        self.assertEqual(starts[0], 60)            # first forward section
        self.assertEqual(starts[1], 120.0)         # jumped to the seek target
        self.assertTrue(sched.coverage.is_complete(200))  # still finishes the rest

    def test_no_progress_section_forces_advance(self):
        # A processor that returns end==start (no VAD gap found) must not spin.
        calls = {"n": 0}

        def process_fn(start):
            calls["n"] += 1
            return FakeSection(index=calls["n"], start=start, end=start)  # zero-length

        ph = Playhead(t=0.0)
        sched = StreamScheduler(
            duration=90, prelude_end=30, process_fn=process_fn,
            playhead_fn=lambda: ph, section_seconds=30,
        )
        sched.run()
        self.assertTrue(sched.coverage.is_complete(90))
        self.assertLess(calls["n"], 20)            # terminated, didn't spin

    def test_failed_section_still_terminates(self):
        # process_fn returning None (section failed) must force progress.
        def process_fn(start):
            return None

        ph = Playhead(t=0.0)
        sched = StreamScheduler(
            duration=60, prelude_end=20, process_fn=process_fn,
            playhead_fn=lambda: ph, section_seconds=30,
        )
        sched.run()
        self.assertTrue(sched.coverage.is_complete(60))

    def test_stop_halts_loop(self):
        def process_fn(start):
            sched.stop()                            # stop after first section
            return FakeSection(index=1, start=start, end=start + 30)

        ph = Playhead(t=0.0)
        sched = StreamScheduler(
            duration=600, prelude_end=60, process_fn=process_fn,
            playhead_fn=lambda: ph, section_seconds=30,
        )
        sched.run()
        self.assertFalse(sched.coverage.is_complete(600))  # bailed early


if __name__ == "__main__":
    unittest.main()
