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
    # idle_fn=lambda: False → when caught up (buffer full to lookahead), end the
    # run instead of waiting for the playhead to advance, so tests terminate
    # deterministically without wall-clock waits.
    def _run(self, *, duration, prelude_end, playhead, section_len=30.0, lookahead=120.0):
        starts: list[float] = []

        def process_fn(start):
            starts.append(round(start, 3))
            return FakeSection(index=len(starts), start=start, end=min(start + section_len, duration))

        sched = StreamScheduler(
            duration=duration, prelude_end=prelude_end,
            process_fn=process_fn, playhead_fn=playhead,
            section_seconds=section_len, lookahead_seconds=lookahead,
            idle_fn=lambda: False,
        )
        sched.run()
        return starts, sched

    def test_dubs_only_a_bounded_window_ahead(self):
        # The whole point of the cost fix: with the playhead pinned at 0 and a
        # 120s lookahead, the scheduler must NOT dub the entire 600s video — only
        # ~[0, 120] — so it doesn't burn cloud TTS/STT on minutes you never reach.
        ph = Playhead(t=0.0)
        starts, sched = self._run(duration=600, prelude_end=0, playhead=lambda: ph, lookahead=120)
        self.assertEqual(starts[0], 0)
        self.assertLessEqual(max(starts), 120.0 + 1e-6)   # never raced far ahead
        self.assertFalse(sched.coverage.is_complete(600))  # did NOT dub the whole video

    def test_forward_from_prelude_end(self):
        ph = Playhead(t=0.0)
        starts, _ = self._run(duration=600, prelude_end=60, playhead=lambda: ph, lookahead=120)
        self.assertEqual(starts[0], 60)                    # starts right after prelude
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))  # monotonic

    def test_seek_reprioritizes_to_playhead(self):
        state = {"t": 0.0}
        ph_obj = Playhead()

        def playhead():
            ph_obj.t = state["t"]
            return ph_obj

        starts: list[float] = []

        def process_fn(start):
            starts.append(round(start, 1))
            if len(starts) == 1:
                state["t"] = 300.0                         # seek far ahead
            return FakeSection(index=len(starts), start=start, end=min(start + 30, 600))

        sched = StreamScheduler(
            duration=600, prelude_end=60, process_fn=process_fn,
            playhead_fn=playhead, section_seconds=30, lookahead_seconds=120,
            idle_fn=lambda: False,
        )
        sched.run()
        self.assertEqual(starts[0], 60)
        self.assertEqual(starts[1], 300.0)                 # jumped to the seek target

    def test_no_progress_section_forces_advance(self):
        calls = {"n": 0}

        def process_fn(start):
            calls["n"] += 1
            return FakeSection(index=calls["n"], start=start, end=start)  # zero-length

        ph = Playhead(t=0.0)
        sched = StreamScheduler(
            duration=600, prelude_end=30, process_fn=process_fn,
            playhead_fn=lambda: ph, section_seconds=30, lookahead_seconds=120,
            idle_fn=lambda: False,
        )
        sched.run()
        self.assertLess(calls["n"], 40)                    # terminated, didn't spin

    def test_failed_section_still_terminates(self):
        ph = Playhead(t=0.0)
        sched = StreamScheduler(
            duration=600, prelude_end=20, process_fn=lambda start: None,
            playhead_fn=lambda: ph, section_seconds=30, lookahead_seconds=60,
            idle_fn=lambda: False,
        )
        sched.run()                                        # must not hang
        self.assertFalse(sched.coverage.is_complete(600))

    def test_stop_halts_loop(self):
        def process_fn(start):
            sched.stop()                                   # stop after first section
            return FakeSection(index=1, start=start, end=start + 30)

        ph = Playhead(t=0.0)
        sched = StreamScheduler(
            duration=600, prelude_end=60, process_fn=process_fn,
            playhead_fn=lambda: ph, section_seconds=30, idle_fn=lambda: False,
        )
        sched.run()
        self.assertFalse(sched.coverage.is_complete(600))  # bailed early

    def test_default_idle_abandons_when_playhead_stale(self):
        # No playhead update for > abandon_seconds → end the stream (viewer left).
        import time as _t
        stale = Playhead(t=0.0, updated_at=_t.time() - 999)
        sched = StreamScheduler(
            duration=600, prelude_end=0, process_fn=lambda s: None,
            playhead_fn=lambda: stale, abandon_seconds=45,
        )
        self.assertFalse(sched._default_idle())            # abandoned → don't keep waiting


if __name__ == "__main__":
    unittest.main()
