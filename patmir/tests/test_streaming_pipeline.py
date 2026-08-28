"""Tests for the pipelined scheduler: a section's dub runs under the NEXT section's
analysis (2026-08-28 watch-along 2x work).

The serial loop spent every section's wall on `analyze + dub` in sequence. Measured
on `runs/bench/sample_1-cuda-watch-fixed.json`, that is 77.20 s of analysis against
140.48 s of dub over nine steady-state sections — so with the halves overlapped the
loop costs `max(analyze, dub)` per section instead of their sum.

What these tests pin is the part that is easy to get wrong and invisible when it is:
the overlap must not reorder sections, must not let a dub read registry state that
the next analysis has already moved on from, and must not drop an in-flight dub when
the stream stops. The load-bearing test is `test_dub_overlaps_next_analysis`, whose
NEGATIVE CONTROL asserts the serial path does not overlap — without that control the
overlap test passes just as well against a scheduler that never pipelined anything.

Everything is driven with fakes: no models, no audio, no GPU.
"""
from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass, field

from patmir.pipeline.streaming.scheduler import Playhead, StreamScheduler


@dataclass
class FakeAnalysis:
    """Stands in for `SectionAnalysis`; the scheduler only reads these two fields."""
    index: int
    start: float
    end_rel: float
    snapshot: dict = field(default_factory=dict)


@dataclass
class FakeSection:
    index: int
    start: float
    end: float
    dub_wav: str = "x.wav"


class _Recorder:
    """Records interleaving of analyze/dub as an event log with wall times."""

    def __init__(self, analyze_s: float = 0.0, dub_s: float = 0.0,
                 section_len: float = 30.0, duration: float = 600.0):
        self.events: list[tuple[str, int]] = []
        self.lock = threading.Lock()
        self.analyze_s, self.dub_s = analyze_s, dub_s
        self.section_len, self.duration = section_len, duration
        self.n = 0
        self.concurrent = False
        self._running_dub = False

    def _log(self, what: str, idx: int) -> None:
        with self.lock:
            self.events.append((what, idx))

    def analyze(self, start, max_end=None):
        self.n += 1
        idx = self.n
        self._log("analyze_start", idx)
        with self.lock:
            if self._running_dub:
                self.concurrent = True      # a dub was in flight during an analysis
        time.sleep(self.analyze_s)
        self._log("analyze_end", idx)
        return FakeAnalysis(index=idx, start=start,
                            end_rel=min(self.section_len, self.duration - start))

    def dub(self, analysis):
        with self.lock:
            self._running_dub = True
        self._log("dub_start", analysis.index)
        time.sleep(self.dub_s)
        self._log("dub_end", analysis.index)
        with self.lock:
            self._running_dub = False
        return FakeSection(index=analysis.index, start=analysis.start,
                           end=analysis.start + analysis.end_rel)

    def process(self, start, max_end=None):
        """Serial equivalent, for the negative control."""
        return self.dub(self.analyze(start, max_end))


def _sched(rec: _Recorder, *, pipelined: bool, emitted: list, duration=600.0,
           prelude_end=30.0, lookahead=1e9):
    kw = dict(analyze_fn=rec.analyze, dub_fn=rec.dub) if pipelined else {}
    return StreamScheduler(
        duration=duration, prelude_end=prelude_end,
        process_fn=rec.process, playhead_fn=lambda: Playhead(t=0.0),
        on_section=emitted.append, section_seconds=rec.section_len,
        lookahead_seconds=lookahead, idle_fn=lambda: False, **kw,
    )


class PipelinedSchedulerTests(unittest.TestCase):
    def test_dub_overlaps_next_analysis(self):
        """The whole point. Plus the control that proves the test can fail."""
        rec = _Recorder(analyze_s=0.05, dub_s=0.05, duration=150)
        emitted: list = []
        _sched(rec, pipelined=True, emitted=emitted, duration=150).run()
        self.assertTrue(rec.concurrent,
                        "pipelined scheduler ran no dub under an analysis")

        # NEGATIVE CONTROL: same fakes, serial path, must NOT overlap.
        rec2 = _Recorder(analyze_s=0.05, dub_s=0.05, duration=150)
        _sched(rec2, pipelined=False, emitted=[], duration=150).run()
        self.assertFalse(rec2.concurrent,
                         "serial scheduler overlapped — the overlap test proves nothing")

    def test_sections_emitted_in_order(self):
        """One dub worker, drained before the next submit: order cannot slip."""
        rec = _Recorder(analyze_s=0.0, dub_s=0.01, duration=300)
        emitted: list = []
        _sched(rec, pipelined=True, emitted=emitted, duration=300).run()
        self.assertEqual([s.index for s in emitted], sorted(s.index for s in emitted))
        self.assertEqual([s.start for s in emitted], sorted(s.start for s in emitted))

    def test_every_section_is_emitted_including_the_last(self):
        """The final dub is in flight when the loop runs out of work; `finally`
        must drain it. Without that the last section of every stream is lost."""
        rec = _Recorder(analyze_s=0.0, dub_s=0.01, duration=150)
        emitted: list = []
        _sched(rec, pipelined=True, emitted=emitted, duration=150).run()
        analyzed = [i for what, i in rec.events if what == "analyze_end"]
        self.assertEqual([s.index for s in emitted], analyzed)

    def test_coverage_advances_on_analysis_not_dub(self):
        """Coverage has to be recorded as soon as the boundary is known, because
        it decides where the next section starts. If it waited for the dub the
        loop would serialize again — this asserts the next analysis starts before
        the previous dub finishes."""
        rec = _Recorder(analyze_s=0.0, dub_s=0.08, duration=150)
        _sched(rec, pipelined=True, emitted=[], duration=150).run()
        order = [f"{w}:{i}" for w, i in rec.events]
        self.assertIn("dub_start:1", order)
        self.assertLess(order.index("analyze_start:2"), order.index("dub_end:1"))

    def test_failed_analysis_forces_progress(self):
        """A None analysis must advance coverage by _MIN_STEP, not spin forever."""
        seen: list[float] = []

        def analyze(start, max_end=None):
            seen.append(start)
            if len(seen) > 20:
                raise AssertionError("scheduler spun on a failing analysis")
            return None

        sched = StreamScheduler(
            duration=60, prelude_end=30, process_fn=lambda *a, **k: None,
            analyze_fn=analyze, dub_fn=lambda a: None,
            playhead_fn=lambda: Playhead(t=0.0), section_seconds=30,
            lookahead_seconds=1e9, idle_fn=lambda: False,
        )
        sched.run()
        self.assertGreater(len(seen), 1)
        self.assertTrue(all(b > a for a, b in zip(seen, seen[1:])))

    def test_dub_exception_propagates(self):
        """A dub that raises must reach the caller. `JobCancelled` travels this
        path, and swallowing it is exactly the bug the serial `_process` once had:
        Stop got downgraded to 'one section failed' and the stream carried on."""
        class Boom(Exception):
            pass

        rec = _Recorder(duration=150)

        def dub(analysis):
            raise Boom()

        sched = StreamScheduler(
            duration=150, prelude_end=30, process_fn=rec.process,
            analyze_fn=rec.analyze, dub_fn=dub,
            playhead_fn=lambda: Playhead(t=0.0), section_seconds=30,
            lookahead_seconds=1e9, idle_fn=lambda: False,
        )
        with self.assertRaises(Boom):
            sched.run()

    def test_dub_reads_a_snapshot_not_the_live_registry(self):
        """`SectionAnalysis` carries registry snapshots taken on the scheduler
        thread. This pins the contract the split relies on: whatever the dub is
        handed must not change once the next analysis starts mutating state.

        Modelled on the real hazard — `_resolve_identities` registers new speakers
        during analysis N+1 while the dub of N is choosing voices.
        """
        live = {"speakers": ["A"]}
        seen: list[list[str]] = []
        n = {"i": 0}

        def analyze(start, max_end=None):
            n["i"] += 1
            live["speakers"] = live["speakers"] + [f"S{n['i']}"]   # registry mutates
            return FakeAnalysis(index=n["i"], start=start, end_rel=30.0,
                                snapshot={"speakers": list(live["speakers"])})

        def dub(analysis):
            seen.append(analysis.snapshot["speakers"])
            return FakeSection(index=analysis.index, start=analysis.start,
                               end=analysis.start + analysis.end_rel)

        sched = StreamScheduler(
            duration=150, prelude_end=30, process_fn=lambda *a, **k: None,
            analyze_fn=analyze, dub_fn=dub, playhead_fn=lambda: Playhead(t=0.0),
            section_seconds=30, lookahead_seconds=1e9, idle_fn=lambda: False,
        )
        sched.run()
        # Each dub saw the registry as of ITS OWN analysis, not the latest one.
        for i, speakers in enumerate(seen, start=1):
            self.assertEqual(speakers, ["A"] + [f"S{k}" for k in range(1, i + 1)])

    def test_stop_drains_the_in_flight_dub(self):
        """Stop must not abandon a section whose dub already ran. It is written to
        the cache and is the free-rewatch store, so losing it loses real audio."""
        rec = _Recorder(analyze_s=0.0, dub_s=0.02, duration=600)
        emitted: list = []
        sched = _sched(rec, pipelined=True, emitted=emitted, duration=600)

        original = rec.analyze

        def analyze_then_stop(start, max_end=None):
            a = original(start, max_end)
            if a is not None and a.index == 2:
                sched.stop()
            return a

        sched.analyze_fn = analyze_then_stop
        sched.run()
        self.assertEqual([s.index for s in emitted], [1, 2])


if __name__ == "__main__":
    unittest.main()
