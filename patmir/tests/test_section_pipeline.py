"""Tests for the pipelined section driver (`streaming/pipeline.py`).

Watch-along spends 44.2 s of a 153.9 s wall in translate — one cloud round-trip
per section, with the GPU idle throughout. This driver analyses the NEXT
section during that wait.

The properties worth pinning are all about what must NOT happen:

  * **No two GPU phases at once.** The TTS decode captures a CUDA graph, and
    capture fails or corrupts if another thread allocates on the device. This
    is the constraint that shapes the whole design, so it is asserted with a
    live overlap detector rather than reasoned about.
  * **A prefetch that no longer matches is discarded, not used.** After a seek
    the scheduler asks for a different start, or the same start with a `max_end`
    clamp that changes where the section ends. Trusting stale analysis there
    would emit audio for the wrong span — the class of bug behind quality
    issues #22 and #24.
  * **Numbering stays under the caller's control**, because a resumed stream
    must number past the highest cached index or it overwrites
    `section_NNN.wav` (issue #22).
"""
from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from patmir.pipeline.context import JobCancelled
from patmir.pipeline.streaming import pipeline as pipe_mod
from patmir.pipeline.streaming.pipeline import SectionPipeline
from patmir.pipeline.streaming.section import AnalyzedSection, SectionResult


class _GpuWatch:
    """Detects any moment when two 'GPU' phases are in flight together."""

    def __init__(self):
        self.lock = threading.Lock()
        self.live = 0
        self.max_live = 0
        self.order: list[str] = []

    def enter(self, tag):
        with self.lock:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
            self.order.append(tag)

    def leave(self):
        with self.lock:
            self.live -= 1


def _fakes(watch, *, analyze_s=0.02, translate_s=0.05, speak_s=0.02,
           fail_at=None, cancel_at=None):
    """Replace the three phases with instrumented stubs."""
    calls = {"analyze": [], "translate": [], "speak": []}

    def analyze(source, cfg, registry, *, index, start, **kw):
        if cancel_at is not None and abs(start - cancel_at) < 0.01:
            raise JobCancelled()
        if fail_at is not None and abs(start - fail_at) < 0.01:
            raise RuntimeError("analysis blew up")
        watch.enter(f"analyze@{start:.1f}")
        try:
            time.sleep(analyze_s)
        finally:
            watch.leave()
        calls["analyze"].append((index, start, kw.get("max_end")))
        return AnalyzedSection(
            index=index, start=start, end=start + 30.0, end_rel=30.0,
            job_dir=Path(f"/tmp/nope/{index}"), local_to_global={},
            new_speakers=[], started_at=time.perf_counter(),
        )

    def translate(analyzed, cfg, registry, **kw):
        # NOT under the GPU lock: this is the network wait the whole design
        # exists to fill.
        calls["translate"].append(analyzed.start)
        time.sleep(translate_s)

    def speak(analyzed, cfg, registry, **kw):
        watch.enter(f"speak@{analyzed.start:.1f}")
        try:
            time.sleep(speak_s)
        finally:
            watch.leave()
        calls["speak"].append(analyzed.start)
        return SectionResult(
            index=analyzed.index, start=analyzed.start, end=analyzed.end,
            dub_wav=Path("/tmp/nope/dub.wav"), job_dir=analyzed.job_dir,
            wall_s=0.1,
        )

    return analyze, translate, speak, calls


class _Cfg:
    class options:
        section_prefetch = True

    class storage:
        runs_root = Path("/tmp/nope")


def _pipeline(watch, monkey, **kw):
    a, t, sp, calls = _fakes(watch, **{k: v for k, v in kw.items()
                                       if k in ("analyze_s", "translate_s",
                                                "speak_s", "fail_at",
                                                "cancel_at")})
    monkey(pipe_mod, "analyze_section", a)
    monkey(pipe_mod, "translate_section", t)
    monkey(pipe_mod, "speak_section", sp)
    p = SectionPipeline(
        Path("/tmp/nope/src.mkv"), _Cfg(), object(),
        duration=kw.get("duration", 3000.0),
        prefetch=kw.get("prefetch", True),
        index_start=kw.get("index_start", 0),
    )
    return p, calls


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self._saved = {}
        self.watch = _GpuWatch()

    def _monkey(self, mod, name, val):
        self._saved.setdefault((mod, name), getattr(mod, name))
        setattr(mod, name, val)

    def tearDown(self):
        for (mod, name), val in self._saved.items():
            setattr(mod, name, val)

    def test_gpu_phases_never_overlap(self):
        # THE constraint: CUDA graph capture cannot tolerate another thread
        # allocating on the device. A long translate gives the prefetch every
        # chance to run alongside a speak; max_live must still be 1.
        p, calls = _pipeline(self.watch, self._monkey,
                             analyze_s=0.05, translate_s=0.12, speak_s=0.05)
        with p:
            r0 = p.process(0.0)
            r1 = p.process(r0.end)
            p.process(r1.end)
        self.assertEqual(self.watch.max_live, 1,
                         f"GPU phases overlapped: {self.watch.order}")
        self.assertEqual(len(calls["speak"]), 3)

    def test_the_prefetch_actually_runs_during_translate(self):
        # The win itself: by the time section 0 is spoken, section 1's analysis
        # is already done, so the second process() is a prefetch HIT.
        p, calls = _pipeline(self.watch, self._monkey,
                             analyze_s=0.02, translate_s=0.15)
        with p:
            r0 = p.process(0.0)
            p.process(r0.end)
        self.assertEqual(p.stats["prefetch_hits"], 1)
        self.assertEqual(p.stats["prefetch_misses"], 1)   # the first section

    def test_sections_are_spoken_in_order(self):
        p, _calls = _pipeline(self.watch, self._monkey, translate_s=0.1)
        with p:
            r0 = p.process(0.0)
            r1 = p.process(r0.end)
            r2 = p.process(r1.end)
        self.assertEqual([r0.index, r1.index, r2.index], [1, 2, 3])
        self.assertLess(r0.start, r1.start)
        self.assertLess(r1.start, r2.start)

    def test_a_seek_discards_the_prefetch(self):
        # The scheduler asks for somewhere else entirely. Reusing analysis of
        # the wrong span would emit audio for the wrong seconds.
        p, calls = _pipeline(self.watch, self._monkey, translate_s=0.05)
        with p:
            p.process(0.0)
            p.process(900.0)              # seek, not the prefetched start
        self.assertEqual(p.stats["prefetch_discarded"], 1)
        self.assertEqual(p.stats["prefetch_hits"], 0)
        self.assertIn(900.0, [c[1] for c in calls["analyze"]])

    def test_a_clamped_window_discards_the_prefetch(self):
        # Same start, but `max_end` changes where the section ends, so the
        # prefetched analysis snapped to the wrong boundary.
        p, calls = _pipeline(self.watch, self._monkey, translate_s=0.05)
        with p:
            r0 = p.process(0.0)
            p.process(r0.end, 45.0)
        self.assertEqual(p.stats["prefetch_discarded"], 1)
        self.assertIn(45.0, [c[2] for c in calls["analyze"]])

    def test_a_prefetch_is_never_started_with_a_clamp(self):
        # The next window's clamp comes from coverage the scheduler has not
        # updated yet, so a speculative analysis must be unclamped.
        p, calls = _pipeline(self.watch, self._monkey, translate_s=0.08)
        with p:
            p.process(0.0)
        prefetched = [c for c in calls["analyze"] if c[1] > 0.0]
        self.assertTrue(all(c[2] is None for c in prefetched))

    def test_prefetch_disabled_is_fully_serial(self):
        p, calls = _pipeline(self.watch, self._monkey, prefetch=False,
                             translate_s=0.05)
        with p:
            r0 = p.process(0.0)
            p.process(r0.end)
        self.assertEqual(p.stats["prefetch_hits"], 0)
        self.assertEqual(p.stats["prefetch_misses"], 2)
        self.assertEqual(self.watch.max_live, 1)

    def test_indexing_starts_where_the_caller_says(self):
        # A resumed stream must number PAST the highest cached index or it
        # overwrites section_NNN.wav — quality issue #22.
        p, _calls = _pipeline(self.watch, self._monkey, index_start=7,
                              translate_s=0.05)
        with p:
            r0 = p.process(0.0)
            r1 = p.process(r0.end)
        self.assertEqual([r0.index, r1.index], [8, 9])

    def test_no_prefetch_past_the_end_of_the_video(self):
        p, calls = _pipeline(self.watch, self._monkey, duration=30.5,
                             translate_s=0.05)
        with p:
            p.process(0.0)             # section ends at 30.0, duration 30.5
        self.assertEqual([c[1] for c in calls["analyze"]], [0.0])

    def test_a_failed_prefetch_is_redone_inline(self):
        # A broken speculative analysis must cost only the wasted work.
        p, calls = _pipeline(self.watch, self._monkey, fail_at=30.0,
                             translate_s=0.05)
        with p:
            r0 = p.process(0.0)
            with self.assertRaises(RuntimeError):
                p.process(r0.end)      # inline retry hits the same failure
        self.assertGreaterEqual(p.stats["prefetch_misses"], 1)

    def test_cancellation_propagates(self):
        # JobCancelled must not be swallowed into "one section failed".
        p, _calls = _pipeline(self.watch, self._monkey, cancel_at=0.0)
        with p:
            with self.assertRaises(JobCancelled):
                p.process(0.0)

    def test_close_waits_for_an_in_flight_prefetch(self):
        # A torch forward cannot be interrupted, so leaving the thread running
        # would hand the GPU to the next job while we still hold it.
        p, calls = _pipeline(self.watch, self._monkey, analyze_s=0.15,
                             translate_s=0.01, speak_s=0.01)
        p.process(0.0)
        p.close()
        self.assertEqual(self.watch.live, 0)


if __name__ == "__main__":
    unittest.main()
