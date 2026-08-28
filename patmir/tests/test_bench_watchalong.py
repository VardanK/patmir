"""Tests for the scheduler-driven watch-along bench (spec 2026-08-25-1931).

`bench_watch_along` used to walk sections with its own forward `while` loop, so
coverage, lookahead, seek priority, idling and the abandon timer — all of
`StreamScheduler`, i.e. every mechanism that decides WHEN a section is produced
— were absent from the benchmark meant to characterise them. A straight-line
loop also cannot stall, so the freeze the author reported was unmeasurable by
construction.

These pin the replacement: the synthetic viewer, and the CLI wiring that lets a
URL be benchmarked the way the server runs it.
"""
from __future__ import annotations

import argparse
import time
import unittest
from unittest import mock

from patmir import bench as benchmod
from patmir import cli
from patmir.pipeline.streaming.coverage import Coverage


class GreedyPlayheadTest(unittest.TestCase):
    """Greedy sits at the buffer edge: wall time is pure production throughput."""

    def test_jumps_to_the_end_of_coverage(self):
        ph = benchmod.BenchPlayhead("greedy")
        cov = Coverage()
        cov.add(0.0, 40.0)
        ph.coverage = cov
        self.assertEqual(ph().t, 40.0)

    def test_follows_coverage_as_it_grows(self):
        ph = benchmod.BenchPlayhead("greedy")
        cov = Coverage()
        cov.add(0.0, 40.0)
        ph.coverage = cov
        ph()
        cov.add(40.0, 70.0)
        self.assertEqual(ph().t, 70.0)

    def test_stops_at_a_gap(self):
        """Coverage beyond a hole is unreachable — the viewer stalls at the hole."""
        ph = benchmod.BenchPlayhead("greedy")
        cov = Coverage()
        cov.add(0.0, 40.0)
        cov.add(100.0, 200.0)
        ph.coverage = cov
        self.assertEqual(ph().t, 40.0)

    def test_never_stalls(self):
        ph = benchmod.BenchPlayhead("greedy")
        cov = Coverage()
        cov.add(0.0, 40.0)
        ph.coverage = cov
        for _ in range(5):
            ph()
        self.assertEqual(ph.stalls, [])

    def test_keeps_the_abandon_timer_fed(self):
        """A synthetic viewer never goes away; without this the scheduler would
        end the stream after 45 s of idling (scheduler.py `_default_idle`)."""
        ph = benchmod.BenchPlayhead("greedy")
        cov = Coverage()
        cov.add(0.0, 40.0)
        ph.coverage = cov
        before = time.time()
        self.assertGreaterEqual(ph().updated_at, before)


class RealtimePlayheadTest(unittest.TestCase):
    """Realtime advances with the clock and records every freeze."""

    def _ph(self, cov: Coverage, elapsed: float) -> benchmod.BenchPlayhead:
        ph = benchmod.BenchPlayhead("realtime")
        ph.coverage = cov
        # Pretend the stream started `elapsed` seconds ago.
        ph._t0 = time.perf_counter() - elapsed
        return ph

    def test_advances_with_the_clock_while_buffered(self):
        cov = Coverage()
        cov.add(0.0, 100.0)
        ph = self._ph(cov, elapsed=5.0)
        self.assertAlmostEqual(ph().t, 5.0, places=1)
        self.assertEqual(ph.stalls, [])

    def test_clamps_to_the_buffer_edge_when_production_falls_behind(self):
        cov = Coverage()
        cov.add(0.0, 10.0)
        ph = self._ph(cov, elapsed=30.0)   # viewer wants 30 s, only 10 s dubbed
        self.assertEqual(ph().t, 10.0)

    def test_records_the_stall_once_production_catches_up(self):
        cov = Coverage()
        cov.add(0.0, 10.0)
        ph = self._ph(cov, elapsed=30.0)
        ph()                               # stalls at 10.0
        self.assertEqual(ph.stalls, [])    # still stalled — not recorded yet
        cov.add(10.0, 500.0)               # production catches up
        ph()
        self.assertEqual(len(ph.stalls), 1)
        self.assertEqual(ph.stalls[0]["at_s"], 10.0)
        self.assertIn("waited_s", ph.stalls[0])

    def test_a_smooth_stream_records_no_stalls(self):
        cov = Coverage()
        cov.add(0.0, 600.0)
        ph = self._ph(cov, elapsed=1.0)
        for _ in range(3):
            ph()
        self.assertEqual(ph.stalls, [])


class BenchUrlWiringTest(unittest.TestCase):
    """`--url` is the difference between benchmarking the reported video and
    benchmarking something else."""

    def _args(self, **kw) -> argparse.Namespace:
        base = dict(clip=None, url=None, mode="watch-along", config=cli.DEFAULT_CONFIG,
                    repeat=1, warm_cache=True, max_sections=1, report=None, label=None,
                    profile=None, scheduler=True, playhead="greedy", prewarm=False,
                    traces=True, serial_sections=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_rejects_both_clip_and_url(self):
        from pathlib import Path
        rc = cli.cmd_bench(self._args(clip=Path("x.mkv"), url="https://y/z"))
        self.assertEqual(rc, 2)

    def test_rejects_neither(self):
        self.assertEqual(cli.cmd_bench(self._args()), 2)

    def test_url_is_resolved_through_the_servers_download_path(self):
        """Must reuse JobManager._download_url, not a private yt-dlp call, so a
        video already fetched by a real submission is not downloaded twice."""
        from pathlib import Path
        seen = {}

        def fake_fetch(url, cfg, config_path, mode):
            seen["url"], seen["mode"] = url, mode
            return Path("/tmp/fake-source.webm")

        with mock.patch.object(cli, "_bench_fetch_url", fake_fetch), \
             mock.patch("patmir.bench.bench_watch_along", return_value=[]) as run, \
             mock.patch("patmir.bench.format_report", return_value=""):
            rc = cli.cmd_bench(self._args(url="https://www.youtube.com/watch?v=os-jXiYRihI"))

        self.assertEqual(rc, 0)
        self.assertEqual(seen["url"], "https://www.youtube.com/watch?v=os-jXiYRihI")
        self.assertEqual(seen["mode"], "watch-along")
        self.assertEqual(run.call_args.args[0], Path("/tmp/fake-source.webm"))

    def test_download_failure_is_reported_not_raised(self):
        def boom(*a, **k):
            raise RuntimeError("yt-dlp not installed")
        with mock.patch.object(cli, "_bench_fetch_url", boom):
            rc = cli.cmd_bench(self._args(url="https://y/z"))
        self.assertEqual(rc, 2)

    def test_watch_along_flags_reach_the_bench(self):
        from pathlib import Path
        with mock.patch.object(cli, "_bench_fetch_url",
                               lambda *a, **k: Path("/tmp/f.webm")), \
             mock.patch("patmir.bench.bench_watch_along", return_value=[]) as run, \
             mock.patch("patmir.bench.format_report", return_value=""):
            cli.cmd_bench(self._args(url="https://y/z", playhead="realtime",
                                     prewarm=True, scheduler=False, traces=False))
        kw = run.call_args.kwargs
        self.assertEqual(kw["playhead"], "realtime")
        self.assertTrue(kw["prewarm"])
        self.assertFalse(kw["scheduler"])
        self.assertFalse(kw["traces"])


class SharedTimingTest(unittest.TestCase):
    def test_bench_and_live_path_share_one_reader(self):
        """If these ever diverge, bench and production can report different
        numbers for the same run — which is how the 2026-08-25 session had to be
        reconstructed by hand in the first place."""
        from patmir.pipeline.streaming import timing as timing_mod
        self.assertIs(benchmod.ANALYSIS_STAGES, timing_mod.ANALYSIS_STAGES)
        self.assertIs(benchmod.DUB_STAGES, timing_mod.DUB_STAGES)
        self.assertIs(benchmod._stage_wall, timing_mod.stage_wall)


if __name__ == "__main__":
    unittest.main()
