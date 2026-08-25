"""Tests for per-section watch-along telemetry (spec 2026-08-25-1931).

Watch-along used to persist NO timing: `sections.json` held
`{index, start, end, file}` and nothing else, so a slow live stream could only
be explained by reading `runs/` afterwards. These pin the contract of the
replacement.

Three things are load-bearing and each has a test:

  * `other_s` — wall MINUS the sum of stage walls. On the 2026-08-25 session
    this was ~1.6 s on every section and 15.5 s on the first prelude, and it is
    invisible to every per-stage view because no stage owns it (ffmpeg
    slice/trim, ECAPA embed, age-gender profile, and the lazy model loads that
    `prewarm_stage_cache` does not cover).
  * Telemetry can never fail a section that already succeeded — a missing or
    unreadable manifest yields zeros, not an exception.
  * The `sections.json` addition is ADDITIVE: entries written by an older build
    (no timing keys) stay valid, and `index`/`start`/`end` always come from the
    caller's arguments so the audio and its timing cannot disagree.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patmir.pipeline.manifest import Manifest, StageStatus
from patmir.pipeline.stage import STAGE_ORDER
from patmir.pipeline.streaming.scheduler import Playhead, StreamScheduler
from patmir.pipeline.streaming.timing import (
    ANALYSIS_STAGES,
    DUB_STAGES,
    section_timing,
)
from patmir.ui import cache as cache_mod
from patmir.ui.stream import StreamManager, StreamRecord

STAGE_WALL_S = 2.0


def _write_manifest(job_dir: Path, *, build_s: float = 0.0,
                    only: list[str] | None = None) -> None:
    """A section job dir with one manifest carrying both run_job halves."""
    job_dir.mkdir(parents=True, exist_ok=True)
    names = only or list(STAGE_ORDER)
    t0 = 1000.0
    stages = {}
    for i, name in enumerate(names):
        stages[name] = StageStatus(
            name=name, state="done",
            started_at=t0 + i * STAGE_WALL_S,
            finished_at=t0 + (i + 1) * STAGE_WALL_S,
            build_s=build_s,
        )
    Manifest(job_id=job_dir.name, source_path="section.wav",
             created_at=t0, stages=stages).to_path(job_dir / "manifest.json")


class SectionTimingTest(unittest.TestCase):
    def test_splits_analysis_and_dub_halves(self):
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "20260825-000000-section"
            _write_manifest(jd)
            rec = section_timing(index=3, start=100.0, end=130.0, job_dir=jd,
                                 wall_s=30.0)
        self.assertEqual(rec["analysis_wall_s"], len(ANALYSIS_STAGES) * STAGE_WALL_S)
        self.assertEqual(rec["dub_wall_s"], len(DUB_STAGES) * STAGE_WALL_S)
        # The two halves partition STAGE_ORDER exactly — no stage unaccounted.
        self.assertEqual(sorted(ANALYSIS_STAGES + DUB_STAGES), sorted(STAGE_ORDER))
        self.assertEqual(rec["section_s"], 30.0)
        self.assertEqual(rec["rtf"], 1.0)

    def test_other_s_is_wall_minus_stage_sum(self):
        """The whole point: work no stage owns has to be visible."""
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "sec"
            _write_manifest(jd)
            stage_sum = len(STAGE_ORDER) * STAGE_WALL_S
            rec = section_timing(index=1, start=0.0, end=30.0, job_dir=jd,
                                 wall_s=stage_sum + 1.6)
        self.assertAlmostEqual(rec["other_s"], 1.6, places=2)

    def test_other_s_never_negative(self):
        """Stage walls use time.time() and the section wall perf_counter, so a
        clock adjustment can make the sum exceed the wall. Report 0, not -0.3."""
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "sec"
            _write_manifest(jd)
            rec = section_timing(index=1, start=0.0, end=30.0, job_dir=jd,
                                 wall_s=1.0)
        self.assertEqual(rec["other_s"], 0.0)

    def test_missing_wall_yields_none_not_zero(self):
        """None means 'not measured'. Zero would read as 'took no time'."""
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "sec"
            _write_manifest(jd)
            rec = section_timing(index=1, start=0.0, end=30.0, job_dir=jd,
                                 wall_s=None)
        self.assertIsNone(rec["wall_s"])
        self.assertIsNone(rec["other_s"])
        self.assertIsNone(rec["rtf"])

    def test_zero_length_section_has_no_rtf(self):
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "sec"
            _write_manifest(jd)
            rec = section_timing(index=1, start=50.0, end=50.0, job_dir=jd,
                                 wall_s=10.0)
        self.assertIsNone(rec["rtf"])
        self.assertEqual(rec["section_s"], 0.0)

    def test_missing_manifest_does_not_raise(self):
        """Telemetry must never fail a dub that already landed on disk."""
        with tempfile.TemporaryDirectory() as td:
            rec = section_timing(index=0, start=0.0, end=40.0,
                                 job_dir=Path(td) / "nonexistent", wall_s=72.4)
        self.assertEqual(rec["analysis_wall_s"], 0.0)
        self.assertEqual(rec["wall_s"], 72.4)
        # With no stage data the whole wall is unattributed, which is honest.
        self.assertEqual(rec["other_s"], 72.4)

    def test_corrupt_manifest_does_not_raise(self):
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "sec"
            jd.mkdir()
            (jd / "manifest.json").write_text("{not json")
            rec = section_timing(index=0, start=0.0, end=40.0, job_dir=jd,
                                 wall_s=10.0)
        self.assertEqual(rec["build_s"], 0.0)

    def test_partial_manifest_counts_only_stages_that_ran(self):
        """Watch-along sections stop at mux — `output` never runs."""
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "sec"
            _write_manifest(jd, only=["ingest", "separate", "diarize",
                                      "emotion", "asr", "translate", "tts", "mux"])
            rec = section_timing(index=1, start=0.0, end=30.0, job_dir=jd,
                                 wall_s=20.0)
        self.assertEqual(rec["stage_wall_s"]["output"], 0.0)
        self.assertEqual(rec["dub_wall_s"], 3 * STAGE_WALL_S)

    def test_optional_fields_only_present_when_given(self):
        with tempfile.TemporaryDirectory() as td:
            jd = Path(td) / "sec"
            _write_manifest(jd)
            bare = section_timing(index=1, start=0.0, end=30.0, job_dir=jd,
                                  wall_s=20.0)
            full = section_timing(index=1, start=0.0, end=30.0, job_dir=jd,
                                  wall_s=20.0, buffer_ahead_s=12.5,
                                  produced_at=1787667776.5)
        self.assertNotIn("buffer_ahead_s", bare)
        self.assertNotIn("produced_at", bare)
        self.assertEqual(full["buffer_ahead_s"], 12.5)
        self.assertEqual(full["produced_at"], 1787667776.5)


class AppendSectionTimingTest(unittest.TestCase):
    """The `sections.json` addition has to be additive and non-authoritative."""

    def _wav(self, td: Path) -> Path:
        p = td / "chunk.wav"
        p.write_bytes(b"RIFF....WAVEfmt ")
        return p

    def test_timing_is_merged_into_the_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root, key = Path(td), "k"
            cache_mod.stream_cache_dir(root, key)
            m = cache_mod.append_section(
                root, key, index=1, start=40.0, end=70.0,
                src_wav=self._wav(Path(td)),
                timing={"wall_s": 31.9, "other_s": 1.6, "rtf": 1.06},
            )
        entry = m["sections"][0]
        self.assertEqual(entry["wall_s"], 31.9)
        self.assertEqual(entry["other_s"], 1.6)
        self.assertEqual(entry["file"], "section_001.wav")

    def test_identity_fields_come_from_args_not_timing(self):
        """A timing dict that disagrees must never move the audio."""
        with tempfile.TemporaryDirectory() as td:
            root, key = Path(td), "k"
            cache_mod.stream_cache_dir(root, key)
            m = cache_mod.append_section(
                root, key, index=2, start=70.0, end=100.0,
                src_wav=self._wav(Path(td)),
                timing={"index": 99, "start": -1.0, "end": -1.0, "file": "bogus.wav"},
            )
        entry = m["sections"][0]
        self.assertEqual(entry["index"], 2)
        self.assertEqual(entry["start"], 70.0)
        self.assertEqual(entry["end"], 100.0)
        self.assertEqual(entry["file"], "section_002.wav")

    def test_no_timing_still_writes_a_valid_entry(self):
        """Older builds and any caller that skips telemetry stay valid."""
        with tempfile.TemporaryDirectory() as td:
            root, key = Path(td), "k"
            cache_mod.stream_cache_dir(root, key)
            m = cache_mod.append_section(root, key, index=0, start=0.0, end=40.0,
                                         src_wav=self._wav(Path(td)))
        entry = m["sections"][0]
        self.assertEqual(set(entry), {"index", "start", "end", "file"})

    def test_replacement_by_start_still_works_with_timing(self):
        with tempfile.TemporaryDirectory() as td:
            root, key = Path(td), "k"
            cache_mod.stream_cache_dir(root, key)
            wav = self._wav(Path(td))
            cache_mod.append_section(root, key, index=1, start=40.0, end=70.0,
                                     src_wav=wav, timing={"wall_s": 30.0})
            m = cache_mod.append_section(root, key, index=1, start=40.0, end=75.0,
                                         src_wav=wav, timing={"wall_s": 12.0})
        self.assertEqual(len(m["sections"]), 1)
        self.assertEqual(m["sections"][0]["end"], 75.0)
        self.assertEqual(m["sections"][0]["wall_s"], 12.0)


class BufferAheadTest(unittest.TestCase):
    """`buffer_ahead_s` crossing zero IS the freeze, so its arithmetic matters."""

    def _rec(self, playhead_t: float, duration: float = 600.0) -> StreamRecord:
        rec = StreamRecord(
            job_id="j", cache_key="k", source_id="s", source_url=None,
            source_path="/tmp/x.mkv", source_lang=None, status="streaming",
            submitted_at=0.0,
        )
        rec.duration = duration
        rec.playhead = Playhead(t=playhead_t)
        return rec

    def _sched(self, prelude_end: float, duration: float = 600.0) -> StreamScheduler:
        return StreamScheduler(
            duration=duration, prelude_end=prelude_end,
            process_fn=lambda _s: None, playhead_fn=Playhead,
            idle_fn=lambda: False,
        )

    def test_none_before_the_scheduler_exists(self):
        """Section 0 is emitted before the scheduler is built."""
        rec = self._rec(0.0)
        self.assertIsNone(StreamManager._buffer_ahead(None, rec))

    def test_seconds_of_continuous_audio_ahead_of_the_playhead(self):
        rec = self._rec(10.0)
        rec._scheduler = self._sched(40.0)
        # Dubbed [0,40); playhead at 10 → 30 s of runway.
        self.assertEqual(StreamManager._buffer_ahead(None, rec), 30.0)

    def test_zero_when_the_playhead_has_caught_up(self):
        rec = self._rec(40.0)
        rec._scheduler = self._sched(40.0)
        self.assertEqual(StreamManager._buffer_ahead(None, rec), 0.0)

    def test_a_gap_ahead_bounds_the_buffer(self):
        """Coverage past a hole does not count — the client stalls at the hole."""
        rec = self._rec(10.0)
        sched = self._sched(40.0)
        sched.coverage.add(100.0, 200.0)   # dubbed later, but unreachable
        rec._scheduler = sched
        self.assertEqual(StreamManager._buffer_ahead(None, rec), 30.0)

    def test_fully_covered_to_the_end_reports_the_remainder(self):
        rec = self._rec(10.0, duration=100.0)
        rec._scheduler = self._sched(100.0, duration=100.0)
        self.assertEqual(StreamManager._buffer_ahead(None, rec), 90.0)


if __name__ == "__main__":
    unittest.main()


class LanguagePinningTest(unittest.TestCase):
    """B1/B4 of spec 2026-08-25-1931: the prelude's detected language is reused.

    Qwen3-ASR probes for the language when `ctx.source_lang` is unset, and that
    probe cost 3.5-4.6 s on EVERY section of the 2026-08-25 session (12-15% of
    section wall) to re-answer a question the prelude had already answered about
    the same soundtrack.
    """

    def test_prelude_result_carries_the_detected_language(self):
        from patmir.pipeline.streaming.prelude import PreludeResult
        import dataclasses
        names = [f.name for f in dataclasses.fields(PreludeResult)]
        self.assertIn("source_lang", names)

    def test_process_section_forwards_the_language_to_run_job(self):
        """The analysis half is where ASR runs, so that is the call that must
        carry it."""
        import inspect
        from patmir.pipeline.streaming import section as section_mod
        src = inspect.getsource(section_mod.process_section)
        self.assertIn("source_lang=source_lang", src)

    def test_the_live_stream_passes_the_preludes_language(self):
        import inspect
        from patmir.ui import stream as stream_mod
        src = inspect.getsource(stream_mod.StreamManager._run_blocking)
        self.assertIn("source_lang=prelude.source_lang", src)
