"""Unit tests for `patmir bench` (`patmir/bench.py`).

Nothing covered this module before 2026-08-24, which is how it went unnoticed
that `bench_batch` passed no `stage_cache` while `bench_watch_along` did — so
`--repeat N` was timing every stage model being rebuilt N times. On MLX that
was invisible (mmap loads, `build_s` 0-3 s); on torch it was ~110 s of a 242 s
run, 45% of wall.

The models are never built here. `run_job` is stubbed with something that
honours the cache contract (`build_s = 0.0` on a hit) so the tests assert on the
contract rather than on a call counter.
"""
from __future__ import annotations

import inspect
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from patmir import bench
from patmir.bench import (
    BenchResult,
    StageTiming,
    _tts_stats,
    bench_batch,
    format_report,
)
from patmir.pipeline.manifest import Manifest, StageStatus
from patmir.pipeline.stage import STAGE_ORDER

BUILD_S = 5.0
STAGE_WALL_S = 10.0
SOURCE_S = 300.0


def _write_job(runs_root: Path, job_id: str, *, build_s: float,
               tts_rows: list[dict[str, Any]] | None = None,
               sentences: int = 24) -> Manifest:
    """Materialize the on-disk artifacts bench reads back."""
    job_dir = runs_root / job_id
    (job_dir / "tts").mkdir(parents=True, exist_ok=True)
    (job_dir / "audio.meta.json").write_text(json.dumps({"duration": SOURCE_S}))
    (job_dir / "transcript.json").write_text(json.dumps({
        "sentences": [{"idx": i, "text": f"s{i}"} for i in range(sentences)],
    }))
    if tts_rows is not None:
        (job_dir / "tts" / "timings.jsonl").write_text(
            "\n".join(json.dumps(r) for r in tts_rows))
    (job_dir / "tts" / "voice_cloning_summary.json").write_text(
        json.dumps({"engine": "test", "sentences": sentences, "overflows": 0}))

    t0 = 1000.0
    stages = {}
    for i, name in enumerate(STAGE_ORDER):
        stages[name] = StageStatus(
            name=name, state="done",
            started_at=t0 + i * STAGE_WALL_S,
            finished_at=t0 + (i + 1) * STAGE_WALL_S,
            build_s=build_s,
        )
    m = Manifest(job_id=job_id, source_path="clip.mkv", source_lang="ru",
                 created_at=t0, stages=stages)
    m.to_path(job_dir / "manifest.json")
    return m


class _FakeConfig:
    """Only the attributes bench touches."""
    profile = "cuda"
    entries = {name: {"id": f"cand_{name}"} for name in STAGE_ORDER}


def _run_job_stub(captured: list[Any], runs_root: Path):
    """A `run_job` that respects the stage_cache contract: a stage already in
    the dict costs nothing to build, otherwise it builds and is inserted."""
    calls = {"n": 0}

    def _run_job(clip, cfg, *, extra=None, stage_cache=None, **kw):
        captured.append(stage_cache)
        job_id = f"job{calls['n']}"
        calls["n"] += 1
        if stage_cache is None:
            build = BUILD_S
        else:
            build = 0.0 if all(n in stage_cache for n in STAGE_ORDER) else BUILD_S
            for n in STAGE_ORDER:
                stage_cache[n] = object()
        return _write_job(runs_root, job_id, build_s=build)

    return _run_job


class BatchWarmCacheTest(unittest.TestCase):
    def _run(self, repeat: int, warm_cache: bool):
        captured: list[Any] = []
        with TemporaryDirectory() as td:
            runs_root = Path(td)
            with mock.patch.object(bench, "run_job",
                                   _run_job_stub(captured, runs_root)), \
                 mock.patch.object(bench, "_free_accelerator_cache", lambda: None):
                results = bench_batch(Path("clip.mkv"), _FakeConfig(),
                                      runs_root=runs_root, repeat=repeat,
                                      warm_cache=warm_cache)
        return results, captured

    def test_all_repeats_share_one_stage_cache(self):
        _, captured = self._run(3, warm_cache=True)
        self.assertEqual(len(captured), 3)
        self.assertIsNotNone(captured[0])
        for c in captured[1:]:
            self.assertIs(c, captured[0], "every repeat must get the SAME dict")

    def test_no_warm_cache_passes_none_every_repeat(self):
        _, captured = self._run(3, warm_cache=False)
        self.assertEqual(captured, [None, None, None])

    def test_repeat_zero_is_cold_and_the_rest_warm(self):
        results, _ = self._run(3, warm_cache=True)
        self.assertFalse(results[0].warm)
        self.assertGreater(results[0].build_total_s, 0.0)
        for r in results[1:]:
            self.assertTrue(r.warm, f"repeat {r.repeat_index} should be warm")
            self.assertEqual(r.build_total_s, 0.0)

    def test_every_repeat_is_cold_without_the_cache(self):
        results, _ = self._run(2, warm_cache=False)
        for r in results:
            self.assertFalse(r.warm)
            self.assertGreater(r.build_total_s, 0.0)

    def test_warm_flag_is_measured_not_declared(self):
        """`warm` must agree with this run's own build total, so the JSON says
        what happened rather than what was requested."""
        results, _ = self._run(2, warm_cache=True)
        for r in results:
            self.assertEqual(r.warm,
                             r.warm_cache and r.build_total_s <= bench.WARM_BUILD_EPSILON_S)

    def test_warm_cache_flag_and_profile_are_recorded(self):
        results, _ = self._run(2, warm_cache=True)
        for r in results:
            self.assertTrue(r.warm_cache)
            self.assertEqual(r.profile, "cuda")

    def test_default_is_warm(self):
        """Guards the default drifting away from the CLI's."""
        self.assertIs(
            inspect.signature(bench_batch).parameters["warm_cache"].default, True)
        self.assertIs(
            inspect.signature(bench.bench_watch_along).parameters["warm_cache"].default,
            True)

    def test_cache_is_released_after_the_last_repeat(self):
        """~15 GB of models must not stay pinned while the report is written."""
        _, captured = self._run(2, warm_cache=True)
        self.assertEqual(captured[0], {})

    def test_build_pct_of_wall_is_reported(self):
        results, _ = self._run(2, warm_cache=True)
        self.assertGreater(results[0].notes["build_pct_of_wall"], 0.0)
        self.assertEqual(results[1].notes["build_pct_of_wall"], 0.0)


class BenchResultCompatTest(unittest.TestCase):
    """The 2026-08-24 fields must not break the stored reports."""

    PRE_CHANGE_KWARGS = dict(
        clip="c.mkv", mode="batch", label=None, repeat_index=0, job_id="j",
        source_s=300.0, total_wall_s=100.0, agg_rtf=0.33, stages=[],
        sentence_count=1, tts=None, translate=None, models={},
    )

    def test_new_fields_have_defaults(self):
        r = BenchResult(**self.PRE_CHANGE_KWARGS)
        self.assertIsNone(r.profile)
        self.assertFalse(r.warm_cache)
        self.assertFalse(r.warm)
        self.assertIsNone(r.build_total_s)

    def test_asdict_output_is_a_superset(self):
        keys = set(asdict(BenchResult(**self.PRE_CHANGE_KWARGS)))
        self.assertTrue(set(self.PRE_CHANGE_KWARGS).issubset(keys))

    def test_stored_reports_still_parse(self):
        roots = [
            Path("agent-os/specs/2026-06-20-1656-pipeline-benchmark-perf/results"),
            Path("runs/bench"),
        ]
        files = [f for root in roots if root.is_dir() for f in sorted(root.glob("*.json"))]
        if not files:
            self.skipTest("no stored bench reports in this checkout")
        for f in files:
            payload = json.loads(f.read_text())
            if "results" not in payload:
                continue
            for row in payload["results"]:
                stages = [StageTiming(**s) for s in row.pop("stages", [])]
                BenchResult(stages=stages, **row)


class TtsStatUnitsTest(unittest.TestCase):
    """Under batched decode a timings row is a CHUNK, not a sentence. The CUDA
    report read `count: 3` beside `sentences: 24` with `p90_s: 51.2` — chunk
    numbers that are trivially misread as the per-sentence p90 the
    Apple-Silicon rows quote."""

    def _stats(self, rows, sentences=24):
        with TemporaryDirectory() as td:
            runs_root = Path(td)
            _write_job(runs_root, "j", build_s=0.0, tts_rows=rows,
                       sentences=sentences)
            return _tts_stats(runs_root / "j")

    CHUNKS = [
        {"kind": "chunk", "sentences": 8, "elapsed_s": 10.0, "audio_s": 40.0},
        {"kind": "chunk", "sentences": 8, "elapsed_s": 20.0, "audio_s": 40.0},
        {"kind": "chunk", "sentences": 8, "elapsed_s": 30.0, "audio_s": 40.0},
    ]
    SENTENCES = [
        {"idx": i, "elapsed_s": 2.0, "audio_s": 4.0} for i in range(6)
    ]

    def test_chunk_rows_report_unit_chunk(self):
        out = self._stats(self.CHUNKS)
        self.assertEqual(out["unit"], "chunk")
        self.assertEqual(out["count"], 3)
        self.assertEqual(out["chunks"], 3)
        self.assertEqual(out["sentences_per_chunk"], 8.0)

    def test_sentence_rows_report_unit_sentence(self):
        out = self._stats(self.SENTENCES)
        self.assertEqual(out["unit"], "sentence")
        self.assertNotIn("chunks", out)
        self.assertNotIn("sentences_per_chunk", out)

    def test_summary_sentence_total_is_not_shadowed(self):
        """The per-job `sentences` total from voice_cloning_summary.json must
        keep winning that key while the chunk figures sit beside it."""
        out = self._stats(self.CHUNKS, sentences=24)
        self.assertEqual(out["sentences"], 24)
        self.assertEqual(out["chunk_sentences"], 24)
        self.assertEqual(out["sentences_per_chunk"], 8.0)

    def test_totals_and_rtf_are_unchanged_under_batching(self):
        out = self._stats(self.CHUNKS)
        self.assertEqual(out["total_s"], 60.0)
        self.assertEqual(out["audio_s"], 120.0)
        self.assertEqual(out["tts_rtf"], 0.5)

    def test_percentiles_describe_rows(self):
        out = self._stats(self.CHUNKS)
        self.assertEqual(out["max_s"], 30.0)
        self.assertEqual(out["median_s"], 20.0)


class ReportFormattingTest(unittest.TestCase):
    def _batch(self, **over):
        base = dict(
            clip="samples/sample_1.mkv", mode="batch", label=None,
            repeat_index=0, job_id="j0", source_s=300.0, total_wall_s=240.0,
            agg_rtf=0.8, stages=[], sentence_count=24, tts=None,
            translate=None, models={}, notes={"build_pct_of_wall": 45.0},
            profile="cuda", warm_cache=True, warm=False, build_total_s=110.0,
        )
        base.update(over)
        return BenchResult(**base)

    def test_cold_and_warm_repeats_are_labelled(self):
        cold = self._batch(repeat_index=0, warm=False, build_total_s=110.0)
        warm = self._batch(repeat_index=1, warm=True, build_total_s=0.0,
                           job_id="j1", notes={"build_pct_of_wall": 0.0})
        text = format_report([cold, warm])
        self.assertIn("repeat=0 (cold: builds models)", text)
        self.assertIn("repeat=1 (warm)", text)
        self.assertIn("build 110.0s", text)

    def test_no_warm_cache_is_labelled(self):
        a = self._batch(warm_cache=False, warm=False)
        b = self._batch(warm_cache=False, warm=False, repeat_index=1, job_id="j1")
        self.assertIn("(no warm cache)", format_report([a, b]))

    def test_cold_and_warm_are_summarized_separately(self):
        cold = self._batch(repeat_index=0, warm=False, build_total_s=110.0)
        warm = self._batch(repeat_index=1, warm=True, build_total_s=0.0,
                           total_wall_s=130.0, agg_rtf=0.43, job_id="j1")
        text = format_report([cold, warm])
        self.assertIn("cold repeat 0:", text)
        self.assertIn("warm repeat 1:", text)

    def test_chunk_stats_are_labelled_chunks_not_sentences(self):
        res = self._batch(tts={
            "unit": "chunk", "count": 3, "chunks": 3, "chunk_sentences": 24,
            "sentences_per_chunk": 8.0, "sentences": 24, "mean_s": 27.8,
            "p90_s": 51.2, "max_s": 51.2, "tts_rtf": 0.39, "overflows": 0,
        })
        text = format_report([res])
        self.assertIn("3 chunks over 24 sentences", text)
        self.assertIn("p90 51.2s/chunk", text)
        self.assertNotIn("3 sent,", text)

    def test_sentence_stats_are_not_labelled_per_chunk(self):
        res = self._batch(tts={
            "unit": "sentence", "count": 25, "mean_s": 9.65, "p90_s": 21.4,
            "max_s": 30.4, "tts_rtf": 1.136, "overflows": 0,
        })
        text = format_report([res])
        self.assertIn("25 sentences", text)
        self.assertNotIn("/chunk", text)

    def test_watch_along_prints_both_rtf_denominators(self):
        """The historical watch-along numbers were computed against dubbed
        audio; the stated convention is source seconds. Both must be visible."""
        res = BenchResult(
            clip="samples/sample_1.mkv", mode="watch-along", label=None,
            repeat_index=0, job_id="j", source_s=300.0, total_wall_s=471.9,
            agg_rtf=1.5, stages=[], sentence_count=None, tts=None,
            translate=None, models={}, sections=[],
            notes={
                "n_sections": 12, "prelude_wall_s": 22.6, "total_build_s": 15.9,
                "build_pct_of_wall": 3.4, "dubbed_audio_s": 314.7,
                "rtf_denominator": "dubbed_audio_s", "agg_rtf_source": 1.573,
            },
        )
        text = format_report([res])
        self.assertIn("per dubbed_audio_s 314.7s", text)
        self.assertIn("per source 1.573", text)


class WatchAlongOptOutTest(unittest.TestCase):
    def test_no_warm_cache_passes_none_into_the_section_loop(self):
        """And the default passes one shared dict per repeat — deliberately NOT
        hoisted across repeats, so each repeat pays its own section-0 builds and
        stays comparable with the stored 1.71 / 1.611 / 1.47 baselines."""
        src = inspect.getsource(bench.bench_watch_along)
        self.assertIn("{} if warm_cache else None", src)
        # created inside `for r in range(repeat)`, not before it
        loop_at = src.index("for r in range(repeat)")
        cache_at = src.index("{} if warm_cache else None")
        self.assertGreater(cache_at, loop_at,
                           "watch-along cache must be created inside the repeat loop")


class CliWiringTest(unittest.TestCase):
    def test_no_warm_cache_flag_reaches_bench_batch(self):
        from patmir import cli

        with TemporaryDirectory() as td:
            clip = Path(td) / "clip.mkv"
            clip.write_bytes(b"x")
            cfgfile = Path(td) / "patmir.toml"
            cfgfile.write_text("")

            captured: dict[str, Any] = {}

            def _spy(*a, **kw):
                captured.update(kw)
                return []

            class _Cfg(_FakeConfig):
                class storage:
                    runs_root = Path(td) / "runs"
                options = None

            for flag, expected in ((["--no-warm-cache"], False), ([], True)):
                captured.clear()
                argv = ["bench", str(clip), "--config", str(cfgfile), *flag]
                with mock.patch.object(cli, "load_pipeline_config", lambda p: _Cfg()), \
                     mock.patch.object(bench, "bench_batch", _spy), \
                     mock.patch.object(bench, "format_report", lambda r: ""):
                    rc = cli.main(argv)
                self.assertEqual(rc, 0, flag)
                self.assertIs(captured.get("warm_cache"), expected, flag)


if __name__ == "__main__":
    unittest.main()
