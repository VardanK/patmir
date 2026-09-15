"""A translator outage fails the dub instead of muting it (quality issue #35).

On 2026-09-13 09:49 the OpenRouter account ran out of credits. For two days
every watch-along section was produced with zero translated lines: the batched
candidate turned each 402 into "", the adapter reported "N of N line(s) came
back empty" and said `done`, the TTS synthesized nothing, and the mux emitted
the music bed plus the original vocals at the gap floor. The extension played
that over the muted element and counted the sections as translated. Nine cache
keys had to be purged.

The candidate- and adapter-level rules are pinned in `test_translate_batch.py`
(`OutagePolicy`) and `test_audience.py` (`EmptyLineGuard`). This file pins what
happens ABOVE them:

  * `TranslatorUnavailable` ends a stream / job `failed` with the plain message
    as `error` (the extension prints it under the button), where any other
    exception keeps the traceback form.
  * The cache never holds a voiceless section, and a cached section records
    `lines` / `voiced_lines` so one can be recognised from the manifest.
  * A skipped section is counted (`sections_failed`) and exposed.
"""
from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from patmir.pipeline.context import TranslatorUnavailable
from patmir.pipeline.streaming.section import SectionResult, line_counts
from patmir.pipeline.streaming.timing import section_timing
from patmir.tests.test_accelerator_lock import _job_record, _stream_record
from patmir.tests.test_cancellation import _managers
from patmir.ui import cache as cache_mod
from patmir.ui.stream import StreamRecord, _is_voiceless

MSG = "OpenRouter: HTTP 402 from m — credits exhausted"


def _result(job_dir: Path, *, lines, voiced, dub: Path | None) -> SectionResult:
    return SectionResult(index=3, start=100.0, end=136.5, dub_wav=dub,
                         job_dir=job_dir, wall_s=1.0, lines=lines,
                         voiced_lines=voiced)


class ExceptionShape(unittest.TestCase):
    def test_is_a_runtime_error_with_message_and_status(self):
        exc = TranslatorUnavailable(MSG, status=402)
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(exc.message, MSG)
        self.assertEqual(exc.status, 402)
        self.assertEqual(str(exc), MSG)

    def test_status_is_optional(self):
        # The adapter's all-empty rule has no HTTP status to give.
        self.assertIsNone(TranslatorUnavailable("no text for any of 7 lines").status)


class VoicelessRule(unittest.TestCase):
    def test_two_or_more_lines_and_none_voiced_is_voiceless(self):
        self.assertTrue(_is_voiceless(_result(Path("."), lines=7, voiced=0, dub=None)))
        self.assertTrue(_is_voiceless(_result(Path("."), lines=2, voiced=0, dub=None)))

    def test_one_empty_line_is_a_reported_gap_not_an_outage(self):
        self.assertFalse(_is_voiceless(_result(Path("."), lines=1, voiced=0, dub=None)))

    def test_no_lines_is_music_not_voiceless(self):
        self.assertFalse(_is_voiceless(_result(Path("."), lines=0, voiced=0, dub=None)))

    def test_any_voiced_line_passes(self):
        self.assertFalse(_is_voiceless(_result(Path("."), lines=7, voiced=1, dub=None)))

    def test_unknown_counts_are_trusted(self):
        # The stage would have raised on an outage it could see; a missing
        # count must not turn telemetry into a gate.
        self.assertFalse(_is_voiceless(_result(Path("."), lines=None, voiced=None, dub=None)))


class LineCounts(unittest.TestCase):
    def test_reads_translated_json(self):
        with TemporaryDirectory() as td:
            (Path(td) / "translated.json").write_text(json.dumps({"sentences": [
                {"idx": 0, "target_text": "Ողջույն"},
                {"idx": 1, "target_text": ""},
                {"idx": 2, "target_text": "   "},
            ]}))
            self.assertEqual(line_counts(Path(td)), (3, 1))

    def test_the_incident_artifact_shape(self):
        # `runs/20260914-210424-section/translated.json`: 7 lines, all "".
        with TemporaryDirectory() as td:
            (Path(td) / "translated.json").write_text(json.dumps({"sentences": [
                {"idx": i, "target_text": ""} for i in range(7)]}))
            self.assertEqual(line_counts(Path(td)), (7, 0))

    def test_missing_or_unreadable_is_none_not_an_error(self):
        with TemporaryDirectory() as td:
            self.assertEqual(line_counts(Path(td)), (None, None))
            (Path(td) / "translated.json").write_text("{not json")
            self.assertEqual(line_counts(Path(td)), (None, None))


class ManifestFields(unittest.TestCase):
    def test_line_counts_are_written_additively(self):
        with TemporaryDirectory() as td:
            rec = section_timing(index=1, start=40.0, end=76.5, job_dir=Path(td),
                                 wall_s=20.0, lines=9, voiced_lines=9)
            self.assertEqual((rec["lines"], rec["voiced_lines"]), (9, 9))
            bare = section_timing(index=1, start=40.0, end=76.5, job_dir=Path(td),
                                  wall_s=20.0)
            self.assertNotIn("lines", bare)
            self.assertNotIn("voiced_lines", bare)

    def test_public_record_carries_sections_failed(self):
        rec = _stream_record()
        rec.sections_failed = 2
        self.assertEqual(rec.to_public()["sections_failed"], 2)


class EmitRefusesVoiceless(unittest.TestCase):
    """Last line of defence: with the candidates and the adapter raising this
    should be unreachable, but the cache is the free-rewatch store and a
    voiceless section written once plays wrong on every viewing."""

    def _emit(self, td: str, *, lines, voiced):
        _, streamer = _managers(td)
        rec = _stream_record()
        rec.duration = 600.0
        streamer._records[rec.job_id] = rec
        streamer._by_cache_key[rec.cache_key] = rec.job_id
        job_dir = Path(td) / "job"
        job_dir.mkdir()
        dub = job_dir / "section_dub.wav"
        dub.write_bytes(b"RIFF....WAVEfmt ")           # only copied, never parsed
        cache_mod.stream_cache_dir(streamer._cache_root, rec.cache_key).mkdir(
            parents=True, exist_ok=True)
        streamer._emit(rec, _result(job_dir, lines=lines, voiced=voiced, dub=dub))
        manifest = cache_mod.read_stream_manifest(streamer._cache_root, rec.cache_key)
        return rec, manifest

    def test_voiceless_section_is_not_cached_and_is_counted(self):
        with TemporaryDirectory() as td:
            rec, manifest = self._emit(td, lines=7, voiced=0)
            self.assertEqual(rec.sections_done, 0)
            self.assertEqual(rec.dubbed_s, 0.0)
            self.assertEqual(rec.sections_failed, 1)
            self.assertEqual((manifest or {}).get("sections", []), [])

    def test_voiced_section_is_cached_with_its_counts(self):
        with TemporaryDirectory() as td:
            rec, manifest = self._emit(td, lines=7, voiced=6)
            self.assertEqual(rec.sections_done, 1)
            self.assertAlmostEqual(rec.dubbed_s, 36.5)
            self.assertEqual(rec.sections_failed, 0)
            entry = manifest["sections"][0]
            self.assertEqual((entry["lines"], entry["voiced_lines"]), (7, 6))
            self.assertEqual(entry["index"], 3)


class StreamEndsFailedWithTheReason(unittest.IsolatedAsyncioTestCase):
    async def _run(self, exc: BaseException) -> StreamRecord:
        with TemporaryDirectory() as td:
            _, streamer = _managers(td)
            rec = _stream_record()
            streamer._records[rec.job_id] = rec

            def body(rec_, force, cancel=None):
                raise exc

            streamer._run_blocking = body  # type: ignore[method-assign]
            await streamer._run(rec, False)
        return rec

    async def test_unavailable_translator_fails_the_stream_with_a_plain_message(self):
        rec = await self._run(TranslatorUnavailable(MSG, status=402))
        self.assertEqual(rec.status, "failed")
        self.assertEqual(rec.error, MSG)          # no traceback, no type name
        self.assertIsNotNone(rec.finished_at)

    async def test_other_exceptions_keep_the_traceback_form(self):
        rec = await self._run(ValueError("boom"))
        self.assertEqual(rec.status, "failed")
        self.assertTrue(rec.error.startswith("ValueError: boom"))
        self.assertIn("Traceback", rec.error)

    async def test_the_manifest_overlay_carries_status_and_error(self):
        # This is what the content script's sections poll reads.
        with TemporaryDirectory() as td:
            _, streamer = _managers(td)
            rec = _stream_record()
            streamer._records[rec.job_id] = rec
            streamer._by_cache_key[rec.cache_key] = rec.job_id
            rec.status, rec.error = "failed", MSG
            m = streamer.sections_manifest(rec.cache_key)
        self.assertEqual(m["status"], "failed")
        self.assertEqual(m["error"], MSG)


class JobEndsFailedWithTheReason(unittest.IsolatedAsyncioTestCase):
    async def _run(self, exc: BaseException):
        with TemporaryDirectory() as td:
            jobs, _ = _managers(td)
            rec = _job_record()
            jobs._records[rec.job_id] = rec

            def body(rec_, cancel=None):
                raise exc

            jobs._run_one = body  # type: ignore[method-assign]
            await jobs.start()
            await jobs._queue.put(rec.job_id)
            deadline = time.monotonic() + 5.0
            while rec.status not in ("done", "failed"):
                if time.monotonic() > deadline:
                    self.fail(f"never settled (status={rec.status})")
                await asyncio.sleep(0.01)
            await jobs.stop()
        return rec

    async def test_quality_mode_reports_the_same_plain_message(self):
        rec = await self._run(TranslatorUnavailable(MSG, status=402))
        self.assertEqual(rec.status, "failed")
        self.assertEqual(rec.error, MSG)

    async def test_other_exceptions_keep_the_traceback_form(self):
        rec = await self._run(ValueError("boom"))
        self.assertEqual(rec.status, "failed")
        self.assertIn("Traceback", rec.error)


if __name__ == "__main__":
    unittest.main()
