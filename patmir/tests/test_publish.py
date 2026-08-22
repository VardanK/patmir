"""Tests for `patmir.pipeline.publish` — the library_root export.

Properties that matter:

* The deliverables land under `library_root/<job_id>/`.
* The job dir is untouched (copy, not move) — `--resume-from` and
  `cache.materialize` both still need it.
* No `.partial` file survives a successful publish, and none survives a failed
  one either (a half-copied `out.mkv` on the share would look valid).
* Publishing is idempotent — a rerun overwrites in place.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from patmir.pipeline.publish import DELIVERABLES, publish_job


def _make_job(job_dir: Path, names=DELIVERABLES) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (job_dir / name).write_text(f"contents of {name}")
    # Intermediates that must NOT be published.
    (job_dir / "audio.wav").write_text("intermediate")
    (job_dir / "manifest.json").write_text("{}")


class PublishJobTest(unittest.TestCase):
    def test_copies_deliverables_into_per_job_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "runs" / "20260801-120000-clip"
            _make_job(job_dir)

            dest = publish_job(job_dir, "20260801-120000-clip", root / "library")

            self.assertEqual(dest, root / "library" / "20260801-120000-clip")
            for name in DELIVERABLES:
                self.assertEqual((dest / name).read_text(), f"contents of {name}")
            # Intermediates stay behind.
            self.assertFalse((dest / "audio.wav").exists())
            self.assertFalse((dest / "manifest.json").exists())

    def test_job_dir_is_left_intact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "runs" / "job"
            _make_job(job_dir)
            before = sorted(p.name for p in job_dir.iterdir())

            publish_job(job_dir, "job", root / "library")

            self.assertEqual(sorted(p.name for p in job_dir.iterdir()), before)

    def test_no_partial_files_left_behind(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "runs" / "job"
            _make_job(job_dir)

            dest = publish_job(job_dir, "job", root / "library")

            self.assertEqual([p for p in dest.iterdir() if p.suffix == ".partial"], [])

    def test_missing_deliverables_are_skipped(self):
        """An audio-only or subtitle-less run still publishes what it has."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "runs" / "job"
            _make_job(job_dir, names=("dub.wav",))

            dest = publish_job(job_dir, "job", root / "library")

            self.assertTrue((dest / "dub.wav").is_file())
            self.assertFalse((dest / "out.mkv").exists())

    def test_raises_when_nothing_to_publish(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "runs" / "job"
            job_dir.mkdir(parents=True)
            (job_dir / "audio.wav").write_text("intermediate only")

            with self.assertRaises(FileNotFoundError):
                publish_job(job_dir, "job", root / "library")

    def test_rerun_overwrites_in_place(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "runs" / "job"
            _make_job(job_dir)
            publish_job(job_dir, "job", root / "library")

            (job_dir / "dub.wav").write_text("regenerated")
            dest = publish_job(job_dir, "job", root / "library")

            self.assertEqual((dest / "dub.wav").read_text(), "regenerated")
            self.assertEqual([p for p in dest.iterdir() if p.suffix == ".partial"], [])

    def test_failed_copy_leaves_no_partial(self):
        """An interrupted transfer must not leave a truncated file behind."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_dir = root / "runs" / "job"
            _make_job(job_dir)

            with mock.patch("shutil.copyfile", side_effect=OSError("share went away")):
                with self.assertRaises(OSError):
                    publish_job(job_dir, "job", root / "library")

            dest = root / "library" / "job"
            self.assertEqual(list(dest.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
