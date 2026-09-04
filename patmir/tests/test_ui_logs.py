"""The server keeps a cumulative, self-cleaning log.

Until 2026-09-04 it logged to stderr only, so `log.exception` from a failing
job, the prewarm summary and uvicorn's request lines all lived in whichever
terminal started the server and died with it. When the WSL2 GPU bridge took
the process down mid-job, the only surviving traceback was in a `/tmp`
redirect that `wsl --shutdown` then wiped.

Rotation rather than a file per run: a crash loop reads as one story, and the
ceiling is structural — no cleanup pass to forget to run.
"""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from patmir.ui.logs import BACKUPS, MAX_BYTES, log_path, setup_file_logging


class SetupTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.root = logging.getLogger()
        self._saved = list(self.root.handlers)
        self._level = self.root.level

    def tearDown(self):
        for h in list(self.root.handlers):
            if h not in self._saved:
                self.root.removeHandler(h)
                h.close()
        self.root.setLevel(self._level)

    def test_the_log_lives_under_state_root(self):
        self.assertEqual(log_path(self.dir), self.dir / "logs" / "server.log")

    def test_the_directory_is_created(self):
        p = setup_file_logging(self.dir)
        self.assertTrue(p.parent.is_dir())

    def test_a_record_reaches_the_file(self):
        p = setup_file_logging(self.dir)
        logging.getLogger("patmir.test").warning("a line")
        self.assertIn("a line", p.read_text())

    def test_a_record_appears_exactly_once(self):
        """The bug this file exists to prevent.

        Attaching the same handler to `uvicorn`, `uvicorn.error` and root
        wrote every request line three times — a record is handled once per
        level while it propagates. Handlers go on root and nowhere else.
        """
        p = setup_file_logging(self.dir)
        logging.getLogger("uvicorn.access").info("GET / 200")
        self.assertEqual(p.read_text().count("GET / 200"), 1)

    def test_uvicorn_records_are_captured(self):
        # They only reach root because the server starts uvicorn with
        # `log_config=None`; uvicorn would otherwise set propagate = False.
        p = setup_file_logging(self.dir)
        logging.getLogger("uvicorn.error").info("Application startup complete.")
        self.assertIn("Application startup complete.", p.read_text())

    def test_setup_is_idempotent(self):
        setup_file_logging(self.dir)
        before = len(self.root.handlers)
        setup_file_logging(self.dir)
        setup_file_logging(self.dir)
        self.assertEqual(len(self.root.handlers), before)

    def test_the_console_keeps_working(self):
        # Python's last-resort stderr handler stops firing once root has a
        # handler, so a console handler is added explicitly. Without it the
        # terminal would go silent the moment file logging was switched on.
        setup_file_logging(self.dir)
        streams = [h for h in self.root.handlers
                   if isinstance(h, logging.StreamHandler)
                   and not isinstance(h, logging.FileHandler)]
        self.assertTrue(streams, "a console handler must remain")

    def test_a_lower_level_is_not_raised(self):
        self.root.setLevel(logging.DEBUG)
        setup_file_logging(self.dir, level=logging.INFO)
        self.assertEqual(self.root.level, logging.DEBUG)

    def test_a_higher_level_is_lowered(self):
        self.root.setLevel(logging.CRITICAL)
        setup_file_logging(self.dir, level=logging.INFO)
        self.assertEqual(self.root.level, logging.INFO)

    def test_it_is_bounded_and_self_cleaning(self):
        """Cumulative but with a hard ceiling, so nothing has to clean up."""
        p = setup_file_logging(self.dir)
        handler = [h for h in self.root.handlers
                   if isinstance(h, logging.handlers.RotatingFileHandler)][0]
        self.assertEqual(handler.maxBytes, MAX_BYTES)
        self.assertEqual(handler.backupCount, BACKUPS)
        self.assertLessEqual(MAX_BYTES * (BACKUPS + 1), 64 * 1024 * 1024)
        self.assertEqual(Path(handler.baseFilename), p)

    def test_rotation_actually_drops_the_oldest(self):
        import logging.handlers
        p = setup_file_logging(self.dir)
        handler = [h for h in self.root.handlers
                   if isinstance(h, logging.handlers.RotatingFileHandler)][0]
        handler.maxBytes = 512          # force rotation without writing 8 MiB
        for i in range(400):
            logging.getLogger("patmir.test").info("line %d padding padding", i)
        rotated = sorted(pp.name for pp in p.parent.iterdir())
        self.assertIn("server.log", rotated)
        self.assertLessEqual(len(rotated), BACKUPS + 1,
                             f"unbounded growth: {rotated}")


if __name__ == "__main__":
    unittest.main()
