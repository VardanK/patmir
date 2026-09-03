"""One `patmir ui` per machine.

Two servers do not fit: each prewarms the whole stage set at boot (~16 GB on
the CUDA profile) and a 24 GB card holds one. The `asyncio.Lock` from
`decisions/2026-08-25-one-accelerator-at-a-time.md` serializes a batch job
against a stream inside ONE process and cannot see another process at all.

Observed 2026-09-03: a second `patmir ui` started on a different port starved
the running one, and the symptom arrived as "the extension won't connect" —
nothing anywhere mentioned memory. Hence the guard, and hence the error
message naming the holder.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from patmir.ui.single_instance import ServerAlreadyRunning, acquire


class AcquireTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.lock = self.dir / "ui.lock"
        self._open = []

    def tearDown(self):
        for h in self._open:
            try:
                h.close()
            except Exception:
                pass

    def _acquire(self, host="127.0.0.1", port=8765):
        h = acquire(self.lock, host=host, port=port)
        self._open.append(h)
        return h

    def test_the_first_server_gets_the_lock_and_records_itself(self):
        self._acquire()
        rec = json.loads(self.lock.read_text())
        self.assertEqual(rec["pid"], os.getpid())
        self.assertEqual(rec["bind"], "127.0.0.1:8765")

    def test_a_second_server_is_refused(self):
        self._acquire()
        with self.assertRaises(ServerAlreadyRunning):
            self._acquire(host="0.0.0.0", port=8000)

    def test_the_refusal_names_the_holder_not_the_newcomer(self):
        """The message has to be actionable: which process, on which port."""
        self._acquire(host="127.0.0.1", port=8765)
        with self.assertRaises(ServerAlreadyRunning) as cm:
            self._acquire(host="0.0.0.0", port=8000)
        msg = str(cm.exception)
        self.assertIn(str(os.getpid()), msg)
        self.assertIn("127.0.0.1:8765", msg)
        self.assertIn(str(self.lock), msg)
        # ...and it says WHY, because "already running" alone sends the reader
        # looking for a port clash, which is not the problem.
        self.assertIn("prewarms", msg)

    def test_releasing_lets_the_next_server_start(self):
        h = self._acquire()
        h.close()
        self._acquire(port=9000)
        self.assertEqual(json.loads(self.lock.read_text())["bind"],
                         "127.0.0.1:9000")

    def test_the_directory_is_created_if_absent(self):
        # First run on a fresh box: `state_root` may not exist yet.
        deep = self.dir / "nested" / "state" / "ui.lock"
        h = acquire(deep, host="127.0.0.1", port=8765)
        self._open.append(h)
        self.assertTrue(deep.exists())

    def test_a_corrupt_lock_file_still_refuses_clearly(self):
        """A hand-edited or half-written file must not turn a clear refusal
        into a traceback — the guard exists to produce a good message."""
        self._acquire()
        self.lock.write_text("not json at all")
        with self.assertRaises(ServerAlreadyRunning) as cm:
            self._acquire(port=8000)
        self.assertIn("already running", str(cm.exception))

    def test_a_stale_file_from_a_dead_process_is_reclaimable(self):
        """No stale-lock cleanup needed, which is why this is flock and not a
        pidfile: the kernel drops the lock when the holder dies."""
        self.lock.write_text(json.dumps({"pid": 999999, "bind": "127.0.0.1:8765"}))
        h = acquire(self.lock, host="127.0.0.1", port=8765)
        self._open.append(h)
        self.assertEqual(json.loads(self.lock.read_text())["pid"], os.getpid())


class StoragePathTest(unittest.TestCase):
    def test_server_lock_lives_under_state_root(self):
        from patmir.paths import StoragePaths
        d = Path("/tmp/x")
        st = StoragePaths(runs_root=d, cache_root=d, downloads_root=d,
                          state_root=d / "state", library_root=None)
        self.assertEqual(st.server_lock, d / "state" / "ui.lock")


if __name__ == "__main__":
    unittest.main()
