"""Cooperative job cancellation.

Cancellation cannot be preemptive here. The heavy stages run in a worker thread
via `asyncio.to_thread`, and a Python thread cannot be killed — `task.cancel()`
only abandons the await while the thread keeps running, holding the GPU *and* the
shared accelerator lock (ADR `2026-08-25-one-accelerator-at-a-time`), so the next
job would queue behind a job the user believes is dead. Hence a token checked at
explicit checkpoints.

The two cases are genuinely different and tested separately:

  * **not started** (`queued` / `waiting`) — exact and immediate; the body must
    never run at all. The accelerator lock made `waiting` common.
  * **running** — cooperative; `cancelling` until the thread returns, and the
    lock must be released at that point so the next job can start.

`test_cancelled_run_releases_the_lock_for_the_next_job` is the one that matters
most: it is the property that a naive `task.cancel()` implementation would
silently violate.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from patmir.pipeline.context import JobCancelled, JobContext, PipelineOptions
from patmir.pipeline.config import PipelineConfig
from patmir.tests.test_accelerator_lock import _job_record, _storage, _stream_record
from patmir.ui.jobs import JobManager
from patmir.ui.stream import StreamManager


def _managers(td: str, *, lock: asyncio.Lock | None = None):
    storage = _storage(Path(td))
    cfg = PipelineConfig(entries={}, options=PipelineOptions(), storage=storage)
    jobs = JobManager(cfg, storage=storage, accelerator_lock=lock)
    streamer = StreamManager(Path("patmir.toml"), jobs=jobs, storage=storage,
                             accelerator_lock=lock)
    return jobs, streamer


class ContextTokenTest(unittest.TestCase):
    """The primitive every checkpoint calls."""

    def _ctx(self, ev):
        return JobContext(job_dir=Path("/tmp"), source_path=Path("/tmp/x.mkv"),
                          options=PipelineOptions(), cancel=ev)

    def test_raises_only_once_set(self):
        ev = threading.Event()
        ctx = self._ctx(ev)
        self.assertFalse(ctx.cancelled())
        ctx.raise_if_cancelled()          # must not raise
        ev.set()
        self.assertTrue(ctx.cancelled())
        with self.assertRaises(JobCancelled):
            ctx.raise_if_cancelled()

    def test_absent_token_never_cancels(self):
        """Every non-server caller (CLI `patmir dub`, bench, tests) passes None."""
        ctx = self._ctx(None)
        self.assertFalse(ctx.cancelled())
        ctx.raise_if_cancelled()


class CancelBeforeStartTest(unittest.IsolatedAsyncioTestCase):
    """Not-yet-started jobs cancel exactly, and must never reach the GPU."""

    async def test_queued_job_never_runs(self):
        ran: list[str] = []
        with TemporaryDirectory() as td:
            jobs, _ = _managers(td)
            rec = _job_record()
            jobs._records[rec.job_id] = rec
            jobs._run_one = lambda r, cancel=None: ran.append("ran")  # type: ignore[method-assign]

            self.assertTrue(jobs.cancel(rec.job_id))
            self.assertEqual(rec.status, "cancelled")
            await jobs.start()
            await jobs._queue.put(rec.job_id)
            await asyncio.sleep(0.2)          # give the worker a chance to run it
            await jobs.stop()

        self.assertEqual(ran, [], "a cancelled job must not execute")
        self.assertEqual(rec.status, "cancelled")

    async def test_waiting_job_cancels_without_taking_the_lock(self):
        """A job blocked on the accelerator lock cancels immediately, and stays
        cancelled once the lock frees — it must not sneak through."""
        ran: list[str] = []
        with TemporaryDirectory() as td:
            lock = asyncio.Lock()
            jobs, _ = _managers(td, lock=lock)
            rec = _job_record()
            jobs._records[rec.job_id] = rec
            jobs._run_one = lambda r, cancel=None: ran.append("ran")  # type: ignore[method-assign]

            await lock.acquire()              # stand in for a running stream
            await jobs.start()
            await jobs._queue.put(rec.job_id)
            deadline = time.monotonic() + 5.0
            while rec.status != "waiting":
                if time.monotonic() > deadline:
                    self.fail(f"never reached waiting (status={rec.status})")
                await asyncio.sleep(0.01)

            self.assertTrue(jobs.cancel(rec.job_id))
            self.assertEqual(rec.status, "cancelled")
            lock.release()
            await asyncio.sleep(0.2)
            await jobs.stop()

        self.assertEqual(ran, [], "cancelled-while-waiting must not execute")
        self.assertEqual(rec.status, "cancelled")


class CancelRunningTest(unittest.IsolatedAsyncioTestCase):
    """Running jobs stop cooperatively, and report it as a stop, not a failure."""

    async def test_running_job_observes_token_and_reports_cancelled(self):
        started = threading.Event()

        def body(rec, cancel=None):
            started.set()
            for _ in range(500):              # ~5s ceiling
                if cancel is not None and cancel.is_set():
                    raise JobCancelled()
                time.sleep(0.01)
            raise AssertionError("token was never observed")

        with TemporaryDirectory() as td:
            jobs, _ = _managers(td)
            rec = _job_record()
            jobs._records[rec.job_id] = rec
            jobs._run_one = body  # type: ignore[method-assign]
            await jobs.start()
            await jobs._queue.put(rec.job_id)

            await asyncio.get_running_loop().run_in_executor(None, started.wait, 5.0)
            self.assertTrue(jobs.cancel(rec.job_id))
            self.assertEqual(rec.status, "cancelling")

            deadline = time.monotonic() + 5.0
            while rec.status == "cancelling":
                if time.monotonic() > deadline:
                    self.fail("never settled after cancel")
                await asyncio.sleep(0.01)
            await jobs.stop()

        self.assertEqual(rec.status, "cancelled")
        # A deliberate stop is not an error: no traceback recorded.
        self.assertIsNone(rec.error)

    async def test_cancelled_run_releases_the_lock_for_the_next_job(self):
        """The property a `task.cancel()` implementation would break.

        If a cancelled job's thread kept running, it would hold the accelerator
        lock and the next job would wait behind a job the user thinks is dead.
        """
        order: list[str] = []
        first_started = threading.Event()

        def body(rec, cancel=None):
            if rec.job_id == "job-first":
                first_started.set()
                for _ in range(500):
                    if cancel is not None and cancel.is_set():
                        order.append("first:cancelled")
                        raise JobCancelled()
                    time.sleep(0.01)
                raise AssertionError("token never observed")
            order.append("second:ran")

        with TemporaryDirectory() as td:
            lock = asyncio.Lock()
            jobs, _ = _managers(td, lock=lock)
            first, second = _job_record(), _job_record()
            first.job_id, second.job_id = "job-first", "job-second"
            for r in (first, second):
                jobs._records[r.job_id] = r
            jobs._run_one = body  # type: ignore[method-assign]

            await jobs.start()
            await jobs._queue.put(first.job_id)
            await asyncio.get_running_loop().run_in_executor(None, first_started.wait, 5.0)
            await jobs._queue.put(second.job_id)

            jobs.cancel(first.job_id)
            deadline = time.monotonic() + 5.0
            while second.status != "done":
                if time.monotonic() > deadline:
                    self.fail(f"second job blocked behind a cancelled one "
                              f"(first={first.status}, second={second.status})")
                await asyncio.sleep(0.01)
            await jobs.stop()

        self.assertEqual(order, ["first:cancelled", "second:ran"])
        self.assertEqual(first.status, "cancelled")
        self.assertFalse(lock.locked(), "lock leaked after a cancelled run")


class CancelIdempotencyTest(unittest.IsolatedAsyncioTestCase):

    async def test_unknown_job_is_false(self):
        with TemporaryDirectory() as td:
            jobs, streamer = _managers(td)
            self.assertFalse(jobs.cancel("nope"))
            self.assertFalse(streamer.cancel("nope"))

    async def test_terminal_states_are_noops(self):
        with TemporaryDirectory() as td:
            jobs, _ = _managers(td)
            for state in ("done", "failed", "cancelled"):
                rec = _job_record()
                rec.job_id, rec.status = f"job-{state}", state
                jobs._records[rec.job_id] = rec
                self.assertTrue(jobs.cancel(rec.job_id),
                                "terminal cancel should report success")
                self.assertEqual(rec.status, state, "must not overwrite a terminal state")

    async def test_double_cancel_of_a_queued_job(self):
        with TemporaryDirectory() as td:
            jobs, _ = _managers(td)
            rec = _job_record()
            jobs._records[rec.job_id] = rec
            self.assertTrue(jobs.cancel(rec.job_id))
            self.assertTrue(jobs.cancel(rec.job_id))
            self.assertEqual(rec.status, "cancelled")


class StopStreamTest(unittest.IsolatedAsyncioTestCase):
    """`POST /stream/<key>/stop` (what the extension calls) must now also trip
    the token, so the in-flight section aborts instead of running to completion
    while the scheduler sits stopped."""

    async def test_stop_stream_sets_the_cancel_token(self):
        with TemporaryDirectory() as td:
            _, streamer = _managers(td)
            rec = _stream_record()
            rec.status = "streaming"
            streamer._records[rec.job_id] = rec
            streamer._by_cache_key[rec.cache_key] = rec.job_id
            ev = threading.Event()
            streamer._cancels[rec.job_id] = ev

            self.assertTrue(streamer.stop_stream(rec.cache_key))
            self.assertTrue(ev.is_set(), "in-flight section was not asked to stop")
            self.assertEqual(rec.status, "cancelling")

    async def test_stop_stream_unknown_key_is_false(self):
        with TemporaryDirectory() as td:
            _, streamer = _managers(td)
            self.assertFalse(streamer.stop_stream("no-such-key"))


if __name__ == "__main__":
    unittest.main()
