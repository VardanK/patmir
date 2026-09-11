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


class WaitingStreamCancelTest(unittest.IsolatedAsyncioTestCase):
    """Quality issue #33 (2026-09-04). A stream parked on the accelerator lock
    is `waiting`, and `StreamManager.cancel` only knew `queued` as the
    not-started status — so the Stop fell into the running branch, found no
    token (none exists before `_run_locked`), and left the record
    `cancelling`. `_run` skips only `cancelled`, so the "cancelled" stream ran
    in full when the lock freed. The batch side had `waiting` from day one."""

    async def test_cancelling_a_waiting_stream_is_instant_and_the_body_never_runs(self):
        ran: list[str] = []
        with TemporaryDirectory() as td:
            lock = asyncio.Lock()
            _, streamer = _managers(td, lock=lock)

            async def fake_locked(rec, force):
                ran.append(rec.job_id)
            streamer._run_locked = fake_locked  # type: ignore[method-assign]

            rec = _stream_record()
            streamer._records[rec.job_id] = rec
            streamer._by_cache_key[rec.cache_key] = rec.job_id

            await lock.acquire()                     # a stream is on the GPU
            task = asyncio.create_task(streamer._run(rec, False))
            deadline = time.monotonic() + 5.0
            while rec.status != "waiting":
                if time.monotonic() > deadline:
                    self.fail(f"never reported waiting (status={rec.status})")
                await asyncio.sleep(0.01)

            self.assertTrue(streamer.cancel(rec.job_id))
            self.assertEqual(rec.status, "cancelled")
            self.assertIsNotNone(rec.finished_at)

            lock.release()
            await asyncio.wait_for(task, 5.0)
        self.assertEqual(ran, [], "a cancelled waiting stream must never start")

    async def test_stop_stream_reaches_every_live_record_for_the_key(self):
        """`_by_cache_key` points at the newest submission only. After a Stop
        + re-submit the record actually holding the accelerator was no longer
        reachable from `/stream/<key>/stop`; on 2026-09-04 three records for
        one video stacked up (running-but-stuck, cancelling-never-started,
        waiting) and only the last could be stopped."""
        with TemporaryDirectory() as td:
            _, streamer = _managers(td)
            older = _stream_record()
            older.job_id = "stream-older"
            older.status = "prelude"
            ev = threading.Event()
            streamer._cancels[older.job_id] = ev
            newer = _stream_record()
            newer.job_id = "stream-newer"
            newer.status = "waiting"
            finished = _stream_record()
            finished.job_id = "stream-finished"
            finished.status = "done"
            for r in (older, newer, finished):
                streamer._records[r.job_id] = r
            streamer._by_cache_key[newer.cache_key] = newer.job_id

            self.assertTrue(streamer.stop_stream(newer.cache_key))
            self.assertTrue(ev.is_set(), "the record holding the accelerator was not told to stop")
            self.assertEqual(older.status, "cancelling")
            self.assertEqual(newer.status, "cancelled")
            self.assertEqual(finished.status, "done")


class _FakeYtDlp:
    """Stand-in for the `yt_dlp` module: records the options `_download_url`
    passes and lets a test script what `extract_info` does."""

    class utils:  # noqa: N801 — mirrors the real module layout
        class DownloadCancelled(Exception):
            def __init__(self, msg=None):
                super().__init__(msg)
                self.msg = msg

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.constructed: list[dict] = []
        outer = self

        class YoutubeDL:
            def __init__(self, opts):
                outer.constructed.append(opts)
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=True):
                return outer.behaviour(self.opts, url)

            def prepare_filename(self, info):
                return info["_filename"]

        self.YoutubeDL = YoutubeDL


class DownloadCancelTest(unittest.TestCase):
    """The yt-dlp download runs INSIDE the accelerator lock and used to be the
    one place the token was never consulted. On 2026-09-04 a stream sat in
    `extract_info` for 6+ minutes on a `watch?v=..&list=RD..` (YouTube Mix)
    URL — yt-dlp walking the whole mix, one entry after another — with the GPU
    idle and every later submission `waiting` behind it."""

    def _manager(self, td: str) -> JobManager:
        jobs, _ = _managers(td)
        return jobs

    def _install(self, fake):
        import sys
        saved = {k: sys.modules.get(k) for k in ("yt_dlp", "yt_dlp.utils")}
        sys.modules["yt_dlp"] = fake
        sys.modules["yt_dlp.utils"] = fake.utils
        def restore():
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.addCleanup(restore)

    def test_single_video_options_and_cancel_hooks_are_wired(self):
        with TemporaryDirectory() as td:
            jobs = self._manager(td)
            def ok(opts, url):
                out = Path(opts["outtmpl"] % {"ext": "m4a"})
                out.write_bytes(b"x")
                return {"_type": "video", "_filename": str(out)}
            fake = _FakeYtDlp(ok)
            self._install(fake)
            path = jobs._download_url("https://www.youtube.com/watch?v=abc&list=RDabc",
                                      "k", None, cancel=threading.Event())
            self.assertTrue(path.is_file())
            opts = fake.constructed[0]
            self.assertIs(opts.get("noplaylist"), True,
                          "a watch URL with list= must download ONE video")
            self.assertTrue(callable(opts.get("match_filter")))
            self.assertTrue(opts.get("progress_hooks"))

    def test_token_set_before_the_download_aborts_without_touching_yt_dlp(self):
        with TemporaryDirectory() as td:
            jobs = self._manager(td)
            fake = _FakeYtDlp(lambda opts, url: self.fail("must not extract"))
            self._install(fake)
            ev = threading.Event(); ev.set()
            with self.assertRaises(JobCancelled):
                jobs._download_url("https://youtu.be/abc", "k", None, cancel=ev)
            self.assertEqual(fake.constructed, [])

    def test_token_set_mid_download_surfaces_as_job_cancelled(self):
        """yt-dlp's abort path: a hook raises `DownloadCancelled`, which
        `extract_info` re-raises verbatim. The boundary must translate it so
        the server records `cancelled`, not `failed`."""
        with TemporaryDirectory() as td:
            jobs = self._manager(td)
            ev = threading.Event()
            def mid_download(opts, url):
                # The per-entry filter passes while the token is clear...
                self.assertIsNone(opts["match_filter"]({"id": "abc"}, incomplete=False))
                ev.set()                      # ...then Stop lands mid-transfer
                for hook in opts["progress_hooks"]:
                    hook({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2})
                self.fail("hook did not abort the download")
            fake = _FakeYtDlp(mid_download)
            self._install(fake)
            with self.assertRaises(JobCancelled):
                jobs._download_url("https://youtu.be/abc", "k", None, cancel=ev)

    def test_playlist_result_fails_fast_instead_of_source_na(self):
        """`noplaylist` cannot help a bare playlist / channel / home-page URL —
        yt-dlp still returns a playlist, `prepare_filename` on it gives
        `source.NA`, and before this the error came only after N entries were
        fetched (observed: `https://www.youtube.com/` submitted by the
        extension)."""
        with TemporaryDirectory() as td:
            jobs = self._manager(td)
            fake = _FakeYtDlp(lambda opts, url: {"_type": "playlist", "entries": [],
                                                 "_filename": "source.NA"})
            self._install(fake)
            with self.assertRaisesRegex(RuntimeError, "playlist"):
                jobs._download_url("https://www.youtube.com/", "k", None)

    def test_no_token_downloads_as_before(self):
        with TemporaryDirectory() as td:
            jobs = self._manager(td)
            def ok(opts, url):
                out = Path(opts["outtmpl"] % {"ext": "webm"})
                out.write_bytes(b"x")
                return {"_type": "video", "_filename": str(out)}
            self._install(_FakeYtDlp(ok))
            self.assertTrue(jobs._download_url("https://youtu.be/abc", "k").is_file())


if __name__ == "__main__":
    unittest.main()
