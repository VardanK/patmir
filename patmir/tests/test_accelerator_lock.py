"""The shared accelerator lock: a batch job and a watch-along stream must never
run inference at the same time.

Regression test for the 2026-08-25 OOM. `JobManager` and `StreamManager` share
one `stage_cache` but had *separate* locks, so each serialized its own work and
nothing serialized them against each other. Submitting a quality-mode dub and a
watch-along stream ~27s apart ran both pipelines concurrently and exhausted a
24 GB card: resident weights are already ~15-16 GB (tts ~8.5, asr+aligner ~5.5,
diarize 0.9, emotion, and *two* Demucs models because batch selects
`htdemucs_vocals` while watch-along selects `htdemucs_fast`), and two concurrent
passes add a second set of activations. Three OOMs landed in 230 ms, the last on
a *tiny* `.to(device)` — once the card is full, allocation size stops mattering.

`test_without_the_lock_they_do_overlap` is the negative control. Without it a
passing suite would prove nothing: the serialization could come from an
incidental single-threaded executor rather than the lock.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from patmir.paths import StoragePaths
from patmir.pipeline.config import PipelineConfig, PipelineOptions
from patmir.ui.jobs import JobManager, JobRecord
from patmir.ui.stream import StreamManager, StreamRecord

# Long enough that genuinely concurrent bodies reliably overlap, short enough to
# keep the suite fast. The negative control depends on this being >> task
# start-up skew (sub-millisecond).
BODY_SECONDS = 0.25


class _Tracker:
    """Records peak simultaneous occupancy across threads."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.order: list[str] = []

    def body(self, label: str) -> None:
        """Stand-in for a stage pipeline. Runs on a worker thread."""
        with self._mu:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
            self.order.append(f"{label}:enter")
        time.sleep(BODY_SECONDS)
        with self._mu:
            self.inside -= 1
            self.order.append(f"{label}:exit")


def _storage(root: Path) -> StoragePaths:
    for name in ("runs", "cache", "downloads", "state"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return StoragePaths(
        runs_root=root / "runs",
        cache_root=root / "cache",
        downloads_root=root / "downloads",
        state_root=root / "state",
    )


def _job_record() -> JobRecord:
    return JobRecord(
        job_id="job-test", cache_key="k-job", source_id="src-job",
        source_url=None, source_path="/nonexistent.mkv", source_lang="ru",
        status="queued", submitted_at=time.time(),
    )


def _stream_record() -> StreamRecord:
    return StreamRecord(
        job_id="stream-test", cache_key="k-stream", source_id="src-stream",
        source_url=None, source_path="/nonexistent.mkv", source_lang="ru",
        status="queued", submitted_at=time.time(),
    )


class SharedAcceleratorLockTest(unittest.IsolatedAsyncioTestCase):

    async def _run_both(self, *, shared: bool) -> _Tracker:
        """Drive one batch job and one stream concurrently.

        With `shared=True` both managers get the SAME lock (production wiring
        from `app.create_app`); with False they get none, reproducing the bug.
        """
        tracker = _Tracker()
        with TemporaryDirectory() as td:
            storage = _storage(Path(td))
            cfg = PipelineConfig(entries={}, options=PipelineOptions(),
                                 storage=storage)
            lock = asyncio.Lock() if shared else None

            jobs = JobManager(cfg, storage=storage, accelerator_lock=lock)
            streamer = StreamManager(Path("patmir.toml"), jobs=jobs,
                                     storage=storage, accelerator_lock=lock)

            # Replace the pipeline bodies: we are testing the serialization
            # around them, not the stages themselves.
            jobs._run_one = lambda rec, cancel=None: tracker.body("job")   # type: ignore[method-assign]
            streamer._run_blocking = (                                # type: ignore[method-assign]
                lambda rec, force, cancel=None: tracker.body("stream")
            )

            jrec, srec = _job_record(), _stream_record()
            jobs._records[jrec.job_id] = jrec
            await jobs.start()

            # Stream first, then the batch job, mirroring the real incident
            # (the stream was already running when the dub was submitted).
            stream_task = asyncio.create_task(streamer._run(srec, False))
            await asyncio.sleep(0.05)          # let the stream take the lock
            await jobs._queue.put(jrec.job_id)

            await stream_task
            deadline = time.monotonic() + 10.0
            while jrec.status not in ("done", "failed"):
                if time.monotonic() > deadline:
                    self.fail(f"batch job never finished (status={jrec.status})")
                await asyncio.sleep(0.01)
            await jobs.stop()

            self.assertEqual(jrec.status, "done", jrec.error)
            self.assertEqual(srec.status, "done", srec.error)
        return tracker

    async def test_shared_lock_serializes_batch_and_stream(self):
        """The fix: peak occupancy of one, and neither body interleaves."""
        t = await self._run_both(shared=True)
        self.assertEqual(t.peak, 1, f"pipelines overlapped: {t.order}")
        # Every enter is immediately followed by its own exit.
        for i in range(0, len(t.order), 2):
            label = t.order[i].split(":")[0]
            self.assertEqual(t.order[i], f"{label}:enter", t.order)
            self.assertEqual(t.order[i + 1], f"{label}:exit", t.order)

    async def test_without_the_lock_they_do_overlap(self):
        """Negative control — proves the assertion above has teeth.

        If this ever fails, the serialization in the test above is coming from
        somewhere other than the lock and that test has stopped guarding
        anything.
        """
        t = await self._run_both(shared=False)
        self.assertEqual(t.peak, 2, f"expected concurrent bodies: {t.order}")

    async def test_contended_job_reports_waiting(self):
        """A job blocked behind a stream must not look hung."""
        seen: list[str] = []
        with TemporaryDirectory() as td:
            storage = _storage(Path(td))
            cfg = PipelineConfig(entries={}, options=PipelineOptions(),
                                 storage=storage)
            lock = asyncio.Lock()
            jobs = JobManager(cfg, storage=storage, accelerator_lock=lock)
            jrec = _job_record()
            jobs._records[jrec.job_id] = jrec
            jobs._run_one = lambda rec, cancel=None: seen.append(rec.status)  # type: ignore[method-assign]

            await lock.acquire()               # stand in for a running stream
            await jobs.start()
            await jobs._queue.put(jrec.job_id)
            deadline = time.monotonic() + 5.0
            while jrec.status != "waiting":
                if time.monotonic() > deadline:
                    self.fail(f"never reported waiting (status={jrec.status})")
                await asyncio.sleep(0.01)
            lock.release()
            deadline = time.monotonic() + 5.0
            while jrec.status != "done":
                if time.monotonic() > deadline:
                    self.fail(f"never completed (status={jrec.status})")
                await asyncio.sleep(0.01)
            await jobs.stop()

        # It ran only after the lock was free, and by then was "running".
        self.assertEqual(seen, ["running"])

    async def test_absent_lock_keeps_single_status_transition(self):
        """`accelerator_lock=None` (the test/legacy path) must not report
        waiting — the uncontended production path must not either."""
        statuses: list[str] = []
        with TemporaryDirectory() as td:
            storage = _storage(Path(td))
            cfg = PipelineConfig(entries={}, options=PipelineOptions(),
                                 storage=storage)
            jobs = JobManager(cfg, storage=storage)   # no lock
            jrec = _job_record()
            jobs._records[jrec.job_id] = jrec
            jobs._run_one = lambda rec, cancel=None: statuses.append(rec.status)  # type: ignore[method-assign]
            await jobs.start()
            await jobs._queue.put(jrec.job_id)
            deadline = time.monotonic() + 5.0
            while jrec.status != "done":
                if time.monotonic() > deadline:
                    self.fail(f"never completed (status={jrec.status})")
                await asyncio.sleep(0.01)
            await jobs.stop()
        self.assertEqual(statuses, ["running"])


if __name__ == "__main__":
    unittest.main()
