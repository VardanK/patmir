"""Watch-along (D3) stream manager.

The D1 `JobManager` runs the whole batch pipeline and caches one final
`dub.wav`. Watch-along is the additive Mode B (ADR `2026-06-13-near-realtime-
d3-spike`): analyze a first-chunk prelude to establish the cast, start playback
behind a short buffer, then stream the rest in VAD-snapped sections while the
viewer watches — prioritizing the section at the playhead, continuing while
paused, and jumping on seek.

This manager owns the server side of that. Per submitted watch-along job it
spins one background thread that:

  1. runs the prelude (`run_prelude`) → a populated `SpeakerRegistry`,
  2. dubs the prelude window (`dub_prelude_window`) → section 0,
  3. drives a `StreamScheduler` over the rest, with `on_section` writing each
     finished chunk into the per-section stream cache as it lands.

The chosen mode is folded into the cache key, so watch-along and quality caches
never collide and a rewatch in watch-along mode replays the cached sections
instantly. Playhead/seek/pause updates arrive via `POST /stream/.../playhead`
and mutate the record's `Playhead`, which the scheduler reads each tick.

Source resolution + URL download are delegated to the existing `JobManager`
(one yt-dlp path, shared download dir keyed by cache_key) so the two managers
agree on where a downloaded source lives.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patmir.paths import StoragePaths
from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import JobCancelled
from patmir.pipeline.streaming.prelude import run_prelude
from patmir.pipeline.streaming.runner import source_duration
from patmir.pipeline.streaming.scheduler import Playhead, StreamScheduler
from patmir.pipeline.streaming.section import (
    SectionResult,
    dub_prelude_window,
    process_section,
)
from patmir.ui import cache as cache_mod
from patmir.ui.jobs import JobManager, JobSpec

log = logging.getLogger(__name__)

MODE = "watch-along"
StreamStatus = str  # queued|prelude|streaming|cancelling|cancelled|done|failed


@dataclass
class StreamRecord:
    job_id: str
    cache_key: str
    source_id: str
    source_url: str | None
    source_path: str | None
    source_lang: str | None
    status: StreamStatus
    submitted_at: float
    mode: str = MODE
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    duration: float | None = None
    window_end: float | None = None       # first section boundary (prelude end)
    sections_done: int = 0
    cached: bool = False
    speakers: dict[str, Any] = field(default_factory=dict)
    # Mutated by the playhead POST handler; read by the scheduler loop.
    playhead: Playhead = field(default_factory=Playhead)
    _scheduler: StreamScheduler | None = field(default=None, repr=False)

    def to_public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "cache_key": self.cache_key,
            "mode": self.mode,
            "status": self.status,
            "source_url": self.source_url,
            "source_path": self.source_path,
            "source_lang": self.source_lang,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "duration": self.duration,
            "window_end": self.window_end,
            "sections_done": self.sections_done,
            "cached": self.cached,
            "speakers": self.speakers,
            "playhead": {"t": self.playhead.t, "paused": self.playhead.paused},
        }


class StreamManager:
    def __init__(
        self,
        config_path: Path,
        *,
        jobs: JobManager,
        storage: StoragePaths,
        stage_cache: dict | None = None,
        accelerator_lock: asyncio.Lock | None = None,
    ) -> None:
        self._config_path = config_path
        # Serializes accelerator work against JobManager — see the matching
        # comment there for why (one pipeline at a time is all a 24 GB card
        # fits). Held for the whole stream, so a batch job submitted mid
        # watch-along waits until the stream ends rather than OOMing it.
        self._accel_lock = accelerator_lock
        # Warm-model cache shared with JobManager and owned by the app. It used
        # to be a local in `_run_blocking`, so every model was rebuilt when a
        # new stream started; hoisting it here keeps them resident. Keys are
        # (stage, candidate_id), so sharing with the batch config is safe even
        # where the streaming overrides select a different candidate.
        self._stage_cache = stage_cache
        self._jobs = jobs            # delegate: source resolution + URL download
        self._storage = storage
        self._runs_root = storage.runs_root
        self._cache_root = storage.cache_root
        self._cfg: PipelineConfig | None = None  # streaming config, lazy
        self._records: OrderedDict[str, StreamRecord] = OrderedDict()
        self._by_cache_key: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        # Per-stream cooperative cancel tokens, job_id → Event. See
        # JobManager._cancels; StreamRecord isn't persisted, but keeping the
        # tokens here matches the batch side and keeps the record serializable.
        self._cancels: dict[str, threading.Event] = {}

    # --- lifecycle ------------------------------------------------------

    async def stop(self) -> None:
        for rec in self._records.values():
            if rec._scheduler is not None:
                rec._scheduler.stop()
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    log.exception("Stream task shutdown raised.")
        self._tasks.clear()

    # --- config ---------------------------------------------------------

    def _streaming_cfg(self) -> PipelineConfig:
        if self._cfg is None:
            from patmir.pipeline.streaming.runner import streaming_config
            self._cfg = streaming_config(self._config_path)
        return self._cfg

    def cache_key_for_url(self, url: str) -> str:
        return cache_mod.cache_key(cache_mod.source_id_for_url(url), self._streaming_cfg(), MODE)

    def lookup_cache_by_url(self, url: str) -> dict[str, Any] | None:
        ck = self.cache_key_for_url(url)
        return cache_mod.read_stream_manifest(self._cache_root, ck)

    # --- public API -----------------------------------------------------

    async def submit(self, spec: JobSpec) -> StreamRecord:
        source_id, source_url = self._jobs._resolve_source_id(spec)
        ck = cache_mod.cache_key(source_id, self._streaming_cfg(), MODE)

        async with self._lock:
            existing_id = self._by_cache_key.get(ck)
            if existing_id and existing_id in self._records and not spec.force:
                rec = self._records[existing_id]
                if rec.status in {"queued", "prelude", "streaming", "done"}:
                    return rec

            manifest = None if spec.force else cache_mod.read_stream_manifest(self._cache_root, ck)
            if manifest is not None and manifest.get("complete"):
                # Fully streamed before → instant rewatch, no work.
                rec = StreamRecord(
                    job_id=f"stream-cache-{ck}",
                    cache_key=ck, source_id=source_id, source_url=source_url,
                    source_path=spec.source_path, source_lang=spec.source_lang,
                    status="done", submitted_at=time.time(),
                    started_at=manifest.get("created_at"),
                    finished_at=manifest.get("finalized_at"),
                    duration=manifest.get("duration"),
                    sections_done=len(manifest.get("sections", [])),
                    cached=True,
                )
                self._record(rec)
                return rec

            job_id = f"stream-{int(time.time()*1000)}-{ck[:6]}"
            rec = StreamRecord(
                job_id=job_id, cache_key=ck, source_id=source_id, source_url=source_url,
                source_path=spec.source_path, source_lang=spec.source_lang,
                status="queued", submitted_at=time.time(),
            )
            self._record(rec)

        task = asyncio.create_task(self._run(rec, spec.force), name=f"patmir-stream-{job_id}")
        self._tasks[rec.job_id] = task
        return rec

    def get(self, job_id: str) -> StreamRecord | None:
        return self._records.get(job_id)

    def get_by_cache_key(self, ck: str) -> StreamRecord | None:
        jid = self._by_cache_key.get(ck)
        return self._records.get(jid) if jid else None

    def set_playhead(self, ck: str, t: float, paused: bool) -> bool:
        rec = self.get_by_cache_key(ck)
        if rec is None:
            return False
        rec.playhead.t = max(0.0, float(t))
        rec.playhead.paused = bool(paused)
        rec.playhead.updated_at = time.time()  # resets the abandon timer
        return True

    def stop_stream(self, ck: str) -> bool:
        """Stop a stream's scheduler immediately (viewer hit Stop / closed the
        tab). Prevents an abandoned stream from dubbing — and paying ElevenLabs
        + Scribe for — the rest of the video.

        Keyed by cache_key, kept because the extension calls it. It now also
        trips the cancel token, so the section *currently* being dubbed aborts at
        its next checkpoint instead of running to completion (~35s of section)
        while the scheduler sits stopped. Sections already written to the cache
        stay there — that is the free-rewatch feature, not leftover garbage.
        """
        rec = self.get_by_cache_key(ck)
        if rec is None:
            return False
        if rec._scheduler is not None:
            rec._scheduler.stop()
        self.cancel(rec.job_id)
        return True

    def cancel(self, job_id: str) -> bool:
        """Ask a stream to stop. Returns False only for an unknown job_id.

        Mirrors `JobManager.cancel`: instant if it never started, cooperative if
        it is running. Also stops the scheduler, so no *further* sections are
        scheduled even before the in-flight one notices the token.
        """
        rec = self._records.get(job_id)
        if rec is None:
            return False
        if rec.status in ("done", "failed", "cancelled"):
            return True
        if rec._scheduler is not None:
            rec._scheduler.stop()
        if rec.status == "queued":
            rec.status = "cancelled"
            rec.finished_at = time.time()
            return True
        ev = self._cancels.get(job_id)
        if ev is not None:
            ev.set()
        rec.status = "cancelling"
        return True

    def sections_manifest(self, ck: str) -> dict[str, Any] | None:
        manifest = cache_mod.read_stream_manifest(self._cache_root, ck)
        rec = self.get_by_cache_key(ck)
        if manifest is None and rec is None:
            return None
        manifest = manifest or {"cache_key": ck, "complete": False, "sections": []}
        # Overlay live job status so the client knows the stream is still
        # producing (vs. a stale partial manifest from a killed run).
        if rec is not None:
            manifest = {
                **manifest,
                "status": rec.status,
                "duration": manifest.get("duration") or rec.duration,
                "window_end": rec.window_end,
                "error": rec.error,
            }
        return manifest

    def section_file(self, ck: str, index: int) -> Path | None:
        manifest = cache_mod.read_stream_manifest(self._cache_root, ck)
        if manifest is None:
            return None
        for s in manifest.get("sections", []):
            if int(s.get("index", -1)) == index:
                path = self._cache_root / ck / s["file"]
                return path if path.is_file() else None
        return None

    # --- internals ------------------------------------------------------

    def _record(self, rec: StreamRecord) -> None:
        self._records[rec.job_id] = rec
        self._by_cache_key[rec.cache_key] = rec.job_id
        while len(self._records) > 50:
            self._records.popitem(last=False)

    async def _run(self, rec: StreamRecord, force: bool) -> None:
        # Surface the wait only when there is one, so an uncontended stream
        # goes straight to "prelude" as it always did.
        if self._accel_lock is not None and self._accel_lock.locked():
            rec.status = "waiting"
        async with (self._accel_lock or contextlib.nullcontext()):
            if rec.status == "cancelled":   # cancelled while queued for the lock
                return
            await self._run_locked(rec, force)

    async def _run_locked(self, rec: StreamRecord, force: bool) -> None:
        ev = threading.Event()
        self._cancels[rec.job_id] = ev
        rec.status = "prelude"
        rec.started_at = time.time()
        try:
            await asyncio.to_thread(self._run_blocking, rec, force, ev)
            rec.status = "done"
        except JobCancelled:
            # Deliberate stop. Sections already in the cache remain playable.
            rec.status = "cancelled"
            log.info("Stream %s cancelled.", rec.job_id)
        except asyncio.CancelledError:
            rec.status = "failed"
            rec.error = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            rec.status = "failed"
            log.exception("Stream %s failed.", rec.job_id)
        finally:
            rec.finished_at = time.time()
            self._cancels.pop(rec.job_id, None)

    def _run_blocking(self, rec: StreamRecord, force: bool,
                      cancel: threading.Event | None = None) -> None:
        """Prelude → opening dub → playhead-driven streaming. Runs on a worker
        thread; the heavy stages are synchronous."""
        cfg = self._streaming_cfg()
        ck = rec.cache_key

        # Resolve the local source (download URLs via the shared JobManager path).
        if rec.source_url:
            local_path = self._jobs._download_url(rec.source_url, ck)
        else:
            local_path = Path(rec.source_path or "").expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"resolved source file missing: {local_path}")

        rec.duration = source_duration(local_path)

        # Resume from any already-cached sections (partial stream / rewatch). On
        # `force`, ignore the prior manifest and start fresh. Only (re)initialize
        # the manifest when there's nothing to resume — otherwise we'd wipe the
        # already-cached sections list and orphan their WAVs.
        prior = None if force else cache_mod.read_stream_manifest(self._cache_root, ck)
        if prior is None:
            cache_mod.init_stream_manifest(
                self._cache_root, ck,
                source_id=rec.source_id, source_url=rec.source_url,
                duration=rec.duration,
                pipeline_summary=cache_mod.pipeline_summary(cfg),
            )
        else:
            rec.sections_done = len(prior.get("sections", []))

        # Warm-model cache: each stage model is built once and reused, killing
        # the measured 11-13% per-section rebuild tax (performance.md roadmap
        # #2). Shared app-wide when the server provided one, so it also survives
        # across streams and is shared with batch jobs; falls back to a
        # stream-local dict when constructed without one.
        stage_cache: dict = self._stage_cache if self._stage_cache is not None else {}

        # 1) Prelude — establish the cast + voices + embeddings. Window size is
        # a config knob (smaller → faster first audio, fewer characters up front).
        prelude = run_prelude(
            local_path, cfg, runs_root=self._runs_root,
            window_seconds=cfg.options.prelude_window_seconds,
            stage_cache=stage_cache, cancel=cancel,
        )
        rec.window_end = prelude.window_end
        rec.speakers = prelude.speakers

        # 2) Dub the prelude window (section 0), reusing the prelude analysis.
        rec.status = "streaming"
        opening = dub_prelude_window(cfg, prelude, runs_root=self._runs_root,
                                     stage_cache=stage_cache, cancel=cancel)
        self._emit(rec, opening)

        # 3) Stream the rest, playhead-driven. Seed the abandon timer so the
        # scheduler doesn't bail before the client's first playhead POST lands.
        rec.playhead.updated_at = time.time()
        index_counter = {"n": opening.index}

        def _process(start: float) -> SectionResult | None:
            index_counter["n"] += 1
            try:
                return process_section(
                    local_path, cfg, prelude.registry,
                    index=index_counter["n"], start=start, runs_root=self._runs_root,
                    stage_cache=stage_cache, cancel=cancel,
                )
            except JobCancelled:
                # MUST propagate. This except-Exception used to swallow it and
                # report a failed section, so Stop would be silently downgraded
                # to "one section didn't work" and the stream would carry on.
                raise
            except Exception:  # noqa: BLE001
                log.exception("Section at %.1fs failed.", start)
                return None

        scheduler = StreamScheduler(
            duration=rec.duration,
            prelude_end=prelude.window_end,
            process_fn=_process,
            playhead_fn=lambda: rec.playhead,
            on_section=lambda r: self._emit(rec, r),
        )
        # Re-seed prior cached sections so a resumed stream skips them.
        if prior:
            for s in prior.get("sections", []):
                scheduler.coverage.add(float(s["start"]), float(s["end"]))
        rec._scheduler = scheduler
        scheduler.run()

        if scheduler.coverage.is_complete(rec.duration):
            cache_mod.finalize_stream(self._cache_root, ck)

    def _emit(self, rec: StreamRecord, result: SectionResult) -> None:
        if result.dub_wav is None:
            return
        cache_mod.append_section(
            self._cache_root, rec.cache_key,
            index=result.index, start=result.start, end=result.end,
            src_wav=result.dub_wav,
        )
        rec.sections_done += 1
