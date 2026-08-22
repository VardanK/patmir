"""In-process job queue + worker.

A single asyncio worker drains a queue serially (concurrency=1) — Demucs +
MPS + cloud-API rate limits make parallel runs unsafe. Each job:

1. Resolves `(source_id, cache_key)` from the request.
2. Returns immediately with status=`done` if the cache already has it.
3. Otherwise enqueues; the worker downloads URLs via yt-dlp (slice 1
   handler-side; slice 3 will move this into a real ingest candidate),
   then calls `patmir.pipeline.runner.run_job` on the resolved local path,
   then `cache.materialize`s the finished `runs/<job>/` into `~/.patmir/cache/`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from patmir.paths import StoragePaths
from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.runner import run_job
from patmir.pipeline.stage import STAGE_ORDER
from patmir.ui import cache as cache_mod

log = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]
_INDEX_MAX_ENTRIES = 200


@dataclass
class JobSpec:
    source_url: str | None = None
    source_path: str | None = None
    source_lang: str | None = None
    force: bool = False  # if True, invalidate cache + delete downloaded source before running
    # "quality" (D1 batch, default) or "watch-along" (D3 streaming). The server
    # routes watch-along specs to the StreamManager; the mode is also folded
    # into the cache key so the two modes never collide.
    mode: str = "quality"

    def __post_init__(self) -> None:
        if not self.source_url and not self.source_path:
            raise ValueError("source_url or source_path is required")
        if self.mode not in {"quality", "watch-along"}:
            raise ValueError(f"unknown mode: {self.mode!r}")


@dataclass
class JobRecord:
    job_id: str
    cache_key: str
    source_id: str
    source_url: str | None
    source_path: str | None
    source_lang: str | None
    status: JobStatus
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    run_job_id: str | None = None
    stage_states: dict[str, str] = field(default_factory=dict)
    cached: bool = False
    download_pct: float | None = None  # 0..1 while yt-dlp is downloading
    live_stage: str | None = None      # most recent progress.jsonl stage
    live_pct: float | None = None
    live_msg: str | None = None

    def to_public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "cache_key": self.cache_key,
            "status": self.status,
            "source_url": self.source_url,
            "source_path": self.source_path,
            "source_lang": self.source_lang,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "run_job_id": self.run_job_id,
            "stage_states": self.stage_states,
            "cached": self.cached,
            "download_pct": self.download_pct,
            "live_stage": self.live_stage,
            "live_pct": self.live_pct,
            "live_msg": self.live_msg,
        }


# Display order including the synthetic "download" pseudo-stage. Used by
# the popup to render stage chips; URL jobs surface "download", local-file
# jobs leave it absent.
DISPLAY_STAGE_ORDER: tuple[str, ...] = ("download", *STAGE_ORDER)


class JobManager:
    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        storage: StoragePaths,
        stage_cache: dict | None = None,
    ) -> None:
        self._cfg = cfg
        # Warm-model cache shared with StreamManager and owned by the app, so
        # models survive across submissions instead of being rebuilt per job.
        # None keeps the old build-per-job behavior (used by the tests).
        self._stage_cache = stage_cache
        self._storage = storage
        self._runs_root = storage.runs_root
        self._cache_root = storage.cache_root
        self._downloads_dir = storage.downloads_root
        self._records: OrderedDict[str, JobRecord] = OrderedDict()
        self._by_cache_key: dict[str, str] = {}  # cache_key → job_id (active or last)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # --- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker(), name="patmir-job-worker")

    async def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("Worker shutdown raised.")
        self._worker_task = None

    # --- public API -----------------------------------------------------

    async def submit(self, spec: JobSpec) -> JobRecord:
        """Resolve cache + enqueue (or no-op if cached / already queued).

        With `spec.force=True`, the existing cache entry is invalidated and
        any cached download is removed so the pipeline reruns from scratch.
        """
        source_id, source_url = self._resolve_source_id(spec)
        ck = cache_mod.cache_key(source_id, self._cfg)
        if spec.force:
            cache_mod.invalidate(self._cache_root, ck)
            # Drop the cached source download so a re-translate also picks
            # up a fresh source (useful if YouTube updated the audio or the
            # download was partial).
            dest = self._downloads_dir / ck
            if dest.is_dir():
                import shutil as _shutil
                _shutil.rmtree(dest, ignore_errors=True)
        async with self._lock:
            # Existing record (queued / running / done / failed-recently) for the same key?
            existing_id = self._by_cache_key.get(ck)
            if existing_id and existing_id in self._records and not spec.force:
                rec = self._records[existing_id]
                if rec.status in {"queued", "running", "done"}:
                    return rec
            # Cache hit on disk? Materialize a synthetic "done" record.
            entry = None if spec.force else cache_mod.lookup(self._cache_root, ck)
            if entry is not None:
                rec = JobRecord(
                    job_id=f"cache-{ck}",
                    cache_key=ck,
                    source_id=source_id,
                    source_url=source_url,
                    source_path=spec.source_path,
                    source_lang=spec.source_lang,
                    status="done",
                    submitted_at=time.time(),
                    started_at=entry.created_at,
                    finished_at=entry.created_at,
                    cached=True,
                )
                self._record(rec)
                return rec
            # New job — enqueue.
            job_id = f"job-{int(time.time()*1000)}-{ck[:6]}"
            rec = JobRecord(
                job_id=job_id,
                cache_key=ck,
                source_id=source_id,
                source_url=source_url,
                source_path=spec.source_path,
                source_lang=spec.source_lang,
                status="queued",
                submitted_at=time.time(),
            )
            # URL jobs surface a synthetic `download` pseudo-stage so the
            # popup can show progress while yt-dlp is still running and
            # `run_job` hasn't been called yet.
            states: dict[str, str] = {}
            if source_url:
                states["download"] = "pending"
            states.update({n: "pending" for n in STAGE_ORDER})
            rec.stage_states = states
            self._record(rec)
        await self._queue.put(rec.job_id)
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        rec = self._records.get(job_id)
        if rec is None:
            return None
        # While the pipeline is running, the worker thread won't update
        # `stage_states` / `live_*` until `run_job` returns. Read the live
        # manifest + last progress line every time the popup polls so the
        # user sees real per-stage progress.
        if rec.status == "running" and rec.run_job_id:
            self._merge_live_state(rec)
        return rec

    def _merge_live_state(self, rec: JobRecord) -> None:
        run_dir = self._runs_root / (rec.run_job_id or "")
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                m = json.loads(manifest_path.read_text())
                live: dict[str, str] = {}
                if "download" in rec.stage_states:
                    live["download"] = rec.stage_states["download"]
                stages_obj = m.get("stages")
                # `stages` is keyed by name on disk; tolerate the list shape
                # too in case the manifest format ever changes.
                if isinstance(stages_obj, dict):
                    items = stages_obj.items()
                elif isinstance(stages_obj, list):
                    items = ((st.get("name"), st) for st in stages_obj if isinstance(st, dict))
                else:
                    items = ()
                for name, st in items:
                    if isinstance(st, dict):
                        state = st.get("state")
                        if name and state:
                            live[name] = state
                if live:
                    rec.stage_states = live
            except (OSError, json.JSONDecodeError):
                pass
        prog_path = run_dir / "progress.jsonl"
        if prog_path.is_file():
            try:
                last = _tail_last_line(prog_path)
                if last:
                    data = json.loads(last)
                    rec.live_stage = data.get("stage") or rec.live_stage
                    rec.live_pct = data.get("pct")
                    rec.live_msg = data.get("msg") or rec.live_msg
            except (OSError, json.JSONDecodeError):
                pass

    def lookup_cache_by_url(self, url: str) -> cache_mod.CacheEntry | None:
        return cache_mod.lookup_by_url(self._cache_root, url, self._cfg)

    def cache_key_for_url(self, url: str) -> str:
        return cache_mod.cache_key(cache_mod.source_id_for_url(url), self._cfg)

    # --- internals ------------------------------------------------------

    def _resolve_source_id(self, spec: JobSpec) -> tuple[str, str | None]:
        if spec.source_url:
            return cache_mod.source_id_for_url(spec.source_url), spec.source_url
        path = Path(spec.source_path or "").expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"source_path does not exist: {path}")
        return cache_mod.source_id_for_file(path), None

    def _record(self, rec: JobRecord) -> None:
        self._records[rec.job_id] = rec
        self._by_cache_key[rec.cache_key] = rec.job_id
        self._persist_index()
        # Bound the index size.
        while len(self._records) > _INDEX_MAX_ENTRIES:
            stale_id, _ = self._records.popitem(last=False)
            # Don't unbind from _by_cache_key if it's still the canonical mapping.

    def _persist_index(self) -> None:
        try:
            index_path = self._storage.jobs_index
            index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [asdict(r) for r in self._records.values()]
            index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except OSError:
            log.warning("Failed to persist jobs index.", exc_info=True)

    async def _worker(self) -> None:
        log.info("JobManager worker started.")
        while True:
            job_id = await self._queue.get()
            rec = self._records.get(job_id)
            if rec is None:
                continue
            rec.status = "running"
            rec.started_at = time.time()
            self._persist_index()
            try:
                await asyncio.to_thread(self._run_one, rec)
                rec.status = "done"
            except Exception as exc:  # noqa: BLE001
                rec.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                rec.status = "failed"
                log.exception("Job %s failed.", rec.job_id)
            finally:
                rec.finished_at = time.time()
                self._persist_index()

    def _run_one(self, rec: JobRecord) -> None:
        """Synchronous job execution — runs in a worker thread."""
        if rec.source_url:
            rec.stage_states["download"] = "running"
            rec.live_stage = "download"
            rec.live_pct = 0.0
            rec.live_msg = "starting yt-dlp"
            try:
                local_path = self._download_url(rec.source_url, rec.cache_key, rec)
            except Exception:
                rec.stage_states["download"] = "failed"
                raise
            rec.stage_states["download"] = "done"
            rec.download_pct = 1.0
        else:
            local_path = Path(rec.source_path or "").expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"resolved source file missing: {local_path}")

        def _on_job_dir(job_dir: Path, job_id_: str) -> None:
            rec.run_job_id = job_id_

        # Run the existing batch pipeline unchanged.
        manifest = run_job(
            local_path,
            self._cfg,
            runs_root=self._runs_root,
            source_lang=rec.source_lang,
            on_job_dir_created=_on_job_dir,
            stage_cache=self._stage_cache,
        )
        rec.run_job_id = manifest.job_id
        # Keep download state in stage_states post-success.
        final_states = {n: manifest.stage(n).state for n in STAGE_ORDER}
        if "download" in rec.stage_states:
            final_states = {"download": rec.stage_states["download"], **final_states}
        rec.stage_states = final_states
        # Materialize finished artifacts into the global cache.
        job_dir = self._runs_root / manifest.job_id
        duration = _read_duration(job_dir)
        cache_mod.materialize(
            self._cache_root,
            rec.cache_key,
            job_dir=job_dir,
            source_id=rec.source_id,
            source_url=rec.source_url,
            job_id=manifest.job_id,
            duration=duration,
            pipeline_summary=cache_mod.pipeline_summary(self._cfg),
        )

    def _download_url(self, url: str, cache_key: str, rec: "JobRecord | None" = None) -> Path:
        """yt-dlp pre-download to a known dir keyed by cache_key.

        Re-uses already-downloaded files between retries so a re-submit after
        pipeline failure doesn't re-fetch from YouTube. When `rec` is given,
        updates `rec.download_pct` + `rec.live_*` on each yt-dlp progress
        tick so the popup can show progress while waiting.
        """
        dest = self._downloads_dir / cache_key
        dest.mkdir(parents=True, exist_ok=True)
        # If we already have a source file from a previous attempt, reuse it.
        existing = next((p for p in dest.iterdir() if p.name.startswith("source.")), None)
        if existing and existing.is_file() and existing.stat().st_size > 0:
            if rec is not None:
                rec.download_pct = 1.0
                rec.live_msg = "reusing cached download"
            return existing

        try:
            import yt_dlp  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "yt-dlp not installed. Install patmir[local] to enable URL ingest."
            ) from exc

        def _hook(info: dict) -> None:
            if rec is None:
                return
            st = info.get("status")
            if st == "downloading":
                total = info.get("total_bytes") or info.get("total_bytes_estimate")
                downloaded = info.get("downloaded_bytes")
                if total and downloaded:
                    pct = min(1.0, downloaded / total)
                    rec.download_pct = pct
                    rec.live_stage = "download"
                    rec.live_pct = pct
                    speed = info.get("speed")
                    speed_s = f" — {speed / 1e6:.1f} MB/s" if speed else ""
                    rec.live_msg = f"{downloaded / 1e6:.1f} / {total / 1e6:.1f} MB{speed_s}"
            elif st == "finished":
                rec.download_pct = 1.0
                rec.live_pct = 1.0
                rec.live_msg = "download complete"

        # Audio-only — the ingest stage extracts PCM via ffmpeg and never
        # touches the video track. `bestaudio/best` falls back to a muxed
        # stream only when no audio-only format is available.
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(dest / "source.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "retries": 3,
            "progress_hooks": [_hook],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        path = Path(filename)
        if not path.is_file():
            # yt-dlp may rewrite the extension post-merge; fall back to scanning.
            candidates = [p for p in dest.iterdir() if p.name.startswith("source.")]
            if not candidates:
                raise RuntimeError(f"yt-dlp finished but no file at {path}")
            path = candidates[0]
        return path


def _tail_last_line(path: Path, *, max_seek: int = 4096) -> str | None:
    """Return the last non-empty line of a file. Reads only the tail."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    with path.open("rb") as f:
        f.seek(max(0, size - max_seek))
        chunk = f.read()
    text = chunk.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _read_duration(job_dir: Path) -> float | None:
    meta = job_dir / "audio.meta.json"
    if not meta.is_file():
        return None
    try:
        return float(json.loads(meta.read_text()).get("duration"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


