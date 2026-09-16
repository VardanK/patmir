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
from time import perf_counter
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from patmir.paths import StoragePaths
from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import (
    NEUTRAL_AUDIENCE,
    JobCancelled,
    TranslatorUnavailable,
    resolve_audience,
)
from patmir.pipeline.streaming.pipeline import SectionPipeline
from patmir.pipeline.streaming.prelude import run_prelude
from patmir.pipeline.streaming.runner import source_duration
from patmir.pipeline.streaming.scheduler import Playhead, StreamScheduler
from patmir.pipeline.streaming.timing import section_timing
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
    # Resolved audience level this stream runs at. Folded into the cache key, so
    # a level change is a new stream rather than a mid-flight switch; a running
    # stream keeps the level it started with.
    audience: str = NEUTRAL_AUDIENCE
    # Audience-free key, used only to name the yt-dlp download dir — the source
    # file is the same at every level.
    download_key: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    duration: float | None = None
    window_end: float | None = None       # first section boundary (prelude end)
    sections_done: int = 0
    # Seconds of the video actually dubbed. `sections_done` alone cannot make a
    # fraction (sections vary in length and the prelude window is its own
    # size), and a stream record has no `stage_states` — which is why the
    # in-page progress bar showed nothing at all in watch-along once it became
    # the default mode.
    dubbed_s: float = 0.0
    # Sections that raised and were skipped (the stream carried on). Before
    # this a skipped section left no trace outside the server log; the client
    # saw `streaming` and a section count that simply did not grow.
    sections_failed: int = 0
    cached: bool = False
    # Download progress. These four names are what JobManager's yt-dlp hook
    # writes; carrying them here is what lets a watch-along stream pass `rec`
    # into the shared `_download_url` and finally report progress instead of
    # sitting silent for the whole fetch.
    # Last time the client asked for anything. Feeds the scheduler's abandon
    # timer, which used to key off playhead movement and so could not tell a
    # paused viewer from a departed one.
    last_seen_at: float | None = None
    download_pct: float | None = None
    live_stage: str | None = None
    live_pct: float | None = None
    live_msg: str | None = None
    # Startup breakdown, in seconds. `first_audio_s` is the author-facing
    # number: submit -> section 0 playable.
    download_s: float | None = None
    prelude_s: float | None = None
    opening_dub_s: float | None = None
    first_audio_s: float | None = None
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
            "audience": self.audience,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "duration": self.duration,
            "window_end": self.window_end,
            "sections_done": self.sections_done,
            # The progress bar divides this by `duration`; without it the
            # extension read `undefined` and hid the bar in watch-along.
            "dubbed_s": self.dubbed_s,
            "sections_failed": self.sections_failed,
            "cached": self.cached,
            "speakers": self.speakers,
            "playhead": {"t": self.playhead.t, "paused": self.playhead.paused},
            "download_pct": self.download_pct,
            "live_stage": self.live_stage,
            "live_pct": self.live_pct,
            "live_msg": self.live_msg,
            "timing": {
                "download_s": self.download_s,
                "prelude_s": self.prelude_s,
                "opening_dub_s": self.opening_dub_s,
                "first_audio_s": self.first_audio_s,
            },
        }




def _is_voiceless(result: SectionResult) -> bool:
    """A section the translate stage saw at least two lines of and gave text to
    none of them. Two, not one: one empty line is what a per-line provider
    refusal looks like and the audience ADR keeps that as a reported gap. A
    section with NO lines (music, silence) is not voiceless — there was nothing
    to say. Unknown counts (`None`) are trusted, since the stage would have
    raised on an outage it could see."""
    if result.lines is None or result.voiced_lines is None:
        return False
    return result.lines >= 2 and result.voiced_lines == 0


def _dubbed_seconds(prior: dict[str, Any] | None) -> float:
    """Seconds already on disk, from a cached manifest.

    Summed rather than counted: sections differ in length, and the prelude
    window is its own size. A malformed entry is skipped rather than raising —
    this feeds a progress bar, and a bar is never worth failing a resume for.
    """
    total = 0.0
    for s in (prior or {}).get("sections", []):
        try:
            total += max(0.0, float(s["end"]) - float(s["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    return total


def _cached_opening(prior: dict[str, Any] | None,
                    window_end: float,
                    tolerance: float = 1.0,
                    start_tolerance: float = 0.05) -> dict[str, Any] | None:
    """The cached section that already covers the opening window, if any.

    Matched on the SPAN rather than on `index == 0`, because a manifest
    repaired by `append_section`'s index dedup (issue #22) can hold the opening
    window under any index. The END may fall short of `window_end` by
    `tolerance` (VAD snapping moves that boundary, and coverage is seeded from
    the cached span, so the next section starts exactly where it ends). The
    START may not: coverage is seeded from the cached span too, so a section
    that begins at 0.9 s leaves `[0, 0.9)` uncovered and the scheduler dubs a
    sub-second section at 0 for it. `start_tolerance` is float slack only.
    """
    for s in (prior or {}).get("sections", []):
        try:
            start = float(s["start"])
            end = float(s["end"])
            index = int(s["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if index < 0:
            continue
        if start <= start_tolerance and end >= window_end - tolerance:
            return s
    return None


def _index_seed(prior: dict[str, Any] | None, opening_index: int) -> int:
    """Highest section index already in the cache, so a resumed stream numbers
    PAST it instead of overwriting.

    `append_section` names the file `section_{index:03d}.wav`, so reusing an
    index overwrites that file. `dub_prelude_window` always returns index 0, so
    a restarted stream restarted numbering at 0 while `append_section` deduped
    by START and kept both entries — two windows, one file. Measured on cache
    key d5af5653d745ed66: indices 1 and 2 each appeared twice with different
    windows, so the wrong audio played at 154.9 s and 190.3 s, live and on every
    rewatch.
    """
    indices = [
        int(x["index"]) for x in (prior or {}).get("sections", [])
        if isinstance(x, dict) and str(x.get("index", "")).lstrip("-").isdigit()
    ]
    return max([opening_index, *indices])


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

    def cache_key_for_url(self, url: str, audience: str | None = None) -> str:
        return cache_mod.cache_key(
            cache_mod.source_id_for_url(url), self._streaming_cfg(), MODE,
            self.resolved_audience(audience),
        )

    def lookup_cache_by_url(
        self, url: str, audience: str | None = None,
    ) -> dict[str, Any] | None:
        ck = self.cache_key_for_url(url, audience)
        return cache_mod.read_stream_manifest(self._cache_root, ck)

    def resolved_audience(self, requested: str | None) -> str:
        """The level a submission with this `requested` value would run at."""
        return resolve_audience(requested, self._streaming_cfg().options)

    def _cfg_for(self, audience: str) -> PipelineConfig:
        """Streaming config carrying this stream's audience level.

        Options only — `entries` is shared by reference so the warm stage cache,
        keyed `(stage, candidate_id)` and shared with JobManager, is untouched.
        Same reasoning as `JobManager._cfg_for`; the prelude, the opening dub and
        every `process_section` already take `cfg`, so deriving once here is the
        whole of the watch-along wiring.
        """
        cfg = self._streaming_cfg()
        if audience == cfg.options.audience:
            return cfg
        return replace(cfg, options=replace(cfg.options, audience=audience))

    # --- public API -----------------------------------------------------

    async def submit(self, spec: JobSpec) -> StreamRecord:
        source_id, source_url = self._jobs._resolve_source_id(spec)
        # Resolved once and carried on the record; the key and the run must not
        # be able to disagree.
        audience = self.resolved_audience(spec.audience)
        ck = cache_mod.cache_key(source_id, self._streaming_cfg(), MODE, audience)
        dl_key = cache_mod.cache_key(
            source_id, self._streaming_cfg(), MODE, NEUTRAL_AUDIENCE,
        )

        async with self._lock:
            existing_id = self._by_cache_key.get(ck)
            if existing_id and existing_id in self._records and not spec.force:
                rec = self._records[existing_id]
                # `waiting` (parked on the accelerator lock) is as live as
                # `queued`; without it here a re-submit during the wait created
                # a second record for the same video, queued behind the first.
                if rec.status in {"queued", "waiting", "prelude", "streaming"}:
                    return rec
                if rec.status == "done":
                    # `done` does NOT imply the video is dubbed: the scheduler
                    # also returns when it thinks the viewer left, which a pause
                    # used to trigger. Handing that record back left the viewer
                    # frozen at the buffer edge with no way to restart short of
                    # a server bounce. Only a genuinely complete stream is
                    # reusable; anything else falls through and resumes from the
                    # sections already cached.
                    done_manifest = cache_mod.read_stream_manifest(self._cache_root, ck)
                    if done_manifest is not None and done_manifest.get("complete"):
                        return rec
                    log.info("Stream %s was done but incomplete — restarting.", rec.job_id)

            manifest = None if spec.force else cache_mod.read_stream_manifest(self._cache_root, ck)
            if manifest is not None and manifest.get("complete"):
                # Fully streamed before → instant rewatch, no work.
                rec = StreamRecord(
                    job_id=f"stream-cache-{ck}",
                    cache_key=ck, source_id=source_id, source_url=source_url,
                    source_path=spec.source_path, source_lang=spec.source_lang,
                    audience=audience, download_key=dl_key,
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
                audience=audience, download_key=dl_key,
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
        rec.playhead.updated_at = time.time()
        rec.last_seen_at = time.time()  # resets the abandon timer
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
        # EVERY live record for the key, not just the newest. `_by_cache_key`
        # points at the latest submission, and `submit()` creates a fresh
        # record whenever the existing one is `cancelling` — so after one
        # Stop + re-submit the record actually holding the accelerator was no
        # longer reachable from here, and the viewer's next Stop hit only the
        # `waiting` newcomer. Three records for one video piled up that way on
        # 2026-09-04 (quality issue #33).
        live = [r for r in self._records.values()
                if r.cache_key == ck and r.status not in ("done", "failed", "cancelled")]
        if not live and self.get_by_cache_key(ck) is None:
            return False
        for rec in live:
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
        if rec.status in ("queued", "waiting"):
            # Not started: nothing has touched the accelerator, so this is
            # exact and immediate, and `_run` skips the body when the lock
            # frees. `waiting` was missing here — it fell through to the
            # running branch, found no token (none exists before
            # `_run_locked`), and left the record `cancelling`, which `_run`
            # does NOT skip: the "cancelled" stream would have run in full the
            # moment the lock was released. `JobManager.cancel` had both
            # statuses from the start.
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
            rec.last_seen_at = time.time()
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
        except TranslatorUnavailable as exc:
            # The translator cannot translate anything (no credits, bad key,
            # every line empty). A diagnosed condition with a one-line reason,
            # so the record carries THAT — the extension prints `error` under
            # the button, and a traceback there is noise. Ends `failed`, not
            # `done`: a stream that produced nothing translated is not done.
            rec.error = exc.message
            rec.status = "failed"
            log.error("Stream %s: translator unavailable — %s", rec.job_id, exc.message)
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
        cfg = self._cfg_for(rec.audience)
        ck = rec.cache_key
        t_start = perf_counter()

        # Resolve the local source (download URLs via the shared JobManager path).
        # `rec` is passed so yt-dlp's progress hook fires: without it the hook
        # no-ops and a URL stream reported nothing at all for the whole fetch,
        # which on a feature-length video is the longest silent stretch of the
        # entire run.
        if rec.source_url:
            t_dl = perf_counter()
            local_path = self._jobs._download_url(
                rec.source_url, rec.download_key or ck, rec, cancel=cancel,
            )
            rec.download_s = perf_counter() - t_dl
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
            rec.dubbed_s = _dubbed_seconds(prior)

        # Warm-model cache: each stage model is built once and reused, killing
        # the measured 11-13% per-section rebuild tax (performance.md roadmap
        # #2). Shared app-wide when the server provided one, so it also survives
        # across streams and is shared with batch jobs; falls back to a
        # stream-local dict when constructed without one.
        stage_cache: dict = self._stage_cache if self._stage_cache is not None else {}

        # 1) Prelude — establish the cast + voices + embeddings. Window size is
        # a config knob (smaller → faster first audio, fewer characters up front).
        t_prelude = perf_counter()
        prelude = run_prelude(
            local_path, cfg, runs_root=self._runs_root,
            window_seconds=cfg.options.prelude_window_seconds,
            stage_cache=stage_cache, cancel=cancel,
        )
        rec.prelude_s = perf_counter() - t_prelude
        rec.window_end = prelude.window_end
        rec.speakers = prelude.speakers

        # 2) Dub the prelude window (section 0), reusing the prelude analysis —
        # unless it is ALREADY CACHED, in which case re-doing it is pure loss.
        #
        # A restart (issue #21: a `done` stream whose manifest is not
        # `complete` is resumed by `submit()`) used to re-translate and
        # re-synthesize the opening window every time, on top of re-running the
        # prelude analysis. The analysis has to happen — the `SpeakerRegistry`
        # it builds is not persisted, and every later section resolves speaker
        # identity against it — but the *audio* is on disk and byte-for-byte
        # what we would produce again.
        #
        # The cost is not just wasted GPU: `scheduler.run()` does not start
        # until this returns, so nothing NEW is produced for its duration and
        # the viewer sits frozen at the edge of the audio they already have.
        # Measured on cache key 8534a9320e5492c2 (a 3471 s video): two prelude
        # runs 53 s apart with exactly one section cached, section 0 costing
        # 22.4 s of that — the author reported the video "stuck at 35 sec",
        # which is inside the 0-39.99 s the cache already held.
        rec.status = "streaming"
        cached_opening = _cached_opening(prior, prelude.window_end)
        if cached_opening is not None:
            rec.opening_dub_s = 0.0
            rec.sections_done = len(prior.get("sections", []))
            rec.dubbed_s = _dubbed_seconds(prior)
            opening_index = int(cached_opening["index"])
            log.info("Opening window %.2f-%.2f already cached (section %d) — "
                     "skipping the re-dub and going straight to streaming.",
                     float(cached_opening["start"]), float(cached_opening["end"]),
                     opening_index)
        else:
            opening = dub_prelude_window(cfg, prelude, runs_root=self._runs_root,
                                         stage_cache=stage_cache, cancel=cancel)
            rec.opening_dub_s = opening.wall_s
            self._emit(rec, opening)
            opening_index = opening.index
        # The number the viewer actually feels: submit -> first playable audio.
        # With a cached opening that is the prelude alone, because the audio was
        # playable before this call started.
        rec.first_audio_s = perf_counter() - t_start

        # 3) Stream the rest, playhead-driven. Seed the abandon timer so the
        # scheduler doesn't bail before the client's first playhead POST lands.
        rec.playhead.updated_at = time.time()
        rec.last_seen_at = time.time()
        # Seed PAST any already-cached section. `opening.index` is always 0, so
        # a resumed stream used to restart numbering at 0 and overwrite
        # `section_001.wav`, `section_002.wav`, ... while `append_section`
        # deduped by START and kept both entries — leaving two windows pointing
        # at one file. Measured on cache key d5af5653d745ed66, where indices 1
        # and 2 each appeared twice with different windows, so the wrong audio
        # played at 154.9 s and 190.3 s, live and on every rewatch.
        index_counter = {"n": _index_seed(prior, opening_index)}

        # Analyse the next section while this one waits on the translate
        # round-trip. Translate is 29% of watch-along wall with the GPU idle
        # throughout, and analysis is the GPU work that fits in that gap. One
        # lock inside the driver keeps GPU phases serialized, which the TTS
        # CUDA-graph capture requires. `section_prefetch = false` restores the
        # fully serial path.
        pipeline = SectionPipeline(
            local_path, cfg, prelude.registry,
            runs_root=self._runs_root, stage_cache=stage_cache, cancel=cancel,
            source_lang=prelude.source_lang, duration=rec.duration,
            prefetch=bool(getattr(cfg.options, "section_prefetch", True)),
            index_start=index_counter["n"],
        )

        def _process(start: float, max_end: float | None = None) -> SectionResult | None:
            try:
                return pipeline.process(start, max_end)
            except JobCancelled:
                # MUST propagate. This except-Exception used to swallow it and
                # report a failed section, so Stop would be silently downgraded
                # to "one section didn't work" and the stream would carry on.
                raise
            except TranslatorUnavailable:
                # MUST propagate too. Every later section would fail the same
                # way, so "skip this one and carry on" means a stream that
                # reports `streaming` forever and produces nothing — or, before
                # the candidates raised at all, one that produced voiceless
                # sections and called them translated (quality issue #35).
                raise
            except Exception:  # noqa: BLE001
                log.exception("Section at %.1fs failed.", start)
                rec.sections_failed += 1
                return None

        scheduler = StreamScheduler(
            duration=rec.duration,
            prelude_end=prelude.window_end,
            process_fn=_process,
            playhead_fn=lambda: rec.playhead,
            on_section=lambda r: self._emit(rec, r),
            alive_fn=lambda: rec.last_seen_at or 0.0,
        )
        # Re-seed prior cached sections so a resumed stream skips them.
        if prior:
            for s in prior.get("sections", []):
                scheduler.coverage.add(float(s["start"]), float(s["end"]))
        rec._scheduler = scheduler
        try:
            scheduler.run()
        finally:
            # Drop any in-flight prefetch before returning. A running analysis
            # cannot be interrupted (it is a torch forward), so this waits for
            # it rather than leaving a thread on the GPU after the stream is
            # gone — which would meet the next job holding the accelerator.
            pipeline.close()
            # The only place the prefetch telemetry reaches production logs.
            # `prefetch_discarded` climbing on a stream means every section is
            # paying a wasted analysis (a resumed stream used to, see
            # `_take_analysis`).
            log.info("stream %s section pipeline: %s", ck, pipeline.stats)

        if scheduler.coverage.is_complete(rec.duration):
            cache_mod.finalize_stream(self._cache_root, ck)

    def _emit(self, rec: StreamRecord, result: SectionResult) -> None:
        if result.dub_wav is None:
            return
        if _is_voiceless(result):
            # Last line of defence. With the candidates raising on 401/402 and
            # the adapter raising on all-empty, this should be unreachable;
            # it exists because the cache is the free-rewatch store, so a
            # voiceless section written once plays wrong on every viewing
            # (nine keys had to be purged after 2026-09-13, quality issue #35).
            log.error(
                "Section %d (%.1f-%.1f s) has %s lines and none with text — "
                "not caching it.", result.index, result.start, result.end,
                result.lines,
            )
            rec.sections_failed += 1
            return
        cache_mod.append_section(
            self._cache_root, rec.cache_key,
            index=result.index, start=result.start, end=result.end,
            src_wav=result.dub_wav,
            timing=self._section_timing(rec, result),
        )
        rec.sections_done += 1
        rec.dubbed_s += max(0.0, float(result.end) - float(result.start))

    def _section_timing(self, rec: StreamRecord,
                        result: SectionResult) -> dict[str, Any] | None:
        """Per-section timing for the cache manifest, or None if it can't be
        built. Telemetry must never fail a section that already succeeded, so
        every failure here is swallowed — the dub is on disk either way."""
        try:
            return section_timing(
                index=result.index,
                start=result.start,
                end=result.end,
                job_dir=result.job_dir,
                wall_s=result.wall_s,
                new_speakers=result.new_speakers,
                buffer_ahead_s=self._buffer_ahead(rec),
                produced_at=time.time(),
                lines=result.lines,
                voiced_lines=result.voiced_lines,
            )
        except Exception:  # noqa: BLE001 — see docstring
            log.exception("Section timing failed for %s.", rec.job_id)
            return None

    def _buffer_ahead(self, rec: StreamRecord) -> float | None:
        """Seconds of continuous dubbed audio ahead of the playhead.

        The single number that predicts a freeze: the client pauses when it
        reaches audio that isn't there yet, so this crossing zero IS the stall.
        Read after the scheduler has added the new section to coverage (its
        loop is process -> coverage.add -> on_section), so it reflects the
        section just emitted. None before the scheduler exists (section 0).
        """
        sched = rec._scheduler
        if sched is None:
            return None
        t = rec.playhead.t
        gap = sched.coverage.first_uncovered(t)
        # None means everything from the playhead on is dubbed.
        return (rec.duration or t) - t if gap is None else max(0.0, gap - t)
