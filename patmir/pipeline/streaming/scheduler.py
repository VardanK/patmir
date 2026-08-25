"""Playhead-driven streaming scheduler (Slice 3 #10).

`stream_sections` (the #9 engine) is a strictly-forward iterator: it always
processes the next section after the last. Watch-along needs the engine to
track the *viewer* — process the section at the playhead first, jump to a new
region on seek, and keep filling forward while paused. This scheduler is that
control loop. It owns the `Coverage` bookkeeping and decides which global start
time to hand to `process_section` next; the heavy per-section work
(analyze→identity→translate→tts→mux) is unchanged.

The loop, each tick:

  1. Read the latest playhead (a float seconds + paused flag) via `playhead_fn`.
  2. Ask `Coverage.first_uncovered(after=playhead)` for the nearest undubbed
     point ahead of the viewer — that's the highest-priority section. If the
     viewer is sitting in already-dubbed audio (or the whole tail ahead is
     done), fall back to the earliest remaining gap anywhere (`after=0`) so we
     still finish the video (and fully populate the rewatch cache).
  3. `process_fn(start)` → a `SectionResult`; record its `[start, end)` in
     coverage and emit it via `on_section`.
  4. Stop when coverage spans the whole duration or `stop()` is called.

Everything heavy is injected (`process_fn`, `playhead_fn`), so the priority
logic is unit-tested with fakes — no models, no audio, no network.

Thread model: the scheduler runs the loop on one worker thread (the server's
`asyncio.to_thread`). `playhead_fn` is read on that thread but written from the
event-loop thread when a playhead POST arrives; a single float/bool read under
the GIL is atomic enough for this (a slightly stale playhead only costs one
mis-prioritized section, self-corrects next tick).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from patmir.pipeline.streaming.coverage import Coverage
from patmir.pipeline.streaming.section import SectionResult

# Minimum forward step when a section reports no usable progress (no VAD gap in
# the window) — keeps the loop from spinning on the same start.
_MIN_STEP = 5.0
# Watch-along only dubs a bounded window AHEAD of the playhead, never the whole
# video — dubbing minute 30 while you watch minute 2 is wasted ElevenLabs/Scribe
# spend (each section is paid cloud TTS + STT). When the buffer is full this far
# ahead, the scheduler idles until the playhead advances.
DEFAULT_LOOKAHEAD_SECONDS = 120.0
# If no playhead update arrives for this long, the viewer has left (tab closed,
# crash) — stop processing so an abandoned stream doesn't keep burning credits.
DEFAULT_ABANDON_SECONDS = 45.0


@dataclass
class Playhead:
    """Latest known viewer position. Mutated by the server on each POST; read
    by the scheduler loop. Plain attributes — atomic enough under the GIL."""
    t: float = 0.0
    paused: bool = False
    updated_at: float = 0.0   # wall-clock of the last update (abandon detection)


class StreamScheduler:
    def __init__(
        self,
        *,
        duration: float,
        prelude_end: float,
        process_fn: Callable[[float], SectionResult | None],
        playhead_fn: Callable[[], Playhead],
        on_section: Callable[[SectionResult], None] | None = None,
        section_seconds: float = 30.0,
        lookahead_seconds: float = DEFAULT_LOOKAHEAD_SECONDS,
        abandon_seconds: float = DEFAULT_ABANDON_SECONDS,
        idle_fn: Callable[[], bool] | None = None,
        alive_fn: Callable[[], float] | None = None,
    ) -> None:
        self.duration = duration
        self.process_fn = process_fn
        self.playhead_fn = playhead_fn
        self.on_section = on_section
        self.section_seconds = section_seconds
        self.lookahead_seconds = lookahead_seconds
        self.abandon_seconds = abandon_seconds
        # Called when caught up (nothing urgent to dub). Returns True to keep
        # waiting for the playhead to advance, False to end the stream. Default
        # idles briefly and ends on stop() or viewer-abandonment. Injectable for
        # tests so run() terminates deterministically without wall-clock waits.
        self._idle_fn = idle_fn or self._default_idle
        # "Is the viewer still there?" — deliberately NOT "is the playhead
        # moving?". The browser client stops POSTing the playhead while paused,
        # so keying abandonment off playhead movement ended the stream after
        # `abandon_seconds` of pause and `submit()` then handed the viewer back
        # the finished record instead of restarting it: the stream was
        # unrecoverably dead at the buffer edge. Polling `/sections` continues
        # while the tab is open, so it separates "paused" from "left".
        # None keeps the old playhead-based behaviour (tests, bench).
        self._alive_fn = alive_fn
        self.coverage = Coverage()
        # The prelude already dubbed [0, prelude_end].
        if prelude_end > 0:
            self.coverage.add(0.0, prelude_end)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

    def _next_start(self) -> float | None:
        """Start time of the highest-priority section to dub now, or None if the
        buffer is already full to `lookahead_seconds` past the playhead (so we
        should idle rather than race ahead dubbing the whole video)."""
        ph = self.playhead_fn()
        target = self.coverage.first_uncovered(after=ph.t, until=self.duration - 1.0)
        if target is None:
            return None
        # Buffered far enough ahead of the viewer → don't dub further yet.
        if target > ph.t + self.lookahead_seconds:
            return None
        return target

    def _default_idle(self) -> bool:
        """Idle while caught up. End the stream if stopped or the viewer left.

        A paused viewer has NOT left: the scheduler keeps filling forward, which
        is what this class's docstring always claimed and what the code did not
        do.
        """
        last_seen = self._alive_fn() if self._alive_fn is not None else None
        if last_seen is None:
            ph = self.playhead_fn()
            last_seen = ph.updated_at
        if last_seen and (time.time() - last_seen) > self.abandon_seconds:
            return False
        # Interruptible 2s wait; returns True (keep waiting) unless stop() fired.
        return not self._stop.wait(2.0)

    def run(self) -> None:
        """Track the playhead, dubbing a bounded window ahead. Ends on stop(),
        viewer-abandonment, or when the rest of the video within range is done."""
        while not self._stop.is_set():
            start = self._next_start()
            if start is None:
                if not self._idle_fn():
                    break          # stopped or abandoned
                continue           # caught up; wait for the playhead to advance
            result = self.process_fn(start)
            if result is None:
                # Processing failed for this window; force progress so we don't
                # retry the same start forever.
                self.coverage.add(start, start + _MIN_STEP)
                continue
            end = result.end
            if end <= start + 0.5:
                end = min(start + self.section_seconds, self.duration)
            self.coverage.add(start, end)
            if self.on_section is not None and result.dub_wav is not None:
                self.on_section(result)
