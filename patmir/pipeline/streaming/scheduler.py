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
from collections.abc import Callable
from dataclasses import dataclass

from patmir.pipeline.streaming.coverage import Coverage
from patmir.pipeline.streaming.section import SectionResult

# Minimum forward step when a section reports no usable progress (no VAD gap in
# the window) — keeps the loop from spinning on the same start.
_MIN_STEP = 5.0


@dataclass
class Playhead:
    """Latest known viewer position. Mutated by the server on each POST; read
    by the scheduler loop. Plain attributes — atomic enough under the GIL."""
    t: float = 0.0
    paused: bool = False


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
    ) -> None:
        self.duration = duration
        self.process_fn = process_fn
        self.playhead_fn = playhead_fn
        self.on_section = on_section
        self.section_seconds = section_seconds
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
        """The global start time of the highest-priority undubbed section, or
        None if the whole video is dubbed."""
        ph = self.playhead_fn()
        # 1) nearest gap ahead of the viewer.
        target = self.coverage.first_uncovered(after=ph.t, until=self.duration - 1.0)
        if target is not None:
            return target
        # 2) viewer's forward region is fully dubbed — backfill the earliest
        #    remaining gap anywhere so the video finishes (free rewatch cache).
        return self.coverage.first_uncovered(after=0.0, until=self.duration - 1.0)

    def run(self) -> None:
        """Drive the loop until the video is fully dubbed or `stop()` fires."""
        while not self._stop.is_set():
            start = self._next_start()
            if start is None:
                break  # fully covered
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
