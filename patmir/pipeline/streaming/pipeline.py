"""Pipelined section driver: analyse the next section while this one translates.

Watch-along spends **44.2 s of a 153.9 s wall (29 %) in translate** — one
structured cloud round-trip per section, measured between 1.75 s and 13.34 s,
with the GPU completely idle throughout. Every other stage is GPU work. So the
sections were serial not because the work is serial but because nobody had
separated "waiting on a network" from "using the accelerator".

This driver overlaps exactly those two things:

    section k    [analyse][translate .......][speak]
    section k+1           [analyse]                 [translate ....][speak]
                          ^^^^^^^^
                          the GPU, busy during k's network wait

**The GPU is never used by two phases at once.** That is not a nicety, it is a
hard requirement: the TTS decode captures a CUDA graph, and capture fails — or
corrupts — if another thread allocates on the device while it runs. So one lock
serializes every GPU phase (`analyse` and `speak`), and `translate` is the only
phase that runs outside it. The win is bounded by
`min(translate_k, analyse_k+1)`, which on measured numbers is the whole
translate call: analysis is ~8 s against translate's ~4 s.

The registry is mutated by `analyse` (resolve-or-register), and the same lock
covers it, so identity resolution stays serialized in section order.

Emission order is preserved by construction: a section is emitted after its
`speak`, `speak` holds the lock, and the driver never speaks section k+1 before
k. Prefetch is one section deep — enough to cover a translate, and bounded so a
seek throws away at most one section's work.

Falls back to fully serial on a seek (the prefetched start no longer matches),
on cancellation, and when `section_prefetch` is off.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import JobCancelled
from patmir.pipeline.streaming.registry import SpeakerRegistry
from patmir.pipeline.streaming.section import (
    DEFAULT_SECTION_SECONDS,
    AnalyzedSection,
    SectionResult,
    analyze_section,
    speak_section,
    translate_section,
)

log = logging.getLogger(__name__)


class SectionPipeline:
    """Drop-in replacement for `process_section` as the scheduler's `process_fn`.

    `process(start, max_end)` has the same contract: return a finished
    `SectionResult`, or raise. What changes is that by the time it returns, the
    NEXT section's analysis may already be done.
    """

    def __init__(
        self,
        source: Path,
        cfg: PipelineConfig,
        registry: SpeakerRegistry,
        *,
        runs_root: Path | None = None,
        stage_cache: dict | None = None,
        cancel: threading.Event | None = None,
        bench: bool = False,
        source_lang: str | None = None,
        section_seconds: float = DEFAULT_SECTION_SECONDS,
        duration: float | None = None,
        prefetch: bool = True,
        index_start: int = 0,
    ) -> None:
        self.source = source
        self.cfg = cfg
        self.registry = registry
        self.runs_root = runs_root
        self.stage_cache = stage_cache
        self.cancel = cancel
        self.bench = bench
        self.source_lang = source_lang
        self.section_seconds = section_seconds
        self.duration = duration
        self.prefetch = prefetch

        # Serializes every phase that touches the GPU **or** the registry.
        # Non-negotiable: CUDA graph capture in the TTS decode cannot tolerate
        # another thread allocating on the device.
        self._gpu = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="patmir-prefetch")
        self._pending: Future[AnalyzedSection] | None = None
        self._pending_start: float | None = None
        # The caller owns numbering. A resumed stream must number PAST the
        # highest cached index or it overwrites `section_NNN.wav` while the
        # manifest keeps both entries — quality issue #22, which put the wrong
        # audio at two timestamps live and on every rewatch.
        self._index = int(index_start)
        # Telemetry: how much wall the overlap actually hid.
        self.stats: dict[str, Any] = {
            "prefetch_hits": 0, "prefetch_misses": 0, "prefetch_discarded": 0,
        }

    # -- lifecycle -- #
    def close(self) -> None:
        """Drop any in-flight prefetch and stop the worker."""
        pending, self._pending = self._pending, None
        self._pending_start = None
        if pending is not None:
            pending.cancel()
            # A running analysis cannot be interrupted (it is a torch forward),
            # so wait for it rather than leaving a thread on the GPU after the
            # stream is gone.
            try:
                pending.result(timeout=300)
            except Exception:  # noqa: BLE001
                pass
        self._pool.shutdown(wait=True)

    def __enter__(self) -> SectionPipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the scheduler's process_fn -- #
    def process(self, start: float, max_end: float | None = None) -> SectionResult:
        analyzed = self._take_analysis(start, max_end)

        # Kick the next section's analysis off BEFORE translating, so the GPU
        # has work for the whole network wait. `max_end` is deliberately not
        # forwarded: the next window's clamp is the scheduler's to compute from
        # coverage it has not updated yet, so a prefetch is only ever started
        # unclamped and is discarded if the scheduler asks for a clamped one.
        self._start_prefetch(analyzed.end)

        translate_section(analyzed, self.cfg, self.registry,
                          runs_root=self.runs_root, stage_cache=self.stage_cache,
                          cancel=self.cancel, bench=self.bench)

        with self._gpu:
            return speak_section(analyzed, self.cfg, self.registry,
                                 runs_root=self.runs_root,
                                 stage_cache=self.stage_cache,
                                 cancel=self.cancel, bench=self.bench)

    # -- internals -- #
    def _take_analysis(self, start: float,
                       max_end: float | None) -> AnalyzedSection:
        """The prefetched analysis if it is the one asked for, else a fresh one.

        A prefetch is only usable when the scheduler asks for the exact start it
        was launched at AND wants no clamp. After a seek the scheduler asks for
        somewhere else entirely, or for the same place with a `max_end` that
        would have changed where the section ends — either way the prefetched
        work is wrong and is thrown away rather than trusted.
        """
        pending, pending_start = self._pending, self._pending_start
        self._pending, self._pending_start = None, None

        usable = (
            pending is not None
            and max_end is None
            and pending_start is not None
            and abs(pending_start - start) < 0.01
        )
        if pending is not None and not usable:
            self.stats["prefetch_discarded"] += 1
            log.debug("Discarding prefetch at %.2fs (asked %.2fs, max_end=%s)",
                      pending_start or -1.0, start, max_end)
            pending.cancel()
            try:
                pending.result(timeout=300)   # cannot interrupt a torch forward
            except Exception:  # noqa: BLE001
                pass
            pending = None

        if usable and pending is not None:
            try:
                analyzed = pending.result()
                self.stats["prefetch_hits"] += 1
                return analyzed
            except JobCancelled:
                raise
            except Exception:  # noqa: BLE001
                # A failed prefetch must not fail the section: redo it inline
                # so the only cost is the wasted work.
                log.exception("Prefetched analysis at %.1fs failed; redoing.",
                              start)

        self.stats["prefetch_misses"] += 1
        self._index += 1
        with self._gpu:
            return self._analyze(self._index, start, max_end)

    def _start_prefetch(self, next_start: float) -> None:
        if not self.prefetch or self._pending is not None:
            return
        if self.cancel is not None and self.cancel.is_set():
            return
        # Nothing left to dub — do not spend a GPU pass on the tail.
        if self.duration is not None and next_start >= self.duration - 1.0:
            return
        self._index += 1
        index = self._index
        self._pending_start = next_start

        def _work() -> AnalyzedSection:
            with self._gpu:
                return self._analyze(index, next_start, None)

        self._pending = self._pool.submit(_work)

    def _analyze(self, index: int, start: float,
                 max_end: float | None) -> AnalyzedSection:
        return analyze_section(
            self.source, self.cfg, self.registry,
            index=index, start=start, target_seconds=self.section_seconds,
            runs_root=self.runs_root, stage_cache=self.stage_cache,
            cancel=self.cancel, bench=self.bench,
            source_lang=self.source_lang, max_end=max_end,
        )
