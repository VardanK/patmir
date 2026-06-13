"""Streaming runner — forward section iterator (Slice 2 #9c).

After the prelude establishes the cast, this walks the rest of the video in
VAD-snapped sections, yielding dubbed chunks in order with voices kept
consistent across sections via the shared SpeakerRegistry.

This is the forward processing *engine*. The live playhead-driven scheduling
(prioritize the section at the playhead, keep processing while paused, jump on
seek) wraps this iterator at the client/server boundary (#10) — the engine just
needs to produce ordered sections faster than playback, which the throughput
spike confirmed it can.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

from patmir.pipeline.config import PipelineConfig, load_pipeline_config
from patmir.pipeline.streaming.prelude import PreludeResult
from patmir.pipeline.streaming.section import (
    DEFAULT_SECTION_SECONDS,
    SectionResult,
    process_section,
)


def streaming_config(config_path: Path) -> PipelineConfig:
    """Build the watch-along pipeline config: the active pipeline with the
    no-clone ElevenLabs pool swapped in for TTS (full quality + clears realtime
    + no celebrity-clone block). See ADR 2026-06-13-near-realtime-d3-spike."""
    return load_pipeline_config(config_path, overrides={"tts": "elevenlabs_pool"})


def source_duration(source: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        capture_output=True, text=True, check=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def stream_sections(
    source: Path,
    cfg: PipelineConfig,
    prelude: PreludeResult,
    *,
    section_seconds: float = DEFAULT_SECTION_SECONDS,
    runs_root: Path | None = None,
    max_sections: int | None = None,
) -> Iterator[SectionResult]:
    """Yield dubbed sections in order, starting from the prelude's window end."""
    duration = source_duration(source)
    t = prelude.window_end
    index = 1
    while t < duration - 0.5:
        result = process_section(
            source, cfg, prelude.registry,
            index=index, start=t, target_seconds=section_seconds, runs_root=runs_root,
        )
        yield result
        # Advance to this section's VAD-snapped end; guard against no-forward-
        # progress (e.g. a section with no usable VAD gap) by forcing a step.
        t = result.end if result.end > t + 0.5 else min(t + section_seconds, duration)
        index += 1
        if max_sections is not None and index > max_sections:
            break
