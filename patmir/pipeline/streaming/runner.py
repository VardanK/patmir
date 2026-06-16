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
    """Build the watch-along pipeline config. Overrides vs the active pipeline:

      - tts → `omnivoice_mlx_bf16_fast`: bf16 OmniVoice + voice cloning at
        num_steps=16 (ear-test 2026-06-16: bf16@16 beat q8@16; Piper rejected as
        too low, num_steps=8 too low). OmniVoice diffusion is serial on MLX so at
        16 steps (RTF ~0.46) this does NOT clear realtime per-section on its own
        — watch-along relies on a pre-buffer / section pipelining. Replaced
        `elevenlabs_pool` (credits exhausted 2026-06-15).
      - translate → `openrouter_gemini_flash_batch`: translates the section's
        sentences in one batched call instead of one round-trip each — cuts the
        opening's translate time and keeps character voice coherent per scene.

      - separate → `htdemucs_mlx_fast`: base single-model Demucs on MLX, ~4×
        faster than the `htdemucs_ft` 4-model bag (which was ~34s/section, the
        throughput wall). Slightly more bleed, fine for watch-along.

    asr + diarize are NOT overridden here — they inherit the now-local
    [pipeline] defaults (nemotron_mlx + sortformer_mlx). All overrides shrink
    per-section time so the stream keeps ahead of playback. See ADRs
    2026-06-13-near-realtime-d3-spike and 2026-06-15-local-asr-diarize-via-mlx-audio."""
    return load_pipeline_config(config_path, overrides={
        "separate": "htdemucs_mlx_fast",
        "tts": "omnivoice_mlx_bf16_fast",
        "translate": "openrouter_gemini_flash_batch",
    })


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
