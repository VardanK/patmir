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

      - translate → `openrouter_gemini_flash_batch`: translates the section's
        sentences in one batched call instead of one round-trip each — cuts the
        opening's translate time and keeps character voice coherent per scene.

      - separate → `htdemucs_mlx_fast`: base single-model Demucs on MLX, ~4×
        faster than the `htdemucs_ft` 4-model bag (which was ~34s/section, the
        throughput wall). Slightly more bleed, fine for watch-along.

    tts, asr and diarize are NOT overridden — they inherit the [pipeline]
    defaults (`higgs_v3_mlx_q8` + `qwen3_asr_1_7b_mlx_8bit` + `sortformer_mlx`).

    **tts override dropped 2026-08-23.** It used to point at
    `higgs_v3_mlx_q8_fast`, whose *only* difference from `higgs_v3_mlx_q8` was
    `max_retries=1` vs 2 — same repo, quant, temperature, top_p and seed. So it
    was never a faster model: a sentence that generates cleanly costs exactly
    the same under both, and the cap only bounded the worst case when the
    non-termination guardrail reseeds (measured 0 overflows / 25 on Reacher,
    0/37 masha, 0/60 Fallout). Sharing one candidate id with batch lets a single
    4B instance serve both modes from the warm-model cache instead of two —
    which matters on a 24 GB discrete GPU, where two would not fit. If
    overflows ever show up in watch-along, tune `max_retries` on the shared
    candidate rather than reintroducing a second one.

    All remaining overrides shrink per-section time so the stream keeps ahead of
    playback. See ADRs 2026-06-13-near-realtime-d3-spike and
    2026-06-15-local-asr-diarize-via-mlx-audio.

    The overrides above name MLX candidates, so they are only right for the
    Apple-Silicon profile. `layer="streaming"` lets an active machine profile
    replace them via `[pipeline.profiles.<name>.streaming]` — that sub-table is
    applied *after* this dict, so the CUDA profile can point `separate` at the
    torch candidate instead of `htdemucs_mlx_fast`, which does not exist off
    Apple Silicon."""
    return load_pipeline_config(config_path, layer="streaming", overrides={
        "separate": "htdemucs_mlx_fast",
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
    # One warm-model cache for the whole stream: stages are built on the first
    # section and reused after (kills the 11–13% per-section rebuild tax).
    stage_cache: dict = {}
    while t < duration - 0.5:
        result = process_section(
            source, cfg, prelude.registry,
            index=index, start=t, target_seconds=section_seconds, runs_root=runs_root,
            stage_cache=stage_cache,
        )
        yield result
        # Advance to this section's VAD-snapped end; guard against no-forward-
        # progress (e.g. a section with no usable VAD gap) by forcing a step.
        t = result.end if result.end > t + 0.5 else min(t + section_seconds, duration)
        index += 1
        if max_sections is not None and index > max_sections:
            break
