# Test current pipeline on masha-5min — Shaping Notes

## Scope

Run the Phase 1 pipeline (as currently wired in `patmir.toml [pipeline]`) end-to-end on a 5-minute trimmed clip of `samples/masha.mkv`. Goal is an Armenian-dubbed `out.mkv` the author can play and judge by ear. No code changes; no model swaps.

## Decisions

- **Clip length:** 5 minutes (00:01:00 → 00:06:00 of `samples/masha.mkv`). Long enough to exercise every stage; short enough to iterate within an hour.
- **No CLI duration flag:** the `dub` subcommand only takes a video path + `--resume-*`; trimming is done offline with `ffmpeg -ss … -t 300 -c copy`.
- **Accept config as-is:** stack is `ffmpeg → htdemucs → qwen3_asr_1_7b_mlx_8bit → qwen3_6_27b_unsloth_ud_mlx_4bit → omnivoice → ffmpeg_pad_stretch → mkv_default`. The `enabled = false` on the omnivoice eval candidate is an eval-loop flag only; `patmir/pipeline/config.py` does not consult it when resolving `[pipeline].tts`.
- **Through `uv run`:** mlx_lm / mlx_qwen3_asr / omnivoice live in the uv venv, not system Python. Every patmir invocation goes through `uv run patmir …`.
- **Success criterion:** pipeline completes; author opens `out.mkv` and judges it by ear. No automated metrics.

## Context

- **Visuals:** None.
- **References:** `patmir/pipeline/runner.py`, `patmir/pipeline/stages/**`, `patmir.toml [pipeline]` block, prior Phase 1 spec at `agent-os/specs/2026-05-24-1704-phase-1-mvp-pipeline/`.
- **Product alignment:** matches `mission.md` — Eastern-Armenian end-to-end, local-first, audio dub (not subtitles) as the product.

## Standards Applied

- `agent-os/standards/index.yml` is empty — no standards to apply.
