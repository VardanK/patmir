# References for masha-5min pipeline test

## Pipeline orchestration

- `patmir/cli.py` — `dub` subcommand entry; tails `progress.jsonl`.
- `patmir/pipeline/runner.py` — creates `runs/<ts-stem>/`, walks STAGE_ORDER, handles `--resume-from`.
- `patmir/pipeline/stage.py` — STAGE_ORDER = ("ingest", "separate", "asr", "translate", "tts", "mux", "output").
- `patmir/pipeline/config.py` — loads `[pipeline]` block from `patmir.toml`; resolves each stage id against its candidate list.
- `patmir/pipeline/context.py`, `patmir/pipeline/manifest.py` — per-job context + manifest writer.

## Stage implementations (matching current `[pipeline]`)

- ingest=ffmpeg → `patmir/pipeline/stages/ingest/ffmpeg.py`
- separate=htdemucs → `patmir/pipeline/stages/separate/htdemucs.py`
- asr=qwen3_asr_1_7b_mlx_8bit → `patmir/pipeline/stages/asr/qwen3_asr_mlx.py`
- translate=qwen3_6_27b_unsloth_ud_mlx_4bit → `patmir/pipeline/stages/translate/adapter.py` + `patmir/eval/candidates/translation/qwen3_instruct_mlx.py`
- tts=omnivoice → `patmir/pipeline/stages/tts/adapter.py` (eval candidate `patmir/eval/candidates/tts/omnivoice.py`)
- mux=ffmpeg_pad_stretch → `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py`
- output=mkv_default → `patmir/pipeline/stages/output/mkv.py`

## Prior spec

- `agent-os/specs/2026-05-24-1704-phase-1-mvp-pipeline/` — Phase 1 MVP spec; this run is the first end-to-end smoke test on a real clip.

## Voice cloning reference

- OmniVoice ref clip: `eval_runs/20260524-160019/tts/piper_hye/hy-cartoon-01.wav` (Eastern-Armenian Piper output saying "Արկածի գնանք").
