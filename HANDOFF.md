# Handoff — End-to-end test the Patmir product (local pipeline)

## Goal
Test Patmir **end-to-end** in both execution modes and confirm it produces a watchable Eastern-Armenian dub:
1. **Batch / Quality mode** (`patmir dub <video>`) — the primary product.
2. **Watch-along mode** (D3 streaming via `patmir ui` + Chrome extension).

The author's real content is **Russian cartoons (masha)** for his hearing-aid-wearing son. English clips have been tested; **Russian has NOT been ear-tested yet** — that's the most important gap.

## What changed in the previous session (context)
ElevenLabs credits were exhausted, so the pipeline was moved **fully local except translation**. New ADR: `docs/architecture/decisions/2026-06-15-local-asr-diarize-via-mlx-audio.md`. All architecture docs are updated (current-models, stage docs, quality-issues). The `mlx-audio` library (github.com/Blaizzy/mlx-audio) is installed and now backs ASR, diarize, and TTS.

Current pipeline (read live from `patmir.toml`; verify with the snippet below):

| Stage | Batch candidate | Model | Watch-along override |
|---|---|---|---|
| ingest | `ffmpeg` | ffmpeg | — |
| separate | `htdemucs_mlx` | Demucs htdemucs_ft (MLX) | `htdemucs_mlx_fast` (base model) |
| diarize | `sortformer_mlx` | NVIDIA Sortformer 4-spk (MLX) | — |
| emotion | `emotion2vec_plus_base` | emotion2vec | — |
| asr | `nemotron_mlx` | Nemotron 3.5 streaming (MLX) | — |
| translate | `openrouter_gemini_flash` | gemini-3.5-flash (**CLOUD**) | `openrouter_gemini_flash_batch` |
| tts | `omnivoice_mlx_bf16` | OmniVoice **bf16**, 32 steps | `omnivoice_mlx_bf16_fast` (16 steps) |
| mux | `ffmpeg_pad_stretch` | ffmpeg | — |
| output | `mkv_default` | MKV | — |

New code files: `patmir/pipeline/stages/asr/nemotron_mlx.py`, `.../diarize/sortformer_mlx.py`, `.../separate/htdemucs_mlx.py`. TTS uses the existing `patmir/eval/candidates/tts/omnivoice_mlx.py` (repo_id swapped to bf16). **bf16 was ear-verified as the best TTS** (bf16@32 > q8@32 > bf16@16 > q8@16).

## Critical things before/while testing
- **Translate is the only cloud stage** — needs `OPENROUTER_API_KEY` (already in `.env`, auto-loaded by `patmir/env.py`). No ElevenLabs anywhere.
- **Restart `patmir ui` before testing watch-along** — a running server caches imported code; stale servers caused confusion last session.
- **Cache:** results are keyed by source + pipeline-config hash at `~/.patmir/cache/<key>/`. Config changes auto-invalidate. To force a clean run, delete `~/.patmir/cache/<key>/` (and `~/.patmir/downloads/<key>/` for URLs).
- **Watch-along is NOT truly realtime.** TTS (OmniVoice diffusion, serial on MLX) plus the serial per-section stages exceed playback, so the player will **stall at section boundaries**. The known fix (NOT yet built) is a **pre-buffer / warm-model loop / section pipelining** — see "open follow-ups." Don't treat stalls as a bug; treat smoothness as out of scope until that's built.

## How to test

### Batch mode
```
uv run patmir dub /path/to/clip.mp4
# prints the run dir, e.g. runs/<id>/
open runs/<id>/out.mkv        # video + Armenian audio + hy subs
afplay runs/<id>/dub.wav      # just the dub
```
Verify in `runs/<id>/`:
- `transcript.json` — sentences split sensibly, `language` correct (en/ru), word timings present.
- `diarize/turns.json` — distinct speakers, short turns not merged, overlaps present.
- `translated.json` — **`target_text`** populated with Armenian (NOTE: the field is `target_text`, not `translated_text`).
- `tts/*.wav` + `dub.wav` non-silent; voices distinct per character.

### Watch-along mode
```
uv run patmir ui          # FastAPI server, port 8765 (matches extension default)
# Chrome: chrome://extensions -> Developer mode -> Load unpacked -> ./extension
# open a YouTube video -> Patmir popup -> "Watch-along"
```
Confirm: sections produce real Armenian speech (not music-bed-only), voices stay consistent across sections (ECAPA registry), no ElevenLabs calls.

### The key gap: Russian
Run a **Russian (masha)** clip through batch mode and ear-judge. If transcription/dub is bad, the likely culprit is Nemotron's Russian — flip `[pipeline].asr = "whisper_large_v3_mlx_fp16"` and re-run to compare. whisper is the documented fallback.

## Verify current config (paste into the new session)
```
uv run python -c "
from pathlib import Path
from patmir.pipeline.config import load_pipeline_config
from patmir.pipeline.streaming.runner import streaming_config
b=load_pipeline_config(Path('patmir.toml')).entries
w=streaming_config(Path('patmir.toml')).entries
for s in ['ingest','separate','diarize','emotion','asr','translate','tts','mux','output']:
    print(f'{s:10} {b[s][\"id\"]}')
print('watch-along overrides:', {s:w[s]['id'] for s in b if w[s]['id']!=b[s]['id']})
"
```

## One-line reverts if a stage misbehaves
- ASR: `[pipeline].asr = "whisper_large_v3_mlx_fp16"` (Russian quality)
- Diarize: `[pipeline].diarize = "pyannote_3_1"` (needs HF_TOKEN; >4 speakers)
- Separate: `[pipeline].separate = "htdemucs"` (PyTorch; if MLX regresses)
- TTS: `[pipeline].tts = "omnivoice_mlx_q8"` (q8 fallback)

## Open follow-ups (not blocking the E2E test, but known)
1. **Watch-along realtime** — build the pre-buffer / warm-model loop / section pipelining so playback doesn't stall. Per-section cost is dominated by cold model rebuilds (~10s) + TTS (~RTF 0.46 of dubbed audio). `process_section` rebuilds all models twice per section (`patmir/pipeline/streaming/section.py`).
2. **Nemotron emits no punctuation** — the candidate re-segments timed words with gap + 10s/40-word caps (`_segment_words`). If transcripts ever come out as one giant run-on, that's regressed.
3. **Uncommitted work** — all the above is uncommitted on `main`. Repo convention: commit directly to main (no PRs). Commit when the E2E test passes. Modified: `patmir.toml`, `streaming/runner.py`, many `docs/`; new: the 3 candidate `.py` files + 4 stage docs + the ADR.
4. **Spike cleanup** — `spike/OmniVoice-MLX/` (ailuntx clone + venv, large), `spike/cmp_*.wav`, `spike/eartest_*.wav`, `spike/q4_*.wav`, `spike/mlx_audio_asr_diar_probe.py` are throwaway; delete when done. (A cached `~/.cache/huggingface/.../OmniVoice-4bit/config.json` was patched during testing — harmless, that repo doesn't load in our pipeline anyway.)

## First steps for the new session
1. Run the verify-config snippet above; confirm it matches the table.
2. Pick a short clip (English first to sanity-check, then **Russian**).
3. Run batch mode, inspect artifacts + ear-test the dub.
4. Restart `patmir ui`, test watch-along (expect boundary stalls — judge dub *quality*, not smoothness).
5. Report what works / what's broken; decide on commit + whether to build the watch-along pre-buffer.
