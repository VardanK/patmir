# htdemucs-mlx (separate)

- **Repo / library:** `demucs-mlx` (`demucs_mlx.Separator`) — Apple-Silicon
  MLX port of Demucs
- **Model:** `htdemucs_ft` (fine-tuned 4-model bag) for batch; base `htdemucs`
  single model for watch-along (`htdemucs_mlx_fast`)
- **Apple Silicon (MLX):** ✅ runs on the Mac GPU + unified memory (not the
  PyTorch MPS path). CUDA/CPU: n/a (use the PyTorch `htdemucs` candidate).
- **License / gating:** MIT (Demucs); non-gated
- **Integration surface:** `patmir/pipeline/stages/separate/htdemucs_mlx.py`
- **Status in pipeline:** **active** (2026-06-15) — replaced the PyTorch
  `htdemucs` candidate as the default separator. PyTorch candidate stays
  registered as the fallback.
- **Last evaluated:** 2026-06-15 (API probe + 90s end-to-end run).

## How it works

`Separator(model, shifts=1, overlap=0.25, split=True).separate_audio_file(path)`
returns `(info, stems_dict)`; stems are keyed `drums/bass/other/vocals` with
`(channels, samples)` float32 arrays at 44.1 kHz. The candidate sums the
non-vocal stems into `accompaniment.wav`, writes `vocals.wav`, then reuses the
PyTorch candidate's `_split_vocals_with_vad` (Silero VAD → `speech_vocals.wav`,
`non_speech_vocals.wav`, `vad_intervals`). Output is the identical
`SeparatedAudio` contract — only the separation engine changed.

## Knobs (`[[stages.separate.candidates]]`)

| Key | Default | Effect |
| --- | --- | --- |
| `model` | `htdemucs_ft` | `htdemucs_ft` (4-model bag, quality) or `htdemucs` (single, ~4× faster) |
| `shifts` | `1` | test-time augmentation shifts (more = slower, marginally better) |
| `overlap` | `0.25` | segment overlap |

## Revisit

- If MLX separation regresses vs PyTorch on a clip, flip
  `[pipeline].separate = "htdemucs"` (PyTorch MPS fallback).
- On a CUDA box, prefer the PyTorch `htdemucs` candidate (MLX is Apple-only).
