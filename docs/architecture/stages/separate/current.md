# Active separate config

```toml
[pipeline]
separate = "htdemucs_mlx"

[[stages.separate.candidates]]
id      = "htdemucs_mlx"
module  = "patmir.pipeline.stages.separate.htdemucs_mlx"
model   = "htdemucs_ft"
shifts  = 1
overlap = 0.25
```

**MLX** (2026-06-15) — Demucs `htdemucs_ft` run through `demucs-mlx`
(`demucs_mlx.Separator`), Apple-Silicon-native on the Mac GPU. Same model +
output contract as the PyTorch candidate; only the separation engine differs.
See `htdemucs-mlx.md`.

- Demucs runs on its native sample rate (44.1 kHz stereo).
- Silero VAD (shared `_split_vocals_with_vad` helper) produces `speech_vocals.wav`
  (20 ms cosine fades) and `non_speech_vocals.wav` (the inverse).
- `vad_intervals` is the `(start, end)` speech-window list downstream stages use
  as a post-filter.

## Watch-along

`streaming_config()` overrides separate → `htdemucs_mlx_fast` (base single-model
`htdemucs`, ~4× faster than the `_ft` 4-model bag), also on MLX.

## Fallback (registered)

PyTorch Demucs stays registered as `htdemucs` (+ `htdemucs_fast`) — flip
`[pipeline].separate` (or the streaming override) to revert to the MPS path.
