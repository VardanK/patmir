# Active separate config

```toml
[pipeline]
separate = "htdemucs_mlx_vocals"

[[stages.separate.candidates]]
id          = "htdemucs_mlx_vocals"
module      = "patmir.pipeline.stages.separate.htdemucs_mlx"
model       = "04573f0d"      # htdemucs_ft vocals-specialist sub-model
vocals_only = true
shifts      = 1
overlap     = 0.25
```

**Vocals-only MLX** (2026-06-19) — runs only the `htdemucs_ft` **vocals-specialist
sub-model** (Demucs sig `04573f0d`) through `demucs-mlx`, instead of the full
4-model bag, and derives `accompaniment = mix − vocals`. The ft bag's vocals
output is 100% this sub-model (identity weight matrix), so vocals are
mathematically identical (ear-confirmed corr 0.998) at ~3.6× less stage time
(39→11 s/60s). ADR `2026-06-19-vocals-only-separation`. See `htdemucs-mlx.md`.

The full `htdemucs_mlx` bag (`model="htdemucs_ft"`) stays registered as a
fallback if the `mix − vocals` residual bed is ever dirty on a clip.

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
