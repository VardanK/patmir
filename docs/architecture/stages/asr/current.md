# Active asr config

```toml
[pipeline]
asr = "whisper_large_v3_mlx_fp16"

[[stages.asr.candidates]]
id        = "whisper_large_v3_mlx_fp16"
module    = "patmir.pipeline.stages.asr.whisper_mlx"
model_id  = "mlx-community/whisper-large-v3-mlx"
dtype     = "float16"
language  = null   # auto-detect
word_timestamps = true
```

## Side-car configuration

- VAD post-filter via `ctx.extra["vad_intervals"]` — words outside any VAD interval get dropped (catches Whisper hallucinations on scream/laugh regions).
- Hallucination blocklist — phrases like `"Субтитры сделал DimaTorzok"`, `"thanks for watching"`, etc. removed at sentence assembly time (Phase 2 addition).
- Gap-filler runs after the primary pass: any turn ≥1.5 s containing only non-lexical words is re-ASR'd at higher temperature with the blocklist applied. Recovers ~10+ dialogue lines per masha episode.
