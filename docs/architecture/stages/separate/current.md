# Active separate config

```toml
[pipeline]
separate = "htdemucs"

[[stages.separate.candidates]]
id = "htdemucs"
module = "patmir.pipeline.stages.separate.htdemucs"
model_id = "htdemucs_ft"
device = "auto"   # auto → mps if available, else cpu
```

- Demucs runs on its native sample rate (44.1 kHz stereo typically).
- Silero VAD is applied to the vocal stem to produce `speech_vocals.wav` (with 20 ms cosine fades) and `non_speech_vocals.wav` (the inverse — atmospheric vocals only).
- `vad_intervals` is the list of `(start, end)` speech windows downstream stages use as a post-filter.
