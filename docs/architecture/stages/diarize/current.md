# Active diarize config

```toml
[pipeline]
diarize = "pyannote_3_1"

[[stages.diarize.candidates]]
id           = "pyannote_3_1"
module       = "patmir.pipeline.stages.diarize.pyannote"
model_id     = "pyannote/speaker-diarization-3.1"
device       = "auto"
min_speakers = 0     # 0 = let pyannote estimate
max_speakers = 8
hf_token_env = "HF_TOKEN"
```

## One-time setup

1. Create an HF token at `huggingface.co/settings/tokens`.
2. Accept the license on all three gated repos:
   - `huggingface.co/pyannote/segmentation-3.0`
   - `huggingface.co/pyannote/speaker-diarization-3.1`
   - `huggingface.co/pyannote/speaker-diarization-community-1`
3. Either `export HF_TOKEN=hf_...` or run `hf auth login` (cached token works too).

The candidate raises a `RuntimeError` with an actionable message if no token is found and the model isn't already in the HF cache.

## Performance

~1-3 min on a 20-min cartoon. MPS coverage in pyannote 3.1 is partial — some ops fall back to CPU. Set `device = "cpu"` if MPS bugs surface.
