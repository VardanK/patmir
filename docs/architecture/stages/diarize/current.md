# Active diarize config

```toml
[pipeline]
diarize = "elevenlabs_scribe_v2"

[[stages.diarize.candidates]]
id                     = "elevenlabs_scribe_v2"
module                 = "patmir.pipeline.stages.diarize.elevenlabs_scribe"
api_key_env            = "ELEVENLABS_API_KEY"
model_id               = "scribe_v2"
num_speakers           = 4
timeout_seconds        = 600.0
```

Shares the API response with the `asr` candidate (same id, also `elevenlabs_scribe_v2`) — single upload per job, cached at `<job>/scribe/result.json`. The diarize candidate uploads first; the asr candidate reads from the cached JSON.

Picked 2026-05-27 after pyannote (3.1 + community-1) merged short cartoon turns from different characters into one `SPEAKER_NN` on masha-2-5min. Briefly reverted to pyannote community-1 on 2026-05-28 PM after the OmniVoice TTS swap, but ear-test showed the residual short-turn merging is still load-bearing — reverted back to Scribe v2 the same day.

## Cloud cost / scope

Diarize + asr share one Scribe upload per job. Cost is per minute of audio uploaded — small for the kinds of clips in scope. Drops the local HF-token gate (no `pyannote/speaker-diarization-community-1` license acceptance flow needed). Trade-off: no overlap-interval output (Scribe v2 doesn't emit overlaps; accepted regression — `quality-issues.md` #5).

## Local alternate (registered, disabled by selector)

`pyannote_3_1` stays registered with `model_id = pyannote/speaker-diarization-community-1` (newer embeddings than 3.1). Flip `[pipeline].diarize = "pyannote_3_1"` to revert to fully local diarization. Requires accepting the license on:

- `huggingface.co/pyannote/segmentation-3.0`
- `huggingface.co/pyannote/speaker-diarization-3.1`
- `huggingface.co/pyannote/speaker-diarization-community-1`

and either `export HF_TOKEN=hf_...` or `hf auth login`.

## Performance

- Scribe v2 upload + diarize/asr response: ~30–60 s for a 5-min clip; cost scales linearly with duration.
- Pyannote local fallback: ~1–3 min on a 20-min clip on Apple Silicon (MPS partial, some CPU fallback).
