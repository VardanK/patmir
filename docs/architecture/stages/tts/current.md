# Active tts config

```toml
[pipeline]
tts = "omnivoice_mlx_q8"

[[eval.tts.candidates]]
id              = "omnivoice_mlx_q8"
enabled         = true
module          = "patmir.eval.candidates.tts.omnivoice_mlx"
repo_id         = "lightsofapollo/omnivoice-mlx-q8-g64"
ref_audio       = "samples/ref-hy.wav"     # global default ref (used as fallback)
ref_text        = "..."                     # ref's source transcript
num_steps       = 32
guidance_scale  = 2.0
language        = "None"
lang_code       = "None"
```

## Phase 2.x emotion routing

Per-sentence overrides at call time:

- **ref_audio / ref_text:** picked from `refs[(speaker_id, emotion)]` with fallback chain (exact → speaker-neutral → global default).
- **guidance_scale:** looked up in `[emotion_guidance]` per-emotion table (`neutral=0.7`, `happy/angry=0.9`, `sad=0.6`, `fearful=0.8`).

See [omnivoice-mlx.md](omnivoice-mlx.md) for the full API surface and `stages/tts/adapter.py` for the lookup logic.

## Fallback

When `refs` has no entry for a `speaker_id` at all (rare — a character with no qualifying turn), the adapter switches that sentence to the **piper_hye** fallback candidate. Same EA-floor quality from Phase 0 (3/3/2.9, 20/20 watchable).
