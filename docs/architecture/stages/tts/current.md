# Active tts config

```toml
[pipeline]
tts = "openrouter_gemini_flash_tts"

[[eval.tts.candidates]]
id              = "openrouter_gemini_flash_tts"
enabled         = true
module          = "patmir.eval.candidates.tts.openrouter"
model           = "google/gemini-3.1-flash-tts-preview"
api_key_env     = "OPENROUTER_API_KEY"
voice           = "Kore"
voices          = ["Kore", "Puck", "Aoede", "Charon", "Leda", "Orus"]
response_format = "pcm"     # Gemini TTS rejects mp3; candidate wraps PCM in WAV @ 24 kHz
speed           = 1.0
timeout_seconds = 90.0
```

API key in `.env` (auto-loaded by `patmir/env.py`): `OPENROUTER_API_KEY=...` — the same key used by the translate stage.

OmniVoice MLX (`omnivoice_mlx_q8`) stays registered as the **local fallback** — flip `[pipeline].tts = "omnivoice_mlx_q8"` to revert to fully-offline operation. ElevenLabs is registered (`enabled = false`) as a future cloud option requiring a Creator-tier key.

## Phase 2.x emotion routing

The TTS adapter passes per-sentence overrides at call time:

- **ref_audio / ref_text:** picked from `refs[(speaker_id, emotion)]` with a 4-step fallback chain (exact → speaker-neutral → any-emotion-for-same-speaker → global). The first-picked clip per *speaker* drives the IVC clone identity; emotion-driven expression comes from `voice_settings` (ElevenLabs) or `guidance_scale` (OmniVoice fallback path).
- **emotion** (new in Phase 2.x): forwarded to candidates that accept it. ElevenLabs uses it to look up its per-emotion `voice_settings` (stability + style); OmniVoice ignores the kwarg and uses `guidance_scale` instead.
- **guidance_scale:** looked up in `[emotion_guidance]` per-emotion table (`neutral=1.8`, `sad=2.0`, `happy=2.8`, `angry=3.0`, `fearful=2.6` — anchored at OmniVoice's `2.0` constructor default). Used by OmniVoice; ignored by ElevenLabs and Piper.

See [omnivoice-mlx.md](omnivoice-mlx.md) for the full API surface and `stages/tts/adapter.py` for the lookup logic.

## Fallback

When `refs` has no entry for a `speaker_id` at all (rare — a character with no qualifying turn), the adapter switches that sentence to the **piper_hye** fallback candidate. Same EA-floor quality from Phase 0 (3/3/2.9, 20/20 watchable).
