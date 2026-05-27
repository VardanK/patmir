# Active tts config

```toml
[pipeline]
tts = "elevenlabs_multilingual_v2"

[[eval.tts.candidates]]
id               = "elevenlabs_multilingual_v2"
enabled          = true
module           = "patmir.eval.candidates.tts.elevenlabs"
model_id         = "eleven_multilingual_v2"
api_key_env      = "ELEVENLABS_API_KEY"
default_voice_id = "21m00Tcm4TlvDq8ikWAM"   # "Rachel" — fallback when no IVC ref
cleanup_voices   = true
timeout_seconds  = 90.0
output_format    = "mp3_44100_128"
```

API key in `.env` (auto-loaded by `patmir/env.py`): `ELEVENLABS_API_KEY=...`. The key must carry the `text_to_speech` + `voices_read` + `create_instant_voice_clone` permission scopes — IVC scope is the one that's easy to miss when creating a scoped key.

OpenRouter Gemini TTS (`openrouter_gemini_flash_tts`) stays registered as a cloud alternate — flip `[pipeline].tts = "openrouter_gemini_flash_tts"` to revert. OmniVoice MLX (`omnivoice_mlx_q8`) stays as the offline-fallback path. ElevenLabs Scribe v2 ASR is still registered `enabled = false` (separate stage).

## Phase 2.x emotion + voice routing

ElevenLabs combines per-speaker cloning + per-emotion expression in one shot:

- **Per-speaker voice (IVC):** the candidate uploads each `diarize/refs/<speaker>/*.wav` clip via `/v1/voices/add` and caches the resulting `voice_id` for that speaker. All emotions for the same speaker reuse the same upload (one IVC voice per character). Uploads are deleted at process exit (`atexit`) to free slots.
- **Per-emotion expression:** `voice_settings.stability` (lower → more variation) + `voice_settings.style` (higher → more dramatic). Mapped per bin in `_DEFAULT_VOICE_SETTINGS` inside the candidate:
  - `neutral` → stab 0.55 / style 0.20
  - `happy`   → stab 0.35 / style 0.55
  - `sad`     → stab 0.55 / style 0.40
  - `angry`   → stab 0.25 / style 0.70
  - `fearful` → stab 0.30 / style 0.60
- **Adapter call passes:** `ref_audio` (chooses IVC voice), `emotion` (chooses voice_settings preset). `ref_text` and `guidance_scale` are accepted for adapter uniformity and ignored — ElevenLabs has no ref-transcript input and `guidance_scale` is an OmniVoice concept.

## Fallback chain

- If IVC upload fails for a specific speaker (key tier, voice-slot cap, etc.), that speaker's lines fall back to `default_voice_id` (Rachel) and a single warning is logged. All other speakers keep their IVC clones.
- If `refs` has no entry for a `speaker_id` at all (rare — a character with no qualifying turn), the adapter switches that sentence to the **piper_hye** fallback candidate.

## Notes

- Output format `mp3_44100_128` works on every tier; the candidate decodes MP3 → WAV (PCM-16 mono 44.1k) via ffmpeg subprocess (~50 ms / sentence; ffmpeg is already a project dep). Override to `pcm_44100` on Creator+ to skip the decode.
- ElevenLabs voice cleanup runs on process exit; if the run is killed (SIGKILL), the IVC slots leak and need manual cleanup via `/v1/voices/{voice_id}` DELETE on the dashboard.
