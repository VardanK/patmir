# ElevenLabs v3 (tts — cloud)

> Doc filename retained for git history; current model is `eleven_v3`.
> `eleven_multilingual_v2` is rejected for Armenian (see "Model selection" below).

- **Repo / version:** ElevenLabs cloud API, model `eleven_v3` (with `language_code="hy"`)
- **Apple Silicon / CUDA:** n/a (cloud)
- **Languages:** broad multilingual; **includes Armenian.** Dialect ear-verified at activation — produces Eastern Armenian when fed `hy_AM` text via IVC-cloned voices.
- **License / gating:** ElevenLabs ToS + per-character cost. The API key must carry the `text_to_speech` + `voices_read` + `create_instant_voice_clone` permission scopes — IVC scope is the easy one to miss when creating a scoped key (results in HTTP 401 `missing_permissions: create_instant_voice_clone` on `/v1/voices/add`).
- **Download source:** n/a (cloud)
- **API key:** read from `ELEVENLABS_API_KEY` env var (loaded from `.env` at CLI startup by `patmir/env.py`).
- **Integration surface:** `patmir/eval/candidates/tts/elevenlabs.py`
  ```python
  synthesize(
      text: str,
      *,
      ref_audio: Path | str | None = None,   # selects IVC-cloned voice (uploaded on first use, cached per speaker)
      ref_text:  str | None = None,           # ignored (ElevenLabs has no ref-transcript input)
      guidance_scale: float | None = None,    # ignored (OmniVoice concept)
      emotion: str | None = None,             # selects voice_settings preset
  ) -> bytes  # WAV (PCM-16 mono 44.1k)
  ```
  The TTS adapter probes the signature and forwards `ref_audio`, `ref_text`, `guidance_scale`, and `emotion` — this candidate accepts all four, ignoring the two that don't apply to ElevenLabs.
- **Status in pipeline:** **active** since 2026-05-26 (key scopes broadened — IVC permission added; supersedes the OpenRouter Gemini Flash TTS active path).
- **Last evaluated:** 2026-05-26 (activation re-run on masha-2-5min after key-scope fix)
- **Our experience:** First activation attempt (2026-05-26 morning) blocked on Free-tier limitations. Re-activated same day after the key was switched to a higher-tier scope including `create_instant_voice_clone`. First pipeline run used `eleven_multilingual_v2` (the natural default for Armenian-on-paper); ear-test exposed wrong phonemes — the model was reading Cyrillic Armenian as English. API probe confirmed only `eleven_v3` accepts `language_code="hy"`; v2 / turbo_v2_5 / flash_v2_5 all 400 with `Model 'X' does not support language_code 'hy'.` Switched to `eleven_v3` + explicit language code; pipeline tts stage runs in ~58 s on the masha-2-5min job (33 sentences). Ear-test pending.

## Model selection

| Model id                    | language_code="hy" | notes                                                |
| --------------------------- | ------------------ | ---------------------------------------------------- |
| `eleven_v3`                 | accepted           | **active.** Only ElevenLabs model that officially supports Armenian. |
| `eleven_multilingual_v2`    | 400 invalid_parameters | Reads Armenian text but applies English phonemes. Sounds wrong. |
| `eleven_turbo_v2_5`         | 400 invalid_parameters | Same problem.                                        |
| `eleven_flash_v2_5`         | 400 invalid_parameters | Same problem.                                        |

The candidate `build(config)` now defaults to `model_id="eleven_v3"` + `language_code="hy"`. Override via the toml entry if needed.

## Per-emotion voice_settings

Lower `stability` + higher `style` = more emotional variation. Defaults baked in the candidate (override via the `voice_settings` config block):

| Emotion  | stability | similarity_boost | style |
| -------- | --------- | ---------------- | ----- |
| neutral  | 0.55      | 0.80             | 0.20  |
| happy    | 0.35      | 0.80             | 0.55  |
| sad      | 0.55      | 0.80             | 0.40  |
| angry    | 0.25      | 0.80             | 0.70  |
| fearful  | 0.30      | 0.80             | 0.60  |

`similarity_boost` is held flat at 0.80 so identity (the IVC-cloned voice) stays consistent across emotions. `use_speaker_boost=True` is always set.

## Instant Voice Cloning (IVC)

The Phase 2.x ref_picker writes `diarize/refs/<speaker>/<emotion>.wav` for each `(speaker, emotion)` cell that has a clean ≥3 s turn. The ElevenLabs candidate uploads **one clip per speaker** as an IVC sample (keyed by `speaker_id` extracted from the ref path's parent dir name), receives a `voice_id`, and uses that voice for every line of that speaker. Per-emotion variation is then driven by `voice_settings`, not by uploading multiple voices per speaker.

Uploaded `voice_id`s are deleted via `atexit` so ElevenLabs voice-slot caps (10-30 depending on tier) don't fill up across runs. If a run crashes between upload and cleanup, voices leak and need manual cleanup via the ElevenLabs UI.

## Output

ElevenLabs returns `audio/wav` at `pcm_44100` (PCM-16 mono 44.1k). The candidate wraps the raw PCM in a WAV header in-memory; the mux stage resamples it to the source's sample rate during the time-stretch + pad pass.

## ADR

[2026-05-26 — ElevenLabs Multilingual v2 as the active TTS (cloud opt-in)](../../decisions/2026-05-26-elevenlabs-tts-cloud-opt-in.md)
