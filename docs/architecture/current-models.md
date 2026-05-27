# Currently active models

Last updated: 2026-05-26

The single source of truth is `patmir.toml [pipeline]`. This table mirrors it for quick reference; if the two ever disagree, `patmir.toml` wins and this file is stale.

| Stage     | Active candidate id              | Model                                       | Device | Key knobs                                                              |
| --------- | -------------------------------- | ------------------------------------------- | ------ | ---------------------------------------------------------------------- |
| ingest    | `ffmpeg`                         | n/a (ffmpeg + ffprobe)                      | cpu    | 16 kHz mono PCM                                                        |
| separate  | `htdemucs`                       | `htdemucs_ft` (Demucs)                      | auto   | Silero VAD masking, 20 ms cosine fades on speech_vocals                |
| diarize   | `pyannote_3_1`                   | `pyannote/speaker-diarization-3.1`          | auto   | `min_speakers=0`, `max_speakers=8`, requires `HF_TOKEN` + license accept |
| emotion   | `emotion2vec_plus_base`          | `iic/emotion2vec_plus_base` (via funasr)    | auto   | 9→5 bin map; confidence floor 0.5 → neutral                            |
| asr       | `openrouter_gpt_audio`           | `openai/gpt-audio` via OpenRouter `/v1/chat/completions` (multimodal audio) | cloud  | Audio sent as base64 WAV in `input_audio` content blocks; chunked at `chunk_seconds=240`; word-level timestamps reconstructed locally by MMS-FA forced aligner (`_aligner.py`). Voxtral Small 24B + GPT Audio Mini registered as disabled alternates. ElevenLabs Scribe v2 registered as a policy-gated future option. Local `whisper_large_v3_mlx_fp16` (mlx-whisper) is the offline fallback — flip `[pipeline].asr` to revert. Requires `OPENROUTER_API_KEY`. |
| translate | `openrouter_gemini_flash`        | `google/gemini-3.5-flash` via OpenRouter    | cloud  | Casual-register system prompt + per-sentence emotion hint; 3.5-flash locked 2026-05-26 evening after A/B vs 3.1-flash-lite (3.5 produced more emotive/playful Armenian). Claude Opus 4.7 is not a viable alternate (user-rejected). GPT-5 registered but unreliable (None-content errors). NLLB-3.3B is the offline fallback — flip `[pipeline].translate` to revert. Requires `OPENROUTER_API_KEY`. |
| tts       | `elevenlabs_multilingual_v2`     | `eleven_v3` (ElevenLabs) + `language_code="hy"` | cloud  | **Only `eleven_v3` accepts Armenian as a `language_code`** — older multilingual models (v2 / turbo_v2_5 / flash_v2_5) reject `hy` and silently produce English-phoneme Armenian. Per-speaker IVC clones (one upload per character, reused across emotions, cleaned up on exit); per-emotion `voice_settings` (stability + style) mapped per bin in the candidate; default "Rachel" voice for speakers without a clean ref. Requires `ELEVENLABS_API_KEY` with `text_to_speech` + `voices_read` + `create_instant_voice_clone` scopes. OpenRouter Gemini Flash TTS stays registered as a cloud alternate; OmniVoice MLX is the offline fallback — flip `[pipeline].tts` to revert. |
| mux       | `ffmpeg_pad_stretch`             | n/a (ffmpeg + pyrubberband)                 | cpu    | `stretch_cap=0.20`, `music_bed_gain=0.6`, 50 ms afade in/out on synthesized track |
| output    | `mkv_default`                    | n/a (ffmpeg mux)                            | cpu    | Armenian default audio, original retained, Armenian SRT default subs   |

## Pipeline options (`patmir.toml [pipeline.options]`)

| Knob                | Default | Effect                                                    |
| ------------------- | ------- | --------------------------------------------------------- |
| `stretch_cap`       | 0.20    | Max ±20% time-stretch ratio per sentence (pitch-preserving) |
| `asr_int8`          | true    | ASR runs in int8 quantization when supported              |
| `keep_intermediate` | true    | Don't delete per-sentence WAVs, transcripts, etc. after run |
| `music_bed_gain`    | 0.6     | Accompaniment level under the AR voice (1.0 = unchanged)  |
| `voice_gain_db`     | 0.0     | Extra dB applied to the AR voice over its native level    |

## How to swap a model

1. Add or enable a candidate in the relevant `[[stages.<stage>.candidates]]` (or legacy `[[eval.<stage>.candidates]]`) section of `patmir.toml`.
2. Change `[pipeline].<stage>` to the new candidate id.
3. Update this file's row + `docs/architecture/stages/<stage>/current.md` + the relevant model file in `docs/architecture/stages/<stage>/<model>.md`.
4. If the swap is non-trivial (different model family, different language coverage, etc.), add a new ADR in `docs/architecture/decisions/`.
