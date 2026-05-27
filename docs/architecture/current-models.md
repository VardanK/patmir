# Currently active models

Last updated: 2026-05-26

The single source of truth is `patmir.toml [pipeline]`. This table mirrors it for quick reference; if the two ever disagree, `patmir.toml` wins and this file is stale.

| Stage     | Active candidate id              | Model                                       | Device | Key knobs                                                              |
| --------- | -------------------------------- | ------------------------------------------- | ------ | ---------------------------------------------------------------------- |
| ingest    | `ffmpeg`                         | n/a (ffmpeg + ffprobe)                      | cpu    | 16 kHz mono PCM                                                        |
| separate  | `htdemucs`                       | `htdemucs_ft` (Demucs)                      | auto   | Silero VAD masking, 20 ms cosine fades on speech_vocals                |
| diarize   | `pyannote_3_1`                   | `pyannote/speaker-diarization-3.1`          | auto   | `min_speakers=0`, `max_speakers=8`, requires `HF_TOKEN` + license accept |
| emotion   | `emotion2vec_plus_base`          | `iic/emotion2vec_plus_base` (via funasr)    | auto   | 9→5 bin map; confidence floor 0.5 → neutral                            |
| asr       | `whisper_large_v3_mlx_fp16`      | `mlx-whisper large-v3` (fp16)               | mlx    | VAD post-filter + gap-filler for laughter-dominated regions + hallucination blocklist |
| translate | `openrouter_gemini_flash`        | `google/gemini-2.5-flash` via OpenRouter    | cloud  | emotion tag injected per sentence; requires `OPENROUTER_API_KEY`. NLLB-3.3B is the local fallback (`nllb_200_3.3b` — flip `[pipeline].translate` to revert). |
| tts       | `openrouter_gemini_flash_tts`    | `google/gemini-3.1-flash-tts-preview` via OpenRouter `/v1/audio/speech` | cloud  | per-speaker voice rotation (Kore / Puck / Aoede / Charon / Leda / Orus); inline `[happy]`/`[sad]`/`[angry]`/`[whispers]` audio tags from `s.emotion`; requires `OPENROUTER_API_KEY`. OmniVoice MLX (`omnivoice_mlx_q8`) is the local fallback — flip `[pipeline].tts` to revert. ElevenLabs Multilingual v2 registered as a future cloud option (needs Creator-tier key). |
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
