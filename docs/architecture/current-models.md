# Currently active models

Last updated: 2026-05-26

The single source of truth is `patmir.toml [pipeline]`. This table mirrors it for quick reference; if the two ever disagree, `patmir.toml` wins and this file is stale.

| Stage     | Active candidate id              | Model                                       | Device | Key knobs                                                              |
| --------- | -------------------------------- | ------------------------------------------- | ------ | ---------------------------------------------------------------------- |
| ingest    | `ffmpeg`                         | n/a (ffmpeg + ffprobe)                      | cpu    | 16 kHz mono PCM                                                        |
| separate  | `htdemucs`                       | `htdemucs_ft` (Demucs)                      | auto   | Silero VAD masking, 20 ms cosine fades on speech_vocals                |
| diarize   | `elevenlabs_scribe_v2`           | `scribe_v2` (ElevenLabs Speech-to-Text)     | cloud  | Shares the API response with the asr candidate (single upload, cached at `<job>/scribe/result.json`). Upload is downsampled to **16 kHz mono** (`upload_sample_rate`=16000, `upload_mono`=true) — ~6× faster than the 44.1 kHz stereo stem (measured 2026-06-14: 120s slice 658s/5.48× realtime as stereo vs 110s/0.92× as 16k-mono), no quality change since STT runs at 16 kHz internally; word timestamps are in seconds so the timeline is unaffected. See the D3 ADR. `num_speakers=4` tuned for masha-2-5min — 98% advertised speaker-label accuracy. Re-activated 2026-05-28 PM after a brief revert to pyannote community-1 caused an ear-tested quality regression on the active clip (short turns still merging). No overlap-interval output (regression vs pyannote; accepted). ADR: `2026-05-27-scribe-v2-for-asr-and-diarize`. Pyannote 3.1 / community-1 stays registered as the local fallback — flip `[pipeline].diarize` to revert. |
| emotion   | `emotion2vec_plus_base`          | `iic/emotion2vec_plus_base` (via funasr)    | auto   | 9→5 bin map; confidence floor 0.5 → neutral                            |
| asr       | `elevenlabs_scribe_v2`           | `scribe_v2` (ElevenLabs Speech-to-Text)     | cloud  | Same Scribe response the diarize candidate uses (cached at `<job>/scribe/result.json`); ASR pulls words from cache, no second upload. Sentence grouping via the existing `group_into_sentences` (Scribe doesn't emit prosodic segments). No `transcribe_slice` (no gap-filler — Scribe's per-call latency makes per-slice re-transcription cost-prohibitive). Local `whisper_large_v3_mlx_fp16` (mlx-whisper large-v3 fp16 + VAD post-filter + gap-filler + hallucination blocklist + non-lexical letter-run filter) stays registered as the offline fallback; `openrouter_gpt_audio` + alternates also still registered. Flip `[pipeline].asr` to revert. |
| translate | `openrouter_gemini_flash`        | `google/gemini-3.5-flash` via OpenRouter    | cloud  | Casual-register system prompt + per-sentence emotion hint; 3.5-flash locked 2026-05-26 evening after A/B vs 3.1-flash-lite (3.5 produced more emotive/playful Armenian). Claude Opus 4.7 is not a viable alternate (user-rejected). GPT-5 registered but unreliable (None-content errors). NLLB-3.3B is the offline fallback — flip `[pipeline].translate` to revert. Requires `OPENROUTER_API_KEY`. |
| tts       | `omnivoice_mlx_q8`               | `lightsofapollo/omnivoice-mlx-q8-g64` (MLX, int8 q8 g64) | mps    | Zero-shot per-speaker voice cloning via `diarize/refs/<speaker>/<emotion>.wav`. `guidance_scale` mapped per emotion bin in `[emotion_guidance]` (neutral 1.8, sad 2.0, happy 2.8, angry 3.0, fearful 2.6). No explicit emotion knob — emotion fidelity is bounded by what's audible in the source character's range (ADR `2026-05-26-emotion-routed-through-ref-clip-binning`). Switched back from `elevenlabs_multilingual_v2` on 2026-05-27 PM after ElevenLabs' captcha system blocked cloned voices of real celebrities (HTTP 403 `voice_access_denied` / `detected_captcha_voice`) on YouTube content. ElevenLabs stays registered + the in-code per-speaker fallback-to-default-voice retry stays as a hedge; flip `[pipeline].tts` to revert. Piper `hy_AM-gor-medium` is the no-ref fallback. |
| mux       | `ffmpeg_pad_stretch`             | n/a (ffmpeg + pyrubberband)                 | cpu    | `stretch_cap=0.20`, `music_bed_gain=0.6`, 50 ms afade in/out on synthesized track, per-sentence RMS loudness-match against source vocals (±12 dB cap) |
| output    | `mkv_default`                    | n/a (ffmpeg mux)                            | cpu    | Armenian default audio, original retained, Armenian SRT default subs   |

## Pipeline options (`patmir.toml [pipeline.options]`)

| Knob                | Default | Effect                                                    |
| ------------------- | ------- | --------------------------------------------------------- |
| `stretch_cap`       | 0.20    | Max ±20% time-stretch ratio per sentence (pitch-preserving) |
| `asr_int8`          | true    | ASR runs in int8 quantization when supported              |
| `keep_intermediate` | true    | Don't delete per-sentence WAVs, transcripts, etc. after run |
| `music_bed_gain`    | 0.6     | Accompaniment level under the AR voice (1.0 = unchanged)  |
| `voice_gain_db`     | -3.0    | Global trim on top of the per-sentence RMS loudness match. Negative = quieter. Set to `-3.0` on 2026-05-27 after ear-test on masha-2-5min (default `0.0` was too hot once the per-sentence match was added). |
| `min_ref_seconds`   | 1.5     | Shortest clean turn (after 100 ms edge trim) that qualifies as a `(speaker, emotion)` ref clip. Lowered from 3.0 → 1.5 on 2026-05-27 — ElevenLabs IVC accepts ≥1 s, and 3.0 was excluding most of `SPEAKER_00`'s short cartoon-pace turns on the masha clip. |
| `sentence_afade_seconds` | 0.05 | Per-sentence fade in/out on synthesized track (voice-leak guard, quality issue #4) |

## How to swap a model

1. Add or enable a candidate in the relevant `[[stages.<stage>.candidates]]` (or legacy `[[eval.<stage>.candidates]]`) section of `patmir.toml`.
2. Change `[pipeline].<stage>` to the new candidate id.
3. Update this file's row + `docs/architecture/stages/<stage>/current.md` + the relevant model file in `docs/architecture/stages/<stage>/<model>.md`.
4. If the swap is non-trivial (different model family, different language coverage, etc.), add a new ADR in `docs/architecture/decisions/`.
