# Language support — Eastern Armenian

Eastern Armenian (`hy_AM`, ISO 639-3 `hye`) is the project's single output language. Western Armenian (`hyw`) outputs are automatically rejected. See [decisions/2026-05-24-eastern-armenian-only.md](decisions/2026-05-24-eastern-armenian-only.md) for why.

## Models grouped by Armenian support

### Produces Eastern Armenian (usable)

| Model                                          | Stage     | How                                                                   | Status                          |
| ---------------------------------------------- | --------- | --------------------------------------------------------------------- | ------------------------------- |
| `davit312/piper-TTS-Armenian (hy_AM-gor-medium)` | tts       | Native Eastern-Armenian Piper voice (community-trained)               | Fallback (was Phase 1 default)  |
| OmniVoice MLX (q8)                             | tts       | Zero-shot ref-clip cloning — EA in / EA out                           | **Active**                      |
| `facebook/nllb-200-distilled-600M`             | translate | NLLB target lang `hye_Armn` — Eastern-Armenian-labeled                | Superseded (kept for legacy)    |
| `facebook/nllb-200-3.3B`                       | translate | Same as 600M but larger; idiom + RU-narration noticeably better       | **Active**                      |

### Produces Eastern Armenian (untested at quality bar / future)

| Model                       | Stage     | Notes                                                                                  | Status                  |
| --------------------------- | --------- | -------------------------------------------------------------------------------------- | ----------------------- |
| `Gen2B/HyGPT-10b-it`        | translate | Gemma-2-9b base + 10B AR tokens. Likely strongest local translator. ~20 GB VRAM, GPU only. | Future-candidate (GPU re-eval) |
| `Helsinki-NLP/opus-mt-en-hy` | translate | OPUS-MT 2020. Explicit Eastern Armenian. ~80 MB. Modest BLEU but tiny.                  | Future-candidate (fast-path A/B) |
| `Helsinki-NLP/opus-mt-ru-hy` | translate | Same family for RU→hy.                                                                  | Future-candidate         |
| `google/madlad400-3b-mt`    | translate | MADLAD-400, includes Armenian.                                                          | Future-candidate (GPU re-eval) |
| Bark                        | tts       | Multilingual. Armenian dialect handling unverified.                                     | Future-candidate (GPU re-eval, ear-check before scoring) |
| ElevenLabs Multilingual v2  | tts       | Cloud. Armenian + SSML + emotion control. Dialect must be ear-verified.                | Future-candidate (cloud — local-first exception required) |
| Azure `hy-AM-AnahitNeural`, `hy-AM-HaykNeural` | tts | Cloud, named Eastern Armenian. Enterprise-tier auth.                          | Future-candidate (cloud) |

### Produces Western Armenian only (rejected on dialect)

| Model                                            | Stage     | Why rejected                            |
| ------------------------------------------------ | --------- | --------------------------------------- |
| `facebook/mms-tts-hyw`                           | tts       | Western Armenian only                   |
| `ArthurMalajyan/seamless-m4t-v2-large-asr-hyw`   | asr       | Western Armenian only (and ASR target language is wrong anyway — we transcribe source, not Armenian) |

### Does not produce Armenian (rejected on language)

| Model                              | Stage     | Notes                                                                                                     |
| ---------------------------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| Coqui XTTS-v2                      | tts       | 17 listed languages; Armenian not among them. Earlier project notes had this wrong; corrected in Phase 1. |
| F5-TTS / F5-TTS-MLX                | tts       | EN + ZH base; community FT for AR/FI/etc; no EA FT.                                                       |
| IndexTTS-2                         | tts       | ZH/EN/JA/ES only. PyTorch-only, no MLX port.                                                              |
| MeloTTS / OpenVoice v2 / Parler-TTS | tts       | None list Armenian.                                                                                       |
| `facebook/mms-tts-hye`             | tts       | **Does not exist on HF.** Returns 404. Only `mms-tts-hyw` exists.                                         |

### Wrong target task

| Model                                       | Why excluded                                                            |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `Chillarmo/whisper-large-v3-turbo-armenian` | Armenian-input ASR, but Patmir transcribes the *source* language (EN/RU), not Armenian. |

## Dialect ambiguity policy

Most multilingual TTS engines do not distinguish Eastern vs Western Armenian in their language tags. Any model that "supports Armenian" without specifying the dialect must be ear-verified by a fluent Eastern-Armenian speaker (the author) before adoption. If the output sounds Western, the candidate is rejected — full stop.

## Source

- `agent-os/product/mission.md` §"Dialect: Eastern Armenian only"
- `agent-os/product/tech-stack.md` §"Stage 4 — Eastern-Armenian TTS"
- `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/verdict.md`
- Phase 1 plan's post-Phase-0 model scan
- 2026-05-26 F5-TTS / IndexTTS-2 investigation (see `stages/tts/f5-tts-mlx.md`, `indextts-2.md`)
