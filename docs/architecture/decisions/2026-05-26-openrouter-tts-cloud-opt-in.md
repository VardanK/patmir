# OpenRouter TTS (Gemini 3.1 Flash) as the active TTS

**Date:** 2026-05-26
**Status:** locked (per-stage cloud opt-in for TTS, via OpenRouter)

## Context

Two prior emotion-routing iterations through OmniVoice MLX (Phase 2.x rounds 1 and 2 of `guidance_scale` tuning) ear-tested flat. The first cloud-TTS attempt — ElevenLabs Multilingual v2 with Instant Voice Cloning — was blocked by ElevenLabs's Free-tier API restrictions (library voices and IVC both require Creator tier or above, ~$22/mo). See [elevenlabs-tts-cloud-opt-in](2026-05-26-elevenlabs-tts-cloud-opt-in.md) for the attempt.

While checking whether OpenRouter could route TTS as well as LLMs, an `openclaw/openclaw#71268` GitHub issue + OpenRouter's live `GET /api/v1/models?output_modalities=speech` endpoint confirmed: **OpenRouter added TTS in late 2025**. They expose an OpenAI-compatible `/v1/audio/speech` endpoint with the same `OPENROUTER_API_KEY` we already use for translate. Nine TTS models on offer at 2026-05-26; relevant ones:

| Model id (OpenRouter)                       | Cost (per OpenRouter, 2026-05-26) | Notes                                            |
| ------------------------------------------- | --------------------------------- | ------------------------------------------------ |
| `google/gemini-3.1-flash-tts-preview`       | $1 in / $20 out per 1M tokens     | **200+ inline audio tags**, 70+ languages, multi-speaker |
| `openai/gpt-4o-mini-tts-2025-12-15`         | $0.60 / 1M chars                  | 6 OpenAI voices, cheapest                        |
| `mistralai/voxtral-mini-tts-2603`           | $16 / 1M chars                    | Zero-shot voice cloning                          |
| `hexgrad/kokoro-82m`                        | $0.62 / 1M chars                  | Apache-2.0 open weights routed via OpenRouter    |
| `x-ai/grok-voice-tts-1.0`                   | $15 / 1M chars                    | xAI Grok Voice                                   |
| `zyphra/zonos-v0.1-{transformer,hybrid}`    | $7 / 1M chars                     | Zyphra Zonos                                     |
| `sesame/csm-1b`                             | $7 / 1M chars                     | Sesame CSM                                       |
| `canopylabs/orpheus-3b-0.1-ft`              | $7 / 1M chars                     | Orpheus 3B                                       |

## Decision

Activate **`google/gemini-3.1-flash-tts-preview`** via the new OpenRouter TTS candidate. Rationale:

- **Inline audio tags = explicit emotion control.** Gemini's TTS reads bracketed directives like `[happy]`, `[sad]`, `[angry]`, `[whispers]`, `[shouting]` as prosody hints for the following text. We inject the right tag per sentence based on `s.emotion` (mapped from emotion2vec's per-turn predictions). This is the cleanest cloud-side emotion-control story available — closest analog to what IndexTTS-2's 8-dim emotion vector would have given us, but actually accessible (no Armenian gap, no tier wall).
- **Same auth as translate.** Reuses `OPENROUTER_API_KEY` from `.env` — no separate vendor account, no tier wall (OpenRouter is pay-per-token from $5 minimum top-up).
- **Per-speaker voice rotation.** Gemini exposes prebuilt voices (Kore, Puck, Aoede, Charon, Leda, Orus, …). The candidate maintains a `(speaker_id → voice)` cache: speakers get voices from a rotation pool in encounter order; once bound, a speaker keeps the same voice for the rest of the run. Preserves Phase 2 multi-voice behavior without requiring voice cloning.
- **Cost.** ~$0.05-0.10 per 5-min clip; ~$0.40-0.80 per 20-min episode. Manageable.

`response_format = "pcm"` (Gemini TTS rejects `mp3`; candidate wraps the raw PCM in a WAV header at 24 kHz). Other models on the same candidate code (`openai/gpt-4o-mini-tts`, etc.) accept `mp3` and use ffmpeg to decode — the candidate handles both formats by switch.

## Implementation

- New candidate `patmir/eval/candidates/tts/openrouter.py`. OpenAI-compatible `/v1/audio/speech` over httpx. Same `synthesize(text, *, ref_audio, ref_text, guidance_scale, emotion)` contract as ElevenLabs — adapter forwards `ref_audio` to extract `speaker_id` from the ref-path's parent dir, and `emotion` to drive the inline-tag prefix.
- Three OpenRouter TTS candidates registered in `patmir.toml`: `openrouter_gemini_flash_tts` (active), `openrouter_gpt4o_mini_tts` + `openrouter_voxtral_mini_tts` (alternates).
- ElevenLabs candidate stays registered (`enabled = false`) with the original ADR; useful evidence for "tried this, hit tier wall." If a Creator-tier ElevenLabs key arrives later, flip the selector to switch.
- Per-emotion inline-tag table in the candidate, configurable from `patmir.toml [[eval.tts.candidates]].emotion_tags`. Defaults: `happy → [happy]`, `sad → [sad]`, `angry → [angry]`, `fearful → [whispers]`, `neutral → ""` (no tag).

## Consequences

- **Network dependency.** Translate (OpenRouter LLM) + TTS (OpenRouter TTS) are both cloud now. Going fully offline = flip both selectors back to local (`nllb_200_3.3b` + `omnivoice_mlx_q8`). Both local options stay registered.
- **Cost.** Per-run ~$0.05-0.10 for the 5-min masha clip; ~$0.40-0.80 for a full 20-min episode.
- **Voice character.** With the rotation pool, every speaker gets a distinct Gemini voice. The voice is *not* a clone of the source character — Masha's character voice is replaced with one of Gemini's prebuilt voices ("Kore", "Puck", etc.). This is a regression from OmniVoice's source-cloning behavior; cloud voices aren't your child's mom's voice. Accepted because emotion delivery matters more than identity match for the target use case (watching the show in Armenian, not preserving the source voice).
- **Emotion-tag quality.** Gemini's interpretation of `[happy]`, `[angry]`, etc. is not as fine-grained as ElevenLabs's per-emotion `voice_settings` (per user ear-test: "much better than what we had, not as good as ElevenLabs"). Accepted constraint until a Creator-tier ElevenLabs key or an Armenian-capable emotion-controllable local TTS arrives.

## Revisit trigger

- A Creator-tier ElevenLabs key becomes available — switch back to `elevenlabs_multilingual_v2` (or run a 30-sentence A/B before committing).
- Cost runs higher than expected.
- Gemini Flash TTS's Armenian quality regresses or has prosody bugs.
- An EA fine-tune of an emotion-controllable local TTS ships → revert to local.

## Source

- This conversation (2026-05-26 cloud-TTS escalation: ElevenLabs Free-tier walls → user finds `openclaw/openclaw#71268` → OpenRouter TTS confirmed live).
- OpenRouter live model list (`GET /api/v1/models?output_modalities=speech`).
- Related ADRs: [elevenlabs-tts-cloud-opt-in](2026-05-26-elevenlabs-tts-cloud-opt-in.md), [emotion-routed-through-ref-clip-binning](2026-05-26-emotion-routed-through-ref-clip-binning.md), [cloud-translate-via-openrouter](2026-05-26-cloud-translate-via-openrouter.md), [local-first-strict](2026-05-24-local-first-strict.md).
