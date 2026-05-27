# ElevenLabs Multilingual v2 as the active TTS (cloud opt-in)

**Date:** 2026-05-26
**Status:** locked (per-stage cloud opt-in for TTS)

## Context

Phase 2.x routed emotion through OmniVoice MLX's existing levers — per-`(speaker, emotion)` ref clips + per-emotion `guidance_scale` + emotion-aware translation prompts. Two ear-tests on the masha-2-5min clip:

- **Round 1** (low guidance, 0.6-0.9): flat.
- **Round 2** (high guidance, 1.8-3.0, anchored above OmniVoice's 2.0 default): still flat. Text-level emotional cues (after the OpenRouter / Gemini Flash translate escalation landed) didn't translate to delivery either.

Conclusion: **OmniVoice MLX's zero-shot ref cloning normalizes prosody more aggressively than the model card implies.** No knob tuning gets the angry / happy / sad lines audibly distinct. The [emotion-routed-through-ref-clip-binning](2026-05-26-emotion-routed-through-ref-clip-binning.md) ADR explicitly accepted this as a possible outcome ("emotion fidelity is bounded by what's audible in the source character's range"); the ear-tests confirmed it.

Per the [local-first-strict](2026-05-24-local-first-strict.md) policy, local-TTS-set exhaustion triggers cloud opt-in.

## Decision

Activate **ElevenLabs `eleven_v3`** (NOT `eleven_multilingual_v2`) as the active TTS:

- `model_id = "eleven_v3"` + `language_code = "hy"`. First activation attempt used `eleven_multilingual_v2` (the natural default given the model name); ear-test exposed the model was reading Armenian script with English phonemes. Direct API probe (2026-05-26 evening) confirmed only `eleven_v3` accepts `language_code="hy"` — v2 / turbo_v2_5 / flash_v2_5 all reject it with `400 invalid_parameters`. The v2 model silently English-phonetizes Armenian script when the language code is omitted, which is why the first run "sort of worked" but sounded wrong.
- **Per-emotion `voice_settings`** drive expressive delivery:
  - `stability`: lower → more emotional variation. Intense emotions get 0.25-0.35; neutral / sad get 0.55.
  - `style`: higher → more dramatic. Intense emotions get 0.55-0.70; neutral 0.20.
  - `similarity_boost`: 0.80 across the board (stable identity).
- **Instant Voice Cloning (IVC)** preserves per-character voice identity. The Phase 2.x ref_picker already slices per-`(speaker, emotion)` ref clips at `diarize/refs/<speaker>/<emotion>.wav`. The ElevenLabs candidate uploads one clip per speaker (cached by speaker id), receives a `voice_id`, and uses that voice for every line of that speaker — emotion is reflected via `voice_settings`, not by uploading separate IVC voices per emotion. The previously-picked emotion clips are not wasted: they were already used by OmniVoice and remain available for re-evaluation if we revert.
- **Voice cleanup on shutdown** — uploaded IVC `voice_id`s are deleted on `atexit` so ElevenLabs voice-slot caps don't fill up across runs.
- The Phase 2.x `[emotion_guidance]` table (which targets OmniVoice's `guidance_scale`) becomes dormant under ElevenLabs — ElevenLabs ignores `guidance_scale` and reads `emotion` directly via the new `_supports_emotion` adapter probe.

## Implementation

- New candidate `patmir/eval/candidates/tts/elevenlabs.py`. No new Python deps — uses `httpx` already in the venv.
- TTS adapter (`patmir/pipeline/stages/tts/adapter.py`) extended with a third signature probe (`_supports_emotion`) so it forwards `emotion=` to candidates that declare it; OmniVoice / Piper signatures unchanged.
- ElevenLabs registered as a new `[[eval.tts.candidates]]` entry; `[pipeline].tts = "elevenlabs_multilingual_v2"`.
- API key from `ELEVENLABS_API_KEY` in `.env` (auto-loaded by `patmir/env.py` at CLI startup).

## Consequences

- **Per-call cost** — ElevenLabs is paid per character. A 20-min episode ≈ 200 sentences ≈ ~10k characters of Armenian ≈ ~$0.30-0.60 at Creator-tier rates. Manageable for personal use; budget-track if usage scales.
- **Network dependency.** With this active, both translate (OpenRouter) and tts (ElevenLabs) require network. Going fully offline = revert `[pipeline].tts` to `omnivoice_mlx_q8` and `[pipeline].translate` to `nllb_200_3.3b`.
- **IVC voice-slot management.** Lower-tier ElevenLabs subscriptions cap concurrent custom voices (typically 10-30). The candidate cleans up its uploaded voices on `atexit`; if a run crashes between upload and cleanup, voices leak and need manual cleanup via ElevenLabs UI / API.
- **Privacy.** Sentence text + ref clips are sent to ElevenLabs. Acceptable for personal-use scope.
- **Latency.** ~1-2 s per sentence (HTTP + synthesis). Comparable to or faster than OmniVoice MLX on the local Mac for the same content.
- **OmniVoice MLX stays registered** as the local fallback (`enabled = true`). Flip `[pipeline].tts` back to revert.
- **Piper stays as the no-IVC-ref fallback** at the adapter level if ElevenLabs IVC upload fails for a speaker (rare — but the candidate's `_voice_id_for_ref` catches upload errors and falls back to the configured default voice, not Piper, so we keep IVC-clone behavior). Piper is reachable by flipping the pipeline selector.

## Revisit trigger

- ElevenLabs ear-tested as no better than OmniVoice on emotion delivery (unlikely given explicit per-emotion settings, but possible). Revert to local + reopen the emotion-transfer issue.
- ElevenLabs dialect comes out Western Armenian. Revert.
- Cost runs higher than expected.
- An Eastern-Armenian fine-tune of an emotion-controllable local TTS (F5-TTS, IndexTTS-2) ships. Then revert to local.

## Source

- This conversation (2026-05-26 emotion-transfer cycle: local OmniVoice with 3 emotion-routing iterations all ear-tested flat).
- The local-TTS-side rejection ADRs: [omnivoice-locked-for-ea](2026-05-26-omnivoice-locked-for-ea.md), [emotion-routed-through-ref-clip-binning](2026-05-26-emotion-routed-through-ref-clip-binning.md).
- Local-first escalation policy: [local-first-strict](2026-05-24-local-first-strict.md).
