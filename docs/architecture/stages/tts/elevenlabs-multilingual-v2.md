# ElevenLabs Multilingual v2 (tts — cloud)

- **Repo / version:** ElevenLabs cloud API
- **Apple Silicon / CUDA:** n/a (cloud)
- **Languages:** broad; **includes Armenian** + supports SSML-ish prompt syntax for pause / prosody / emotion control
- **License / gating:** ElevenLabs ToS + per-character cost (paid)
- **Voice cloning:** Instant Voice Cloning (cloud, ~30 s sample)
- **Status in pipeline:** **disabled** per the [local-first strict](../../decisions/2026-05-24-local-first-strict.md) policy
- **Notes:**
  - The primary cloud TTS candidate per `tech-stack.md` and the Phase 0 verdict's "what changes when the GPU workstation arrives" section.
  - Significantly above Piper's quality baseline; supports per-utterance emotion / pause / SSML — would actually unlock the emotion-control story OmniVoice can't deliver.
  - **Dialect must be ear-verified** — ElevenLabs likely doesn't distinguish Eastern vs Western in its language tag.
  - **Activation path:**
    1. Verify all local TTS options have been judged inadequate (currently we're at OmniVoice locked + Piper fallback; ear-tests have not yet been declared "inadequate").
    2. Write a new ADR documenting local exhaustion and the revisit trigger to go back to local.
    3. Flip `enabled = true` and set `[pipeline].tts` to the cloud candidate id.
