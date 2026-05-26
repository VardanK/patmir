# Piper hy_AM-gor-medium as the TTS floor

**Date:** 2026-05-24
**Status:** locked as **fallback** since 2026-05-25 (no longer the default TTS — OmniVoice took over)

## Context

Phase 0 evaluated Eastern-Armenian TTS candidates on a 20-utterance eval set covering cartoon dialogue, narration, idioms, and proper nouns. The de-facto local baseline was the community-trained Piper voice `davit312/piper-TTS-Armenian (hy_AM-gor-medium)` — ~63 MB ONNX, CPU-only, fast.

Other candidates explicitly ruled out at the time:

- `facebook/mms-tts-hye` — does not exist on HuggingFace (only the Western-Armenian `mms-tts-hyw`).
- Coqui XTTS-v2 — its current model card lists 17 supported languages; Armenian is not among them. Earlier notes had been wrong about this.
- Bark — dialect handling unverified, deferred to GPU re-eval.
- MMS / SeamlessM4T / OpenVoice etc. — none publish a `hy_AM` voice.

## Decision

`piper_hye (hy_AM-gor-medium)` is the Phase 1 default TTS and is documented as the **quality floor** for Eastern Armenian. The "watchable for hours?" rubric passed 20/20 across all categories (cartoon, narration, idiom, proper) — uniform 3/3/2.9 on intelligibility/naturalness/prosody. Acceptable but not delightful.

## Consequences

- Piper has no voice variety (one fixed speaker) and no emotion control.
- Phase 2's multi-voice need was the catalyst to look beyond Piper.
- The 20/20 watchable score is the bar future TTS swaps must clear or beat.

## Current status (after 2026-05-25)

Since OmniVoice MLX became the active TTS, Piper is the **fallback** when no ref clip is available for a speaker (e.g. a character who only ever interjects during overlap regions). It still ships in the pipeline; just not as the primary.

## Revisit trigger

- If Piper produces unusable output for a real episode (regression).
- If a meaningfully better EA fine-tune of any TTS appears.
- If OmniVoice is removed for any reason, Piper becomes the default again.

## Source

- `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/verdict.md`
- `agent-os/product/tech-stack.md` §"Stage 4 — Eastern-Armenian TTS"
