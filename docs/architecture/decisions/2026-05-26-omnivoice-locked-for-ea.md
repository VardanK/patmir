# OmniVoice MLX locked as the Eastern-Armenian TTS

**Date:** 2026-05-26
**Status:** locked

## Context

After Phase 2 shipped, ear-testing the masha-2-5min output surfaced the gap that **emotions are not transferred** — Patmir reads emotionally rich source dialogue in a flat Armenian voice. Two contemporary TTS engines stood out as obvious upgrade candidates for explicit emotion control:

- **IndexTTS-2** (2026, IndexTeam / Bilibili). Three-module cascade (T2S = Qwen3-finetuned LM, S2M Zipformer + flow matching, BigVGANv2 vocoder). **Explicit 8-dim emotion vector** (happy / angry / sad / afraid / disgusted / melancholic / surprised / calm, each 0-1), independent of speaker identity. ~16 GB RAM, PyTorch-only, no MLX port, no published GGUF/INT8 variants.
- **F5-TTS** (335M, DiT + flow matching, non-autoregressive). MLX-native via `lucasnewman/f5-tts-mlx`. ~3-8 GB MPS, ~11 s/sample on M3 Max. Emotion only partially carried via the reference clip; no explicit knob.

Both were investigated against the project's hard constraint: **Eastern Armenian (`hy_AM`)** as a supported output language.

## Findings

| Property                       | IndexTTS-2          | F5-TTS-MLX          | OmniVoice MLX (current) |
| ------------------------------ | ------------------- | ------------------- | ----------------------- |
| EA out-of-box                  | ❌ (ZH/EN/JA/ES)    | ❌ (EN+ZH base)     | ✅ verified              |
| MLX / MPS native               | ❌                  | ✅                  | ✅                       |
| Explicit emotion knob          | ✅ (8-dim vector)   | ❌ (ref only)       | ❌ (ref only)            |
| Speaker / emotion decoupling   | ✅                  | ❌                  | ❌                       |
| Community Armenian fine-tune   | None published      | None published      | n/a (zero-shot via ref) |

Both rejection candidates fail the EA gate. F5-TTS has a documented community-fine-tuning path (Arabic, Finnish, Spanish, French, German), but no EA fine-tune exists, and producing one would require building an EA tokenizer + collecting ~50-100 h of EA speech data — a project of its own, not a TTS-swap.

## Decision

**OmniVoice MLX stays as the active TTS.** No engine swap. Emotion transfer must route through OmniVoice's existing levers:

- `ref_audio` + `ref_text` (the dominant signal — picks both speaker identity and the expressive cues present in the clip)
- `guidance_scale` (strength of conditioning)
- `num_steps` (denoising steps, weakly tied to expressiveness)

How emotion is implemented on top of these levers is captured in [emotion-routed-through-ref-clip-binning](2026-05-26-emotion-routed-through-ref-clip-binning.md).

## Consequences

- Emotion fidelity in the dub is bounded by what's audible in the source character's range. A character who is always calm in the source cannot be made to sound angry. **Accepted constraint.**
- Per-`(speaker, emotion)` ref-clip selection becomes the dominant emotion-transfer mechanism.
- IndexTTS-2 and F5-TTS-MLX both get docs files under `stages/tts/` with status `rejected (2026-05-26)` so this investigation isn't re-done.

## Revisit trigger

- An EA fine-tune of F5-TTS-MLX (or any MLX-native multilingual TTS) is published.
- Cloud TTS (ElevenLabs Multilingual v2, or an OpenRouter-mediated emotion-controllable engine) is approved under the local-first exception policy.
- OmniVoice itself becomes unusable for any reason.

## Source

- This conversation's web-research investigation (IndexTTS-2: arXiv 2506.21619, IndexTeam HF model card; F5-TTS: SWivid/F5-TTS GitHub, lucasnewman/f5-tts-mlx).
- `docs/architecture/stages/tts/indextts-2.md` and `f5-tts-mlx.md` for the per-model detail.
