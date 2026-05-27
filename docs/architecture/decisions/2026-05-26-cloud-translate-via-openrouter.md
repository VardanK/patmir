# Cloud translate via OpenRouter (Gemini 2.5 Flash default)

**Date:** 2026-05-26
**Status:** locked (per-stage cloud opt-in for translate)

## Context

Local translate has been stable on NLLB-200-3.3B (Phase 1.3) with Qwen3-Instruct-MLX registered but not active. The Phase 2.x emotion-transfer work added emotion-aware prompting to Qwen3 — dormant under NLLB, since NLLB has no prompt surface.

After Phase 2.x shipped, ear-tests on the masha-2-5min output revealed that emotion transfer **still sounds flat** even after:

- Per-`(speaker, emotion)` reference clips were sliced and used per sentence
- OmniVoice's `guidance_scale` was raised from sub-default (0.6-0.9) to amplified (1.8-3.0) to push more of the ref's prosody through

The dominant signal for "flat" delivery is the TTS side — OmniVoice normalizes prosody more aggressively than its zero-shot reputation suggests. But translate is the **secondary** contributor: NLLB-3.3B's Armenian phrasing is register-neutral; even when TTS later renders it, there's no emotional dynamism in the *word choice* for the TTS to ride on. Frontier LLMs do markedly better here — they pick more dynamic verbs and word order when prompted with the speaker's emotion.

The user explicitly asked to "try OpenRouter and a couple of frontier models where it's necessary."

## Decision

Opt in to **cloud translate via OpenRouter** as the active translator. OpenRouter is a unified gateway (openrouter.ai) to GPT-5, Claude Opus 4.7, Gemini 2.5 Flash/Pro, Grok, etc. behind one API key — convenient for A/B comparison and avoids vendor-specific SDK churn.

**Default model: `google/gemini-2.5-flash`.** Reasoning:

- Cheap per token — supports iteration during quality experimentation.
- Competitive on multilingual translation; Armenian is in its supported language list.
- Single-call latency is comparable to or better than the on-device NLLB pass.

Alternates registered (`enabled = false`):

- `anthropic/claude-opus-4.7` — strongest on register / nuance preservation; most expensive.
- `openai/gpt-5` — comparable quality to Claude; cheaper.

These flip on via a config one-liner when we want a side-by-side comparison.

## Implementation

- New candidate module `patmir/eval/candidates/translation/openrouter.py`. OpenAI-compatible chat-completions over httpx (no new deps; httpx is already in the venv via huggingface_hub / fastapi).
- `translate(text, source_lang, emotion=None) -> str` — same contract as Qwen3-Instruct-MLX, so the existing translate adapter's signature-probe forwards the `emotion` kwarg automatically.
- API key from `OPENROUTER_API_KEY` env var. Candidate raises a clear setup error if missing.
- Reuses the same system + user prompts as Qwen3 — "preserve tone, register, and emotion" + per-sentence emotion-tag injection.
- Three OpenRouter candidates registered in `patmir.toml`: `openrouter_gemini_flash` (active), `openrouter_claude_opus` and `openrouter_gpt5` (alternates).

## Consequences

- **Per-call cost.** Gemini Flash is cheap (~$0.10 / 1M input + $0.40 / 1M output tokens at posted rates), but it's no longer zero. A 20-minute episode ≈ 200 sentences ≈ ~$0.005-0.01 — negligible at this scale, worth budgeting for an experimental phase.
- **Internet required.** Patmir was strictly offline-capable; with this candidate active, the translate stage needs network. The candidate's `RuntimeError` on missing key is the only failure mode if offline.
- **Privacy.** Sentence text is sent to OpenRouter (and onward to the chosen vendor). For Patmir's home-use scope this is acceptable.
- **Latency.** Per-sentence HTTP round-trip is in the 200-800 ms range, similar to or slower than the local NLLB pass. Not a problem for batch dubbing.
- **NLLB-3.3B stays registered** as the local fallback. Flip `[pipeline].translate` back to `"nllb_200_3.3b"` to go fully offline.

## Revisit trigger

- Frontier-translate output ear-tested as not meaningfully better than NLLB-3.3B for our content — revert to local.
- A frontier-quality local Armenian translator appears (HyGPT or an EA fine-tune of a 70B-class open LLM) — revert to local.
- Cost runs higher than expected for our usage volume.

## Source

- This conversation (2026-05-26 emotion-transfer ear-test cycle: low guidance flat, high guidance still flat, user requested cloud escalation).
- See related ADR [emotion-routed-through-ref-clip-binning](2026-05-26-emotion-routed-through-ref-clip-binning.md) for the local-TTS-side story.
- Local-first escalation policy: [local-first-strict](2026-05-24-local-first-strict.md).
