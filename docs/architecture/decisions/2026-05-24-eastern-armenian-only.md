# Eastern Armenian only

**Date:** 2026-05-24
**Status:** locked (foundational; not expected to revisit)

## Context

Armenian splits into two literary standards: Eastern (`hy_AM`, ISO 639-3 `hye`) spoken in Armenia and the Iranian-Armenian community, and Western (`hyw`) spoken by most diaspora communities. The two share the script but differ in pronunciation (notably the consonant voicing system) and vocabulary.

The target user is the author's son, a fluent native Eastern-Armenian speaker. A Western-Armenian voice reading Eastern-Armenian text is intelligible but sounds foreign — which defeats the product's reason for existing.

## Decision

Target locale is `hy_AM` end-to-end. Western-Armenian-only models are not viable candidates. If a model's Armenian output is ambiguous (most TTS engines don't distinguish dialects in their language tags), it must be ear-verified by a fluent Eastern-Armenian speaker before adoption.

## Consequences

- Several otherwise-attractive models are off the table: `facebook/mms-tts-hyw` (Western only), `ArthurMalajyan/seamless-m4t-v2-large-asr-hyw` (Western only).
- Cross-lingual zero-shot voice cloners (F5-TTS, OpenVoice, IndexTTS-2) are filtered hard: none of them list `hy` in their supported languages today.
- The pool of usable local TTS for EA is *very* small — historically just `davit312/piper-TTS-Armenian (hy_AM-gor-medium)` and (Phase 1.3 onward) OmniVoice MLX via zero-shot cloning on a known-good EA ref.

## Revisit trigger

This is a product-scope decision tied to the target user. It does not revisit on its own.

## Source

- `agent-os/product/mission.md` §"Dialect: Eastern Armenian only"
- Memory: `project_eastern_armenian_only`
