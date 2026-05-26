# Decisions log

The index of architectural decisions. Each entry links to a full ADR in `decisions/`. Format: date — short title (link); one-sentence summary + revisit trigger.

If you (agent or human) are about to "investigate" or "consider" a model or pattern, search this file first. If a relevant decision exists, follow it unless the revisit trigger has fired.

## Foundational (project setup)

- **2026-05-24 — [Eastern Armenian only](decisions/2026-05-24-eastern-armenian-only.md).** Target dialect is `hy_AM`; Western Armenian (`hyw`) outputs are automatically rejected. *Revisit:* never (this is a product scope decision tied to the target user).
- **2026-05-24 — [Local-first strict](decisions/2026-05-24-local-first-strict.md).** Cloud APIs are never enabled by default; cloud is a per-stage opt-in fallback only after the local set for that stage is judged inadequate. *Revisit:* if cloud becomes free *and* private *and* offline-capable (unlikely).

## Phase 0 / Phase 1 (model selection — historical)

- **2026-05-24 — [Piper hy_AM-gor-medium as TTS floor](decisions/2026-05-24-piper-hye-as-tts-floor.md).** Phase 0 winner (20/20 watchable); ~63 MB ONNX, CPU-only. Kept as the no-ref fallback after the OmniVoice swap. *Revisit:* if Piper produces unusable output for a real episode, or if a meaningfully-better EA fine-tune appears.
- **2026-05-25 — [NLLB-3.3B supersedes the 600M distilled](decisions/2026-05-25-nllb-3-3b-over-600m.md).** Phase 0 verdict flagged the 600M as "soft go with weak idioms / RU narration"; the 3.3B clears those weaknesses well enough that NLLB-3.3B is the Phase 1 default translator. *Revisit:* when Qwen3-Instruct-MLX (already registered as a candidate) has been ear-judged against NLLB-3.3B on the masha clip.
- **2026-05-25 — [OmniVoice MLX supersedes Piper as the active TTS](decisions/2026-05-25-omnivoice-supersedes-piper.md).** OmniVoice's zero-shot ref-clip cloning produces hy_AM cleanly, gives per-character voice variety automatically, and is the foundation Phase 2 multi-voice depends on. Piper stays as the fallback when no ref clip is available for a speaker. *Revisit:* if a stronger EA-capable local TTS appears (see the on-demand `stages/tts/` rejection files for the current survey).

## Phase 2 (multi-voice)

- **2026-05-26 — [pyannote 3.1 for diarization](decisions/2026-05-26-pyannote-3-1-for-diarization.md).** SOTA accuracy + native overlap detection. HF-gated (token + license accept on segmentation-3.0 + speaker-diarization-3.1 + speaker-diarization-community-1). All inference local after first download. *Revisit:* if a non-gated alternative reaches comparable accuracy, or if pyannote 3.2/4.x drops with relevant breaking changes.

## Phase 2.x (emotion transfer + auto-loaded docs — this spec)

- **2026-05-26 — [OmniVoice locked for Eastern Armenian](decisions/2026-05-26-omnivoice-locked-for-ea.md).** F5-TTS-MLX (335M, MLX-native, but EN+ZH only, no EA fine-tune) and IndexTTS-2 (PyTorch-only, ZH/EN/JA/ES only, 8-dim emotion vector but no Armenian) both fail the EA gate. OmniVoice MLX remains the only verified-working EA local TTS. *Revisit:* an EA fine-tune of F5-TTS-MLX appears, OR cloud TTS (ElevenLabs / OpenRouter Multilingual v2) is approved per the local-first exception policy.
- **2026-05-26 — [Emotion routed through ref-clip binning](decisions/2026-05-26-emotion-routed-through-ref-clip-binning.md).** Since OmniVoice has no explicit emotion knob, emotion transfer routes through (a) detecting per-sentence source emotion with emotion2vec, (b) picking per-`(speaker, emotion_bin)` ref clips, (c) tuning `guidance_scale` per bin, (d) injecting the emotion tag into Qwen3's user prompt when active. Emotion fidelity is bounded by what's audible in the source character's range — accepted constraint. *Revisit:* if OmniVoice exposes an explicit emotion knob, or if the TTS engine is swapped.
- **2026-05-26 — [Auto-loaded architecture docs](decisions/2026-05-26-auto-loaded-architecture-docs.md).** Two-tier `docs/architecture/` structure. Always-loaded tier (`pipeline.md`, `current-models.md`, `decisions.md`, `quality-issues.md`) imported by `CLAUDE.md` via `@`-imports; on-demand tier read by the agent when working in that area. The `CLAUDE.md` "Maintenance contract" names which docs to update for which action. *Revisit:* if `CLAUDE.md` becomes too large to auto-load efficiently, or if a better local-knowledge mechanism replaces `@`-imports in Claude Code.
