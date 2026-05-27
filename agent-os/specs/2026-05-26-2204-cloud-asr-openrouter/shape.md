# Cloud ASR migration — Shaping Notes

## Scope

Migrate the ASR (speech-to-text) stage from local MLX Whisper Large-v3 to a cloud model via OpenRouter. Continues the per-stage cloud escalation that already covered translate (Gemini 2.5 Flash) and TTS (Gemini 3.1 Flash TTS) — quality-baseline play, not a cost play.

## Decisions

- **Active cloud model:** `openai/gpt-4o-transcribe` on OpenRouter. Chosen over `qwen/qwen3-asr-flash` because the Qwen3-ASR family was already tried locally and shelved (`stages/asr/qwen3-asr-mlx.md`: chain-of-thought interleaving on long inputs, commit `c1b02c5`). Best published Russian WER and native `timestamp_granularities=["word"]`.
- **Alternates registered as `enabled = false`:** `qwen/qwen3-asr-flash` (kept as a music-robustness option in case 4o regresses on cartoon SFX), `google/chirp-3`, `mistralai/voxtral-mini-transcribe`. Same OpenRouter module — flip `[pipeline].asr` to A/B.
- **Timestamp gap fallback = forced alignment.** When the cloud transcript lacks word-level timings (or shows attention-drift), run `ctc-forced-aligner` locally on the vocals. User picked this over the segment-proportional approximation despite the added dep — preserves the dub-time stretch contract (±20 % cap stays within 200 ms).
- **OpenRouter only.** Soniox / Gladia / Speechmatics / Deepgram (recommended by external deep research) are off the table per user constraint — keep one cloud key (`OPENROUTER_API_KEY`).
- **Not enterprise-grade.** Single-user pipeline. Concurrency, SLA, 20 k-batch-job, IP-allowlist scenarios are ignored.
- **ElevenLabs Scribe — documented, not wired.** Tier-gated by project policy (mirrors how ElevenLabs Multilingual v2 TTS is shelved at `enabled = false`). Scribe is actually on the Free tier, but we won't activate it until ear-tested and budget-sanctioned. Toml stub only; no candidate module.
- **Local Whisper-MLX stays.** Kept registered for offline runs.
- **No new pipeline stage.** Forced alignment runs inside the ASR candidate when needed — the `ASRStage` Protocol contract stays unchanged.

## Context

- **Visuals:** None.
- **References:**
  - `patmir/eval/candidates/translation/openrouter.py` — HTTP + auth + error pattern to mirror.
  - `patmir/eval/candidates/tts/openrouter.py` — `/v1/audio/*` endpoint pattern (TTS uses `/v1/audio/speech`; STT will use `/v1/audio/transcriptions`).
  - `patmir/eval/candidates/tts/elevenlabs.py` + the `enabled = false` shelving in `patmir.toml` — pattern for documented-but-not-used candidates.
  - `patmir/pipeline/stages/asr/whisper_mlx.py` — current active; shows the `transcribe()` contract + sentence grouping invocation.
- **Product alignment:** Patmir's mission is to set the quality baseline for the author's son's Eastern Armenian dub. Local-first strict policy permits per-stage cloud opt-in when the local set is judged inadequate — ASR meets that bar (Whisper-MLX laughter-region hallucinations, even with gap-filler + blocklist).
- **External research input:** A Gemini deep-research doc surveyed the broader STT market (Deepgram Nova-3, AssemblyAI Universal-3 Pro, Speechmatics, Soniox, Gladia, etc.). Useful for context but constrained out by the OpenRouter-only rule. Noted: Soniox / Gladia natively support Armenian end-to-end, but we already own translation downstream — unified-translation isn't our use case.

## Standards Applied

None — `agent-os/standards/index.yml` is empty.
