# Phase 1 MVP pipeline — Shaping Notes

## Scope

End-to-end offline batch translation pipeline for local video files, producing a watchable Eastern-Armenian-dubbed MKV. Six stages (ingest, ASR, translate, TTS, mux, output) as swappable modules behind typed `Protocol`s, plus a CLI and a minimal localhost FastAPI page.

**Definition of done:** the author's son watches a 20-minute cartoon episode dubbed by the pipeline end-to-end, in Armenian, without asking to switch back to the original audio.

## Decisions

| Decision | Value | Why |
|---|---|---|
| GPU re-eval | Deferred; plumb now on Phase 0 winners | Stages are swappable — re-eval becomes a `patmir.toml` config change. Ships sooner. |
| Hardware | Mac (MPS); device set per stage via config | Phase 0 ran on MPS; CUDA workstation still pending. Code stays device-agnostic. |
| Source langs | EN + RU from day 1 | Whisper auto-detects; NLLB-600M handles both. |
| ASR | `faster-whisper` running `whisper-large-v3-turbo`, int8 | 4-decoder-layer pruned variant; much faster, minor quality loss; same runtime as `large-v3`. |
| Alignment | Sentence-level pad + `pyrubberband` stretch capped at ±20% | Lip sync is not a goal; heavy stretching wrecks TTS audio. |
| Container | MKV first, output stage MP4-ready | AR-default + original 2nd + SRT all clean in MKV. |
| UI | CLI + minimal FastAPI localhost page | Matches `tech-stack.md` §"Local UI". |
| Job dir | `runs/<YYYYMMDD-HHMMSS>-<source-stem>/` | Mirrors `eval_runs/<ts>/` convention. |

## Context

- **Visuals:** None — CLI/FastAPI MVP needs no mockups.
- **References:** Phase 0 candidate-module pattern in `patmir/eval/candidates/` (Translator + TTS interfaces) is reused as-is. The Phase 1 pipeline picks one candidate per stage via a new `[pipeline]` selector in `patmir.toml`. See `references.md`.
- **Product alignment:** Implements `agent-os/product/roadmap.md` §"Phase 1 — MVP: Mode A on local files". Acts on the soft go from `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/verdict.md`.

## Open weakness accepted into Phase 1

NLLB-600M idiom translation is unusable (Phase 0: 1.3/1.3/1.5). The dogfood episode will have wrong idiom translations and that's fine for Phase 1 — idiom quality is a GPU-re-eval problem, not a plumbing problem.

## Modern HuggingFace candidates registered (not enabled)

Added to `patmir.toml` as `enabled = false` so the GPU re-eval picks them up automatically:

- **Translation:** `Gen2B/HyGPT-10b-it` (Eastern-Armenian-centric LLM, Jul 2025); `Helsinki-NLP/opus-mt-en-hy` and `opus-mt-ru-hy` (small fast Eastern-Armenian translators, MPS-friendly).
- **ASR:** `whisper-large-v3` as a non-default fallback for jobs where turbo's quality loss matters.

## TTS landscape correction (vs. Phase 0)

A deeper HuggingFace scan in shaping turned up:

- **XTTS-v2 does NOT support Armenian** in its current 17-language model card. Phase 0's `tech-stack.md` and `patmir.toml` overstated this. Entry stays in `patmir.toml` disabled with a `# no Armenian without fine-tune` comment; `tech-stack.md` gets a matching correction.
- **Bark** stays the only deferred local TTS candidate that hasn't been disqualified — kept for the GPU re-eval pass, not added to Phase 1 plumbing.
- **F5-TTS / MeloTTS / OpenVoice v2 / Parler-TTS** — none claim Armenian; skipped.
- **`facebook/mms-tts-hye`** still doesn't exist; only `mms-tts-hyw` (Western, wrong dialect).
- **`davit312/Armenian-speech-hy` dataset** (Nov 2024) exists and is useful for a future Piper / VITS fine-tune in Phase 2's voice-variety work.

Piper `hy_AM-gor-medium` remains the Phase 1 TTS, unchanged.

## Standards Applied

`agent-os/standards/index.yml` is currently empty. No standards apply yet. See `standards.md`.
