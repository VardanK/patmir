# Emotion transfer + quality docs — shaping notes

## Scope

Address Phase 2 quality issue #3 (emotion not transferred — flagged as biggest by user) plus #4 (voice leak at sentence boundaries — bundled because it's small + independent). Establish a durable architecture-docs scaffold (`docs/architecture/`) so future model investigations and decisions are auto-loaded into every conversation without prompting.

The other three Phase-2 quality issues (#1 cloning fidelity, #2 translation, #5 overlap handling) are explicitly deferred to follow-up specs.

## Locked decisions (11)

1. **TTS engine:** OmniVoice MLX. No swap. F5-TTS-MLX and IndexTTS-2 investigated and rejected on the Eastern-Armenian language gap.
2. **Emotion is detected from source audio**, not translated text.
3. **SER model:** `iic/emotion2vec_plus_base` (9 classes via funasr).
4. **Emotion bins:** 5 — `neutral`, `happy`, `sad`, `angry`, `fearful`. Configurable mapping from the 9 SER classes.
5. **Emotion-aware ref-clip selection:** per-`(speaker, emotion_bin)` with fallback chain (exact → speaker-neutral → global default).
6. **Qwen3 translate prompt** carries emotion when active; NLLB ignores.
7. **OmniVoice `guidance_scale` per-emotion lookup table.**
8. **Voice-leak fix:** per-sentence 50 ms `afade` in/out on synthesized track. Bumped to its own spec if the simple fix doesn't resolve the leak.
9. **All 6 docs-scaffold additions:** `decisions/`, `data-contracts.md`, `language-support.md`, `runs.md`, `quality-issues.md`, `glossary.md`.
10. **Local-first strict** (per existing policy): SER stays local; no cloud SER.
11. **Auto-loaded docs:** two-tier structure (`CLAUDE.md` `@`-imports the always-loaded tier; on-demand tier read when relevant). Maintenance contract in `CLAUDE.md` names which docs to update for which action.

## TTS audit decision

**OmniVoice MLX stays as the active TTS** for Eastern Armenian. Both candidates investigated for emotion control fail the EA gate:

- **F5-TTS-MLX** — 335M, MLX-native, ~3-8 GB MPS — EN+ZH base only, no EA fine-tune published.
- **IndexTTS-2** — explicit 8-dim emotion vector + speaker/emotion decoupling — ZH/EN/JA/ES only, no MLX port.

OmniVoice has no explicit emotion knob, so emotion transfer must route through the levers OmniVoice does have: per-`(speaker, emotion_bin)` ref clips + `guidance_scale` per bin + (when Qwen3 is active) emotion-aware translation prompts.

Accepted constraint: emotion fidelity is bounded by what's audible in the source character's range. A character who is always calm in the source cannot be made to sound angry.

## Context / inputs to the design

- Five quality issues flagged by user after ear-testing Phase 2 output on `samples/masha-2-5min.mkv`. #3 (emotion) is biggest.
- User explicitly asked to ensure we're using "the absolute best local model" before considering OpenRouter/cloud frontier models.
- User asked for `docs/architecture/` and explicitly required auto-loading: "we should also make the docs accessible to the AI agent, so I'm not required to ask it every time to look into the docs."
- User pointed at `agent-os/specs/` as the historical context source — instructed to mine prior specs for decisions/models/rationale rather than starting blank.

## Out of scope (explicit)

- #1 cloning fidelity (re-eval after this lands)
- #2 translation quality (its own spec; likely needs OpenRouter as ceiling reference)
- #5 overlap handling (its own spec)
- OpenRouter integration (gated by per-stage local-ceiling verdicts in `decisions/`)
- Full 20-min episode validation
- YouTube ingest via `yt-dlp`
- Diarization audit (folded into the docs by capturing pyannote 3.1 as the active candidate; no swap)

## Visuals

None.

## References

- This conversation's F5-TTS / IndexTTS-2 investigation (web research + the OmniVoice constraint findings)
- Phase 0 verdict + Phase 1.x + Phase 2 specs (mined for the historical model lineage in `docs/architecture/`)
- `patmir/eval/candidates/tts/omnivoice_mlx.py:48-74` (the per-call ref + guidance_scale surface)
- `patmir/pipeline/stages/diarize/pyannote.py` + `ref_picker.py` (Phase 2 ref-picking pattern; extended in this spec for emotion bins)
- `patmir/pipeline/stages/asr/speaker_tag.py` (algorithm shape that `emotion_tag.py` mirrors)
- `patmir/pipeline/runner.py:131-213` (the `ctx.extra` + post-stage backfill pattern reused for emotion)
