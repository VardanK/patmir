# Phase 2.x — Emotion transfer + voice-leak fix + architecture docs scaffold

## Context

Phase 2 (multi-voice diarization + per-speaker cloning, commit `3908e6c`) shipped a watchable dub of the masha-2-5min clip, but ear-tests on the output surfaced five quality issues. In priority order the user flagged:

1. Voice cloning fidelity sometimes off (#1)
2. Translation quality not there yet (#2)
3. **Emotions not transferred — biggest issue (#3)**
4. Original voice leaks at sentence boundaries (#4)
5. Overlap handling unclear (#5)

This spec tackles **#3 (emotion transfer)** and bundles **#4 (voice leak)** because the voice-leak fix is pipeline-mechanics (mux-stage), independent of model choice, and small. #1, #2, #5 are deferred to follow-up specs.

The spec also establishes **`docs/architecture/`** — a small Markdown knowledge base so future investigations (mine or the user's) don't re-do the work that's already been done. The catalyst was discovering that F5-TTS and IndexTTS-2 — the two obvious TTS upgrades for emotion control — both fail the Eastern-Armenian gate, and that investigation needed somewhere durable to land.

### TTS engine — locked

After investigating F5-TTS (335M, MLX-native, ~3-8 GB MPS, EN+ZH only, no EA fine-tune) and IndexTTS-2 (PyTorch-only, 16 GB RAM min, ZH/EN/JA/ES only, explicit 8-dim emotion vector but no Armenian and no MLX port), **OmniVoice MLX stays as the active TTS**. It's the only EA-capable local TTS we have verified output from. Emotion transfer must route through OmniVoice's existing levers, not through an engine swap.

OmniVoice's available levers for emotion (per `omnivoice_mlx.py:48-74`):
- `ref_audio` + `ref_text` — speaker identity *and* the expressive cues present in the reference clip
- `guidance_scale` — strength of conditioning
- `num_steps` — denoising steps (quality/time tradeoff, weakly tied to expressiveness)
- No explicit emotion vector. No prosody/style kwargs.

So emotion fidelity in EA dubs is bounded by **what's audible in the source character's clips**. A character who is always calm in the source cannot be made to sound angry. Accepted constraint.

## Locked decisions

1. **TTS engine:** OmniVoice MLX. No swap. Documented in `docs/architecture/decisions/2026-05-26-omnivoice-locked-for-ea.md`.
2. **Emotion is detected from source audio**, not from the translated text. The source vocal stem carries the ground truth; translation rephrasing should not redefine the emotion.
3. **SER model:** `emotion2vec_plus_base` (iic/emotion2vec) — multilingual-trained on multiple datasets, outputs 9 categories (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown). PyTorch+MPS. Run as a new stage between `diarize` and `asr`.
4. **Emotion bins:** collapse the 9 SER outputs to **5 ref-clip bins** — `neutral`, `happy`, `sad`, `angry`, `fearful`. `disgusted`/`surprised`/`other`/`unknown` map to the nearest bin (configurable in `patmir.toml`).
5. **Emotion-aware ref-clip selection:** `ref_picker` becomes per-`(speaker, emotion_bin)`. If a speaker has no qualifying turn for a bin, fall back to that speaker's neutral ref. If even neutral is missing, fall back to the global default ref.
6. **Qwen3 translate prompt** carries an emotion tag per sentence. NLLB candidates ignore the field (no API surface). Documented in each translator's docs file.
7. **OmniVoice `guidance_scale` per-emotion:** small lookup table (neutral=0.7 baseline, angry/happy=0.9, sad=0.6, fearful=0.8). Tunable in `patmir.toml`. Not a magic bullet — kept low-effort because the dominant signal is still the ref clip.
8. **Voice-leak fix (#4):** scoped to investigation + a *single* surgical change to `stages/mux/ffmpeg_pad_stretch.py`. Most likely cause: insufficient ducking of the accompaniment / original-stem residual at the start/end of each synthesized sentence. Fix is a per-sentence ~50 ms fade-in/fade-out on the synthesized track and matching fade on the original-stem when used as fallback. If that doesn't resolve it, the fix scope is bumped to a separate spec.
9. **Docs scaffold:** all 6 supporting files included (`decisions/`, `data-contracts.md`, `language-support.md`, `runs.md`, `quality-issues.md`, `glossary.md`) — see Task 1.
10. **Local-first strict:** SER inference stays local. No cloud SER. (Per `feedback_local_first_strict` in memory.)
11. **Auto-loaded docs:** the docs scaffold has a **two-tier** structure. A small "always-loaded" tier (pipeline shape, currently-active models, decisions index, quality-issues tracker) is `@`-imported by `CLAUDE.md` so every conversation sees it without being asked. The rest (per-stage / per-model deep files) is read on demand. This is what makes the scaffold pay off — the agent's defaults stay correct without the user having to remind it.

## Architecture

New pipeline order — inserts one stage between `diarize` and `asr`:

```
ingest → separate → diarize → emotion → asr → translate → tts → mux → output
                       │         │
                       │         └─ writes emotion/segments.json (per-sentence emotion predictions
                       │            keyed by time window, not yet by sentence — backfilled post-ASR)
                       │
                       └── diarize/turns.json + diarize/refs/<spk>/<bin>.wav (per-emotion-bin)
```

### Data flow per sentence

1. `diarize` writes turns + per-`(speaker, emotion_bin)` ref WAVs to disk.
   - `ref_picker` now needs an emotion label *per turn* to bin it. The emotion stage hasn't run yet at this point — so `diarize` writes turns with no ref selection, and `ref_picker` runs **after `emotion`** in the runner (same pattern as the existing post-ASR `ref_text` backfill in the committed Phase 2 code).
2. `emotion` runs `emotion2vec` on each turn window from `vocals.wav`, attaches an `emotion_bin` to each turn, then the runner triggers `ref_picker.pick_refs` to slice per-`(speaker, bin)` ref clips. `Diarization.refs` becomes `dict[(speaker_id, emotion_bin), SpeakerRef]`.
3. `asr` produces sentences. Runner immediately calls `apply_speaker_tags` (existing) and then `apply_emotion_tags(transcript, emotion_predictions)` — same algorithm: dominant-emotion ≥85% of sentence duration, else split or fallback.
4. `translate` (Qwen3 candidate) injects `emotion` into the user prompt: `"Translate this {src} sentence into Eastern Armenian, with emotion '{emotion}': {text}"`. NLLB ignores.
5. `tts` adapter looks up `refs[(speaker_id, emotion_bin)]`; if missing, falls back to `refs[(speaker_id, "neutral")]`; if missing, falls back to global default. Sets `guidance_scale` from `[emotion_guidance]` config table.
6. `mux` applies per-sentence ~50 ms fade-in/fade-out on the synthesized track at sentence boundaries.

### Re-ordering note vs Phase 2

Phase 2's `ref_picker` runs inside the `diarize` stage adapter and writes WAVs immediately. This spec **moves the WAV-slicing pass** out of `diarize` and into a runner-level step that fires **after** `emotion`. `diarize` only writes `turns.json` and overlaps. `ref_picker` becomes runner-driven (similar to the existing post-ASR `_backfill_ref_text` pattern). This avoids a circular dependency without adding a separate stage just for ref-slicing.

## Critical files

### New files (code)

- `patmir/pipeline/stages/emotion/__init__.py`
- `patmir/pipeline/stages/emotion/adapter.py` — stage entry; writes `emotion/segments.json`
- `patmir/pipeline/stages/emotion/emotion2vec.py` — emotion2vec candidate. Lazy-load `iic/emotion2vec_plus_base` via `funasr`. Runs on `vocals.wav` per turn window. Returns `list[EmotionPrediction]`.
- `patmir/pipeline/stages/asr/emotion_tag.py` — `apply_emotion_tags(transcript, predictions) -> Transcript`. Mirrors `speaker_tag.py`.
- `tests/test_emotion_tag.py` — dominant/split/orphan cases (mirror `speaker_tag` tests).
- `tests/test_emotion_ref_picker.py` — per-bin selection, fallback chain.

### Modified files (code)

- `patmir/pipeline/stage.py`
  - Add `Emotion` enum-like literal type: `Literal["neutral","happy","sad","angry","fearful"]`
  - Add `EmotionPrediction{start, end, emotion, confidence}` dataclass
  - Add `emotion: Emotion | None = None` field to `SpeakerTurn`, `Sentence`, `TranslatedSentence`
  - Change `Diarization.refs: dict[str, SpeakerRef]` → `dict[tuple[str, Emotion], SpeakerRef]` (or string-keyed `"spk__bin"` for JSON portability)
  - Add `EmotionStage` Protocol: `detect(separated, diarization, ctx) -> list[EmotionPrediction]`
  - Insert `"emotion"` into `STAGE_ORDER` between `"diarize"` and `"asr"`
- `patmir/pipeline/config.py` — add `"emotion": ("stages", "emotion")` to `_STAGE_NAMESPACE`
- `patmir/pipeline/runner.py`
  - New `_run_stage` branch for `"emotion"`
  - Move `ref_picker.pick_refs` invocation out of the `diarize` stage and into a runner-level call **after** `emotion` returns (analogous to existing `_backfill_ref_text` after ASR)
  - In ASR branch: after `apply_speaker_tags`, also call `apply_emotion_tags(transcript, ctx.extra["emotion_predictions"])`
  - Add `"emotion"` branch to `_load_resume_artifact`
- `patmir/pipeline/stages/diarize/pyannote.py` — strip the `pick_refs` call. Stage now writes only `turns.json` + `overlaps`.
- `patmir/pipeline/stages/diarize/ref_picker.py` — `pick_refs` now takes `predictions: list[EmotionPrediction]`, bins each turn by dominant emotion-overlap, picks the longest non-overlapping qualifying turn per `(speaker, bin)`. Writes `diarize/refs/<spk>/<bin>.wav`.
- `patmir/pipeline/stages/translate/adapter.py` — passthrough `s.emotion` into `TranslatedSentence`.
- `patmir/eval/candidates/translation/qwen3_instruct_mlx.py` — inject `emotion` tag into user prompt when set (don't change system prompt — it already says "preserve tone, register, and emotion"). NLLB candidates: pass-through, no behavior change.
- `patmir/pipeline/stages/tts/adapter.py` — refs lookup becomes `refs[(s.speaker_id, s.emotion or "neutral")]`. Fallback chain: exact → `(speaker, "neutral")` → global default. `guidance_scale` set from `cfg["emotion_guidance"][s.emotion]` table (with default).
- `patmir/eval/candidates/tts/omnivoice_mlx.py` — extend `synthesize` to accept `guidance_scale: float | None = None`; pass through to `model.generate(...)` when set.
- `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py` — per-sentence ~50 ms `afade` on the synthesized track at boundaries (`afade=t=in:st=...:d=0.05` + `afade=t=out:st=...:d=0.05`). Verify it eliminates the audible leak.
- `patmir.toml`
  - `[pipeline]` — add `emotion = "emotion2vec_plus_base"`
  - New `[[stages.emotion.candidates]]` section
  - New `[emotion_bins]` mapping table (9 SER → 5 bins)
  - New `[emotion_guidance]` table (5 bins → guidance_scale floats)
- `pyproject.toml` — add `funasr>=1.1` (for emotion2vec) and `modelscope>=1.20` (its dep).

### New files (docs)

Two tiers. The **always-loaded tier** is `@`-imported by `CLAUDE.md` and lands in every conversation's context. The **on-demand tier** is read by the agent only when working in that area.

```
CLAUDE.md                                  # NEW — project root, @-imports the always-loaded tier

docs/architecture/
# ── always-loaded tier (small; auto-imported by CLAUDE.md) ───────────────
├── pipeline.md                            # Stage order, ASCII diagram, ctx.extra plumbing
├── current-models.md                      # One line per stage: active candidate + key knob values
├── decisions.md                           # Index of every locked decision; one line each + link
├── quality-issues.md                      # The 5 issues; status, owner stage, spec tackling them
# ── on-demand tier (deeper detail; read when working on that stage) ──────
├── index.md                               # Human-facing TOC; not auto-imported
├── data-contracts.md                      # All dataclasses from stage.py, fields + types + meanings
├── language-support.md                    # EA constraint; hy_AM vs hyw vs nothing
├── runs.md                                # What every file in runs/<job>/ is; debugging guide
├── glossary.md                            # SER, VAD, DiT, ref clip, overlap, flow matching, etc.
├── decisions/
│   └── 2026-05-26-omnivoice-locked-for-ea.md  # Full ADR; decisions.md links here
└── stages/
    ├── ingest/{index.md, current.md, ffmpeg.md}
    ├── separate/{index.md, current.md, htdemucs.md}
    ├── diarize/{index.md, current.md, pyannote-3-1.md}
    ├── emotion/{index.md, current.md, emotion2vec-plus-base.md}
    ├── asr/{index.md, current.md, whisper-mlx-large-v3.md, faster-whisper.md, qwen3-asr-mlx.md}
    ├── translate/{index.md, current.md, nllb-200-distilled-600m.md, nllb-200-3-3b.md, qwen3-instruct-mlx.md}
    ├── tts/{index.md, current.md, omnivoice-mlx.md, piper-hye.md, f5-tts-mlx.md, indextts-2.md}
    ├── mux/{index.md, current.md, ffmpeg-pad-stretch.md}
    └── output/{index.md, current.md, ffmpeg-mux.md}
```

**`CLAUDE.md` contract (project root):**

```markdown
# Patmir — project context

[1-paragraph project intro: English/Russian video → Eastern-Armenian dub,
local-first, target audience is the author's child watching cartoons.]

## Always-loaded reference docs

The files below are imported into every conversation's context. Read them
before answering any question about pipeline behavior, model choices, or
quality status. Do not re-investigate model alternatives that already have
a decision recorded.

@docs/architecture/pipeline.md
@docs/architecture/current-models.md
@docs/architecture/decisions.md
@docs/architecture/quality-issues.md

## Maintenance contract

When you (the agent) do any of the following, update the corresponding doc
**in the same change** — without being asked:

| Action                                  | Files to update                                                       |
|-----------------------------------------|-----------------------------------------------------------------------|
| Swap an active model                    | `current-models.md` + `stages/<stage>/current.md` + relevant model file |
| Investigate and reject a model          | `stages/<stage>/<model>.md` (status: rejected, with reason)            |
| Lock a non-trivial decision             | New file in `decisions/<YYYY-MM-DD>-<slug>.md` + one line in `decisions.md` |
| Address or regress a quality issue      | `quality-issues.md` (status + last-touched date)                       |
| Change a data contract (stage.py shape) | `data-contracts.md`                                                    |
| Add a new pipeline stage                | `pipeline.md` + new `stages/<name>/` subdir                            |

The on-demand tier is not auto-loaded. Read those files when the task is
in that area. The `decisions.md` index is the canonical list of "why we
chose X" — if the user asks "why are we using X?" or "should we try Y?",
check `decisions.md` first and only re-investigate if there's a revisit
trigger or new evidence.
```

**`current-models.md` contract** — one row per stage, kept tight on purpose. Example:

```markdown
# Currently active models

Last updated: 2026-05-26

| Stage     | Active candidate                | Key knobs                                          |
|-----------|--------------------------------|-----------------------------------------------------|
| ingest    | ffmpeg                          | 16 kHz mono PCM                                     |
| separate  | htdemucs_ft                     | device=auto; Silero VAD masking on speech_vocals    |
| diarize   | pyannote/speaker-diarization-3.1| min_speakers=0, max_speakers=8                      |
| emotion   | iic/emotion2vec_plus_base       | 9→5 bin map; confidence floor 0.5 → neutral         |
| asr       | mlx-whisper large-v3            | + gap-filler for laughter-dominated regions         |
| translate | qwen3-instruct-mlx              | system prompt preserves tone/register/emotion       |
| tts       | omnivoice-mlx                   | per-(speaker, emotion) ref; guidance_scale per bin  |
| mux       | ffmpeg-pad-stretch              | 50 ms afade-in/out per sentence on synthesized      |
| output    | ffmpeg-mux                      | hy track default, original retained as secondary    |
```

**`decisions.md` contract** — one bullet per ADR, with revisit trigger:

```markdown
# Decisions log

- [2026-05-26 — OmniVoice locked for EA](decisions/2026-05-26-omnivoice-locked-for-ea.md)
  TTS engine is OmniVoice MLX. F5-TTS-MLX and IndexTTS-2 both rejected on
  Eastern-Armenian gap. Revisit if: an EA fine-tune of F5-TTS-MLX appears,
  OR cloud TTS (ElevenLabs / OpenRouter) is approved per the local-first
  exception policy.
```

Model-file template (every model file follows this shape):

```markdown
# <Model name>
- Repo / version:
- Parameters / on-disk size / runtime memory:
- Quantization available:
- Apple Silicon (MLX / MPS) / CUDA / CPU:
- Languages (and Eastern Armenian status):
- License / gating:
- Integration surface (function signature, key kwargs):
- Status in pipeline: active | rejected | benchmarked | future-candidate
- Last evaluated:
- Our experience:
- Notes:
```

### Reusable existing utilities

- `ref_picker.pick_refs` shape (the per-speaker version) — extend, don't rewrite
- `speaker_tag.apply_speaker_tags` algorithm — `emotion_tag` uses the same dominant-≥85% / split / orphan-fallback logic
- `ctx.extra` plumbing pattern (`runner.py` Phase 2 work) — add `ctx.extra["emotion_predictions"]`
- The post-ASR backfill pattern (`_backfill_ref_text`) — reused for the ref-WAV slicing move
- `_resolve_device` (`htdemucs.py:25`) — copy/adapt for the emotion2vec candidate

## Tasks

### Task 1 — Spec docs + architecture scaffold + auto-loading

**1a. Mine historical context first.** Before writing any docs, read every existing spec to extract decisions, model evaluations, and rejections. Source files:

- `agent-os/product/{mission.md, roadmap.md, tech-stack.md}` — the foundational principles (local-first, EA-only, escalation policy)
- `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/{plan.md, shape.md, verdict.md}` — winning candidates per stage, quality floors, RU-narration / idiom failure modes, GPU re-eval queue (NLLB-3.3B, MADLAD-400, XTTS-v2, Bark), cloud fallback ordering (ElevenLabs, Azure, Google)
- `agent-os/specs/2026-05-24-1704-phase-1-mvp-pipeline/{plan.md, shape.md, dogfood.md}` — Phase 1 model choices, dogfood findings
- `agent-os/specs/2026-05-25-1745-test-pipeline-masha-clip/{plan.md, shape.md}` — what was learned testing on the masha clip
- `agent-os/specs/2026-05-26-0215-multi-voice-diarization-cloning/{plan.md, shape.md}` — Phase 2 decisions, the "preset pool over cloning" intent that got overridden by OmniVoice's per-call ref cloning, the Whisper hallucination blocklist

Every model mentioned in any of these specs becomes a file in `docs/architecture/stages/<stage>/<model>.md`, with the `status` field reflecting current reality (`active`, `superseded`, `rejected`, `future-candidate`) and a `Notes` section that *cites the spec it came from*. The point of the docs scaffold is to capture the history these specs already contain in a form the agent auto-loads — not to rewrite history.

**1b. Spec folder.** Create `agent-os/specs/2026-05-26-<HHMM>-emotion-transfer-and-quality-docs/` with `plan.md` (copy of this file), `shape.md` (scope + 11 locked decisions), `standards.md` (note `agent-os/standards/index.yml` is empty), `references.md` (cite the F5/IndexTTS-2 investigation, the four prior specs mined above, and the key code locations: `omnivoice_mlx.py:48-74`, `htdemucs.py:163-245`, `runner.py:131-213`).

**1c. Docs scaffold.** Create the two-tier `docs/architecture/` structure. Populate from historical specs:

- `stages/translate/nllb-200-distilled-600m.md` — status `superseded`, score floor from Phase 0 verdict (idioms 1.33, RU narration 1.20), reason for upgrade, link to verdict.
- `stages/translate/nllb-200-3-3b.md` — status `superseded` (per commit `9c9f219`'s upgrade and the further move to Qwen3), reason and link to the Phase 1.3 commit.
- `stages/translate/qwen3-instruct-mlx.md` — status `active`, system prompt captured verbatim, expected upgrade path documented.
- `stages/tts/piper-hye.md` — status `fallback`, Phase 0 score (3/3/2.9, 20/20 watchable), why kept as the fallback when OmniVoice ref is missing.
- `stages/tts/omnivoice-mlx.md` — status `active`, ref-clip cloning surface, EA support verified, the locked decision.
- `stages/tts/xtts-v2.md` — status `evaluated-not-shipped`, Phase 0 verdict's notes on dialect risk, why it never made it past the GPU queue.
- `stages/tts/elevenlabs-multilingual-v2.md` — status `future-candidate`, cloud-only, gated by local-first escalation policy, what triggers opt-in.
- `stages/tts/f5-tts-mlx.md` + `stages/tts/indextts-2.md` — status `rejected (2026-05-26)`, reason `no Eastern-Armenian support`, link to this spec's investigation as the evidence.
- `stages/asr/{whisper-mlx-large-v3, faster-whisper, qwen3-asr-mlx}.md` — status from current code; capture which is active and why (Phase 1.3 commit).
- `stages/diarize/pyannote-3-1.md` — status `active`, HF gating requirement, MPS coverage caveat from Phase 2 plan, the gap-filler logic that surrounds it.
- `stages/separate/htdemucs.md` — status `active`, VAD masking detail, Demucs sample-rate constraint.

**1d. Decisions log.** Create `decisions/` files for the decisions already implicit in the historical specs *and* the decisions made in this conversation:

- `2026-05-24-eastern-armenian-only.md` — from `mission.md` + memory `project_eastern_armenian_only`
- `2026-05-24-local-first-strict.md` — from `mission.md` + memory `feedback_local_first_strict`
- `2026-05-24-piper-hye-as-tts-floor.md` — from Phase 0 verdict
- `2026-05-25-nllb-3-3b-over-600m.md` — from Phase 1.3 commit history + Phase 0 verdict's "GPU re-eval" recommendation
- `2026-05-25-omnivoice-supersedes-piper.md` — from Phase 1.3 commit `9c9f219`
- `2026-05-26-pyannote-3-1-for-diarization.md` — from Phase 2 plan's locked decisions
- `2026-05-26-omnivoice-locked-for-ea.md` — new (this conversation)
- `2026-05-26-auto-loaded-architecture-docs.md` — new (this conversation, the auto-load mechanism itself)
- `2026-05-26-emotion-routed-through-ref-clip-binning.md` — new (this conversation)

Then `decisions.md` is the one-line index over all of these.

**1e. Always-loaded tier completion.** `current-models.md` (the table) and `quality-issues.md` (the 5 user-flagged issues with owner stage + target spec).

**1f. `CLAUDE.md` at project root.** With the contract spelled out earlier in the plan: `@`-imports for the four always-loaded files + the "Maintenance contract" table. This is the wire that makes the scaffold auto-load. Without this file the docs are dead pages.

**1g. Verify auto-loading works.** Open a fresh Claude Code conversation in the repo root and ask "what TTS are we using and why?" — the answer must come directly from the auto-loaded `current-models.md` + `decisions.md` content with no file-reading and no agent re-investigation. Equivalent smoke-test prompts: "have we tried F5-TTS?" (must cite the rejection ADR), "what's the status of issue #4 voice leak?" (must cite `quality-issues.md`).

### Task 2 — Data contracts + stage protocol

Edit `patmir/pipeline/stage.py`: add `Emotion` literal, `EmotionPrediction`, `EmotionStage` Protocol; add `emotion` field to `SpeakerTurn`, `Sentence`, `TranslatedSentence`; change `Diarization.refs` keying to `(speaker_id, emotion_bin)`; insert `"emotion"` into `STAGE_ORDER`. Edit `config.py` to register the namespace. Update `docs/architecture/data-contracts.md`.

### Task 3 — emotion2vec candidate

Implement `stages/emotion/emotion2vec.py`. Lazy-load `iic/emotion2vec_plus_base` via `funasr.AutoModel`. Loop over `diarization.turns`, slice each turn window from `vocals.wav`, run inference, take argmax over the 9-class output. Resolve to the 5-bin Emotion via `[emotion_bins]` config. Return `list[EmotionPrediction]`. Persist `emotion/segments.json`.

### Task 4 — Runner reordering + ref-WAV slicing relocation

In `runner.py`:
1. Strip `pick_refs` call from the diarize stage; diarize now returns `Diarization` with empty `refs`.
2. Add `"emotion"` branch: `ctx.extra["emotion_predictions"] = stage_obj.detect(state["separate"], state["diarize"], ctx)`.
3. Immediately after, call the re-shaped `pick_refs(turns, overlaps, predictions, speech_vocals, refs_dir, min_ref_seconds)` and assign the result back to `state["diarize"].refs`.
4. Update `apply_speaker_tags` call site to also call `apply_emotion_tags(transcript, ctx.extra["emotion_predictions"])` immediately after.
5. Add `"emotion"` branch to `_load_resume_artifact`.

### Task 5 — Emotion-aware ref picker

Rework `stages/diarize/ref_picker.pick_refs` to take `predictions` and produce `dict[(speaker, bin), SpeakerRef]`. Per `(speaker, bin)`: filter turns whose dominant overlap is that bin, drop those intersecting `OverlapInterval`s, drop turns < `min_ref_seconds`, pick longest, trim 100 ms inward, slice from `speech_vocals`. Write to `diarize/refs/<spk>/<bin>.wav`. Unit test (`tests/test_emotion_ref_picker.py`).

### Task 6 — Emotion sentence tagging

Implement `stages/asr/emotion_tag.apply_emotion_tags(transcript, predictions)`. Same algorithm shape as `speaker_tag` (dominant ≥85% / split / orphan-fallback). Unit test (`tests/test_emotion_tag.py`).

### Task 7 — Translate passthrough + Qwen3 prompt injection

Update `TranslatedSentence` construction in `stages/translate/adapter.py` and Qwen3 candidate to thread `emotion` through. In `qwen3_instruct_mlx.py`, inject the emotion tag into the user message template when `emotion` is set. Confirm NLLB candidates pass-through without behavior change. Grep `TranslatedSentence(` for coverage.

### Task 8 — TTS adapter emotion-aware refs + guidance

In `stages/tts/adapter.py`: lookup ref via `(speaker_id, emotion)`; fallback to `(speaker_id, "neutral")`; fallback to global. Pass per-call `guidance_scale` from `cfg["emotion_guidance"][emotion]` (defaulting to a baseline if missing). Extend `omnivoice_mlx.synthesize` to accept `guidance_scale: float | None = None` and pass to `model.generate(...)`. Update `piper_hye` to accept-and-ignore the new kwargs (uniform adapter signature).

### Task 9 — Voice-leak fix (#4)

Investigate the leak first: listen to existing `runs/20260526-001510-masha-2-5min/dub.wav` boundaries; identify whether the leak is at sentence start, end, or both. Add per-sentence `afade=in:d=0.05` + `afade=out:d=0.05` to the synthesized track in `stages/mux/ffmpeg_pad_stretch.py`. If the leak source turns out to be Demucs accompaniment residual rather than synthesized-track edges, switch the fix to ducking the accompaniment under the synthesized line by ~6-12 dB during the line and crossfading the duck at boundaries. **If neither fix resolves it on a representative clip, bump #4 to its own spec and revert this task** — don't ship a partial fix.

### Task 10 — Config + deps + manifest

`patmir.toml`: add `[pipeline] emotion`, `[[stages.emotion.candidates]]`, `[emotion_bins]`, `[emotion_guidance]` tables. `pyproject.toml`: add `funasr>=1.1`, `modelscope>=1.20`; `uv sync`. Extend `Manifest`/`patmir runs show` to surface per-speaker emotion-bin coverage (how many bins each speaker covered, which were fallbacks).

### Task 11 — End-to-end verification

Run on `samples/masha-2-5min.mkv`:

1. `uv run patmir dub samples/masha-2-5min.mkv` runs all 9 stages without error.
2. `runs/<latest>/emotion/segments.json` exists; spot-check a few high-emotion windows by ear — predicted emotion is plausible.
3. `runs/<latest>/diarize/refs/<spk>/<bin>.wav` — multiple bins for at least one speaker (Masha is animated enough that we expect ≥3 bins). Missing-bin fallbacks are logged.
4. `runs/<latest>/transcript.json` — every sentence has both `speaker_id` and `emotion` non-null.
5. Listen to `runs/<latest>/output/*.mkv` side-by-side with `runs/20260526-001510-masha-2-5min/dub.wav` (existing baseline). Specifically:
   - Emotion: does Masha sound *animated* vs *calm* in lines where the source shows that contrast? Pass if there's any perceptible difference; we are not chasing parity with the source.
   - Voice leak (#4): is the leak at sentence boundaries gone or substantially reduced? Pass = unanimous "better" from a re-listen.
6. `patmir runs show <id>` — summary surfaces emotion-bin coverage.
7. Update `docs/architecture/quality-issues.md` — mark #3 and #4 as "addressed; revisit if regression."

### Task 12 — Docs backfill (always-loaded tier + touched on-demand files)

Update the always-loaded tier first (it's what every future conversation will see):

1. `docs/architecture/current-models.md` — bump `emotion` row (new), update `tts` row (per-(speaker, emotion) ref + guidance_scale), update `mux` row (50 ms afade).
2. `docs/architecture/quality-issues.md` — mark #3 emotion and #4 voice-leak as `addressed (2026-05-26 — spec: emotion-transfer-and-quality-docs)` if verification passes; else mark `partial` with notes.
3. `docs/architecture/decisions.md` — add any new ADRs that came out of implementation (e.g. "emotion2vec confidence floor at 0.5", "emotion-bin = 5 categories", "mux afade duration = 50 ms"); each gets a one-liner here and a full file in `decisions/`.

Then the on-demand tier:

4. `docs/architecture/stages/tts/omnivoice-mlx.md` — document per-call `guidance_scale` lever + emotion-bin ref strategy.
5. `docs/architecture/stages/emotion/current.md` + `emotion2vec-plus-base.md` — fully populate.
6. `docs/architecture/data-contracts.md` — reflect the `Emotion` field additions to `Sentence`/`TranslatedSentence`/`SpeakerTurn` and the `Diarization.refs` key change.
7. `docs/architecture/pipeline.md` — update the ASCII diagram to include the `emotion` stage.

This task is **not optional** — it's the discipline that makes the docs scaffold a one-time investment instead of bit-rot. The CLAUDE.md maintenance contract names this as part of the implementation flow, not a follow-up.

## Verification (summary)

Already in Task 11; reproduced here for visibility:

- `uv run patmir dub samples/masha-2-5min.mkv` — runs 9 stages clean
- Listen side-by-side vs baseline run — emotion perceptibly closer to source; sentence-boundary leak reduced
- All sentences carry `speaker_id` + `emotion`
- `runs show` displays emotion coverage
- Docs scaffold renders cleanly (just open in a Markdown viewer; no build step)

## Risks

- **SER misclassification on Russian source.** emotion2vec is multilingual-trained but biased toward English/Mandarin. Mitigation: confidence threshold — predictions below 0.5 fall to `neutral`.
- **Per-character emotion coverage is sparse.** Some characters may only have neutral and one other bin. Documented and accepted (fallback chain handles it).
- **`guidance_scale` is a blunt instrument.** Real prosody control isn't possible with OmniVoice's API. Accepted; gains will be marginal but compound with the ref-clip switching.
- **Voice-leak fix may not be a single `afade` knob.** Task 9 has an explicit "bump to its own spec" branch if the simple fix doesn't land.
- **`Diarization.refs` key change** ripples through manifest JSON serialization. Migration: bump manifest schema version; old runs simply can't be `--resume-from` past `diarize`.
- **funasr / modelscope are heavy deps.** First-run download is ~500 MB. Document in `docs/architecture/stages/emotion/current.md`.
- **EA tokenizer/phoneme limits** — OmniVoice handles EA via clone but exact phoneme coverage is unmeasured. If the new per-emotion refs land on phonetically-narrow clips, output may regress. Mitigation: `min_ref_seconds` floor stays at 3.0 — same as Phase 2.

## Out of scope (deferred specs)

- **#1 cloning fidelity** — re-eval after emotion-aware refs land; many "wrong character" reports may turn out to be emotion-mismatched refs.
- **#2 translation quality** — separate spec. Likely entails OpenRouter integration as a comparison ceiling once we've documented Qwen3+NLLB as the local floor.
- **#5 overlap handling** — separate spec. Requires deciding how to render simultaneous speakers (mix? duck the lower-priority? duplicate channels?).
- **OpenRouter integration** — separate spec, gated by per-stage local-ceiling verdicts captured in `docs/architecture/decisions/`.
- **Full-episode (20 min) validation** — separate spec; current spec validates on 2.5 min as before.
- **YouTube ingest via yt-dlp** — separate spec.
