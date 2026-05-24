# Phase 0 — Armenian Feasibility Gate

## Context

Patmir's roadmap (`agent-os/product/roadmap.md`) explicitly gates all downstream work on a Phase 0 feasibility check: can currently available models produce Eastern-Armenian (`hy_AM`) translation + Eastern-Armenian audio that is good enough for a child to enjoy? Armenian is lower-resource for most modern translation and TTS systems and is the single largest technical risk for the product. No Phase 1 pipeline plumbing is justified until this gate produces a written go/no-go.

The repository today contains only product docs and an empty standards index; there is no Python code yet. This plan stands up the minimum project scaffolding and a **lightweight, reusable eval harness** so the Phase 0 verdict is reproducible — the roadmap explicitly anticipates re-running Phase 0 as the model ecosystem shifts (Phase 2). The harness is deliberately small: Phase 0 is research, not the pipeline.

**Decisions baked in (from shaping conversation):**

- **Lightweight harness** — small, reusable Python scripts per stage, model candidates declared in config, outputs written to disk.
- **Both EN→HY and RU→HY** language pairs, per the roadmap.
- **Local-first, strictly.** Cloud candidates (GPT/Claude/Google TTS) are declared in config but ship `enabled = false`. The Phase 0 pass evaluates *only* local candidates. Cloud is brought in per-stage **only if** that stage's local set is judged inadequate after the workstation-GPU pass. This is stricter than the roadmap's original "cloud baseline as ceiling reference" framing — it follows `mission.md`'s local-first principle.
- **CPU/Mac first** — the target workstation GPU is not yet available. The first pass evaluates what runs on CPU/Mac. GPU-heavy local candidates (NLLB-3.3B, XTTS-v2, Bark, MADLAD-400) are deferred to a second local pass when hardware arrives; the harness must accept new candidates as config entries without code changes.
- **Scoring: author (fluent Armenian) on an informal 1–5 rubric**, with the target child providing the final "watchable for hours?" judgement on the shortlisted TTS outputs. No additional reviewers, no automated metrics in this round.

**Exit criteria (verbatim from roadmap):** *A written one-pager naming the winning translator + TTS per language pair, with audio samples attached, and an explicit go/no-go on Phase 1.*

**Explicitly out of scope:** any pipeline plumbing (no `Stage` protocol, no ingest/ASR/mux/output stages), no UI, no Phase 1 model swaps in config, no Phase 2 diarization or voice cloning.

---

## Task 1: Save Spec Documentation

Create `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/` with:

- `plan.md` — this full plan.
- `shape.md` — shaping notes (scope from roadmap; decisions on harness shape, language pairs, CPU/Mac-first, scoring model).
- `standards.md` — none apply (`agent-os/standards/index.yml` is empty). File records that explicitly.
- `references.md` — no in-repo references (greenfield); records the roadmap + tech-stack sections that drove decisions.
- `visuals/` — empty (no mockups; Phase 0 is research).

## Task 2: Project scaffolding

Initialize a minimal Python project at the repo root:

- `pyproject.toml` with `uv`-friendly metadata (Python 3.11+, per `tech-stack.md`).
- `.python-version`, `.gitignore` (Python + macOS + audio/large-file patterns + `eval_runs/` outputs).
- Layout:
  - `patmir/eval/` — the eval harness package.
  - `patmir/eval/candidates/` — one module per candidate (translator or TTS), each implementing a small protocol.
  - `eval_data/` — curated eval set (input texts/audio + reference translations).
  - `eval_runs/` — outputs (per-run timestamped folders; gitignored).
- One `patmir.toml` at repo root with two sections: `[eval.translation]` and `[eval.tts]`, each listing candidate IDs to run. Per `tech-stack.md`: *"per-stage config loaded from a single `patmir.toml` so model swaps are config changes, not code changes."*

No package needs to be installable yet; running scripts via `uv run` is enough.

## Task 3: Build the eval set

Curate ~20 short utterances per source language (EN, RU), following the roadmap's categories:

- **Cartoon dialogue** (short, exclamatory, child register) — ~8 items.
- **Simple narration** — ~5 items.
- **Idioms / colloquial** — ~3 items (these stress translation more than TTS).
- **Proper nouns** (Armenian names, foreign names that must be transliterated cleanly) — ~4 items.

Format on disk:

- `eval_data/translation/en.jsonl`, `eval_data/translation/ru.jsonl` — `{id, category, source, reference_hy?}`. `reference_hy` is the author's gold translation, filled in for as many items as is practical to give a chrF-style sanity reference even if not used as the scoring source of truth.
- `eval_data/tts/hy.jsonl` — Armenian utterances to synthesize. Seeded from the author's gold translations (so TTS is judged independently of translation quality).

## Task 4: Translation eval harness

`patmir/eval/translation_eval.py`:

- Defines a `Translator` protocol: `translate(text: str, source_lang: str) -> str`. Target is Armenian; no parameter needed.
- Each candidate is a module in `patmir/eval/candidates/translation/` exporting a factory that builds a `Translator` from its config block.
- Candidates enabled for the CPU/Mac-first pass:
  - **`nllb_200_distilled_600m`** — `facebook/nllb-200-distilled-600M` via `transformers` on CPU/MPS. Smallest NLLB; runs on Mac. This is the *only* default-on candidate for this pass.
- Candidates declared but disabled (`enabled = false`):
  - **Deferred to GPU pass:** `nllb_200_3.3b`, `madlad_400`, prompted local LLM (e.g. Qwen2.5).
  - **Cloud fallbacks:** `gpt_cloud`, `claude_cloud`. Per the local-first policy, turn these on *only* after the local set (CPU pass + GPU pass) has been evaluated and judged inadequate.
- Runner: iterates the eval set × enabled candidates, writes one output file per `(candidate, lang_pair)` plus a single combined `eval_runs/<timestamp>/translation_results.md` with utterances laid out side-by-side per candidate for the author to score.
- Captures per-call latency. No automated quality metric in this round.

## Task 5: TTS eval harness

`patmir/eval/tts_eval.py`:

- Defines a `TTS` protocol: `synthesize(text: str) -> bytes` (returns WAV bytes; harness writes to `.wav`).
- Candidates enabled for the Mac-first pass:
  - **`piper_hye`** — community Piper voice `davit312/piper-TTS-Armenian` (`hy_AM-gor-medium`). ONNX, runs on CPU. **This is currently the only known local Eastern-Armenian TTS** — if it fails the bar there is no obvious local alternative without waiting on the ecosystem.
- Candidates explicitly removed:
  - **`mms_tts_hye`** — does not exist on HuggingFace. `facebook/mms-tts-hye` returns 404; Meta only ships `mms-tts-hyw` (Western Armenian, wrong dialect). The original plan listed MMS-TTS as the local baseline; that's incorrect. See `tech-stack.md` Stage 4.
- Candidates declared but disabled (`enabled = false`):
  - **Deferred to GPU pass:** `xtts_v2`, `bark`. Both list Armenian but do not distinguish Eastern vs Western; dialect must be verified by ear before relying on them.
  - **Cloud fallbacks:** `google_tts_cloud` (Eastern Armenian `hy-AM-*` voice family). Per the local-first policy, turn this on *only* after the local set (CPU pass + GPU pass) has been evaluated and judged inadequate.
- Runner: synthesizes every Armenian eval utterance × enabled candidates, writes WAVs into `eval_runs/<timestamp>/tts/<candidate>/<utt_id>.wav`, plus an `index.html` (single static file, no server) that lays out a 3-column table per utterance (text + audio players per candidate) so the author can A/B/C listen easily.
- Captures per-utterance synthesis time; flags candidates that take >30s/utterance on CPU as "GPU-pass only."

## Task 6: Scoring sheet

`eval_runs/<timestamp>/scoresheet.csv` written by the runner, pre-filled with `(utt_id, category, candidate)` rows and empty score columns:

- **Translation columns:** `meaning_preserved` (1–5), `register` (1–5), `naturalness` (1–5), `notes`.
- **TTS columns:** `intelligibility` (1–5), `naturalness` (1–5), `prosody` (1–5), `child_would_watch_hours` (Y/N/maybe), `notes`.

A small `patmir/eval/summarize.py` reads the filled scoresheet and prints a per-candidate mean per metric, plus per-category breakdowns, to support the go/no-go writeup. No statistical machinery — N ≈ 20 is too small for it to matter.

## Task 7: Go/no-go one-pager

`agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/verdict.md`:

- Winning translator per language pair (EN→HY, RU→HY).
- Winning TTS engine.
- Audio sample links (relative paths into `eval_runs/<timestamp>/tts/`).
- Quality floor reached (per-metric means).
- **Explicit go/no-go on Phase 1**, with reasoning that maps back to *"would I let a child watch hours of this?"*
- A line item on what changes when the GPU workstation arrives: which deferred candidates should be re-evaluated before Phase 1 commits to a TTS choice.

## Verification

- `uv run python -m patmir.eval.translation_eval --config patmir.toml` produces a `translation_results.md` for the active candidates with no errors; manual spot-check confirms outputs are Armenian-script and non-empty.
- `uv run python -m patmir.eval.tts_eval --config patmir.toml` produces playable `.wav` files for every (candidate × utterance) pair; opening `index.html` in a browser plays each candidate inline.
- Author fills `scoresheet.csv` for at least one full (lang_pair × candidate × category) sweep; `summarize.py` prints per-candidate means.
- `verdict.md` exists with audio links resolving to real files and an explicit go/no-go sentence.
- Disabled GPU-pass candidates remain present in `patmir.toml` as commented-out entries (or `enabled = false`) so re-running on the workstation is a config toggle, not a code change.
