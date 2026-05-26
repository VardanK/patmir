# Auto-loaded architecture docs

**Date:** 2026-05-26
**Status:** locked

## Context

The catalyst was discovering, mid-conversation, that F5-TTS and IndexTTS-2 — the two obvious TTS upgrades for emotion control — both fail the Eastern-Armenian gate. The investigation took ~15 minutes of web research and synthesis. Without somewhere durable to land that work, the same investigation would happen again next time the topic comes up.

Memory (`.claude/memory/`) auto-loads but is short-form. Spec folders (`agent-os/specs/`) capture per-feature context but are not auto-loaded. The repository has no general "what are we doing and why" knowledge base that the agent reads on every conversation start.

## Decision

Establish `docs/architecture/` as a two-tier Markdown knowledge base:

**Always-loaded tier (auto-imported by `CLAUDE.md` via `@`-imports):**

- `pipeline.md` — stage order, data flow diagram, `ctx.extra` plumbing
- `current-models.md` — one row per stage: active candidate + key knob values
- `decisions.md` — one-bullet index of every locked decision, with link + revisit trigger
- `quality-issues.md` — live tracker of observed problems, owner stage, status, spec tackling each

**On-demand tier (read by the agent when working in that area):**

- `index.md` — human-facing table-of-contents
- `data-contracts.md` — every dataclass from `stage.py`, fields + meanings
- `language-support.md` — EA-specific notes (hy_AM vs hyw, which models produce which)
- `runs.md` — what every file in `runs/<job>/` is; how to debug a failed run
- `glossary.md` — SER, VAD, DiT, ref clip, flow matching, etc.
- `decisions/<YYYY-MM-DD>-<slug>.md` — full ADRs, one per locked decision
- `stages/<stage>/index.md` + `current.md` + per-model files

`CLAUDE.md` contains the "Maintenance contract" — a table naming which docs to update for which action (model swap, model rejection, new decision, quality-issue change, data-contract change, new stage). This is the discipline that prevents bit-rot.

## Consequences

- Every conversation auto-loads ~200-400 lines: the pipeline shape, the active models, the decisions log, the quality status. Context cost is small but real.
- The on-demand tier is *not* auto-loaded — kept off-tree from the conversation's default context to avoid bloat.
- Model rejections (F5-TTS, IndexTTS-2, XTTS-v2, MMS-hye, etc.) live in `stages/tts/<model>.md` so future "have we tried X?" questions are answered without re-investigation.
- New decisions made mid-conversation become ADRs in the same change, not "I'll write that up later" promises.

## Revisit trigger

- `CLAUDE.md` becomes too large to auto-load efficiently (>~1000 lines of imports). Mitigation: collapse the always-loaded tier further or move some content on-demand.
- Claude Code replaces `@`-imports with a better mechanism. Then this ADR adapts to the new shape.

## Source

- This conversation (2026-05-26 docs-scaffold proposal + user request to "make the docs accessible to the AI agent... done automatically, link the markdown files that way").
