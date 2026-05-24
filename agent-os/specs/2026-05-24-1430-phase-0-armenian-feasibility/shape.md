# Phase 0 — Shaping Notes

## Scope

Stand up a lightweight, reusable Python harness to evaluate local Eastern-Armenian (`hy_AM`) translator candidates and local Eastern-Armenian TTS candidates against a curated ~20-utterance eval set per source language (EN, RU), with the author scoring outputs on an informal 1–5 rubric. Produce a written go/no-go on Phase 1 with audio samples attached.

**Target dialect is Eastern Armenian only.** See `mission.md` — Western Armenian (`hyw`) is the wrong dialect for the target user; a model that only supports Western Armenian is not a viable candidate, even if it's the strongest "Armenian" option on a model card.

This is the feasibility gate the roadmap insists on before any Phase 1 pipeline work is justified.

## Decisions

- **Harness shape: lightweight, reusable.** Small Python scripts per stage with candidates declared in `patmir.toml`. Reasoning: the roadmap explicitly anticipates re-running Phase 0 as the model ecosystem shifts (Phase 2), so a one-off notebook would leave value on the table for a marginal time saving today.
- **Language pairs: both EN→HY and RU→HY.** Per the roadmap; Russian-source content is common and may translate better via Russian as a pivot.
- **Local-first, strictly.** Cloud candidates (GPT, Claude, Google TTS, etc.) are declared in `patmir.toml` but ship `enabled = false`. The first pass evaluates only local candidates. Cloud is brought in per-stage **only if** that stage's local set is judged inadequate after the workstation-GPU pass. Reasoning: `mission.md` makes local-first a *core value* (privacy, offline, zero per-use cost), so enabling cloud APIs by default — even as a research "ceiling reference" — silently violates that principle and forces the user to set up API keys to run the harness. This is stricter than the original roadmap wording; the roadmap has been updated to match.
- **CPU/Mac first.** Target workstation GPU not yet available. The first pass evaluates only CPU/Mac-runnable local candidates. GPU-heavy local candidates (NLLB-3.3B, XTTS-v2, Bark, MADLAD-400, prompted 70B-class LLMs) are declared in config but disabled and form the **second local pass** once hardware arrives. Only after both local passes fall short for a given stage is the cloud fallback for that stage considered.
- **Scoring: author + son, informal 1–5 rubric.** No additional reviewers, no automated metrics (chrF/MOSNet) in this round. The roadmap's hard requirement is *"would I let a child watch hours of this?"* — the target child's reaction is the source of truth on shortlisted TTS outputs.
- **Pipeline plumbing is out of scope.** No `Stage` protocol, no ingest/ASR/mux/output stages, no UI. Phase 0 is research only.

## Mid-flight finding: Eastern-Armenian TTS ecosystem is thin

During harness setup we discovered that `facebook/mms-tts-hye` (the Eastern-Armenian MMS-TTS voice the tech-stack assumed existed) does **not** exist on HuggingFace — Meta only ships `mms-tts-hyw` (Western Armenian). Of the original four TTS candidates listed in `tech-stack.md`, only one Eastern-Armenian local option turned up after a real search: the community Piper voice `davit312/piper-TTS-Armenian` (`hy_AM-gor-medium`).

This raises the stakes of Phase 0: if the Piper voice doesn't clear the "watchable for hours?" bar, the GPU-pass candidates (XTTS-v2, Bark — neither verified for dialect) and the cloud fallback (Google `hy-AM-*`) are the only remaining shots before the product has to wait on the ecosystem. The `tech-stack.md` Stage 4 and `roadmap.md` Phase 0 sections have been updated to reflect this finding.

## Context

- **Visuals:** None. Phase 0 is research and produces no UI.
- **References:** None in-repo (greenfield). The roadmap (`agent-os/product/roadmap.md`) and tech-stack (`agent-os/product/tech-stack.md`) drove every decision. See `references.md` for the specific sections.
- **Product alignment:** Direct execution of Phase 0 as described in `roadmap.md`. The CPU/Mac-first deviation is a sequencing concession (hardware not yet procured), not a scope change — the GPU pass is preserved as deferred config entries so it can run unchanged once the workstation arrives.

## Standards Applied

None. `agent-os/standards/index.yml` is empty; see `standards.md`.
