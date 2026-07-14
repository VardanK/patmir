# Patmir — project context

Patmir (պատմիր, "tell the story") translates foreign-language video (local file → Eastern Armenian dub) for the author's hearing-aid-wearing son. Pipeline runs locally on a 32 GB Apple Silicon Mac; cloud is a per-stage opt-in fallback only. Target dialect is **Eastern Armenian** (`hy_AM`) — Western Armenian (`hyw`) output is automatically rejected. The product is a workstation tool, not a SaaS; a polished UI is not a goal.

## Always-loaded reference docs

The files below are imported into every conversation's context. Read them before answering any question about pipeline behavior, model choices, or quality status. Do not re-investigate model alternatives that already have a decision recorded.

@docs/architecture/pipeline.md
@docs/architecture/current-models.md
@docs/architecture/decisions.md
@docs/architecture/quality-issues.md

## Maintenance contract

When you (the agent) do any of the following, update the corresponding doc **in the same change** — without being asked:

| Action                                  | Files to update                                                                |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| Swap an active model                    | `current-models.md` + `stages/<stage>/current.md` + relevant model file        |
| Investigate and reject a model          | `stages/<stage>/<model>.md` (status: rejected, with reason + date)             |
| Lock a non-trivial decision             | New `decisions/<YYYY-MM-DD>-<slug>.md` + one-line entry in `decisions.md`      |
| Address or regress a quality issue      | `quality-issues.md` (status + last-touched date)                               |
| Change a data contract (stage.py shape) | `data-contracts.md`                                                            |
| Add a new pipeline stage                | `pipeline.md` + new `stages/<name>/` subdir                                    |

The on-demand tier (`stages/*/`, `decisions/*.md`, `data-contracts.md`, `language-support.md`, `runs.md`, `glossary.md`, `performance.md`) is **not** auto-loaded. Read those files when the task is in that area. `decisions.md` is the canonical list of "why we chose X" — if the user asks "why are we using X?" or "should we try Y?", check `decisions.md` first and only re-investigate if there's a revisit trigger or new evidence.

## Repo orientation

- `patmir/pipeline/` — stage protocols + runner + per-stage candidate code
- `patmir/eval/candidates/` — pluggable model candidates (translation, tts, asr)
- `patmir.toml` — single source of truth for which candidate runs each stage
- `agent-os/specs/` — historical spec folders (one per phase / feature)
- `agent-os/product/` — mission, roadmap, tech-stack
- `runs/` — per-job artifact directories
- `docs/architecture/` — this knowledge base
