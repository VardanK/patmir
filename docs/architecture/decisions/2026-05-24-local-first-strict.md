# Local-first strict

**Date:** 2026-05-24
**Status:** locked (foundational policy)

## Context

Patmir runs on a single workstation. Privacy, offline capability, and zero per-use cost are values, not afterthoughts. Cloud APIs are available (GPT, Claude, ElevenLabs, Azure, Google) and are tempting per-stage upgrades, but unconditional cloud use would undermine those values and create a per-job cost.

## Decision

Cloud candidates are declared in `patmir.toml` for documentation, but are **never enabled by default**. Cloud is a per-stage opt-in fallback only, brought in *after* the local set for that specific stage has been tried and judged inadequate.

The escalation order for each stage is:

1. Try the current best local candidate.
2. If inadequate, try other local candidates (other models, larger sizes, different families).
3. If still inadequate, opt in to a cloud fallback **for that specific stage only**. Other stages remain local.
4. If cloud also fails to clear the quality bar, pause and reassess (wait for the ecosystem, or scope down).

Per-stage isolation is the key: enabling cloud TTS does not enable cloud translation; they are independent decisions captured in independent ADRs.

## Consequences

- Anytime "use a cloud model" is proposed, the local-set-exhausted prerequisite must be documented (typically in the relevant stage's evaluation files).
- The `enabled = false` cloud entries in `patmir.toml` exist to make opt-in a one-line config change, not a code-edit task.
- Cloud opt-in *always* requires a new ADR in `decisions/` naming the cloud model, the local exhaustion evidence, and the revisit trigger for going back to local.

## Revisit trigger

If cloud becomes free, private, and offline-capable simultaneously (which is unlikely), the policy might relax. Until then, this stays.

## Source

- `agent-os/product/mission.md` §"Solution"
- `agent-os/product/tech-stack.md` §"Translation" + §"TTS"
- Memory: `feedback_local_first_strict`
