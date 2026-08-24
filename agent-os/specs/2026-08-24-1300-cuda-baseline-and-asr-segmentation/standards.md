# Standards for CUDA baseline + ASR segmentation

`agent-os/standards/index.yml` contains only a header comment
(`# Agent OS Standards Index`) with no entries, so **no standards apply to this
work**.

The governing conventions for this spec come from `CLAUDE.md` instead:

- **The maintenance contract.** Swapping or measuring a model, addressing a quality
  issue, and locking a non-trivial decision each require updating the named docs *in
  the same change*. Task 10 exists to satisfy it: `current-models.md`,
  `stages/<stage>/current.md`, `quality-issues.md`, a new ADR plus its one-line entry
  in `decisions.md`.
- **`decisions.md` is checked before investigating.** Every model choice this spec
  touches already has a recorded decision; none of their revisit triggers had fired
  except the CUDA port ADR's ("the measured 3090 baseline lands far from the
  projection"), which this spec fires deliberately.
- **`patmir.toml` is the single source of truth** for which candidate runs each
  stage; docs mirror it and lose if they disagree.
- **Eastern Armenian (`hy_AM`) only.** Not at risk here — no translate or TTS
  candidate changes.
