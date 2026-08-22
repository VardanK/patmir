# Standards for configurable storage roots

`agent-os/standards/index.yml` contains only a header comment — no standards are indexed
in this repo. Nothing to apply.

The conventions this work follows instead come from `CLAUDE.md` and the existing code:

- **Smallest correct change.** Replace hardcoded roots; don't restructure the job-dir
  layout or the stage protocols.
- **Preserve existing architecture and backward compatibility.** Defaults reproduce
  today's paths exactly, so an install without a `[storage]` table is unchanged.
- **Maintenance contract** (`CLAUDE.md`): a non-trivial locked decision needs a new ADR in
  `docs/architecture/decisions/` plus a one-line entry in `decisions.md`; a data-contract
  change needs `data-contracts.md`.
