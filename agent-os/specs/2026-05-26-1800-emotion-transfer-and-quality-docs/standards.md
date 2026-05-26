# Standards applied

`agent-os/standards/index.yml` is currently empty. No project standards apply to this work yet.

The implicit standards followed (not yet codified anywhere):

- **Local-first strict** — captured in `docs/architecture/decisions/2026-05-24-local-first-strict.md`.
- **Eastern Armenian only** — captured in `docs/architecture/decisions/2026-05-24-eastern-armenian-only.md`.
- **Stage protocols** — typed dataclasses in `patmir/pipeline/stage.py`; one `Protocol` per stage; one candidate selected per stage via `patmir.toml [pipeline]`.
- **Per-stage candidate registries** — `[[stages.<stage>.candidates]]` (or legacy `[[eval.<stage>.candidates]]`) tables with `id` + `module` + per-candidate config; loaded by `patmir/pipeline/config.py::_STAGE_NAMESPACE`.
- **Lazy stage loading** — heavy model loads happen inside the candidate's `build(config)` call, not at module-import time. Memory is freed between stages with `gc.collect()` + device-specific cache clears.

If `agent-os/standards/` is populated later, this file should be regenerated to list the relevant standards with the rationale for each.
