# Standards

No `agent-os/standards/` standards apply to this work — it's local Python pipeline glue with no API surface, no database schema, no frontend.

The relevant repo conventions (followed implicitly):

- Python `from __future__ import annotations` + dataclass + Protocol typing as in existing stage modules.
- One concrete candidate per stage; capabilities are duck-typed via signature inspection plus optional class attributes (existing pattern in `tts/adapter.py` for `ref_audio` / `ref_text` / `guidance_scale` / `emotion` kwargs). The new `ref_text_optional` flag follows this same pattern.
- New JSON artifacts written under `<job_dir>/<stage>/` and indexed in `docs/architecture/runs.md`.
- `pytest` for tests under `patmir/tests/` (the directory is new; existing repo has no tests yet).
