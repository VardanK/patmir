# Standards for Multi-voice (diarization + per-speaker cloning)

`agent-os/standards/index.yml` is currently empty — no project-wide standards are defined yet. Nothing applies to this spec at the standards level.

When standards are introduced, the most likely fits for this work would be:

- A pipeline-stage convention (typed input/output dataclasses, `module.build(config)` entry point, per-candidate sidecar JSON artifacts) — this work already follows the de-facto convention established in Phase 1 (`patmir/pipeline/stage.py`, `patmir/pipeline/config.py`).
- A local-first / model-download policy (HF token + license handling, fail-fast with actionable errors when the local cache is empty) — relevant once we add pyannote's HF gate; we apply it implicitly here.
