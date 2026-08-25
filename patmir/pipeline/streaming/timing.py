"""Per-section timing, shared by the live stream path and `patmir bench`.

Watch-along used to record **no** timing at all in production: `sections.json`
held `{index, start, end, file}` and `StreamRecord.to_public()` exposed
`status`/`sections_done`/`window_end`/`duration`. Every per-section number in
`performance.md` came from `bench` re-reading `manifest.json` afterwards, so a
slow live stream could only be diagnosed by archaeology on `runs/` — which is
exactly what had to be done to explain the 2026-08-25 session.

This module is the one reader. `bench` and `StreamManager` both call
`section_timing`, so the live path and the benchmark cannot report different
numbers for the same run.

## `other_s` is the point

`wall_s - sum(stage_wall_s)` is work that no stage owns and no manifest records:
the ffmpeg slice and trim, the ECAPA embedding of every ref clip, and the
wav2vec2 age-gender profiling in `_resolve_identities`. Measured at ~1.6 s on
every section of the 2026-08-25 session (~5% of section wall) and **15.5 s** on
the first prelude of a fresh server, where it also carries the one-time lazy
load of the two models neither `stage_cache` nor `prewarm_stage_cache` covers.
Surfacing it as its own field is the difference between "the prelude took 72 s"
and "14.3 s of the prelude was a model load nobody prewarms".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from patmir.pipeline.manifest import Manifest
from patmir.pipeline.stage import STAGE_ORDER

# The two `run_job` halves a watch-along section runs (streaming/section.py):
# analysis (ingest→asr) then dub (translate→mux). `output` never runs in
# watch-along — sections stop at mux — but it stays in DUB_STAGES so the split
# covers STAGE_ORDER exactly and a future change cannot silently drop a stage.
ANALYSIS_STAGES = ["ingest", "separate", "diarize", "emotion", "asr"]
DUB_STAGES = ["translate", "tts", "mux", "output"]


def stage_wall(st) -> float | None:
    """Wall seconds for one stage, or None if it never ran."""
    if st is not None and st.started_at and st.finished_at:
        return st.finished_at - st.started_at
    return None


def section_timing(
    *,
    index: int,
    start: float,
    end: float,
    job_dir: Path,
    wall_s: float | None,
    new_speakers: list[str] | None = None,
    buffer_ahead_s: float | None = None,
    produced_at: float | None = None,
) -> dict[str, Any]:
    """One section's timing, read back out of its `manifest.json`.

    `job_dir` holds ONE manifest carrying both halves, because the dub resumes
    into the analysis job dir (`resume_job_dir=`), so a single read covers the
    whole section.

    A missing or unreadable manifest yields zeroed stage walls rather than
    raising: telemetry must never be able to fail a dub that already succeeded.
    """
    try:
        manifest = Manifest.from_path(job_dir / "manifest.json")
        per_stage = {n: (stage_wall(manifest.stage(n)) or 0.0) for n in STAGE_ORDER}
        build_total = sum((manifest.stage(n).build_s or 0.0) for n in STAGE_ORDER)
    except Exception:  # noqa: BLE001 — see docstring
        per_stage = {n: 0.0 for n in STAGE_ORDER}
        build_total = 0.0

    span = max(0.0, end - start)
    stage_sum = sum(per_stage.values())
    rec: dict[str, Any] = {
        "index": index,
        "start": round(start, 2),
        "end": round(end, 2),
        "section_s": round(span, 2),
        "wall_s": round(wall_s, 2) if wall_s is not None else None,
        "analysis_wall_s": round(sum(per_stage[n] for n in ANALYSIS_STAGES), 2),
        "dub_wall_s": round(sum(per_stage[n] for n in DUB_STAGES), 2),
        "build_s": round(build_total, 2),
        "rtf": round(wall_s / span, 3) if (wall_s is not None and span) else None,
        "stage_wall_s": {n: round(per_stage[n], 2) for n in STAGE_ORDER},
        "new_speakers": list(new_speakers or []),
    }
    # Work no stage owns — see the module docstring.
    rec["other_s"] = (
        round(max(0.0, wall_s - stage_sum), 2) if wall_s is not None else None
    )
    if buffer_ahead_s is not None:
        rec["buffer_ahead_s"] = round(buffer_ahead_s, 2)
    if produced_at is not None:
        rec["produced_at"] = round(produced_at, 3)
    return rec
