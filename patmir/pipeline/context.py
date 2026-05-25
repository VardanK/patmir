"""JobContext — per-run state passed to every stage.

Holds the working directory, source path, detected source language (set by
ASR), pipeline options, and a `progress(...)` callback that appends to
`<job_dir>/progress.jsonl` for the CLI tail + FastAPI SSE stream.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineOptions:
    stretch_cap: float = 0.20
    asr_int8: bool = True
    keep_intermediate: bool = True
    # Linear gain applied to the accompaniment bed when mixing the AR voice
    # on top. 1.0 = unchanged source mix volume; 0.6 = -4 dB (a touch quieter
    # so the dialogue sits above the music). The voice itself is normalized
    # to roughly match the original dialogue loudness.
    music_bed_gain: float = 0.6
    # Extra dB applied to the AR voice over its native level. 0 = identity.
    voice_gain_db: float = 0.0


@dataclass
class JobContext:
    job_dir: Path
    source_path: Path
    options: PipelineOptions
    source_lang: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def progress(self, stage: str, pct: float, msg: str = "") -> None:
        record = {
            "ts": time.time(),
            "stage": stage,
            "pct": round(pct, 4),
            "msg": msg,
        }
        path = self.job_dir / "progress.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
