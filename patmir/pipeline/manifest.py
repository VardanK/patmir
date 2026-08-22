"""Job manifest — persisted per-stage status + artifact paths.

Written as `<job_dir>/manifest.json` after every stage transition. The FastAPI
UI reads this for the run-detail page; the CLI reads it for `runs show`.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from patmir.pipeline.stage import STAGE_ORDER


@dataclass
class StageStatus:
    name: str
    state: str = "pending"  # pending | running | done | failed
    started_at: float | None = None
    finished_at: float | None = None
    # Wall-clock seconds spent building this stage's candidate (model load /
    # quantization). Included in [started_at, finished_at]; broken out so the
    # benchmark can separate model-load cost from compute. Dominant in the
    # watch-along path, which rebuilds every stage twice per section.
    build_s: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Manifest:
    job_id: str
    source_path: str
    source_lang: str | None = None
    created_at: float = field(default_factory=time.time)
    stages: dict[str, StageStatus] = field(default_factory=dict)
    # Where the finished deliverables were copied, when [storage].library_root
    # is configured (see patmir.pipeline.publish). None = not published.
    published_dir: str | None = None

    def __post_init__(self) -> None:
        # Always ensure every stage in STAGE_ORDER has an entry. This keeps
        # older manifests forward-compatible when new stages are added between
        # versions (e.g. a job started before `separate` was introduced).
        for name in STAGE_ORDER:
            self.stages.setdefault(name, StageStatus(name=name))

    def stage(self, name: str) -> StageStatus:
        return self.stages[name]

    def to_path(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "source_path": self.source_path,
            "source_lang": self.source_lang,
            "created_at": self.created_at,
            "published_dir": self.published_dir,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def from_path(cls, path: Path) -> "Manifest":
        data = json.loads(path.read_text())
        stages = {k: StageStatus(**v) for k, v in data["stages"].items()}
        return cls(
            job_id=data["job_id"],
            source_path=data["source_path"],
            source_lang=data.get("source_lang"),
            created_at=data.get("created_at", time.time()),
            stages=stages,
            published_dir=data.get("published_dir"),
        )
