"""Local FastAPI progress UI.

Localhost-only, no auth. Lists runs under `runs/`, shows per-stage status from
each manifest, and streams `progress.jsonl` over Server-Sent Events for live
updates while a job is running.

  patmir ui --port 8765
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from patmir.pipeline.manifest import Manifest
from patmir.pipeline.stage import STAGE_ORDER

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _create_app(runs_root: Path) -> FastAPI:
    app = FastAPI(title="Patmir")
    runs_root = runs_root.resolve()

    def _safe_job_dir(job_id: str) -> Path:
        candidate = (runs_root / job_id).resolve()
        if runs_root not in candidate.parents and candidate != runs_root:
            raise HTTPException(status_code=400, detail="invalid job_id")
        if not candidate.is_dir():
            raise HTTPException(status_code=404, detail="no such run")
        return candidate

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        rows = []
        if runs_root.exists():
            for d in sorted(runs_root.iterdir(), reverse=True):
                if not d.is_dir():
                    continue
                mp = d / "manifest.json"
                if not mp.exists():
                    continue
                m = Manifest.from_path(mp)
                rows.append({
                    "job_id": m.job_id,
                    "source": m.source_path,
                    "lang": m.source_lang or "?",
                    "stages": [(n, m.stage(n).state) for n in STAGE_ORDER],
                })
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {"runs": rows},
        )

    @app.get("/run/{job_id}", response_class=HTMLResponse)
    def run_detail(request: Request, job_id: str):
        job_dir = _safe_job_dir(job_id)
        m = Manifest.from_path(job_dir / "manifest.json")
        artifacts = []
        for n in STAGE_ORDER:
            st = m.stage(n)
            for k, v in st.artifacts.items():
                artifacts.append((n, k, v))
        return TEMPLATES.TemplateResponse(
            request,
            "run.html",
            {
                "job_id": job_id,
                "manifest": m,
                "stages": [(n, m.stage(n)) for n in STAGE_ORDER],
                "artifacts": artifacts,
            },
        )

    @app.get("/run/{job_id}/progress")
    async def progress_sse(job_id: str):
        job_dir = _safe_job_dir(job_id)
        path = job_dir / "progress.jsonl"

        async def stream():
            pos = 0
            while True:
                if path.exists():
                    with path.open() as f:
                        f.seek(pos)
                        for line in f:
                            if line.strip():
                                yield f"data: {line.strip()}\n\n"
                        pos = f.tell()
                # Done condition: read manifest, stop if all stages are terminal.
                mp = job_dir / "manifest.json"
                if mp.exists():
                    m = Manifest.from_path(mp)
                    terminal = {"done", "failed"}
                    if all(m.stage(n).state in terminal for n in STAGE_ORDER):
                        yield "data: {\"event\":\"end\"}\n\n"
                        return
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/run/{job_id}/file/{rest:path}")
    def serve_artifact(job_id: str, rest: str):
        job_dir = _safe_job_dir(job_id)
        target = (job_dir / rest).resolve()
        if job_dir not in target.parents and target != job_dir:
            raise HTTPException(status_code=400, detail="path escape")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(target)

    @app.get("/run/{job_id}/transcript.json")
    def transcript(job_id: str):
        job_dir = _safe_job_dir(job_id)
        path = job_dir / "transcript.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no transcript yet")
        return json.loads(path.read_text())

    return app


def serve(port: int = 8765, runs_root: Path | None = None) -> None:
    import uvicorn

    app = _create_app(runs_root or Path("runs"))
    uvicorn.run(app, host="127.0.0.1", port=port)
