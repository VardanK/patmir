"""Local FastAPI server.

Two surfaces live here:

* The original read-only progress UI (HTML + SSE over `runs/`) — unchanged.
* The Phase-2-finalization job-submission API used by the Chrome extension
  and the `patmir submit` CLI: `POST /jobs`, `GET /cache/lookup`,
  `GET /jobs/{id}`, `GET /cache/{key}/dub.wav`, `GET /cache/{key}/metadata.json`.

Bound to `127.0.0.1` only; bearer-token auth gates every new endpoint. The
existing `/`, `/run/{id}`, `/run/{id}/progress` routes stay open for the
browser progress UI on the same loopback origin.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.manifest import Manifest
from patmir.pipeline.stage import STAGE_ORDER
from patmir.ui import auth, cache as cache_mod
from patmir.ui.jobs import JobManager, JobSpec
from patmir.ui.stream import StreamManager

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Cache-key length is 16 hex chars; constrain the path param so a stray
# request can't walk paths under the cache root.
_CACHE_KEY_RE = r"^[0-9a-f]{16}$"


def _create_app(
    runs_root: Path,
    cache_root: Path,
    cfg: PipelineConfig,
    token: str,
    config_path: Path = Path("patmir.toml"),
) -> FastAPI:
    app = FastAPI(title="Patmir")
    runs_root = runs_root.resolve()
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    require_token = auth.TokenAuth(token)
    manager = JobManager(cfg, runs_root=runs_root, cache_root=cache_root)
    streamer = StreamManager(
        config_path, jobs=manager, runs_root=runs_root, cache_root=cache_root
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^chrome-extension://.*$|^moz-extension://.*$|^http://(127\.0\.0\.1|localhost)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    @app.on_event("startup")
    async def _startup() -> None:
        await manager.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await streamer.stop()
        await manager.stop()

    # ---------- legacy progress UI (unchanged) -----------------------------

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
        return TEMPLATES.TemplateResponse(request, "index.html", {"runs": rows})

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

    # ---------- D1: cache + jobs API ---------------------------------------

    def _safe_cache_dir(key: str) -> Path:
        candidate = (cache_root / key).resolve()
        if cache_root not in candidate.parents:
            raise HTTPException(status_code=400, detail="invalid cache_key")
        if not candidate.is_dir():
            raise HTTPException(status_code=404, detail="no such cache entry")
        return candidate

    @app.get("/cache/lookup", dependencies=[Depends(require_token)])
    def cache_lookup(url: str = Query(..., min_length=4), mode: str = Query("quality")):
        if mode == "watch-along":
            manifest = streamer.lookup_cache_by_url(url)
            return JSONResponse({
                "cached": bool(manifest and manifest.get("complete")),
                "partial": bool(manifest and not manifest.get("complete")),
                "cache_key": streamer.cache_key_for_url(url),
                "mode": "watch-along",
                "sections": len((manifest or {}).get("sections", [])),
            })
        entry = manager.lookup_cache_by_url(url)
        if entry is None:
            return JSONResponse({
                "cached": False,
                "cache_key": manager.cache_key_for_url(url),
                "mode": "quality",
            })
        return JSONResponse({
            "cached": True,
            "cache_key": entry.cache_key,
            "duration": entry.duration,
            "created_at": entry.created_at,
            "source_url": entry.source_url,
            "mode": "quality",
        })

    @app.post("/jobs", dependencies=[Depends(require_token)])
    async def jobs_submit(payload: dict):
        try:
            spec = JobSpec(
                source_url=payload.get("source_url"),
                source_path=payload.get("source_path"),
                source_lang=payload.get("source_lang"),
                force=bool(payload.get("force", False)),
                mode=payload.get("mode", "quality"),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            if spec.mode == "watch-along":
                rec = await streamer.submit(spec)
            else:
                rec = await manager.submit(spec)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return rec.to_public()

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
    def jobs_get(job_id: str):
        # Watch-along records live in the StreamManager; their ids are prefixed
        # `stream-`. Try the matching manager first, fall back to the other.
        rec = (streamer.get(job_id) if job_id.startswith("stream-") else manager.get(job_id))
        if rec is None:
            rec = manager.get(job_id) or streamer.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        return rec.to_public()

    # ---------- D3: watch-along streaming ----------------------------------

    @app.get("/stream/{cache_key}/sections", dependencies=[Depends(require_token)])
    def stream_sections_list(cache_key: str):
        if not _is_valid_key(cache_key):
            raise HTTPException(status_code=400, detail="invalid cache_key")
        manifest = streamer.sections_manifest(cache_key)
        if manifest is None:
            raise HTTPException(status_code=404, detail="no such stream")
        return JSONResponse(manifest)

    @app.get("/stream/{cache_key}/section/{index}.wav", dependencies=[Depends(require_token)])
    def stream_section_wav(cache_key: str, index: int):
        if not _is_valid_key(cache_key):
            raise HTTPException(status_code=400, detail="invalid cache_key")
        path = streamer.section_file(cache_key, index)
        if path is None:
            raise HTTPException(status_code=404, detail="no such section")
        return FileResponse(
            path,
            media_type="audio/wav",
            headers={"Cache-Control": "private, immutable, max-age=31536000"},
        )

    @app.post("/stream/{cache_key}/playhead", dependencies=[Depends(require_token)])
    def stream_playhead(cache_key: str, payload: dict):
        if not _is_valid_key(cache_key):
            raise HTTPException(status_code=400, detail="invalid cache_key")
        try:
            t = float(payload.get("t", 0.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="bad playhead t") from exc
        ok = streamer.set_playhead(cache_key, t, bool(payload.get("paused", False)))
        if not ok:
            raise HTTPException(status_code=404, detail="no active stream")
        return JSONResponse({"ok": True})

    @app.get("/cache/{cache_key}/dub.wav", dependencies=[Depends(require_token)])
    def cache_dub(cache_key: str, request: Request):
        if not _is_valid_key(cache_key):
            raise HTTPException(status_code=400, detail="invalid cache_key")
        cache_dir = _safe_cache_dir(cache_key)
        path = cache_dir / "dub.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no dub.wav")
        return FileResponse(
            path,
            media_type="audio/wav",
            headers={"Cache-Control": "private, immutable, max-age=31536000"},
        )

    @app.get("/cache/{cache_key}/metadata.json", dependencies=[Depends(require_token)])
    def cache_metadata(cache_key: str):
        if not _is_valid_key(cache_key):
            raise HTTPException(status_code=400, detail="invalid cache_key")
        cache_dir = _safe_cache_dir(cache_key)
        path = cache_dir / "metadata.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no metadata")
        return JSONResponse(json.loads(path.read_text()))

    return app


def _is_valid_key(key: str) -> bool:
    import re
    return re.fullmatch(_CACHE_KEY_RE, key or "") is not None


def serve(
    *,
    port: int = 8765,
    runs_root: Path | None = None,
    cache_root: Path | None = None,
    cfg: PipelineConfig | None = None,
    token: str | None = None,
    config_path: Path = Path("patmir.toml"),
) -> None:
    import uvicorn

    if cfg is None:
        from patmir.pipeline.config import load_pipeline_config
        cfg = load_pipeline_config(config_path)
    if token is None:
        token = auth.ensure_token()

    app = _create_app(
        runs_root=runs_root or Path("runs"),
        cache_root=cache_root or cache_mod.DEFAULT_CACHE_ROOT,
        cfg=cfg,
        token=token,
        config_path=config_path,
    )
    uvicorn.run(app, host="127.0.0.1", port=port)
