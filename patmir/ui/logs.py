"""Cumulative, size-bounded server log.

Until 2026-09-04 the server logged to stderr only: `log.exception` from a
failing job, uvicorn's request lines, the prewarm summary — all of it lived in
whichever terminal started the server and died with it. That cost a real
debugging session. When the WSL2 GPU bridge took the process down mid-job, the
only reason a traceback survived was that the run happened to be redirected to
a file in `/tmp`, which `wsl --shutdown` then wiped.

What already persisted, and still does: per-job `runs/<job>/progress.jsonl`,
and a failed job's traceback in `rec.error` inside the jobs index. What did
not: anything the server did outside a job, which is exactly where a crash
lives.

`RotatingFileHandler` rather than a date-stamped file per run: cumulative
across restarts (a crash loop is legible as one story), and self-cleaning by
construction — no cron, no cleanup pass, a hard ceiling of
`MAX_BYTES * (BACKUPS + 1)`.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024          # 8 MiB per file
BACKUPS = 4                          # ~40 MiB total, then the oldest is dropped

# Handlers go on the ROOT logger and nowhere else. Attaching the same handler
# to `uvicorn`, `uvicorn.error` AND root wrote every request line three times,
# because a record is handled once per level while it propagates. The server
# instead starts uvicorn with `log_config=None` so uvicorn does not install
# its own handlers, leaving exactly one place that decides where logs go.
_MARKER = "_patmir_file_log"
_CONSOLE = "_patmir_console_log"


def log_path(state_root: Path) -> Path:
    """Where the server log lives. Under `state_root`, which is local by policy."""
    return Path(state_root) / "logs" / "server.log"


def setup_file_logging(state_root: Path, *, level: int = logging.INFO) -> Path:
    """Send this process's logging to a rotating file as well as the console.

    Idempotent: a second call does not double every line. Returns the path so
    the caller can tell the operator where to look — a log nobody can find is
    not much better than no log.
    """
    path = log_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.setLevel(level)
    setattr(handler, _MARKER, True)

    root = logging.getLogger()
    if not any(getattr(h, _MARKER, False) for h in root.handlers):
        root.addHandler(handler)
    # A console handler too, because uvicorn is started with
    # `log_config=None`: without this the terminal would go silent, and
    # Python's last-resort stderr handler stops firing the moment root has a
    # handler of its own.
    if not any(getattr(h, _CONSOLE, False) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        console.setLevel(level)
        setattr(console, _CONSOLE, True)
        root.addHandler(console)
    # Only raise a level that is currently more restrictive: lowering WARNING
    # to INFO is the point, raising DEBUG to INFO is not.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    return path
