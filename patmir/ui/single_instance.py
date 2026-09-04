"""One `patmir ui` per machine.

Two servers do not fit. Each prewarms the whole stage set at boot — ~16 GB of
weights on the CUDA profile (tts ~8.5, asr+aligner ~5.5, diarize, emotion,
Demucs) — and a 24 GB card holds exactly one. The `asyncio.Lock` added on
2026-08-25 (`decisions/2026-08-25-one-accelerator-at-a-time.md`) serializes a
batch job against a stream *inside* one process; it cannot see another process
at all, so a second server silently starves the first. Observed 2026-09-03:
starting a second `patmir ui` on a different port broke the running one, and
the symptom surfaced as "the extension won't connect" rather than as anything
mentioning memory.

`flock` is the whole mechanism: the kernel releases it when the holder dies, so
a crashed server leaves no stale lock to clear by hand — which a pidfile would.
The file's contents are advisory, for the error message only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ServerAlreadyRunning(RuntimeError):
    """Raised when another `patmir ui` holds the lock."""

    def __init__(self, holder: dict[str, Any], lock_path: Path) -> None:
        self.holder = holder
        self.lock_path = lock_path
        pid = holder.get("pid", "?")
        where = holder.get("bind", "?")
        super().__init__(
            f"another patmir ui is already running (pid {pid}, bound {where}). "
            f"Two servers do not fit: each prewarms the whole model set at boot "
            f"and one card holds one. Stop that one first, or point the "
            f"extension at it. Lock: {lock_path}"
        )


def acquire(lock_path: Path, *, host: str, port: int) -> Any:
    """Take the single-instance lock, or raise `ServerAlreadyRunning`.

    Returns the open file object; the caller keeps it alive for the process
    lifetime. Closing it (or exiting) releases the lock.
    """
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.seek(0)
        raw = handle.read().strip()
        handle.close()
        holder: dict[str, Any] = {}
        try:
            holder = json.loads(raw) if raw else {}
        except ValueError:
            # A half-written or hand-edited file must not turn a clear
            # "already running" into a crash.
            holder = {"pid": "?", "bind": raw[:80] or "?"}
        raise ServerAlreadyRunning(holder, lock_path) from None

    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "bind": f"{host}:{port}"}))
    handle.flush()
    return handle
