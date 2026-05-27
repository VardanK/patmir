"""Minimal `.env` loader (stdlib only).

Reads `KEY=value` lines from `.env` (and optionally `.env.local`, which
takes precedence) at the project root and seeds `os.environ`. Existing
shell-exported env vars always win over `.env` — so a value set in
`~/.zshrc` or via `export` in the current shell isn't overridden.

Project-scoped key storage keeps secrets like OPENROUTER_API_KEY out of
the global shell environment (and out of git — `.env` and `.env.local`
are gitignored).

Lines starting with `#` are comments. Inline `# comment` after a value is
NOT stripped — keep values clean (whitespace at the ends IS stripped).
Surrounding single or double quotes are stripped if balanced.

Calling this at CLI startup makes every downstream candidate (translate /
TTS / ASR / diarize) see the keys without each having to load `.env`
itself.
"""
from __future__ import annotations

import os
from pathlib import Path


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, _, raw_value = line.partition("=")
    key = key.strip()
    if not key:
        return None
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_dotenv(project_root: Path | None = None, *, override_shell: bool = False) -> dict[str, str]:
    """Load `.env` (then `.env.local`, which wins) into `os.environ`.

    `override_shell=False` (default) leaves shell-exported vars untouched —
    explicit exports are higher-priority than `.env`. Set to `True` to
    force `.env` values to win.

    Returns the dict of (key, value) pairs that were actually applied,
    for logging / debugging.
    """
    root = project_root or Path.cwd()
    files = [root / ".env", root / ".env.local"]
    applied: dict[str, str] = {}
    for path in files:
        if not path.exists():
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parsed = _parse_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if not override_shell and key in os.environ:
                continue
            os.environ[key] = value
            applied[key] = value
    return applied
