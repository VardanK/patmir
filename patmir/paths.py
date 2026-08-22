"""Storage roots — the single place every filesystem location is decided.

Before this module each root was hardcoded at its point of use, and `runs/` had
five independent `Path.cwd() / "runs"` fallbacks that silently changed meaning
with the invoking directory. Now one `[storage]` table in `patmir.toml` decides
all of them, and `PipelineConfig.storage` carries the result to every entry
point (CLI, server, runner, streaming loop).

```toml
[storage]
runs_root      = "runs"                        # relative → next to patmir.toml
cache_root     = "/Volumes/media/patmir/cache"
downloads_root = "~/.patmir/downloads"
state_root     = "~/.patmir"
library_root   = "/Volumes/media/patmir/library"
```

Every key is optional; the defaults reproduce the pre-change hardcoded paths
exactly, so an install with no `[storage]` table is unchanged.

`library_root` is the one new location: when set, the finished deliverables of
each run are published to `library_root/<job_id>/` (see
`patmir.pipeline.publish`). Unset means don't publish — the dub stays in the job
dir as before.

Which roots are safe to point at network storage: `library_root` and
`cache_root` are written in whole-file copies and are fine. `runs_root` is the
live working directory — `progress.jsonl` is appended and byte-offset
seek-tailed by the SSE loop, `manifest.json` is rewritten per stage, and
ffmpeg/soundfile write in place — so it is best left local. `state_root` holds
the server's bearer token and job index and should stay local too.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HOME_PATMIR = Path.home() / ".patmir"

# Keys accepted under [storage]. An unrecognized key raises rather than being
# ignored, so a typo'd `runs-root` fails loudly instead of silently writing to
# the default location.
_KEYS = ("runs_root", "cache_root", "downloads_root", "state_root", "library_root")


@dataclass(frozen=True)
class StoragePaths:
    """Resolved absolute roots for one pipeline configuration."""

    runs_root: Path
    cache_root: Path
    downloads_root: Path
    state_root: Path
    library_root: Path | None = None

    @property
    def jobs_index(self) -> Path:
        """The local server's job index (was `~/.patmir/jobs.json`)."""
        return self.state_root / "jobs.json"

    @property
    def secret_path(self) -> Path:
        """The local server's bearer token (was `~/.patmir/secret`)."""
        return self.state_root / "secret"


def defaults(config_path: Path | None = None) -> StoragePaths:
    """The pre-`[storage]` hardcoded paths.

    `runs_root` sits next to `patmir.toml` when a config path is known; with no
    config path it falls back to the current directory, matching the old
    `Path("runs")` behavior.
    """
    base = config_path.parent if config_path is not None else Path.cwd()
    return StoragePaths(
        runs_root=_absolute(base / "runs"),
        cache_root=_HOME_PATMIR / "cache",
        downloads_root=_HOME_PATMIR / "downloads",
        state_root=_HOME_PATMIR,
        library_root=None,
    )


def load_storage(toml: dict[str, Any], config_path: Path) -> StoragePaths:
    """Build `StoragePaths` from a parsed `patmir.toml`.

    Relative paths resolve against the config file's directory — NOT the
    current working directory — so `patmir dub` behaves the same whichever
    directory it's invoked from.
    """
    raw = toml.get("storage") or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"[storage] in {config_path} must be a table.")

    unknown = sorted(set(raw) - set(_KEYS))
    if unknown:
        raise RuntimeError(
            f"Unknown key(s) under [storage] in {config_path}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(_KEYS)}."
        )

    base = config_path.parent
    fallback = defaults(config_path)

    def root(key: str) -> Path | None:
        value = raw.get(key)
        if value is None:
            return getattr(fallback, key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                f"[storage].{key} in {config_path} must be a non-empty path string."
            )
        return resolve_root(value, base)

    return StoragePaths(
        runs_root=root("runs_root"),
        cache_root=root("cache_root"),
        downloads_root=root("downloads_root"),
        state_root=root("state_root"),
        library_root=root("library_root"),
    )


def resolve_root(value: str, base: Path) -> Path:
    """Expand `~` and make `value` absolute relative to `base`."""
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else _absolute(base / path)


def _absolute(path: Path) -> Path:
    """Absolute form without resolving symlinks.

    `Path.resolve()` would follow a symlinked root to its target, which breaks
    the server's `resolve()`-based path-containment checks for individual job
    and cache dirs. `os.path.abspath` semantics are enough here — the roots only
    need to stop being CWD-relative.
    """
    return Path(path).absolute()
