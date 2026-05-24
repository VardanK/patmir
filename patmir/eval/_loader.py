"""Shared candidate loader.

Reads `patmir.toml`, returns enabled candidates for a given stage. Each
candidate entry must include `id`, `module`, and `enabled` fields; any other
fields are passed through to the module's `build(config)` factory.
"""
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Any


def load_candidates(config_path: Path, stage: str) -> list[tuple[str, dict[str, Any], Any]]:
    """Return [(id, raw_config, built_candidate), ...] for enabled candidates in `stage`.

    `stage` is "translation" or "tts".
    """
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    section = config.get("eval", {}).get(stage, {}).get("candidates", [])
    if not section:
        raise RuntimeError(
            f"No `[[eval.{stage}.candidates]]` entries in {config_path}."
        )

    built: list[tuple[str, dict[str, Any], Any]] = []
    for entry in section:
        if not entry.get("enabled", False):
            continue
        cid = entry["id"]
        module_name = entry["module"]
        module = importlib.import_module(module_name)
        try:
            candidate = module.build(entry)
        except AttributeError as exc:
            raise RuntimeError(
                f"Candidate module {module_name!r} must export a `build(config)` factory."
            ) from exc
        built.append((cid, entry, candidate))

    if not built:
        raise RuntimeError(
            f"No enabled candidates for stage {stage!r} in {config_path}."
        )
    return built
