"""Pipeline config loader.

Reads `[pipeline]` from `patmir.toml` to learn which candidate id runs each
stage. Stage instances are *built lazily by the runner* — `PipelineConfig`
stores the resolved candidate entries (plus options), and `build_stage(name)`
instantiates one on demand. The runner drops each stage's reference + frees
MPS cache after the stage runs, so on Apple Silicon's unified memory we
never hold more than one model resident at a time.

Translation and TTS candidates live under `[[eval.translation.candidates]]` /
`[[eval.tts.candidates]]` (shared with the eval harness) and expose the raw
Phase-0 interface (`translate(text, src) -> str`, `synthesize(text) -> bytes`).
The translate/tts builds wrap the raw candidate in the corresponding stage
adapter so the runner gets the typed `TranslateStage` / `TTSStage` protocol.

The four pipeline-specific stages (ingest, asr, mux, output) live under
`[[stages.<stage>.candidates]]` and implement the typed protocol directly.
"""
from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patmir.pipeline.context import PipelineOptions

_STAGE_NAMESPACE: dict[str, tuple[str, str]] = {
    "ingest":    ("stages", "ingest"),
    "separate":  ("stages", "separate"),
    "diarize":   ("stages", "diarize"),
    "emotion":   ("stages", "emotion"),
    "asr":       ("stages", "asr"),
    "translate": ("eval", "translation"),
    "tts":       ("eval", "tts"),
    "mux":       ("stages", "mux"),
    "output":    ("stages", "output"),
}


@dataclass
class PipelineConfig:
    """Resolved candidate entries per stage + global options.

    Stage instances are built lazily by the runner via `build_stage(name)`;
    `entries` carries the raw config dicts so each build is cheap to repeat
    (and easy to free between stages on memory-constrained backends).
    """
    entries: dict[str, dict[str, Any]]
    options: PipelineOptions

    def build_stage(self, name: str) -> Any:
        entry = self.entries.get(name)
        if entry is None:
            raise RuntimeError(f"No candidate entry resolved for stage {name!r}.")
        instance = _build_candidate(entry)
        if name == "translate":
            from patmir.pipeline.stages.translate.adapter import SentenceTranslator
            if not isinstance(instance, SentenceTranslator):
                instance = SentenceTranslator(inner=instance)
        elif name == "tts":
            from patmir.pipeline.stages.tts.adapter import SentenceTTS
            if not isinstance(instance, SentenceTTS):
                instance = SentenceTTS(inner=instance)
        return instance


def _find_candidate(toml: dict[str, Any], namespace: tuple[str, str], cid: str) -> dict[str, Any]:
    root, stage = namespace
    candidates = toml.get(root, {}).get(stage, {}).get("candidates", [])
    for entry in candidates:
        if entry.get("id") == cid:
            return entry
    raise RuntimeError(
        f"Candidate id {cid!r} not found under [[{root}.{stage}.candidates]] in patmir.toml."
    )


def _build_candidate(entry: dict[str, Any]) -> Any:
    module = importlib.import_module(entry["module"])
    try:
        return module.build(entry)
    except AttributeError as exc:
        raise RuntimeError(
            f"Candidate module {entry['module']!r} must export `build(config)`."
        ) from exc


def load_pipeline_config(
    config_path: Path,
    *,
    skip_before: str | None = None,
    overrides: dict[str, str] | None = None,
) -> PipelineConfig:
    """Load and instantiate the pipeline configuration.

    `skip_before`, if set to a stage name, instantiates only that stage and
    all later stages — earlier ones are left as `None` so their (possibly
    heavy) models don't get loaded into memory when we're about to resume
    from a later stage. The runner's `_load_resume_artifact` handles earlier-
    stage outputs from disk, so the actual stage instances are not needed.

    `overrides`, a {stage: candidate_id} map, replaces the `[pipeline]`
    selectors — used by the streaming/watch-along path to swap in a different
    candidate (e.g. `tts = "elevenlabs_pool"`) without editing patmir.toml.
    """
    from patmir.pipeline.stage import STAGE_ORDER  # local import to avoid cycle

    with config_path.open("rb") as f:
        toml = tomllib.load(f)

    pipe = toml.get("pipeline")
    if not pipe:
        raise RuntimeError(f"No [pipeline] section in {config_path}.")

    selectors = {stage: pipe.get(stage) for stage in _STAGE_NAMESPACE}
    for stage, cid in (overrides or {}).items():
        if stage in selectors:
            selectors[stage] = cid
    missing = [s for s, v in selectors.items() if not v]
    if missing:
        raise RuntimeError(f"[pipeline] is missing selectors for: {missing}")

    opts_raw = pipe.get("options", {}) or {}
    # Emotion guidance lives in a top-level [emotion_guidance] table (not
    # under [pipeline.options]) because it's a per-bin lookup table, not a
    # scalar option. Merge user overrides on top of the defaults.
    guidance_raw = toml.get("emotion_guidance", {}) or {}
    from patmir.pipeline.context import _DEFAULT_EMOTION_GUIDANCE
    emotion_guidance: dict[str, float] = dict(_DEFAULT_EMOTION_GUIDANCE)
    for k, v in guidance_raw.items():
        try:
            emotion_guidance[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    options = PipelineOptions(
        stretch_cap=float(opts_raw.get("stretch_cap", 0.20)),
        asr_int8=bool(opts_raw.get("asr_int8", True)),
        keep_intermediate=bool(opts_raw.get("keep_intermediate", True)),
        music_bed_gain=float(opts_raw.get("music_bed_gain", 0.6)),
        voice_gain_db=float(opts_raw.get("voice_gain_db", -3.0)),
        min_ref_seconds=float(opts_raw.get("min_ref_seconds", 1.5)),
        sentence_afade_seconds=float(opts_raw.get("sentence_afade_seconds", 0.05)),
        voice_no_overlap=bool(opts_raw.get("voice_no_overlap", True)),
        voice_overlap_gap_seconds=float(opts_raw.get("voice_overlap_gap_seconds", 0.06)),
        voice_max_compress=float(opts_raw.get("voice_max_compress", 1.5)),
        emotion_guidance=emotion_guidance,
    )

    start_idx = STAGE_ORDER.index(skip_before) if skip_before else 0
    needed = set(STAGE_ORDER[start_idx:])

    entries: dict[str, dict[str, Any]] = {}
    for stage, cid in selectors.items():
        if stage not in needed:
            continue
        entries[stage] = _find_candidate(toml, _STAGE_NAMESPACE[stage], cid)

    return PipelineConfig(entries=entries, options=options)
