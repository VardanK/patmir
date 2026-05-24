"""Pipeline config loader.

Reads `[pipeline]` from `patmir.toml` to learn which candidate id runs each
stage, then instantiates the corresponding `[[<namespace>.<stage>.candidates]]`
entry via the candidate module's `build(config)` factory.

Translation and TTS candidates live under `[[eval.translation.candidates]]` /
`[[eval.tts.candidates]]` (shared with the eval harness) and expose the raw
Phase-0 interface (`translate(text, src) -> str`, `synthesize(text) -> bytes`).
The pipeline wraps each in the corresponding stage adapter so the orchestrator
gets the typed `TranslateStage` / `TTSStage` protocol.

The four pipeline-specific stages (ingest, asr, mux, output) live under
`[[stages.<stage>.candidates]]` and implement the typed protocol directly.
"""
from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from patmir.pipeline.context import PipelineOptions
from patmir.pipeline.stage import (
    ASRStage,
    IngestStage,
    MuxStage,
    OutputStage,
    TTSStage,
    TranslateStage,
)

_STAGE_NAMESPACE: dict[str, tuple[str, str]] = {
    "ingest":    ("stages", "ingest"),
    "asr":       ("stages", "asr"),
    "translate": ("eval", "translation"),
    "tts":       ("eval", "tts"),
    "mux":       ("stages", "mux"),
    "output":    ("stages", "output"),
}


@dataclass
class PipelineConfig:
    ingest: IngestStage
    asr: ASRStage
    translate: TranslateStage
    tts: TTSStage
    mux: MuxStage
    output: OutputStage
    options: PipelineOptions


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


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    with config_path.open("rb") as f:
        toml = tomllib.load(f)

    pipe = toml.get("pipeline")
    if not pipe:
        raise RuntimeError(f"No [pipeline] section in {config_path}.")

    selectors = {stage: pipe.get(stage) for stage in _STAGE_NAMESPACE}
    missing = [s for s, v in selectors.items() if not v]
    if missing:
        raise RuntimeError(f"[pipeline] is missing selectors for: {missing}")

    opts_raw = pipe.get("options", {}) or {}
    options = PipelineOptions(
        stretch_cap=float(opts_raw.get("stretch_cap", 0.20)),
        asr_int8=bool(opts_raw.get("asr_int8", True)),
        keep_intermediate=bool(opts_raw.get("keep_intermediate", True)),
    )

    raw: dict[str, Any] = {}
    for stage, cid in selectors.items():
        entry = _find_candidate(toml, _STAGE_NAMESPACE[stage], cid)
        raw[stage] = _build_candidate(entry)

    # translate / tts candidates expose the raw Phase-0 interface; wrap them
    # so the runner sees the typed stage protocol.
    from patmir.pipeline.stages.translate.adapter import SentenceTranslator
    from patmir.pipeline.stages.tts.adapter import SentenceTTS

    translate_stage = raw["translate"]
    if not isinstance(translate_stage, SentenceTranslator):
        translate_stage = SentenceTranslator(inner=translate_stage)

    tts_stage = raw["tts"]
    if not isinstance(tts_stage, SentenceTTS):
        tts_stage = SentenceTTS(inner=tts_stage)

    return PipelineConfig(
        ingest=raw["ingest"],
        asr=raw["asr"],
        translate=translate_stage,
        tts=tts_stage,
        mux=raw["mux"],
        output=raw["output"],
        options=options,
    )
