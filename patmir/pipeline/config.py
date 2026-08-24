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
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patmir import paths as paths_mod
from patmir.paths import StoragePaths, load_storage
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
    """Resolved candidate entries per stage + global options + storage roots.

    Stage instances are built lazily by the runner via `build_stage(name)`;
    `entries` carries the raw config dicts so each build is cheap to repeat
    (and easy to free between stages on memory-constrained backends).

    `storage` carries the resolved filesystem roots (see `patmir.paths`). It is
    deliberately NOT part of the cache key — `ui.cache.config_hash` hashes only
    `entries`, so relocating the cache or the library never orphans an existing
    dub.
    """
    entries: dict[str, dict[str, Any]]
    options: PipelineOptions
    storage: StoragePaths = field(
        default_factory=lambda: paths_mod.defaults(None)
    )
    # Which machine profile produced these entries, if any. Provenance: the
    # selected candidates are already in `entries`, so this adds nothing to the
    # cache key — it exists so build failures can say whether a profile was
    # active, and so a bench report can record what actually ran.
    profile: str | None = None

    def build_stage(self, name: str) -> Any:
        entry = self.entries.get(name)
        if entry is None:
            raise RuntimeError(f"No candidate entry resolved for stage {name!r}.")
        try:
            instance = _build_candidate(entry)
        except Exception as exc:
            raise RuntimeError(
                _build_failure_message(name, entry, exc, self.profile)
            ) from exc
        if name == "translate":
            from patmir.pipeline.stages.translate.adapter import SentenceTranslator
            if not isinstance(instance, SentenceTranslator):
                instance = SentenceTranslator(inner=instance)
        elif name == "tts":
            from patmir.pipeline.stages.tts.adapter import SentenceTTS
            if not isinstance(instance, SentenceTTS):
                instance = SentenceTTS(inner=instance)
        return instance


def _build_failure_message(
    name: str,
    entry: dict[str, Any],
    exc: Exception,
    profile: str | None = None,
) -> str:
    """Explain a stage build failure in terms of the config that caused it.

    A raw ImportError/OSError from deep inside a candidate's dependencies names
    the missing library and nothing else — not which stage, not which candidate
    id, not what to change. The worst case is an Apple-only candidate selected on
    a CUDA box, where the entire message is `libmlx.so: cannot open shared object
    file`: a configuration problem wearing a linker error's clothes.

    Note an unset profile is the dangerous case, not a typo'd one. A typo raises
    in `_profile_overrides` naming the known profiles; omitting the variable
    entirely just falls through to the `[pipeline]` defaults, which off Apple
    Silicon means MLX candidates that cannot load.
    """
    cid = entry.get("id", "?")
    module = str(entry.get("module", "?"))
    lines = [
        f"Stage {name!r} failed to build candidate {cid!r} (module {module}).",
        f"  cause: {type(exc).__name__}: {exc}",
    ]
    looks_mlx = "mlx" in module.lower() or "mlx" in str(exc).lower()
    if looks_mlx and not profile:
        lines += [
            "  That candidate needs MLX, which exists only on Apple Silicon, and no",
            f"  machine profile is active. Set ${PROFILE_ENV} to a profile that selects",
            "  candidates native to this machine:",
            "",
            f"      export {PROFILE_ENV}=cuda",
            "",
            "  Available profiles are the [pipeline.profiles.*] tables in patmir.toml.",
        ]
    elif looks_mlx and profile:
        lines += [
            f"  Profile {profile!r} is active but still resolved an MLX candidate for",
            f"  this stage. Add a {name!r} key to [pipeline.profiles.{profile}] pointing",
            "  at a candidate that runs on this machine.",
        ]
    return "\n".join(lines)


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
    """Import a candidate module and call its `build(config)`.

    The missing-`build` check is a `getattr`, NOT a `try/except AttributeError`
    around the call. The old form wrapped the call itself, so ANY AttributeError
    raised *inside* a candidate's constructor — a renamed attribute after a
    dependency upgrade, say — was reported as "module must export build(config)".
    That is a wrong diagnosis of a real bug, and it sent two investigations down
    the wrong path before it was noticed. Let the constructor's own exception
    through; `_build_failure_message` adds the stage/candidate context around it.
    """
    module = importlib.import_module(entry["module"])
    builder = getattr(module, "build", None)
    if not callable(builder):
        raise RuntimeError(
            f"Candidate module {entry['module']!r} must export `build(config)`."
        )
    return builder(entry)


PROFILE_ENV = "PATMIR_PROFILE"


def active_profile(explicit: str | None = None) -> str | None:
    """Machine profile name: the explicit argument, else `$PATMIR_PROFILE`.

    A profile is how one committed `patmir.toml` serves machines with different
    accelerators — the Apple-Silicon dev laptop selects MLX candidates from
    `[pipeline]`, the CUDA workstation exports `PATMIR_PROFILE=cuda` and gets
    the torch siblings from `[pipeline.profiles.cuda]`.

    Deliberately NOT auto-detected from `torch.cuda.is_available()`: which model
    ran has to be a stated choice, not something the machine picks silently, or
    a benchmark stops being attributable. An unknown name is a hard error for
    the same reason (see `_profile_overrides`).
    """
    if explicit:
        return explicit
    name = os.environ.get(PROFILE_ENV, "").strip()
    return name or None


def _profile_overrides(
    pipe: dict[str, Any],
    profile: str | None,
    layer: str | None,
    *,
    explicit: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    """Stage selectors from `[pipeline.profiles.<profile>]` and its `<layer>`.

    Returns `(base, layered)`. `base` is the profile's own stage map; `layered`
    is the optional `[pipeline.profiles.<profile>.<layer>]` sub-table, used by
    the watch-along path so a profile can override the streaming candidate set
    too (e.g. a faster `separate` for sections).

    An unknown profile name raises — *except* for one case, which the first
    version got wrong. `$PATMIR_PROFILE` is ambient and machine-scoped: once it
    is exported in a shell profile it applies to every config this process
    loads, including small ones that define no `[pipeline.profiles]` table at
    all. Hard-failing those is wrong; the variable is not addressed to them.

    So the rule keys on provenance:

      * asked for explicitly (`--profile` / `profile=`) and not found -> raise,
        the caller named something specific that does not exist;
      * from the environment, and the config defines OTHER profiles but not this
        one -> raise, that is a real mismatch (a typo, or a config that still
        needs the profile added);
      * from the environment, and the config defines no profiles whatsoever ->
        ignore it and use `[pipeline]`.

    The safety property that motivated raising still holds. A profile-less config
    that selects MLX candidates on a CUDA box does not silently succeed — it
    fails at build time, where `_build_failure_message` names the stage, the
    candidate and `$PATMIR_PROFILE`.
    """
    if not profile:
        return {}, {}
    profiles = pipe.get("profiles") or {}
    if profile not in profiles:
        if not explicit and not profiles:
            return {}, {}  # ambient variable, config does not use profiles
        if profiles:
            raise RuntimeError(
                f"Unknown pipeline profile {profile!r} "
                f"(via {'--profile' if explicit else '$' + PROFILE_ENV}). "
                f"Defined profiles: {sorted(profiles)}."
            )
        raise RuntimeError(
            f"Profile {profile!r} was requested explicitly but this config "
            "defines no [pipeline.profiles] table."
        )
    table = profiles[profile] or {}
    base = {s: str(v) for s, v in table.items() if s in _STAGE_NAMESPACE}
    layered: dict[str, str] = {}
    if layer:
        sub = table.get(layer) or {}
        layered = {s: str(v) for s, v in sub.items() if s in _STAGE_NAMESPACE}
    return base, layered


def load_pipeline_config(
    config_path: Path,
    *,
    skip_before: str | None = None,
    overrides: dict[str, str] | None = None,
    profile: str | None = None,
    layer: str | None = None,
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

    `profile` (defaulting to `$PATMIR_PROFILE`) selects a machine profile from
    `[pipeline.profiles.<name>]`, and `layer` an optional sub-table of it. The
    four layers resolve in this order, last wins:

        [pipeline]  <  profile base  <  overrides  <  profile <layer>

    `overrides` sits above the profile base so the watch-along path keeps its
    per-mode candidate choices, and the profile's `<layer>` sits above those so
    a machine that cannot run the overridden candidate at all (an MLX-only
    `separate` on a CUDA box) can still replace it.
    """
    from patmir.pipeline.stage import STAGE_ORDER  # local import to avoid cycle

    with config_path.open("rb") as f:
        toml = tomllib.load(f)

    pipe = toml.get("pipeline")
    if not pipe:
        raise RuntimeError(f"No [pipeline] section in {config_path}.")

    profile_name = active_profile(profile)
    prof_base, prof_layer = _profile_overrides(
        pipe, profile_name, layer, explicit=bool(profile),
    )

    selectors = {stage: pipe.get(stage) for stage in _STAGE_NAMESPACE}
    for source in (prof_base, overrides or {}, prof_layer):
        for stage, cid in source.items():
            if stage in selectors:
                selectors[stage] = cid
    missing = [s for s, v in selectors.items() if not v]
    if missing:
        raise RuntimeError(f"[pipeline] is missing selectors for: {missing}")

    opts_raw = pipe.get("options", {}) or {}
    options = PipelineOptions(
        stretch_cap=float(opts_raw.get("stretch_cap", 0.20)),
        asr_int8=bool(opts_raw.get("asr_int8", True)),
        keep_intermediate=bool(opts_raw.get("keep_intermediate", True)),
        music_bed_gain=float(opts_raw.get("music_bed_gain", 0.6)),
        vocals_bed_gain=float(opts_raw.get("vocals_bed_gain", 0.12)),
        voice_gain_db=float(opts_raw.get("voice_gain_db", -3.0)),
        min_ref_seconds=float(opts_raw.get("min_ref_seconds", 1.5)),
        sentence_afade_seconds=float(opts_raw.get("sentence_afade_seconds", 0.05)),
        tts_pin_speaker_ref=bool(opts_raw.get("tts_pin_speaker_ref", True)),
        duck_lead_seconds=float(opts_raw.get("duck_lead_seconds", 0.25)),
        duck_tail_seconds=float(opts_raw.get("duck_tail_seconds", 0.10)),
        voice_no_overlap=bool(opts_raw.get("voice_no_overlap", True)),
        voice_overlap_gap_seconds=float(opts_raw.get("voice_overlap_gap_seconds", 0.06)),
        voice_max_compress=float(opts_raw.get("voice_max_compress", 1.5)),
        prelude_window_seconds=float(opts_raw.get("prelude_window_seconds", 40.0)),
        prewarm_stages=bool(opts_raw.get("prewarm_stages", True)),
        ref_min_rms_db=float(opts_raw.get("ref_min_rms_db", -33.0)),
        # Top-level [[tts_ref_pool]] tables — a list
        # of curated stand-in ref clips, not a scalar option.
        tts_ref_pool=[e for e in (toml.get("tts_ref_pool") or []) if isinstance(e, dict) and e.get("path")],
    )

    start_idx = STAGE_ORDER.index(skip_before) if skip_before else 0
    needed = set(STAGE_ORDER[start_idx:])

    entries: dict[str, dict[str, Any]] = {}
    for stage, cid in selectors.items():
        if stage not in needed:
            continue
        entries[stage] = _find_candidate(toml, _STAGE_NAMESPACE[stage], cid)

    return PipelineConfig(
        entries=entries,
        options=options,
        storage=load_storage(toml, config_path),
        profile=profile_name,
    )
