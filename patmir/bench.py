"""patmir bench — repeatable per-stage timing + RTF for the pipeline.

Runs a clip end-to-end in **batch** (D1 quality) or **watch-along** (D3
streaming) mode and reports where the wall-clock goes:

  - per-stage wall + model-build time, read from each run's manifest.json
    (`StageStatus.started_at/finished_at/build_s`)
  - RTF per stage and aggregate, against the source duration
    (RTF = wall_seconds / source_seconds; >1.0 = slower than realtime)
  - per-sentence TTS + translate latency traces (the adapters emit
    `tts/timings.jsonl` / `translate/timings.jsonl` when `ctx.extra["bench"]`
    is set, which only `patmir bench` does) + Higgs overflow counts from
    `tts/voice_cloning_summary.json`

This is the canonical, reusable replacement for the throwaway
`spike/d3_throughput/*` scripts. Output is a stdout table + an optional
machine-readable `bench.json` (`--report`).
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from patmir.pipeline.config import PipelineConfig, load_pipeline_config
from patmir.pipeline.manifest import Manifest
from patmir.pipeline.runner import run_job
from patmir.pipeline.stage import STAGE_ORDER

# The two run_job halves the watch-along section path runs (see
# streaming/section.py): analysis (ingest→asr) then dub (translate→mux).
ANALYSIS_STAGES = ["ingest", "separate", "diarize", "emotion", "asr"]
DUB_STAGES = ["translate", "tts", "mux", "output"]


# --------------------------------------------------------------------------- #
# result shapes
# --------------------------------------------------------------------------- #
@dataclass
class StageTiming:
    name: str
    state: str
    wall_s: float | None      # finished_at - started_at
    build_s: float | None     # model load / quantization (subset of wall_s)
    rtf: float | None         # wall_s / source_s
    pct: float | None         # wall_s / total_wall_s


@dataclass
class BenchResult:
    clip: str
    mode: str                 # "batch" | "watch-along"
    label: str | None
    repeat_index: int
    job_id: str
    source_s: float | None
    total_wall_s: float
    agg_rtf: float | None
    stages: list[StageTiming]
    sentence_count: int | None
    tts: dict[str, Any] | None
    translate: dict[str, Any] | None
    models: dict[str, str]
    sections: list[dict[str, Any]] | None = None
    notes: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# small readers
# --------------------------------------------------------------------------- #
def _source_seconds(job_dir: Path) -> float | None:
    meta = job_dir / "audio.meta.json"
    if not meta.exists():
        return None
    try:
        return float(json.loads(meta.read_text()).get("duration"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _sentence_count(job_dir: Path) -> int | None:
    for fn in ("transcript.json", "translated.json"):
        p = job_dir / fn
        if p.exists():
            try:
                return len(json.loads(p.read_text()).get("sentences", []))
            except (OSError, json.JSONDecodeError):
                pass
    return None


def _stage_wall(st) -> float | None:
    if st.started_at and st.finished_at:
        return st.finished_at - st.started_at
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _pctile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _tts_stats(job_dir: Path) -> dict[str, Any] | None:
    rows = _read_jsonl(job_dir / "tts" / "timings.jsonl")
    out: dict[str, Any] = {}
    if rows:
        el = [r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]
        audio = sum(r.get("audio_s", 0.0) for r in rows)
        out.update({
            "count": len(el),
            "total_s": round(sum(el), 2),
            "mean_s": round(statistics.mean(el), 3) if el else 0.0,
            "median_s": round(statistics.median(el), 3) if el else 0.0,
            "p90_s": round(_pctile(el, 0.9), 3),
            "max_s": round(max(el), 3) if el else 0.0,
            "audio_s": round(audio, 2),
            "tts_rtf": round(sum(el) / audio, 3) if audio else None,
        })
    summ = job_dir / "tts" / "voice_cloning_summary.json"
    if summ.exists():
        try:
            data = json.loads(summ.read_text())
            for k in ("engine", "sentences", "overflows"):
                if k in data:
                    out[k] = data[k]
        except (OSError, json.JSONDecodeError):
            pass
    return out or None


def _translate_stats(job_dir: Path) -> dict[str, Any] | None:
    rows = _read_jsonl(job_dir / "translate" / "timings.jsonl")
    if not rows:
        return None
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    if not lat:
        return None
    return {
        "count": len(lat),
        "total_s": round(sum(lat) / 1000, 2),
        "mean_ms": round(statistics.mean(lat), 1),
        "median_ms": round(statistics.median(lat), 1),
        "p90_ms": round(_pctile(lat, 0.9), 1),
        "max_ms": round(max(lat), 1),
    }


def _models(cfg: PipelineConfig) -> dict[str, str]:
    return {stage: str(entry.get("id", "?")) for stage, entry in cfg.entries.items()}


def _stage_timings(manifest: Manifest, source_s: float | None,
                   only: list[str] | None = None) -> tuple[list[StageTiming], float]:
    names = only or STAGE_ORDER
    walls = {n: (_stage_wall(manifest.stage(n)) or 0.0) for n in names}
    total = sum(walls.values())
    rows: list[StageTiming] = []
    for n in names:
        st = manifest.stage(n)
        wall = _stage_wall(st)
        rows.append(StageTiming(
            name=n,
            state=st.state,
            wall_s=round(wall, 2) if wall is not None else None,
            build_s=round(st.build_s, 2) if st.build_s is not None else None,
            rtf=round(wall / source_s, 3) if (wall is not None and source_s) else None,
            pct=round(100 * wall / total, 1) if (wall is not None and total) else None,
        ))
    return rows, total


# --------------------------------------------------------------------------- #
# batch (D1)
# --------------------------------------------------------------------------- #
def bench_batch(clip: Path, cfg: PipelineConfig, *, runs_root: Path,
                repeat: int = 1, label: str | None = None) -> list[BenchResult]:
    results: list[BenchResult] = []
    for r in range(repeat):
        manifest = run_job(clip, cfg, runs_root=runs_root, extra={"bench": True})
        job_dir = runs_root / manifest.job_id
        source_s = _source_seconds(job_dir)
        stages, total = _stage_timings(manifest, source_s)
        results.append(BenchResult(
            clip=str(clip),
            mode="batch",
            label=label,
            repeat_index=r,
            job_id=manifest.job_id,
            source_s=round(source_s, 2) if source_s else None,
            total_wall_s=round(total, 2),
            agg_rtf=round(total / source_s, 3) if source_s else None,
            stages=stages,
            sentence_count=_sentence_count(job_dir),
            tts=_tts_stats(job_dir),
            translate=_translate_stats(job_dir),
            models=_models(cfg),
        ))
    return results


# --------------------------------------------------------------------------- #
# watch-along (D3)
# --------------------------------------------------------------------------- #
def _section_record(index: int, sec, wall_s: float, runs_root: Path) -> dict[str, Any]:
    job_dir = sec.job_dir
    manifest = Manifest.from_path(job_dir / "manifest.json")
    span = max(0.0, sec.end - sec.start)
    per_stage = {n: (_stage_wall(manifest.stage(n)) or 0.0) for n in STAGE_ORDER}
    build_total = sum((manifest.stage(n).build_s or 0.0) for n in STAGE_ORDER)
    analysis = sum(per_stage[n] for n in ANALYSIS_STAGES)
    dub = sum(per_stage[n] for n in DUB_STAGES)
    return {
        "index": index,
        "start": round(sec.start, 2),
        "end": round(sec.end, 2),
        "section_s": round(span, 2),
        "wall_s": round(wall_s, 2),
        "analysis_wall_s": round(analysis, 2),
        "dub_wall_s": round(dub, 2),
        "build_s": round(build_total, 2),
        "rtf": round(wall_s / span, 3) if span else None,
        "stage_wall_s": {n: round(per_stage[n], 2) for n in STAGE_ORDER},
        "new_speakers": list(sec.new_speakers),
    }


def bench_watch_along(clip: Path, config_path: Path, *, runs_root: Path,
                      repeat: int = 1, max_sections: int | None = None,
                      label: str | None = None) -> list[BenchResult]:
    from patmir.pipeline.streaming.prelude import run_prelude
    from patmir.pipeline.streaming.runner import streaming_config, source_duration
    from patmir.pipeline.streaming.section import (
        DEFAULT_SECTION_SECONDS,
        dub_prelude_window,
        process_section,
    )

    results: list[BenchResult] = []
    for r in range(repeat):
        cfg = streaming_config(config_path)
        duration = source_duration(clip)

        sections: list[dict[str, Any]] = []
        # Same warm-model cache the production stream loop uses (ui/stream.py)
        # — bench must measure the warm path, not the legacy rebuild path.
        stage_cache: dict[str, Any] = {}
        t0 = time.perf_counter()
        prelude = run_prelude(
            clip, cfg,
            window_seconds=cfg.options.prelude_window_seconds,
            runs_root=runs_root,
            stage_cache=stage_cache,
        )
        prelude_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        sec0 = dub_prelude_window(cfg, prelude, runs_root=runs_root, stage_cache=stage_cache)
        sec0_wall = time.perf_counter() - t0
        sections.append(_section_record(0, sec0, sec0_wall, runs_root))

        idx = 1
        t = prelude.window_end
        while t < duration - 0.5:
            s0 = time.perf_counter()
            sec = process_section(
                clip, cfg, prelude.registry,
                index=idx, start=t, runs_root=runs_root,
                stage_cache=stage_cache,
            )
            sections.append(_section_record(idx, sec, time.perf_counter() - s0, runs_root))
            t = sec.end if sec.end > t + 0.5 else min(t + DEFAULT_SECTION_SECONDS, duration)
            idx += 1
            if max_sections is not None and idx > max_sections:
                break

        dubbed_audio_s = sum(s["section_s"] for s in sections)
        sections_wall = sum(s["wall_s"] for s in sections)
        total_wall = prelude_wall + sections_wall

        # Aggregate per-stage walls + builds across all sections, so the table
        # is comparable to batch (and exposes total rebuild cost).
        agg_stage = {n: 0.0 for n in STAGE_ORDER}
        for s in sections:
            for n in STAGE_ORDER:
                agg_stage[n] += s["stage_wall_s"].get(n, 0.0)
        stage_rows: list[StageTiming] = []
        total_section_stage = sum(agg_stage.values()) or 1.0
        for n in STAGE_ORDER:
            wall = agg_stage[n]
            stage_rows.append(StageTiming(
                name=n,
                state="done",
                wall_s=round(wall, 2),
                build_s=None,
                rtf=round(wall / dubbed_audio_s, 3) if dubbed_audio_s else None,
                pct=round(100 * wall / total_section_stage, 1),
            ))

        total_build = sum(s["build_s"] for s in sections)
        results.append(BenchResult(
            clip=str(clip),
            mode="watch-along",
            label=label,
            repeat_index=r,
            job_id=prelude.job_dir.name,
            source_s=round(duration, 2),
            total_wall_s=round(total_wall, 2),
            agg_rtf=round(total_wall / dubbed_audio_s, 3) if dubbed_audio_s else None,
            stages=stage_rows,
            sentence_count=None,
            tts=None,
            translate=None,
            models=_models(cfg),
            sections=sections,
            notes={
                "prelude_wall_s": round(prelude_wall, 2),
                "sections_wall_s": round(sections_wall, 2),
                "dubbed_audio_s": round(dubbed_audio_s, 2),
                "n_sections": len(sections),
                "total_build_s": round(total_build, 2),
                "build_pct_of_wall": round(100 * total_build / total_wall, 1) if total_wall else None,
            },
        ))
    return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def to_json(results: list[BenchResult]) -> dict[str, Any]:
    return {"results": [asdict(r) for r in results]}


def _fmt(v: Any, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "-"


def format_report(results: list[BenchResult]) -> str:
    lines: list[str] = []
    for res in results:
        rep = f" repeat={res.repeat_index}" if any(r.repeat_index for r in results) else ""
        lines.append("")
        lines.append(f"=== {res.mode} | {Path(res.clip).name}{rep} "
                     f"| {res.job_id} ===")
        lines.append(f"source {_fmt(res.source_s, 's')}  "
                     f"wall {_fmt(res.total_wall_s, 's')}  "
                     f"agg RTF {_fmt(res.agg_rtf)}  "
                     f"({'slower' if (res.agg_rtf or 0) > 1 else 'faster'} than realtime)")
        if res.sentence_count is not None:
            lines.append(f"sentences: {res.sentence_count}")
        lines.append("")
        lines.append(f"  {'stage':<10} {'wall':>8} {'build':>8} {'rtf':>7} {'%':>6}")
        lines.append(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
        for s in res.stages:
            lines.append(f"  {s.name:<10} {_fmt(s.wall_s):>8} {_fmt(s.build_s):>8} "
                         f"{_fmt(s.rtf):>7} {_fmt(s.pct):>6}")
        if res.tts:
            t = res.tts
            lines.append("")
            lines.append(f"  tts: {t.get('count', '?')} sent, mean {_fmt(t.get('mean_s'), 's')}, "
                         f"p90 {_fmt(t.get('p90_s'), 's')}, max {_fmt(t.get('max_s'), 's')}, "
                         f"tts_rtf {_fmt(t.get('tts_rtf'))}, overflows {t.get('overflows', '?')}")
        if res.translate:
            tr = res.translate
            lines.append(f"  translate: {tr['count']} sent, mean {_fmt(tr['mean_ms'], 'ms')}, "
                         f"p90 {_fmt(tr['p90_ms'], 'ms')}, total {_fmt(tr['total_s'], 's')} (serial)")
        if res.sections is not None:
            n = res.notes
            lines.append("")
            lines.append(f"  watch-along: {n.get('n_sections')} sections, "
                         f"prelude {_fmt(n.get('prelude_wall_s'), 's')}, "
                         f"build {_fmt(n.get('total_build_s'), 's')} "
                         f"({_fmt(n.get('build_pct_of_wall'), '%')} of wall)")
            lines.append(f"  {'sec':>3} {'start':>7} {'span':>6} {'wall':>7} "
                         f"{'anlys':>7} {'dub':>7} {'build':>7} {'rtf':>6}")
            for s in res.sections:
                lines.append(f"  {s['index']:>3} {s['start']:>7} {s['section_s']:>6} "
                             f"{s['wall_s']:>7} {s['analysis_wall_s']:>7} {s['dub_wall_s']:>7} "
                             f"{s['build_s']:>7} {_fmt(s['rtf']):>6}")
    return "\n".join(lines)
