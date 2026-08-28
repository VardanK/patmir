"""patmir bench — repeatable per-stage timing + RTF for the pipeline.

Runs a clip end-to-end in **batch** (D1 quality) or **watch-along** (D3
streaming) mode and reports where the wall-clock goes:

  - per-stage wall + model-build time, read from each run's manifest.json
    (`StageStatus.started_at/finished_at/build_s`)
  - RTF per stage and aggregate, against the source duration
    (RTF = wall_seconds / source_seconds; >1.0 = slower than realtime)
  - TTS + translate latency traces (the adapters emit `tts/timings.jsonl` /
    `translate/timings.jsonl` when `ctx.extra["bench"]` is set, which only
    `patmir bench` does) + Higgs overflow counts from
    `tts/voice_cloning_summary.json`

**`--repeat N` shares one warm model cache** (since 2026-08-24): repeat 0
builds every stage model, repeats 1..N-1 reuse them and report
`build_s == 0.0`. The warm repeats are the baseline — `performance.md` has
always said so, but batch was passing no `stage_cache` at all, so on torch it
was timing nine model loads three times over (~110 s of a 242 s run, 45 % of
wall, where MLX's mmap loads cost 0-3 s and hid the problem). `--no-warm-cache`
restores the old rebuild-every-repeat behaviour, which is also how to collect N
independent cold samples or to stay inside memory on a smaller box.

**TTS percentiles are per row, and a row is not always a sentence.** Under
batched decode the adapter emits one row per *chunk* (`"kind": "chunk"`),
because a per-sentence wall does not exist when the chunk's cost is shared.
`tts["unit"]` says which, and the report labels it; `total_s`, `audio_s` and
`tts_rtf` are correct either way.

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

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.manifest import Manifest
from patmir.pipeline.runner import (
    _free_accelerator_cache,
    prewarm_stage_cache,
    run_job,
)
from patmir.pipeline.stage import STAGE_ORDER
from patmir.pipeline.streaming import timing as timing_mod
from patmir.pipeline.streaming.scheduler import Playhead, StreamScheduler

# The two run_job halves the watch-along section path runs (see
# streaming/section.py): analysis (ingest→asr) then dub (translate→mux).
# Defined in streaming/timing.py, which the LIVE stream path also uses — bench
# and production must not be able to report different numbers for one run.
ANALYSIS_STAGES = timing_mod.ANALYSIS_STAGES
DUB_STAGES = timing_mod.DUB_STAGES

# A stage_cache hit sets build_s to exactly 0.0 (runner.py), so this only has to
# absorb per-stage rounding in the report.
WARM_BUILD_EPSILON_S = 0.05


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
    # Added 2026-08-24. Defaulted so the reports stored under
    # agent-os/specs/2026-06-20-1656-pipeline-benchmark-perf/results/ still
    # deserialize into this dataclass.
    profile: str | None = None      # PATMIR_PROFILE that produced this run
    warm_cache: bool = False        # a shared stage_cache was threaded through
    warm: bool = False              # ...and this repeat paid no model builds
    build_total_s: float | None = None  # sum of per-stage build_s


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


_stage_wall = timing_mod.stage_wall


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


def _build_total(stages: list[StageTiming]) -> float:
    return sum((st.build_s or 0.0) for st in stages)


def _gpu_peak_reset() -> None:
    """Reset CUDA's peak counters so `_resource_peaks` describes one repeat."""
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _resource_peaks() -> dict[str, Any]:
    """Best-effort memory telemetry. Never raises: a four-minute benchmark must
    not die on an instrumentation call.

    Why this is in the report at all: warm-model residency keeps every stage
    model alive for the whole run, and on a 24 GB discrete GPU with 23 GB of
    host RAM the margin is real (~14.5 GB resident, peaks ~18-21 GB). Whoever
    raises `max_batch` needs to read the headroom rather than re-derive it.

    `peak_rss_gb` is process-lifetime, so across `--repeat` it is monotonic and
    describes the whole invocation, not the last repeat. It is here because
    host swap — not VRAM — is what produced the 1.71 -> 3.02 RTF regression
    documented in performance.md.
    """
    out: dict[str, Any] = {}
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            out["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            free_b, _total_b = torch.cuda.mem_get_info()
            out["vram_free_gb"] = round(free_b / 1024 ** 3, 2)
    except Exception:
        pass
    try:
        import resource
        import sys
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KiB on Linux, bytes on Darwin.
        unit = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
        out["peak_rss_gb"] = round(maxrss / unit, 2)
    except Exception:
        pass
    return out


def _tts_stats(job_dir: Path) -> dict[str, Any] | None:
    rows = _read_jsonl(job_dir / "tts" / "timings.jsonl")
    out: dict[str, Any] = {}
    if rows:
        el = [r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]
        audio = sum(r.get("audio_s", 0.0) for r in rows)
        # Under batched decode the adapter emits one row per CHUNK, not per
        # sentence — a per-sentence wall does not exist once the chunk's cost is
        # shared (see stages/tts/adapter.py). So `count` and the percentiles
        # describe whatever a row is, and `unit` has to say which: the CUDA
        # report read `count: 3` beside `sentences: 24` with `p90_s: 51.2`, and
        # those chunk numbers are trivially misread as the per-sentence p90 the
        # Apple-Silicon rows quote.
        chunk_rows = [r for r in rows if r.get("kind") == "chunk"]
        out.update({
            "unit": "chunk" if chunk_rows else "sentence",
            "count": len(el),
            "total_s": round(sum(el), 2),
            "mean_s": round(statistics.mean(el), 3) if el else 0.0,
            "median_s": round(statistics.median(el), 3) if el else 0.0,
            "p90_s": round(_pctile(el, 0.9), 3),
            "max_s": round(max(el), 3) if el else 0.0,
            "audio_s": round(audio, 2),
            # Correct under batching by construction: total compute over total
            # audio does not care how the rows were bucketed.
            "tts_rtf": round(sum(el) / audio, 3) if audio else None,
        })
        if chunk_rows:
            covered = sum(int(r.get("sentences") or 0) for r in chunk_rows)
            out["chunks"] = len(chunk_rows)
            # NOT named `sentences` — that key belongs to the per-job total from
            # voice_cloning_summary.json, merged below, and must keep winning.
            out["chunk_sentences"] = covered
            out["sentences_per_chunk"] = round(covered / len(chunk_rows), 2)
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
                repeat: int = 1, label: str | None = None,
                warm_cache: bool = True) -> list[BenchResult]:
    """Run `clip` through the batch pipeline `repeat` times.

    With `warm_cache` (the default) all repeats share ONE `stage_cache`, so
    repeat 0 builds every stage model and repeats 1..N-1 reuse them — the same
    warm path `bench_watch_along`, the stream loop and the UI server all take.
    The warm repeats are the baseline; repeat 0 is the cold datapoint.

    Before 2026-08-24 no cache was passed at all, so every repeat rebuilt
    everything. That was invisible on MLX (mmap loads, `build_s` 0-3 s) and
    dominant on torch (~110 s of a 242 s run) — `--repeat 3` was timing nine
    model loads three times rather than steady state.

    `warm_cache=False` restores that behaviour. Two reasons to want it: N
    independent cold samples, and memory — every stage model staying resident
    is ~14.5 GB on the cuda profile, which is fine on 24 GB but not free.

    `prewarm_stage_cache` is deliberately NOT used here. It would zero repeat
    0's `build_s` too, destroying the cold number, and it swallows build
    failures by contract — a benchmark has to fail loudly instead of quietly
    reporting a run with a stage missing.
    """
    results: list[BenchResult] = []
    # One dict for every repeat, or None to rebuild each time.
    stage_cache: dict[str, Any] | None = {} if warm_cache else None
    profile = cfg.profile
    try:
        for r in range(repeat):
            _gpu_peak_reset()
            manifest = run_job(
                clip, cfg,
                runs_root=runs_root,
                extra={"bench": True},
                stage_cache=stage_cache,
            )
            job_dir = runs_root / manifest.job_id
            source_s = _source_seconds(job_dir)
            stages, total = _stage_timings(manifest, source_s)
            build_total = _build_total(stages)
            notes: dict[str, Any] = {
                "build_pct_of_wall": round(100 * build_total / total, 1) if total else None,
            }
            notes.update(_resource_peaks())
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
                notes=notes,
                profile=profile,
                warm_cache=warm_cache,
                # Measured, not declared: the JSON should say what actually
                # happened, not what was requested.
                warm=bool(warm_cache and build_total <= WARM_BUILD_EPSILON_S),
                build_total_s=round(build_total, 2),
            ))
    finally:
        # Don't leave ~15 GB of models pinned while the report is written, or
        # while a second mode runs in the same process.
        if stage_cache is not None:
            stage_cache.clear()
            _free_accelerator_cache()
    return results


# --------------------------------------------------------------------------- #
# watch-along (D3)
# --------------------------------------------------------------------------- #
class BenchPlayhead:
    """Synthetic viewer, so the bench exercises the scheduler the server runs.

    `bench_watch_along` used to walk sections with its own forward `while`
    loop, which meant coverage, lookahead, seek priority, idling and the
    abandon timer — every part of `StreamScheduler` — were untested by the
    benchmark that was supposed to characterise them.

    Two viewers, because they answer different questions:

    * `greedy` (default) sits exactly at the buffer edge, so the scheduler
      always has work and wall time measures pure production throughput. Fast,
      and comparable to the historical forward-loop numbers.
    * `realtime` advances with the clock and **stalls when it outruns
      production**, which is precisely what the browser client does. It counts
      those stalls, so a freeze becomes a number instead of a complaint. Costs
      about one video-duration of wall time — use `--max-sections`.
    """

    def __init__(self, mode: str = "greedy") -> None:
        self.mode = mode
        self._t0 = time.perf_counter()
        self._ph = Playhead(t=0.0, paused=False, updated_at=time.time())
        self.coverage = None          # set by the caller once the scheduler exists
        self.stalls: list[dict[str, Any]] = []
        self._stalled_at: float | None = None

    def _covered_end(self, t: float) -> float:
        if self.coverage is None:
            return t
        gap = self.coverage.first_uncovered(t)
        return t if gap is None else gap

    def __call__(self) -> Playhead:
        # Keep the abandon timer fed: a synthetic viewer never goes away, and
        # the scheduler would otherwise end the stream after 45 s of idling.
        self._ph.updated_at = time.time()
        if self.mode == "greedy":
            self._ph.t = self._covered_end(self._ph.t)
            return self._ph

        wanted = time.perf_counter() - self._t0
        edge = self._covered_end(self._ph.t)
        if wanted > edge:
            # Outran production: this is the freeze the author reported.
            if self._stalled_at is None:
                self._stalled_at = time.perf_counter()
            self._ph.t = edge
        else:
            if self._stalled_at is not None:
                self.stalls.append({
                    "at_s": round(self._ph.t, 2),
                    "waited_s": round(time.perf_counter() - self._stalled_at, 2),
                })
                self._stalled_at = None
            self._ph.t = wanted
        return self._ph



def _section_record(index: int, sec, wall_s: float, runs_root: Path) -> dict[str, Any]:
    """One section's timing. Thin wrapper over the shared reader so bench and
    the live stream path cannot drift; `runs_root` is unused (the section's own
    `job_dir` carries the manifest) and kept for call-site compatibility."""
    return timing_mod.section_timing(
        index=index, start=sec.start, end=sec.end, job_dir=sec.job_dir,
        wall_s=wall_s, new_speakers=sec.new_speakers,
    )


def bench_watch_along(clip: Path, config_path: Path, *, runs_root: Path,
                      repeat: int = 1, max_sections: int | None = None,
                      serial_sections: bool = False,
                      label: str | None = None,
                      warm_cache: bool = True,
                      scheduler: bool = True,
                      playhead: str = "greedy",
                      prewarm: bool = False,
                      traces: bool = True) -> list[BenchResult]:
    """Run `clip` through the watch-along section loop `repeat` times.

    Unlike `bench_batch`, the cache is created INSIDE the repeat loop and stays
    there. Watch-along's warmness is per-section — section 0 builds, later
    sections reuse — so each repeat has to pay its own section-0 builds to stay
    comparable with the stored baselines (agg RTF 1.71 / 1.611 / 1.47). Hoisting
    it would make repeats 1+ skip work a real stream cannot skip.

    `warm_cache=False` reproduces the pre-2026-07-14 rebuild-every-section path,
    which is how the historical 1.94-2.31 rows were measured.
    """
    from patmir.pipeline.streaming.prelude import run_prelude
    from patmir.pipeline.streaming.runner import streaming_config, source_duration
    from patmir.pipeline.streaming.section import (
        DEFAULT_SECTION_SECONDS,
        dub_prelude_window,
        analyze_section,
        dub_section,
        process_section,
    )

    results: list[BenchResult] = []
    for r in range(repeat):
        cfg = streaming_config(config_path)
        duration = source_duration(clip)

        sections: list[dict[str, Any]] = []
        # Same warm-model cache the production stream loop uses (ui/stream.py)
        # — bench must measure the warm path, not the legacy rebuild path.
        stage_cache: dict[str, Any] | None = {} if warm_cache else None
        # A live server prewarms at boot, so its section 0 pays no build. The
        # default (no prewarm) keeps the historical baselines comparable; pass
        # --prewarm to measure what the author's server actually does.
        prewarm_s = 0.0
        if prewarm and stage_cache is not None:
            pt = time.perf_counter()
            prewarm_stage_cache(cfg, stage_cache)
            prewarm_s = time.perf_counter() - pt
        _gpu_peak_reset()
        run_t0 = time.perf_counter()
        t0 = time.perf_counter()
        prelude = run_prelude(
            clip, cfg,
            window_seconds=cfg.options.prelude_window_seconds,
            runs_root=runs_root,
            stage_cache=stage_cache,
            bench=traces,
        )
        prelude_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        sec0 = dub_prelude_window(cfg, prelude, runs_root=runs_root,
                                  stage_cache=stage_cache, bench=traces)
        sec0_wall = time.perf_counter() - t0
        sections.append(_section_record(0, sec0, sec0_wall, runs_root))
        # The author-facing number: nothing plays before this.
        time_to_first_audio_s = time.perf_counter() - run_t0

        counter = {"n": 0}
        stalls: list[dict[str, Any]] = []
        # Real elapsed seconds of the streaming phase. Under the pipeline the
        # per-section walls OVERLAP, so summing them double-counts and would
        # report no improvement however well the overlap worked.
        stream_elapsed_s: float | None = None

        def _one(start: float, max_end: float | None = None):
            counter["n"] += 1
            s0 = time.perf_counter()
            sec = process_section(
                clip, cfg, prelude.registry,
                index=counter["n"], start=start, runs_root=runs_root,
                stage_cache=stage_cache, bench=traces,
                source_lang=prelude.source_lang,
                # Only the scheduler branch supplies this; the straight-line
                # loop below never revisits covered ground, so it has no clamp.
                max_end=max_end,
            )
            sections.append(_section_record(counter["n"], sec,
                                            time.perf_counter() - s0, runs_root))
            return sec

        def _process_serial(start: float, max_end: float | None = None):
            if max_sections is not None and counter["n"] >= max_sections:
                sched.stop()
                return None
            return _one(start, max_end)

        if scheduler:
            # Drive the SAME scheduler the server runs, so coverage, lookahead,
            # seek priority and idling are inside the measurement instead of
            # being replaced by a straight-line loop that cannot stall. That
            # includes the PIPELINE: the server overlaps a section's dub with the
            # next section's analysis, so a bench that ran them serially would
            # measure a configuration nobody ships.
            ph = BenchPlayhead(playhead)
            pending: dict[str, Any] = {}

            def _analyze(start: float, max_end: float | None = None):
                if max_sections is not None and counter["n"] >= max_sections:
                    sched.stop()
                    return None
                counter["n"] += 1
                idx = counter["n"]
                pending[str(idx)] = time.perf_counter()
                return analyze_section(
                    clip, cfg, prelude.registry,
                    index=idx, start=start, runs_root=runs_root,
                    stage_cache=stage_cache, bench=traces,
                    source_lang=prelude.source_lang, max_end=max_end,
                )

            def _dub(analysis):
                sec = dub_section(cfg, analysis, runs_root=runs_root,
                                  stage_cache=stage_cache)
                s0 = pending.pop(str(analysis.index), time.perf_counter())
                sections.append(_section_record(analysis.index, sec,
                                                time.perf_counter() - s0, runs_root))
                return sec

            def _bench_idle() -> bool:
                """End the run when the whole video is covered.

                `BenchPlayhead` deliberately keeps `updated_at` fresh — a
                synthetic viewer never leaves — so the scheduler's default idle,
                which ends a stream after `abandon_seconds` without a sign of
                life, can never fire here. Without this the scheduler-driven
                bench spins at 2 s a tick forever once the last section lands and
                the report is never written. Every stored scheduler run used
                `--max-sections`, which stops through a different path, so a
                full-length scheduler run had never reached this state.
                """
                if sched.coverage.first_uncovered(after=0.0,
                                                  until=duration - 1.0) is None:
                    return False
                time.sleep(0.25)
                return True

            sched = StreamScheduler(
                duration=duration,
                prelude_end=prelude.window_end,
                process_fn=_process_serial,
                analyze_fn=None if serial_sections else _analyze,
                dub_fn=None if serial_sections else _dub,
                playhead_fn=ph,
                idle_fn=_bench_idle,
            )
            ph.coverage = sched.coverage
            _stream_t0 = time.perf_counter()
            sched.run()
            stream_elapsed_s = time.perf_counter() - _stream_t0
            stalls = ph.stalls
        else:
            idx = 1
            t = prelude.window_end
            while t < duration - 0.5:
                counter["n"] = idx - 1
                sec = _one(t)
                t = sec.end if sec.end > t + 0.5 else min(t + DEFAULT_SECTION_SECONDS, duration)
                idx += 1
                if max_sections is not None and idx > max_sections:
                    break

        sections.sort(key=lambda s_: s_["index"])
        dubbed_audio_s = sum(s["section_s"] for s in sections)
        # Sum of per-section walls: still the right denominator for per-stage
        # attribution, but NOT the wall the viewer experiences once the halves
        # overlap -- a section's analyze and the previous section's dub run at the
        # same wall-clock time, so the sum counts that time twice.
        sections_wall = sum(s["wall_s"] for s in sections)
        # `stream_elapsed_s` is measured around `sched.run()`, which covers only
        # sections 1..N -- section 0 is the prelude dub and finishes before the
        # scheduler starts. Comparing it against a sum that INCLUDES section 0
        # understates the total by section 0's wall and reports a ~70 s "overlap"
        # on a serial run. So both sides of the comparison are steady-state only.
        steady_wall = sum(s["wall_s"] for s in sections if s["index"] >= 1)
        steady_audio_s = sum(s["section_s"] for s in sections if s["index"] >= 1)
        sec0_wall_s = sum(s["wall_s"] for s in sections if s["index"] == 0)
        steady_elapsed = stream_elapsed_s if stream_elapsed_s is not None else steady_wall
        sections_elapsed = sec0_wall_s + steady_elapsed
        total_wall = prelude_wall + sections_elapsed

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
                # Wall clock actually spent on sections (section 0 + the measured
                # elapsed of the scheduler loop). Differs from `sections_wall_s`
                # by exactly what the pipeline overlapped.
                "sections_elapsed_s": round(sections_elapsed, 2),
                # Steady state = sections 1..N, i.e. excluding the prelude dub.
                # This is the pair to compare across arms: the prelude is not
                # pipelined and its cost is identical either way.
                "steady_wall_sum_s": round(steady_wall, 2),
                "steady_elapsed_s": round(steady_elapsed, 2),
                "steady_audio_s": round(steady_audio_s, 2),
                "steady_rtf": (round(steady_elapsed / steady_audio_s, 3)
                               if steady_audio_s else None),
                "pipeline_overlap_s": round(steady_wall - steady_elapsed, 2),
                "dubbed_audio_s": round(dubbed_audio_s, 2),
                "n_sections": len(sections),
                "total_build_s": round(total_build, 2),
                # Submit -> first playable audio. The author's actual complaint
                # ("the first chunk takes too long") had no metric before this.
                "time_to_first_audio_s": round(time_to_first_audio_s, 2),
                "prewarm_s": round(prewarm_s, 2) if prewarm else None,
                # Work no stage owns, summed: ffmpeg slice/trim, ECAPA embed,
                # age-gender profile, and any lazily-loaded model.
                "other_s": round(sum(s_.get("other_s") or 0.0 for s_ in sections), 2),
                "driver": "scheduler" if scheduler else "forward-loop",
                # Whether a section's dub overlapped the NEXT section's
                # analysis. False is the pre-2026-08-28 behaviour and is the
                # control any pipelined number has to be read against.
                "pipelined": bool(scheduler and not serial_sections),
                "playhead": playhead if scheduler else None,
                # Non-empty only under --playhead realtime: each entry is a
                # freeze the viewer would have seen.
                "stalls": stalls,
                "stall_count": len(stalls),
                "stall_total_s": round(sum(x["waited_s"] for x in stalls), 2),
                "build_pct_of_wall": round(100 * total_build / total_wall, 1) if total_wall else None,
                # `agg_rtf` above divides by DUBBED AUDIO, not source seconds —
                # unlike batch and unlike the convention performance.md states.
                # Keeping it that way is deliberate: every historical
                # watch-along number (1.71, 1.611, 1.47, 1.94-2.31) was computed
                # this way, and silently reflooring the field would leave a
                # reader unable to tell which convention a given row used. So
                # name the denominator and publish the source-second figure
                # beside it. They are within 0.03% today (dubbed 301.0 vs source
                # 301.1); they were 4.9% apart before the pad-dubbing fix.
                "rtf_denominator": "dubbed_audio_s",
                "agg_rtf_source": round(total_wall / duration, 3) if duration else None,
                **_resource_peaks(),
            },
            profile=cfg.profile,
            warm_cache=warm_cache,
            # Never "warm" as a whole: section 0 builds by construction. Warmness
            # here is per-section and lives in notes.build_pct_of_wall.
            warm=False,
            build_total_s=round(total_build, 2),
        ))
    return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def to_json(results: list[BenchResult]) -> dict[str, Any]:
    return {"results": [asdict(r) for r in results]}


def _fmt(v: Any, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "-"


def _cache_tag(res: BenchResult) -> str:
    """Say, in the header, whether this repeat paid for model builds. Cold and
    warm numbers are not interchangeable and must never read as one series."""
    if not res.warm_cache:
        return " (no warm cache)"
    if res.mode != "batch":
        return ""
    return " (warm)" if res.warm else " (cold: builds models)"


def _summary_line(results: list[BenchResult]) -> list[str]:
    """Cold and warm summarized separately — averaging them is the mistake this
    whole change exists to prevent."""
    batch = [r for r in results if r.mode == "batch"]
    if len(batch) < 2:
        return []
    cold = [r for r in batch if not r.warm]
    warm = [r for r in batch if r.warm]
    parts: list[str] = []
    for name, group in (("cold", cold), ("warm", warm)):
        if not group:
            continue
        walls = [r.total_wall_s for r in group]
        rtfs = [r.agg_rtf for r in group if r.agg_rtf is not None]
        builds = [r.build_total_s or 0.0 for r in group]
        which = f"repeat {group[0].repeat_index}" if len(group) == 1 else f"mean of {len(group)}"
        parts.append(
            f"{name} {which}: wall {statistics.mean(walls):.2f}s "
            f"RTF {statistics.mean(rtfs):.3f} (build {statistics.mean(builds):.2f}s)"
            if rtfs else
            f"{name} {which}: wall {statistics.mean(walls):.2f}s (build {statistics.mean(builds):.2f}s)"
        )
    out = [""] + [f"--- {' | '.join(parts)}"] if parts else []
    peaks = batch[-1].notes
    mem = []
    if peaks.get("peak_vram_gb") is not None:
        mem.append(f"peak VRAM {peaks['peak_vram_gb']} GB alloc / "
                   f"{peaks.get('peak_vram_reserved_gb')} GB reserved "
                   f"(free {peaks.get('vram_free_gb')} GB)")
    if peaks.get("peak_rss_gb") is not None:
        mem.append(f"peak RSS {peaks['peak_rss_gb']} GB (process lifetime)")
    if mem:
        out.append(f"--- {' · '.join(mem)}")
    return out


def format_report(results: list[BenchResult]) -> str:
    lines: list[str] = []
    for res in results:
        # `any(r.repeat_index ...)` hid repeat=0, which is exactly the repeat a
        # reader most needs labelled now that it is the cold one.
        rep = f" repeat={res.repeat_index}" if len(results) > 1 else ""
        lines.append("")
        lines.append(f"=== {res.mode} | {Path(res.clip).name}{rep}{_cache_tag(res)} "
                     f"| {res.job_id} ===")
        rtf_extra = ""
        if res.mode == "watch-along" and res.notes.get("agg_rtf_source") is not None:
            rtf_extra = (f" [per {res.notes.get('rtf_denominator', 'dubbed_audio_s')} "
                         f"{res.notes.get('dubbed_audio_s')}s; "
                         f"per source {res.notes['agg_rtf_source']}]")
        lines.append(f"source {_fmt(res.source_s, 's')}  "
                     f"wall {_fmt(res.total_wall_s, 's')}  "
                     f"agg RTF {_fmt(res.agg_rtf)}  "
                     f"({'slower' if (res.agg_rtf or 0) > 1 else 'faster'} than realtime)"
                     f"{rtf_extra}")
        if res.mode == "batch" and res.build_total_s is not None:
            lines.append(f"build {res.build_total_s}s "
                         f"({_fmt(res.notes.get('build_pct_of_wall'), '%')} of wall)"
                         + (f"  profile {res.profile}" if res.profile else ""))
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
            unit = t.get("unit", "sentence")
            lines.append("")
            if unit == "chunk":
                # Spell out the batching so a chunk p90 is never read as the
                # per-sentence p90 the Apple-Silicon rows quote.
                head = (f"  tts: {t.get('chunks', '?')} chunks over "
                        f"{t.get('sentences', t.get('chunk_sentences', '?'))} sentences "
                        f"(~{t.get('sentences_per_chunk', '?')} sent/chunk)")
                per = "/chunk"
            else:
                head = f"  tts: {t.get('count', '?')} sentences"
                per = ""
            lines.append(f"{head}, mean {_fmt(t.get('mean_s'), 's')}{per}, "
                         f"p90 {_fmt(t.get('p90_s'), 's')}{per}, "
                         f"max {_fmt(t.get('max_s'), 's')}{per}, "
                         f"tts_rtf {_fmt(t.get('tts_rtf'))}, "
                         f"overflows {t.get('overflows', '?')}")
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
    lines.extend(_summary_line(results))
    return "\n".join(lines)
