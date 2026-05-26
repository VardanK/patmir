"""Patmir CLI.

Subcommands:
  patmir dub <video> [--resume-from <stage>] [--config <path>]
  patmir runs ls
  patmir runs show <ts>
  patmir ui [--port 8765]

Progress is read from `<job_dir>/progress.jsonl` in a background thread and
rendered as one line per stage in the terminal.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from patmir.pipeline.config import load_pipeline_config
from patmir.pipeline.manifest import Manifest
from patmir.pipeline.runner import run_job
from patmir.pipeline.stage import STAGE_ORDER

DEFAULT_CONFIG = Path("patmir.toml")
RUNS_ROOT = Path("runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="patmir")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dub = sub.add_parser("dub", help="Run the full pipeline on a video file.")
    p_dub.add_argument("video", type=Path)
    p_dub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_dub.add_argument("--resume-from", choices=STAGE_ORDER, default=None)
    p_dub.add_argument("--resume-job", type=Path, default=None,
                       help="Existing job dir under runs/ to resume into.")
    p_dub.add_argument("--source-lang", type=str, default=None,
                       help="ISO source language hint (e.g. 'ru', 'en'). "
                            "Used by ASR as a fallback if it can't detect language.")

    p_runs = sub.add_parser("runs", help="Inspect prior runs.")
    p_runs_sub = p_runs.add_subparsers(dest="runs_cmd", required=True)
    p_runs_sub.add_parser("ls", help="List runs under ./runs.")
    p_show = p_runs_sub.add_parser("show", help="Show a run's manifest.")
    p_show.add_argument("job_id")

    p_ui = sub.add_parser("ui", help="Launch the local FastAPI progress UI.")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.add_argument("--runs-root", type=Path, default=RUNS_ROOT)

    args = parser.parse_args(argv)
    if args.cmd == "dub":
        return cmd_dub(args)
    if args.cmd == "runs":
        if args.runs_cmd == "ls":
            return cmd_runs_ls()
        if args.runs_cmd == "show":
            return cmd_runs_show(args.job_id)
    if args.cmd == "ui":
        return cmd_ui(args.port, args.runs_root)
    parser.print_help()
    return 1


def cmd_dub(args: argparse.Namespace) -> int:
    if not args.video.exists():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 2
    cfg = load_pipeline_config(args.config, skip_before=args.resume_from)

    job_dir = args.resume_job
    progress_stop = threading.Event()
    progress_thread: threading.Thread | None = None

    if job_dir is None:
        # Pre-create runs/ so the run_job mkdir doesn't fail.
        RUNS_ROOT.mkdir(exist_ok=True)
        # The runner makes the per-job dir; we tail progress.jsonl by polling
        # the runs/ root for the freshest dir after a moment.
        # Simpler: have the runner accept a callback to give us the job_dir as
        # soon as it's created.
        # For Phase 1 we just print progress events from manifest after the run.
        pass

    print(f"patmir dub {args.video}", file=sys.stderr)
    print(f"  config: {args.config}", file=sys.stderr)
    if args.resume_from:
        print(f"  resume-from: {args.resume_from}", file=sys.stderr)

    try:
        manifest = run_job(
            args.video,
            cfg,
            runs_root=RUNS_ROOT,
            resume_job_dir=job_dir,
            resume_from=args.resume_from,
            source_lang=args.source_lang,
        )
    except Exception as exc:
        progress_stop.set()
        print(f"\nfailed: {exc}", file=sys.stderr)
        return 1
    finally:
        progress_stop.set()
        if progress_thread:
            progress_thread.join(timeout=0.5)

    _print_summary(manifest)
    return 0


def cmd_runs_ls() -> int:
    if not RUNS_ROOT.exists():
        print("(no runs)")
        return 0
    rows = []
    for child in sorted(RUNS_ROOT.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.exists():
            rows.append((child.name, "?", "-"))
            continue
        m = Manifest.from_path(manifest_path)
        states = " ".join(_state_glyph(m.stage(n).state) for n in STAGE_ORDER)
        rows.append((m.job_id, states, m.source_path))
    width = max((len(r[0]) for r in rows), default=10)
    for job_id, states, src in rows:
        print(f"{job_id:<{width}}  {states}  {src}")
    return 0


def cmd_runs_show(job_id: str) -> int:
    path = RUNS_ROOT / job_id / "manifest.json"
    if not path.exists():
        print(f"no manifest at {path}", file=sys.stderr)
        return 2
    print(path.read_text())
    return 0


def cmd_ui(port: int, runs_root: Path) -> int:
    try:
        from patmir.ui.app import serve
    except ImportError as exc:
        print(f"UI extra not installed: {exc}", file=sys.stderr)
        print("  pip install -e '.[ui]'", file=sys.stderr)
        return 2
    serve(port=port, runs_root=runs_root)
    return 0


def _print_summary(m: Manifest) -> None:
    print()
    print(f"job: {m.job_id}")
    print(f"source: {m.source_path}  lang={m.source_lang}")
    for name in STAGE_ORDER:
        st = m.stage(name)
        dur = (
            f"{(st.finished_at - st.started_at):.1f}s"
            if st.started_at and st.finished_at
            else "-"
        )
        print(f"  {_state_glyph(st.state)} {name:<10} {dur}")
        for k, v in st.artifacts.items():
            print(f"        {k}: {v}")
    _print_diarize_summary(m)


def _print_diarize_summary(m: Manifest) -> None:
    """Surface speaker count + per-(speaker, emotion) ref-clip coverage."""
    job_dir = RUNS_ROOT / m.job_id
    turns_path = job_dir / "diarize" / "turns.json"
    refs_path = job_dir / "diarize" / "refs.json"
    if not turns_path.exists():
        return
    try:
        turns_data = json.loads(turns_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    speakers = turns_data.get("speakers") or sorted({t["speaker_id"] for t in turns_data.get("turns", [])})
    refs: dict[str, dict] = {}
    if refs_path.exists():
        try:
            refs = json.loads(refs_path.read_text())
        except (OSError, json.JSONDecodeError):
            refs = {}
    # Group refs by speaker — keys are "speaker__emotion" strings in Phase 2.x.
    by_spk: dict[str, list[tuple[str, dict]]] = {}
    for key, r in refs.items():
        spk = r.get("speaker_id")
        emo = r.get("emotion") or "neutral"
        if not spk and "__" in key:
            spk, emo = key.rsplit("__", 1)
        elif not spk:
            spk = key
        by_spk.setdefault(spk, []).append((emo, r))

    print()
    print(f"diarize: {len(speakers)} speaker(s)")
    for spk in speakers:
        spk_refs = sorted(by_spk.get(spk, []))
        if not spk_refs:
            print(f"  {spk}: no clean reference clip — using global fallback voice")
            continue
        bins = ", ".join(e for e, _ in spk_refs)
        print(f"  {spk}: {len(spk_refs)} bin(s) — {bins}")
        for emo, r in spk_refs:
            dur = float(r["source_end"]) - float(r["source_start"])
            text = (r.get("ref_text") or "").strip().replace("\n", " ")
            if len(text) > 60:
                text = text[:57] + "..."
            print(f"    [{emo}] {dur:.1f}s — {text}")

    # Per-sentence emotion coverage (from transcript.json).
    transcript_path = job_dir / "transcript.json"
    if transcript_path.exists():
        try:
            td = json.loads(transcript_path.read_text())
            from collections import Counter
            emo_counts = Counter(s.get("emotion") for s in td.get("sentences", []) if s.get("emotion"))
            if emo_counts:
                print(f"emotion (sentences): " + ", ".join(
                    f"{e}={n}" for e, n in emo_counts.most_common()))
        except (OSError, json.JSONDecodeError):
            pass


def _state_glyph(state: str) -> str:
    return {
        "pending": ".",
        "running": "*",
        "done": "+",
        "failed": "X",
    }.get(state, "?")


if __name__ == "__main__":
    raise SystemExit(main())
