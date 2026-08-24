"""Patmir CLI.

Subcommands:
  patmir dub <video> [--resume-from <stage>] [--config <path>]
  patmir runs ls
  patmir runs show <ts>
  patmir ui [--port 8765]
  patmir submit <video|--url URL>

Progress is read from `<job_dir>/progress.jsonl` in a background thread and
rendered as one line per stage in the terminal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from patmir.env import load_dotenv
from patmir.paths import StoragePaths
from patmir.pipeline import config as config_mod
from patmir.pipeline.config import load_pipeline_config
from patmir.pipeline.manifest import Manifest
from patmir.pipeline.runner import run_job
from patmir.pipeline.stage import STAGE_ORDER

DEFAULT_CONFIG = Path("patmir.toml")


def _storage_for(config: Path) -> StoragePaths:
    """Resolve just the storage roots, without requiring a full pipeline config.

    `runs ls`, `runs show`, and `submit` don't build any stage — they only need
    the roots. Before `[storage]` existed they ran with no `patmir.toml` at all,
    so a missing config falls back to the defaults rather than failing. A config
    that exists but is malformed still raises.
    """
    import tomllib

    from patmir.paths import defaults, load_storage

    if not config.is_file():
        return defaults(config)
    with config.open("rb") as f:
        return load_storage(tomllib.load(f), config)


def _require_config(config: Path) -> bool:
    """Print a clean error (instead of a traceback) when the config is missing."""
    if config.is_file():
        return True
    print(f"error: config not found: {config}", file=sys.stderr)
    print("  pass --config <path> or run from the directory holding patmir.toml",
          file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    # Project-scoped secrets (OPENROUTER_API_KEY, HF_TOKEN, etc.) live in
    # `.env` at the project root. Shell-exported env vars always win.
    load_dotenv()

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
    p_dub.add_argument("--profile", type=str, default=None,
                       help='Machine profile from [pipeline.profiles.<name>] in patmir.toml (e.g. "cuda"). Sugar for $PATMIR_PROFILE — it is exported into the environment so the watch-along path picks it up too.')

    p_runs = sub.add_parser("runs", help="Inspect prior runs.")
    p_runs_sub = p_runs.add_subparsers(dest="runs_cmd", required=True)
    # Both need --config to learn [storage].runs_root.
    p_runs_ls = p_runs_sub.add_parser("ls", help="List runs under the configured runs root.")
    p_runs_ls.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_show = p_runs_sub.add_parser("show", help="Show a run's manifest.")
    p_show.add_argument("job_id")
    p_show.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    p_ui = sub.add_parser("ui", help="Launch the local FastAPI progress UI.")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.add_argument("--runs-root", type=Path, default=None,
                      help="Override [storage].runs_root for this server.")
    p_ui.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_ui.add_argument("--profile", type=str, default=None,
                      help='Machine profile from [pipeline.profiles.<name>] in patmir.toml (e.g. "cuda"). Sugar for $PATMIR_PROFILE — it is exported into the environment so the watch-along path picks it up too.')

    p_bench = sub.add_parser(
        "bench",
        help="Benchmark the pipeline: per-stage wall + RTF for a clip.",
    )
    p_bench.add_argument("clip", type=Path)
    p_bench.add_argument("--mode", choices=["batch", "watch-along"], default="batch")
    p_bench.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_bench.add_argument("--repeat", type=int, default=1,
                         help="Run N times in one process sharing a warm model cache: "
                              "repeat 0 is cold (builds every stage model), 1..N-1 are warm. "
                              "The warm repeats are the baseline.")
    p_bench.add_argument("--no-warm-cache", dest="warm_cache",
                         action="store_false", default=True,
                         help="Rebuild every stage model on every repeat (the behaviour "
                              "before 2026-08-24). Use it to collect N independent COLD "
                              "samples, or on a box where keeping all stage models "
                              "resident risks OOM; every reported repeat is then cold.")
    p_bench.add_argument("--max-sections", type=int, default=None,
                         help="watch-along only: cap the number of sections processed.")
    p_bench.add_argument("--report", type=Path, default=None,
                         help="Write machine-readable bench.json to this path.")
    p_bench.add_argument("--label", type=str, default=None)
    p_bench.add_argument("--profile", type=str, default=None,
                         help='Machine profile from [pipeline.profiles.<name>] in patmir.toml (e.g. "cuda"). Sugar for $PATMIR_PROFILE — it is exported into the environment so the watch-along path picks it up too.')

    p_submit = sub.add_parser(
        "submit",
        help="POST a file or URL to the local Patmir server. Server must be running (patmir ui).",
    )
    p_submit.add_argument("video", nargs="?", type=Path,
                          help="Local video file. Omit when using --url.")
    p_submit.add_argument("--url", type=str, default=None,
                          help="Video URL (e.g. YouTube). Cannot be combined with a file path.")
    p_submit.add_argument("--source-lang", type=str, default=None)
    p_submit.add_argument("--server", type=str, default="http://127.0.0.1:8765")
    p_submit.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                          help="Read the bearer token from this config's [storage].state_root.")

    args = parser.parse_args(argv)
    # A profile has to reach every config load in the process, including
    # `streaming_config()` deep inside the watch-along path, which only ever
    # receives a config path. Exporting it makes the flag and the env var the
    # same mechanism instead of the flag silently applying to batch only.
    if getattr(args, "profile", None):
        os.environ[config_mod.PROFILE_ENV] = args.profile
    if args.cmd == "dub":
        return cmd_dub(args)
    if args.cmd == "runs":
        if args.runs_cmd == "ls":
            return cmd_runs_ls(args.config)
        if args.runs_cmd == "show":
            return cmd_runs_show(args.job_id, args.config)
    if args.cmd == "ui":
        return cmd_ui(args.port, args.runs_root, args.config)
    if args.cmd == "bench":
        return cmd_bench(args)
    if args.cmd == "submit":
        return cmd_submit(args)
    parser.print_help()
    return 1


def cmd_dub(args: argparse.Namespace) -> int:
    if not args.video.exists():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 2
    if not _require_config(args.config):
        return 2
    cfg = load_pipeline_config(args.config, skip_before=args.resume_from)
    runs_root = cfg.storage.runs_root

    job_dir = args.resume_job
    progress_stop = threading.Event()
    progress_thread: threading.Thread | None = None

    if job_dir is None:
        # Pre-create the runs root so the run_job mkdir doesn't fail.
        runs_root.mkdir(parents=True, exist_ok=True)
        # The runner makes the per-job dir; we tail progress.jsonl by polling
        # the runs/ root for the freshest dir after a moment.
        # Simpler: have the runner accept a callback to give us the job_dir as
        # soon as it's created.
        # For Phase 1 we just print progress events from manifest after the run.
        pass

    print(f"patmir dub {args.video}", file=sys.stderr)
    print(f"  config: {args.config}", file=sys.stderr)
    print(f"  runs:   {runs_root}", file=sys.stderr)
    if cfg.storage.library_root:
        print(f"  library: {cfg.storage.library_root}", file=sys.stderr)
    if args.resume_from:
        print(f"  resume-from: {args.resume_from}", file=sys.stderr)

    try:
        manifest = run_job(
            args.video,
            cfg,
            runs_root=runs_root,
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

    _print_summary(manifest, runs_root)
    return 0


def cmd_runs_ls(config: Path) -> int:
    runs_root = _storage_for(config).runs_root
    if not runs_root.exists():
        print("(no runs)")
        return 0
    rows = []
    for child in sorted(runs_root.iterdir(), reverse=True):
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


def cmd_runs_show(job_id: str, config: Path) -> int:
    path = _storage_for(config).runs_root / job_id / "manifest.json"
    if not path.exists():
        print(f"no manifest at {path}", file=sys.stderr)
        return 2
    print(path.read_text())
    return 0


def cmd_ui(port: int, runs_root: Path | None, config: Path) -> int:
    if not _require_config(config):
        return 2
    try:
        from patmir.ui.app import serve
        from patmir.ui import auth as ui_auth
    except ImportError as exc:
        print(f"UI extra not installed: {exc}", file=sys.stderr)
        print("  pip install -e '.[ui]'", file=sys.stderr)
        return 2
    cfg = load_pipeline_config(config)
    storage = cfg.storage
    if runs_root is not None:
        storage = replace(storage, runs_root=runs_root.expanduser().absolute())
    token = ui_auth.ensure_token(storage.secret_path)
    print(f"patmir ui — bind 127.0.0.1:{port}", file=sys.stderr)
    print(f"  config:  {config}", file=sys.stderr)
    print(f"  runs:    {storage.runs_root}", file=sys.stderr)
    print(f"  cache:   {storage.cache_root}", file=sys.stderr)
    if storage.library_root:
        print(f"  library: {storage.library_root}", file=sys.stderr)
    print(f"  token:   {token}  (paste into the Chrome extension options page)",
          file=sys.stderr)
    serve(port=port, storage=storage, cfg=cfg, token=token, config_path=config)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    if not args.clip.exists():
        print(f"error: {args.clip} not found", file=sys.stderr)
        return 2
    if not _require_config(args.config):
        return 2
    from patmir import bench as benchmod

    cfg = load_pipeline_config(args.config)
    runs_root = cfg.storage.runs_root
    runs_root.mkdir(parents=True, exist_ok=True)
    cache_mode = "warm cache shared across repeats" if args.warm_cache else "no warm cache (every repeat cold)"
    print(f"patmir bench {args.clip} (mode={args.mode}, repeat={args.repeat}, "
          f"profile={cfg.profile or '-'}, {cache_mode})",
          file=sys.stderr)
    if args.mode == "batch":
        results = benchmod.bench_batch(
            args.clip, cfg, runs_root=runs_root,
            repeat=args.repeat, label=args.label,
            warm_cache=args.warm_cache,
        )
    else:
        results = benchmod.bench_watch_along(
            args.clip, args.config, runs_root=runs_root,
            repeat=args.repeat, max_sections=args.max_sections, label=args.label,
            warm_cache=args.warm_cache,
        )

    print(benchmod.format_report(results))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(benchmod.to_json(results), indent=2, ensure_ascii=False)
        )
        print(f"\nwrote {args.report}", file=sys.stderr)
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    import urllib.request

    if bool(args.video) == bool(args.url):
        print("error: pass exactly one of <video> or --url", file=sys.stderr)
        return 2

    try:
        from patmir.ui import auth as ui_auth
    except ImportError as exc:
        print(f"UI extra not installed: {exc}", file=sys.stderr)
        return 2

    token = ui_auth.ensure_token(_storage_for(args.config).secret_path)
    body: dict[str, str] = {}
    if args.video:
        path = args.video.expanduser().resolve()
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
        body["source_path"] = str(path)
    else:
        body["source_url"] = args.url
    if args.source_lang:
        body["source_lang"] = args.source_lang

    req = urllib.request.Request(
        f"{args.server.rstrip('/')}/jobs",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"error: could not reach {args.server} — is `patmir ui` running?", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def _print_summary(m: Manifest, runs_root: Path) -> None:
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
    if m.published_dir:
        print(f"published: {m.published_dir}")
    _print_diarize_summary(m, runs_root)


def _print_diarize_summary(m: Manifest, runs_root: Path) -> None:
    """Surface speaker count + per-(speaker, emotion) ref-clip coverage."""
    job_dir = runs_root / m.job_id
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
