"""Phase 0 TTS eval runner.

Loads enabled TTS candidates from `patmir.toml`, synthesizes every Armenian
eval utterance with every candidate, and writes:

  eval_runs/<timestamp>/tts/
    <candidate_id>/<utt_id>.wav
    index.html                 # side-by-side A/B/C audio players
    timings.jsonl              # per-(candidate, utt) synthesis time
    scoresheet.csv             # pre-filled, empty score columns

Candidates that take longer than `SLOW_THRESHOLD_S` per utterance on the
first item are flagged as "GPU-pass only" but still synthesized (so the
author has a sample to compare even when slow).

Usage:
    uv run python -m patmir.eval.tts_eval --config patmir.toml
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from patmir.eval._loader import load_candidates

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATA = REPO_ROOT / "eval_data" / "tts"
EVAL_RUNS = REPO_ROOT / "eval_runs"

SLOW_THRESHOLD_S = 30.0
SCORE_COLUMNS = ["intelligibility", "naturalness", "prosody", "child_would_watch_hours", "notes"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def synth_all(
    items: list[dict],
    candidates: list[tuple[str, dict, object]],
    out_dir: Path,
) -> tuple[dict[str, list[dict]], set[str]]:
    """Synthesize every utterance × candidate. Returns (timings_by_candidate, slow_ids)."""
    timings: dict[str, list[dict]] = {}
    slow_candidates: set[str] = set()

    for cid, _cfg, candidate in candidates:
        print(f"\nCandidate: {cid}", file=sys.stderr)
        cand_dir = out_dir / cid
        cand_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for i, item in enumerate(items):
            wav_path = cand_dir / f"{item['id']}.wav"
            t0 = time.perf_counter()
            try:
                wav_bytes = candidate.synthesize(item["text"])
                wav_path.write_bytes(wav_bytes)
                error = None
            except Exception as exc:
                error = repr(exc)
            elapsed_s = time.perf_counter() - t0
            rows.append({
                "candidate": cid,
                "id": item["id"],
                "category": item["category"],
                "elapsed_s": round(elapsed_s, 2),
                "error": error,
            })
            tag = "✗" if error else "✓"
            print(f"  {tag} {item['id']}  ({elapsed_s:.1f}s)", file=sys.stderr)
            # Flag slow candidates on the first utterance only.
            if i == 0 and elapsed_s > SLOW_THRESHOLD_S and not error:
                slow_candidates.add(cid)
                print(
                    f"    ⚠ {cid} took {elapsed_s:.0f}s on first utterance > {SLOW_THRESHOLD_S}s — "
                    "flagging as GPU-pass-only.",
                    file=sys.stderr,
                )
        timings[cid] = rows

    return timings, slow_candidates


def write_timings(path: Path, timings: dict[str, list[dict]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rows in timings.values():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_index_html(
    out_path: Path,
    items: list[dict],
    candidate_ids: list[str],
    slow_candidates: set[str],
) -> None:
    """Single static HTML — opens directly in a browser, plays all WAVs inline."""
    parts: list[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Patmir — Phase 0 TTS eval</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }",
        "table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }",
        "th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; vertical-align: top; }",
        "th { background: #f5f5f5; }",
        "audio { width: 100%; }",
        ".text { font-size: 1.1rem; }",
        ".cat { color: #888; font-size: 0.85rem; }",
        ".slow { color: #b00; font-size: 0.85rem; }",
        "</style></head><body>",
        "<h1>Phase 0 TTS eval</h1>",
        "<p>Listen across rows, score in <code>scoresheet.csv</code>.</p>",
    ]

    if slow_candidates:
        parts.append("<p class='slow'>Flagged as GPU-pass-only on first utterance: " +
                     ", ".join(f"<code>{html.escape(c)}</code>" for c in sorted(slow_candidates)) + "</p>")

    # One table; one row per utterance; one column per candidate.
    header_cells = "".join(f"<th>{html.escape(c)}</th>" for c in candidate_ids)
    parts.append(f"<table><thead><tr><th>utterance</th>{header_cells}</tr></thead><tbody>")
    for item in items:
        cells: list[str] = []
        for cid in candidate_ids:
            rel = f"{cid}/{item['id']}.wav"
            cells.append(f"<td><audio controls preload='none' src='{html.escape(rel)}'></audio></td>")
        parts.append(
            f"<tr><td><div class='text'>{html.escape(item['text'])}</div>"
            f"<div class='cat'>{html.escape(item['id'])} · {html.escape(item['category'])}</div></td>"
            f"{''.join(cells)}</tr>"
        )
    parts.append("</tbody></table></body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_scoresheet(
    out_path: Path, items: list[dict], candidate_ids: list[str]
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "category", "candidate", *SCORE_COLUMNS])
        for item in items:
            for cid in candidate_ids:
                w.writerow([item["id"], item["category"], cid, "", "", "", "", ""])


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase 0 TTS eval.")
    ap.add_argument("--config", default="patmir.toml", type=Path)
    args = ap.parse_args()

    candidates = load_candidates(args.config.resolve(), stage="tts")
    candidate_ids = [cid for cid, _, _ in candidates]
    print(f"Enabled TTS candidates: {', '.join(candidate_ids)}", file=sys.stderr)

    items = load_jsonl(EVAL_DATA / "hy.jsonl")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = EVAL_RUNS / ts / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    timings, slow_candidates = synth_all(items, candidates, out_dir)

    write_timings(out_dir / "timings.jsonl", timings)
    write_index_html(out_dir / "index.html", items, candidate_ids, slow_candidates)
    write_scoresheet(out_dir / "scoresheet.csv", items, candidate_ids)
    print(f"\nWrote: {out_dir}\nOpen: file://{out_dir / 'index.html'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
