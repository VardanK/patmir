"""Phase 0 translation eval runner.

Loads enabled translation candidates from `patmir.toml`, runs them against
the EN and/or RU eval set, and writes:

  eval_runs/<timestamp>/translation/
    <candidate_id>/<lang_pair>.jsonl      # raw outputs + latency
    translation_results.md                # side-by-side, human-readable
    scoresheet.csv                        # pre-filled, empty score columns

Usage:
    uv run python -m patmir.eval.translation_eval --config patmir.toml --pair both
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from patmir.eval._loader import load_candidates

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATA = REPO_ROOT / "eval_data" / "translation"
EVAL_RUNS = REPO_ROOT / "eval_runs"

SCORE_COLUMNS = ["meaning_preserved", "register", "naturalness", "notes"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pair(
    pair: str,
    candidates: list[tuple[str, dict, object]],
    out_dir: Path,
) -> dict[str, list[dict]]:
    """Translate every utterance with every candidate; return {candidate_id: [rows]}."""
    items = load_jsonl(EVAL_DATA / f"{pair}.jsonl")
    print(f"\n[{pair}→hy] {len(items)} utterances × {len(candidates)} candidates", file=sys.stderr)

    by_candidate: dict[str, list[dict]] = {}
    for cid, _cfg, candidate in candidates:
        print(f"  candidate: {cid}", file=sys.stderr)
        rows: list[dict] = []
        for item in items:
            t0 = time.perf_counter()
            try:
                hyp = candidate.translate(item["source"], source_lang=pair)
                error = None
            except Exception as exc:  # capture per-utterance failures, keep going
                hyp = ""
                error = repr(exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "id": item["id"],
                "category": item["category"],
                "source": item["source"],
                "reference_hy": item.get("reference_hy"),
                "translation_hy": hyp,
                "latency_ms": round(elapsed_ms, 1),
                "error": error,
            })
            tag = "✗" if error else "✓"
            print(f"    {tag} {item['id']}  ({elapsed_ms:.0f} ms)", file=sys.stderr)
        write_jsonl(out_dir / cid / f"{pair}.jsonl", rows)
        by_candidate[cid] = rows
    return by_candidate


def write_results_md(
    out_path: Path,
    pair_results: dict[str, dict[str, list[dict]]],
    candidate_ids: list[str],
) -> None:
    """Side-by-side markdown for human scoring."""
    lines: list[str] = ["# Translation eval results\n"]
    for pair, by_candidate in pair_results.items():
        lines.append(f"\n## {pair.upper()} → HY\n")
        # Use the first candidate's row order as canonical.
        canonical = by_candidate[candidate_ids[0]]
        for i, row in enumerate(canonical):
            lines.append(f"\n### `{row['id']}`  ({row['category']})\n")
            lines.append(f"- **Source:** {row['source']}")
            if row.get("reference_hy"):
                lines.append(f"- **Reference:** {row['reference_hy']}")
            lines.append("")
            lines.append("| candidate | translation | latency (ms) |")
            lines.append("|---|---|---:|")
            for cid in candidate_ids:
                r = by_candidate[cid][i]
                cell = r["translation_hy"].replace("\n", " ") if not r["error"] else f"ERROR: {r['error']}"
                lines.append(f"| `{cid}` | {cell} | {r['latency_ms']:.0f} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_scoresheet(out_path: Path, pair_results: dict[str, dict[str, list[dict]]]) -> None:
    """Pre-fill (utt_id, category, candidate) rows with empty score columns."""
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "utt_id", "category", "candidate", *SCORE_COLUMNS])
        for pair, by_candidate in pair_results.items():
            for cid, rows in by_candidate.items():
                for r in rows:
                    w.writerow([pair, r["id"], r["category"], cid, "", "", "", ""])


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase 0 translation eval.")
    ap.add_argument("--config", default="patmir.toml", type=Path)
    ap.add_argument("--pair", choices=["en", "ru", "both"], default="both")
    args = ap.parse_args()

    candidates = load_candidates(args.config.resolve(), stage="translation")
    candidate_ids = [cid for cid, _, _ in candidates]
    print(f"Enabled translation candidates: {', '.join(candidate_ids)}", file=sys.stderr)

    pairs = ["en", "ru"] if args.pair == "both" else [args.pair]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = EVAL_RUNS / ts / "translation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_results: dict[str, dict[str, list[dict]]] = {}
    for pair in pairs:
        pair_results[pair] = run_pair(pair, candidates, out_dir)

    write_results_md(out_dir / "translation_results.md", pair_results, candidate_ids)
    write_scoresheet(out_dir / "scoresheet.csv", pair_results)
    print(f"\nWrote: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
