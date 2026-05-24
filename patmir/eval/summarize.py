"""Summarize a filled-in scoresheet.

Auto-detects translation vs TTS scoresheets by column names, prints
per-candidate means per metric plus a per-category breakdown.

Usage:
    uv run python -m patmir.eval.summarize <path/to/scoresheet.csv>
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

TRANSLATION_METRICS = ["meaning_preserved", "register", "naturalness"]
TTS_METRICS = ["intelligibility", "naturalness", "prosody"]


def summarize(path: Path) -> None:
    # Auto-detect delimiter — spreadsheet apps in some locales save CSVs with `;`.
    sample = path.read_text(encoding="utf-8-sig").splitlines()
    if not sample:
        print(f"No rows in {path}", file=sys.stderr)
        return
    delimiter = ";" if sample[0].count(";") > sample[0].count(",") else ","
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=delimiter))
    if not rows:
        print(f"No rows in {path}", file=sys.stderr)
        return

    headers = list(rows[0].keys())
    is_translation = "meaning_preserved" in headers
    metrics = TRANSLATION_METRICS if is_translation else TTS_METRICS
    stage = "translation" if is_translation else "tts"

    # candidate -> metric -> [scores]
    per_cand: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # (candidate, category) -> metric -> [scores]
    per_cat: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # candidate -> child_watch_count (TTS only)
    child_yes: dict[str, int] = defaultdict(int)
    child_total: dict[str, int] = defaultdict(int)

    for r in rows:
        cid = r["candidate"]
        cat = r["category"]
        for m in metrics:
            v = r.get(m, "").strip()
            if not v:
                continue
            try:
                per_cand[cid][m].append(float(v))
                per_cat[(cid, cat)][m].append(float(v))
            except ValueError:
                pass  # skip malformed
        if not is_translation:
            child = r.get("child_would_watch_hours", "").strip().lower()
            if child:
                child_total[cid] += 1
                if child in ("y", "yes", "true", "1"):
                    child_yes[cid] += 1

    print(f"Stage: {stage}")
    print(f"Source file: {path}")
    print(f"Rows: {len(rows)}")
    print()

    print(f"{'candidate':<28}" + "".join(f"{m:>16}" for m in metrics) +
          ("    child_y/n" if not is_translation else ""))
    print("-" * (28 + 16 * len(metrics) + (14 if not is_translation else 0)))
    for cid in sorted(per_cand):
        cells = "".join(
            f"{mean(per_cand[cid][m]):>16.2f}" if per_cand[cid][m] else f"{'-':>16}"
            for m in metrics
        )
        suffix = ""
        if not is_translation and child_total[cid]:
            suffix = f"    {child_yes[cid]}/{child_total[cid]}"
        print(f"{cid:<28}{cells}{suffix}")

    print("\nPer-category breakdown:\n")
    categories = sorted({cat for _, cat in per_cat})
    for cat in categories:
        print(f"  category: {cat}")
        for cid in sorted(per_cand):
            scores = per_cat.get((cid, cat))
            if not scores:
                continue
            cells = "  ".join(
                f"{m}={mean(scores[m]):.2f}" for m in metrics if scores.get(m)
            )
            print(f"    {cid:<28} {cells}")
        print()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m patmir.eval.summarize <scoresheet.csv>", file=sys.stderr)
        return 2
    summarize(Path(sys.argv[1]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
