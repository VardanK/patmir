"""Side-by-side translator comparison: per-sentence vs batched.

Reads an existing run's `transcript.json` (the ASR output, already
sentence-segmented + speaker/emotion-tagged) and runs both
`openrouter_gemini_flash` (per-sentence baseline) and
`openrouter_gemini_flash_batch` (chunked + structured output) over the
same input. Writes two artifacts to the run dir:

  - `translate_ab.json`  — machine-readable {idx, source, emotion, speaker,
                            current, batch}
  - `translate_ab.tsv`   — tab-separated, one row per sentence, easy to eyeball

The batched call should produce more coherent character voice + better
discourse flow; this script gives you the diff to decide whether to flip
`[pipeline].translate` to the batched variant.

Usage:

  python scripts/eval_translate_batch.py runs/<job_id>

Requires `OPENROUTER_API_KEY` in env (via `.env` or shell). Reads the
candidate configs straight from `patmir.toml` so any reasoning / model /
chunk-size overrides applied there are honored.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from patmir.env import load_dotenv


def _load_candidate_entry(toml_path: Path, candidate_id: str) -> dict[str, Any]:
    with toml_path.open("rb") as f:
        toml = tomllib.load(f)
    for entry in toml.get("eval", {}).get("translation", {}).get("candidates", []):
        if entry.get("id") == candidate_id:
            return entry
    raise SystemExit(f"candidate {candidate_id!r} not in {toml_path}")


def _build(entry: dict[str, Any]):
    import importlib
    module = importlib.import_module(entry["module"])
    return module.build(entry)


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path,
                   help="A run directory under runs/ (must contain transcript.json).")
    p.add_argument("--config", type=Path, default=Path("patmir.toml"))
    p.add_argument("--current", default="openrouter_gemini_flash",
                   help="Candidate id for the per-sentence baseline.")
    p.add_argument("--batch", default="openrouter_gemini_flash_batch",
                   help="Candidate id for the batched variant.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only translate the first N sentences (smoke test).")
    args = p.parse_args()

    transcript_path = args.run_dir / "transcript.json"
    if not transcript_path.is_file():
        print(f"no transcript.json at {transcript_path}", file=sys.stderr)
        return 2
    transcript = json.loads(transcript_path.read_text())
    source_lang = transcript.get("language") or "ru"
    sentences = transcript.get("sentences") or []
    if args.limit > 0:
        sentences = sentences[: args.limit]
    if not sentences:
        print("no sentences in transcript", file=sys.stderr)
        return 2

    current_entry = _load_candidate_entry(args.config, args.current)
    batch_entry = _load_candidate_entry(args.config, args.batch)

    print(f"loading current candidate: {args.current} ({current_entry['model']})",
          file=sys.stderr)
    current = _build(current_entry)
    print(f"loading batch candidate:   {args.batch} ({batch_entry['model']})",
          file=sys.stderr)
    batch = _build(batch_entry)

    # Run per-sentence.
    print(f"running per-sentence translation on {len(sentences)} sentences…",
          file=sys.stderr)
    current_out: dict[int, str] = {}
    for i, s in enumerate(sentences):
        idx = int(s["idx"])
        try:
            current_out[idx] = current.translate(
                s["text"], source_lang=source_lang, emotion=s.get("emotion"),
            )
        except Exception as exc:  # noqa: BLE001
            current_out[idx] = f"<ERROR: {type(exc).__name__}: {exc}>"
        print(f"  [{i + 1}/{len(sentences)}] per-sentence done", file=sys.stderr)

    # Run batched.
    print(f"running batched translation on {len(sentences)} sentences…",
          file=sys.stderr)
    def _progress(done: int, total: int, msg: str) -> None:
        print(f"  [{done}/{total}] batch: {msg}", file=sys.stderr)
    batch_out = batch.translate_batch(sentences, source_lang=source_lang, progress=_progress)

    # Emit side-by-side artifacts.
    rows = []
    for s in sentences:
        idx = int(s["idx"])
        rows.append({
            "idx": idx,
            "speaker_id": s.get("speaker_id") or "",
            "emotion": s.get("emotion") or "neutral",
            "source": s["text"],
            "current": current_out.get(idx, ""),
            "batch": batch_out.get(idx, ""),
        })

    out_json = args.run_dir / "translate_ab.json"
    out_tsv = args.run_dir / "translate_ab.tsv"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    with out_tsv.open("w") as f:
        f.write("idx\tspeaker\temotion\tsource\tcurrent\tbatch\n")
        for r in rows:
            cells = [
                str(r["idx"]),
                r["speaker_id"],
                r["emotion"],
                r["source"].replace("\t", " ").replace("\n", " "),
                r["current"].replace("\t", " ").replace("\n", " "),
                r["batch"].replace("\t", " ").replace("\n", " "),
            ]
            f.write("\t".join(cells) + "\n")

    n_same = sum(1 for r in rows if r["current"].strip() == r["batch"].strip())
    print(file=sys.stderr)
    print(f"wrote {out_json}", file=sys.stderr)
    print(f"wrote {out_tsv}", file=sys.stderr)
    print(f"  {n_same}/{len(rows)} sentences identical between the two translators",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
