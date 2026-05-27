"""Audit voice cloning decisions for a completed run.

Given a `runs/<job>/` directory, this script answers two questions:

1. **Per speaker** — did the ref picker keep a usable ref? If a WAV was sliced
   to disk but never made it into `refs.json`, that's the bug we hit on
   2026-05-27 (`backfill_ref_text` deleted refs with empty backfilled text
   even though ElevenLabs IVC doesn't need the text).
2. **Per sentence** — which ref would the TTS adapter actually use, walking
   the same fallback chain (`exact → speaker-neutral → any speaker ref →
   global default`)?

The output is printable + machine-readable. Use it as the "before" and "after"
snapshots in the spec verification step.

Usage:
    python -m patmir.eval.voice_cloning_audit runs/<job-dir>
    python -m patmir.eval.voice_cloning_audit runs/<job-dir> --markdown audit.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from patmir.pipeline.stages.diarize.ref_picker import EDGE_TRIM_SEC


@dataclass
class RefRow:
    speaker: str
    emotion: str
    on_disk: bool
    in_refs_json: bool
    source_start: float | None
    source_end: float | None
    backfilled_ref_text: str
    backfilled_word_count: int


@dataclass
class SpeakerRow:
    speaker: str
    turn_count: int
    total_clean_seconds: float
    longest_eligible_seconds: float
    refs: list[RefRow]
    cloning_status: str  # "cloned", "would_clone_after_fix", "no_ref_possible"


@dataclass
class SentenceRow:
    idx: int
    speaker: str | None
    emotion: str | None
    fallback_step: str  # "exact", "speaker_neutral", "any_speaker", "default_voice"
    matched_ref_key: str | None


def _intersects_any(start: float, end: float, overlaps: list[dict]) -> bool:
    for ov in overlaps:
        if ov["start"] < end and ov["end"] > start:
            return True
    return False


def _backfill_text(words: list[dict], start: float, end: float) -> tuple[str, int]:
    matched = [w for w in words if start <= 0.5 * (w["start"] + w["end"]) < end]
    return " ".join(w["text"] for w in matched).strip(), len(matched)


def _parse_refs_key(key: str) -> tuple[str, str] | None:
    if "__" not in key:
        return None
    spk, emo = key.split("__", 1)
    return spk, emo


def audit_run(run_dir: Path, min_ref_seconds: float = 3.0) -> tuple[list[SpeakerRow], list[SentenceRow], dict]:
    """Return per-speaker + per-sentence audit, plus aggregate metrics.

    `min_ref_seconds` is the threshold the pipeline used; pass the value
    actually used for that run (default mirrors `JobOptions.min_ref_seconds`).
    """
    turns_path = run_dir / "diarize" / "turns.json"
    refs_path = run_dir / "diarize" / "refs.json"
    transcript_path = run_dir / "transcript.json"
    translated_path = run_dir / "translated.json"
    refs_dir = run_dir / "diarize" / "refs"

    turns_doc = json.loads(turns_path.read_text())
    turns = turns_doc["turns"]
    overlaps = turns_doc.get("overlaps", [])

    refs_doc = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    refs_in_json = {_parse_refs_key(k): v for k, v in refs_doc.items() if _parse_refs_key(k)}

    transcript = json.loads(transcript_path.read_text())
    words = transcript.get("words", [])

    translated = json.loads(translated_path.read_text()) if translated_path.exists() else {"sentences": []}

    # ─── Per speaker ──────────────────────────────────────────────────────
    speakers: dict[str, list[dict]] = {}
    for t in turns:
        speakers.setdefault(t["speaker_id"], []).append(t)

    speaker_rows: list[SpeakerRow] = []
    for spk in sorted(speakers):
        spk_turns = speakers[spk]
        clean_total = 0.0
        longest_eligible = 0.0
        for t in spk_turns:
            s = t["start"] + EDGE_TRIM_SEC
            e = t["end"] - EDGE_TRIM_SEC
            if e - s < min_ref_seconds:
                continue
            if _intersects_any(s, e, overlaps):
                continue
            clean_total += (e - s)
            longest_eligible = max(longest_eligible, e - s)

        # Discover all emotion bins observed for this speaker.
        spk_emotions = sorted({t.get("emotion") or "neutral" for t in spk_turns})
        ref_rows: list[RefRow] = []
        for emo in spk_emotions:
            wav_path = refs_dir / spk / f"{emo}.wav"
            on_disk = wav_path.exists()
            entry = refs_in_json.get((spk, emo))
            in_json = entry is not None
            src_start = entry["source_start"] if in_json else None
            src_end = entry["source_end"] if in_json else None
            if in_json:
                text, count = _backfill_text(words, src_start, src_end)
            elif on_disk:
                # We can't recover source_start/end from disk, so the best
                # estimate is to find the longest eligible turn matching this
                # (speaker, emotion) — that's what pick_refs would have used.
                candidate = None
                for t in spk_turns:
                    if (t.get("emotion") or "neutral") != emo:
                        continue
                    s = t["start"] + EDGE_TRIM_SEC
                    e = t["end"] - EDGE_TRIM_SEC
                    if e - s < min_ref_seconds:
                        continue
                    if _intersects_any(s, e, overlaps):
                        continue
                    if candidate is None or (e - s) > (candidate[1] - candidate[0]):
                        candidate = (s, e)
                if candidate is not None:
                    src_start, src_end = candidate
                    text, count = _backfill_text(words, src_start, src_end)
                else:
                    text, count = "(unrecoverable)", 0
            else:
                text, count = "", 0
            ref_rows.append(RefRow(
                speaker=spk,
                emotion=emo,
                on_disk=on_disk,
                in_refs_json=in_json,
                source_start=src_start,
                source_end=src_end,
                backfilled_ref_text=text,
                backfilled_word_count=count,
            ))

        if any(r.in_refs_json for r in ref_rows):
            status = "cloned"
        elif any(r.on_disk for r in ref_rows):
            status = "would_clone_after_fix"
        elif longest_eligible > 0:
            status = "ref_lost_pre_disk"
        else:
            status = "no_ref_possible"

        speaker_rows.append(SpeakerRow(
            speaker=spk,
            turn_count=len(spk_turns),
            total_clean_seconds=clean_total,
            longest_eligible_seconds=longest_eligible,
            refs=ref_rows,
            cloning_status=status,
        ))

    # ─── Per sentence (simulate adapter._lookup_ref) ──────────────────────
    sentence_rows: list[SentenceRow] = []

    def lookup(spk: str | None, emo: str | None) -> tuple[str, str | None]:
        if not spk:
            return "default_voice", None
        emo = emo or "neutral"
        if (spk, emo) in refs_in_json:
            return "exact", f"{spk}__{emo}"
        if emo != "neutral" and (spk, "neutral") in refs_in_json:
            return "speaker_neutral", f"{spk}__neutral"
        same_speaker = [k for k in refs_in_json if k[0] == spk]
        if same_speaker:
            best = max(
                same_speaker,
                key=lambda k: refs_in_json[k]["source_end"] - refs_in_json[k]["source_start"],
            )
            return "any_speaker", f"{best[0]}__{best[1]}"
        return "default_voice", None

    for s in translated.get("sentences", []):
        step, key = lookup(s.get("speaker_id"), s.get("emotion"))
        sentence_rows.append(SentenceRow(
            idx=s["idx"],
            speaker=s.get("speaker_id"),
            emotion=s.get("emotion"),
            fallback_step=step,
            matched_ref_key=key,
        ))

    fallback_counts: dict[str, int] = {}
    for sr in sentence_rows:
        fallback_counts[sr.fallback_step] = fallback_counts.get(sr.fallback_step, 0) + 1

    metrics = {
        "speakers_detected": len(speaker_rows),
        "speakers_cloned": sum(1 for s in speaker_rows if s.cloning_status == "cloned"),
        "speakers_recoverable_after_fix": sum(
            1 for s in speaker_rows if s.cloning_status == "would_clone_after_fix"
        ),
        "total_sentences": len(sentence_rows),
        "sentences_by_fallback_step": fallback_counts,
        "min_ref_seconds_assumed": min_ref_seconds,
    }
    return speaker_rows, sentence_rows, metrics


def format_markdown(
    run_dir: Path,
    speakers: list[SpeakerRow],
    sentences: list[SentenceRow],
    metrics: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"# Voice-cloning audit — `{run_dir.name}`\n")
    lines.append("## Headline metrics\n")
    lines.append(f"- Speakers detected: **{metrics['speakers_detected']}**")
    lines.append(f"- Speakers actually cloned (entry in `refs.json`): **{metrics['speakers_cloned']}**")
    lines.append(
        f"- Speakers with a ref WAV on disk but no entry in `refs.json` "
        f"(would be recovered by the fix): **{metrics['speakers_recoverable_after_fix']}**"
    )
    lines.append(f"- Total sentences in `translated.json`: **{metrics['total_sentences']}**")
    lines.append(f"- Sentences by adapter fallback step:")
    for step, count in sorted(metrics["sentences_by_fallback_step"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    - `{step}`: {count}")
    lines.append(f"- `min_ref_seconds` assumed: {metrics['min_ref_seconds_assumed']}\n")

    lines.append("## Per speaker\n")
    lines.append("| speaker | turns | clean s | longest eligible s | refs on disk | refs in JSON | status |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in speakers:
        on_disk = ", ".join(r.emotion for r in s.refs if r.on_disk) or "—"
        in_json = ", ".join(r.emotion for r in s.refs if r.in_refs_json) or "—"
        lines.append(
            f"| {s.speaker} | {s.turn_count} | {s.total_clean_seconds:.2f} | "
            f"{s.longest_eligible_seconds:.2f} | {on_disk} | {in_json} | `{s.cloning_status}` |"
        )

    lines.append("\n## Per-(speaker, emotion) ref detail\n")
    lines.append("| speaker | emotion | wav | in refs.json | source window | backfill words | backfill text |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in speakers:
        for r in s.refs:
            window = (
                f"{r.source_start:.2f}–{r.source_end:.2f}"
                if r.source_start is not None
                else "—"
            )
            text = r.backfilled_ref_text if r.backfilled_ref_text else "(empty)"
            text = text.replace("|", "\\|")
            if len(text) > 60:
                text = text[:60] + "…"
            lines.append(
                f"| {r.speaker} | {r.emotion} | {'✓' if r.on_disk else '—'} | "
                f"{'✓' if r.in_refs_json else '—'} | {window} | {r.backfilled_word_count} | {text} |"
            )

    lines.append("\n## Per-sentence fallback path\n")
    lines.append("| idx | speaker | emotion | step | matched ref |")
    lines.append("| --- | --- | --- | --- | --- |")
    for s in sentences:
        lines.append(
            f"| {s.idx} | {s.speaker or '—'} | {s.emotion or '—'} | "
            f"`{s.fallback_step}` | {s.matched_ref_key or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit voice-cloning decisions for a run dir.")
    ap.add_argument("run_dir", type=Path, help="Path to runs/<job>/")
    ap.add_argument("--markdown", type=Path, default=None, help="Write a markdown report to this path.")
    ap.add_argument(
        "--min-ref-seconds",
        type=float,
        default=3.0,
        help="Threshold used by pick_refs at the time of the run (default 3.0).",
    )
    args = ap.parse_args(argv)

    if not args.run_dir.exists():
        print(f"run dir not found: {args.run_dir}", file=sys.stderr)
        return 2

    speakers, sentences, metrics = audit_run(args.run_dir, min_ref_seconds=args.min_ref_seconds)
    report = format_markdown(args.run_dir, speakers, sentences, metrics)
    print(report)
    if args.markdown:
        args.markdown.write_text(report)
        print(f"\n(wrote {args.markdown})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
