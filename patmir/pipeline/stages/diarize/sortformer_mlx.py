"""Speaker diarization via NVIDIA Sortformer 4-speaker streaming on MLX.

Runs through `mlx-audio` (`mlx_audio.vad.load`). `model.generate_stream(path,
chunk_duration=...)` yields chunk results, each with `.segments`
(`DiarizationSegment(.start, .end, .speaker:int)`). End-to-end neural
diarization (no clustering hyperparameters), which resolves short cartoon turns
far better than pyannote — the failure mode that pushed us to cloud Scribe.

Sortformer emits fragmented same-speaker segments, so we merge adjacent
same-speaker pieces (gap ≤ `merge_gap_seconds`) into turns. Because it's
end-to-end it also emits concurrent speakers, so we re-derive overlap intervals
(≥2 distinct speakers active) — restoring the overlap output Scribe dropped.

Output contract mirrors `pyannote.py`: a `Diarization` with `turns` +
`overlaps`, `refs={}` (the runner's `pick_refs` fills refs after the emotion
stage), normalized `SPEAKER_NN` ids, and `diarize/turns.json` on disk.

Capped at 4 speakers (model architecture). VAD post-filter drops turns with
zero intersection with the Silero VAD intervals from the separate stage.

Install once: `uv pip install "git+https://github.com/Blaizzy/mlx-audio.git"`.
"""
from __future__ import annotations

import json
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    Diarization,
    OverlapInterval,
    SeparatedAudio,
    SpeakerTurn,
)


def _passes_vad(start: float, end: float, vad_intervals: list[tuple[float, float]] | None) -> bool:
    if not vad_intervals:
        return True
    for vs, ve in vad_intervals:
        if vs < end and ve > start:
            return True
    return False


def _norm_speaker(spk: Any) -> str:
    """Normalize Sortformer's speaker label (int 0..3 or 'speaker_0') to the
    `SPEAKER_NN` convention `apply_speaker_tags` and the ref-picker expect."""
    s = str(spk)
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        return f"SPEAKER_{int(digits):02d}"
    return s


def _merge_same_speaker(turns: list[SpeakerTurn], gap: float) -> list[SpeakerTurn]:
    """Merge consecutive same-speaker turns separated by ≤ `gap` seconds."""
    merged: list[SpeakerTurn] = []
    for t in sorted(turns, key=lambda x: (x.start, x.end)):
        if merged and merged[-1].speaker_id == t.speaker_id and t.start - merged[-1].end <= gap:
            merged[-1] = SpeakerTurn(
                start=merged[-1].start,
                end=max(merged[-1].end, t.end),
                speaker_id=t.speaker_id,
            )
        else:
            merged.append(SpeakerTurn(start=t.start, end=t.end, speaker_id=t.speaker_id))
    return merged


def _derive_overlaps(turns: list[SpeakerTurn]) -> list[OverlapInterval]:
    """Time regions where ≥2 distinct speakers are simultaneously active.
    Within a speaker the merged turns are non-overlapping, so an active count
    ≥2 necessarily means ≥2 distinct speakers."""
    events: list[tuple[float, int]] = []
    for t in turns:
        events.append((t.start, 1))
        events.append((t.end, -1))
    events.sort()
    overlaps: list[OverlapInterval] = []
    active = 0
    ov_start: float | None = None
    for time, delta in events:
        prev = active
        active += delta
        if prev < 2 <= active:
            ov_start = time
        elif prev >= 2 > active and ov_start is not None:
            if time > ov_start:
                overlaps.append(OverlapInterval(start=ov_start, end=time))
            ov_start = None
    return overlaps


class SortformerMLX:
    id = "sortformer_mlx"

    def __init__(
        self,
        repo_id: str,
        chunk_duration: float = 5.0,
        merge_gap_seconds: float = 0.5,
    ) -> None:
        from mlx_audio.vad import load  # lazy import (heavy)
        self._model = load(repo_id)
        self.repo_id = repo_id
        self.chunk_duration = chunk_duration
        self.merge_gap_seconds = merge_gap_seconds

    def diarize(self, separated: SeparatedAudio, ctx: JobContext) -> Diarization:
        ctx.progress("diarize", 0.05, f"sortformer-mlx ({self.repo_id})")

        raw: list[SpeakerTurn] = []
        for res in self._model.generate_stream(str(separated.vocals), chunk_duration=self.chunk_duration):
            for seg in getattr(res, "segments", []) or []:
                raw.append(SpeakerTurn(
                    start=float(getattr(seg, "start", 0.0)),
                    end=float(getattr(seg, "end", 0.0)),
                    speaker_id=_norm_speaker(getattr(seg, "speaker", getattr(seg, "speaker_id", "0"))),
                ))
        ctx.progress("diarize", 0.8, f"{len(raw)} raw segments")

        merged = _merge_same_speaker(raw, self.merge_gap_seconds)
        kept = [t for t in merged if _passes_vad(t.start, t.end, separated.vad_intervals)]
        overlaps = _derive_overlaps(kept)
        speakers = sorted({t.speaker_id for t in kept})

        diarize_dir = ctx.job_dir / "diarize"
        diarize_dir.mkdir(exist_ok=True)
        (diarize_dir / "turns.json").write_text(json.dumps({
            "model": self.repo_id,
            "device": "mps",
            "speakers": speakers,
            "turns": [{"start": t.start, "end": t.end, "speaker_id": t.speaker_id} for t in kept],
            "overlaps": [{"start": ov.start, "end": ov.end} for ov in overlaps],
        }, indent=2))

        ctx.progress("diarize", 1.0, f"{len(speakers)} speakers, {len(kept)} turns, {len(overlaps)} overlaps")

        # refs intentionally empty — runner's pick_refs fills it after the
        # emotion stage (per-(speaker, emotion) clips need the emotion bin).
        return Diarization(
            turns=kept,
            overlaps=overlaps,
            refs={},
            sample_rate=separated.sample_rate,
        )


def build(config: dict[str, Any]) -> SortformerMLX:
    return SortformerMLX(
        repo_id=config["repo_id"],
        chunk_duration=float(config.get("chunk_duration", 5.0)),
        merge_gap_seconds=float(config.get("merge_gap_seconds", 0.5)),
    )
