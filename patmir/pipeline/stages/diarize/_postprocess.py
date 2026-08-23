"""Shared post-processing for the Sortformer diarize candidates.

Sortformer reaches this pipeline through two runtimes — `mlx-audio` on Apple
Silicon (`sortformer_mlx.py`) and NVIDIA's own NeMo on CUDA
(`sortformer_nemo.py`). Both get raw per-frame speaker segments out of the same
model and then need identical clean-up before the result can satisfy the
`Diarization` contract:

  * normalize the speaker label to the `SPEAKER_NN` convention that
    `apply_speaker_tags` and the ref-picker expect;
  * merge the fragmented same-speaker pieces the model emits into turns;
  * re-derive overlap intervals, which the model supports (it is end-to-end and
    emits concurrent speakers) but does not report directly;
  * drop turns with no intersection with the separate stage's Silero VAD windows.

These live here rather than in either candidate so the two runtimes cannot drift
apart in post-processing and be mistaken for a model difference — the whole
point of having both is that they are the same diarizer.
"""
from __future__ import annotations

from typing import Any

from patmir.pipeline.stage import OverlapInterval, SpeakerTurn


def passes_vad(
    start: float,
    end: float,
    vad_intervals: list[tuple[float, float]] | None,
) -> bool:
    """True if `[start, end)` intersects any VAD speech window.

    With no VAD intervals available every turn passes — the filter is an
    improvement on the diarizer's output, not a precondition for it.
    """
    if not vad_intervals:
        return True
    for vs, ve in vad_intervals:
        if vs < end and ve > start:
            return True
    return False


def norm_speaker(spk: Any) -> str:
    """Normalize a Sortformer speaker label to `SPEAKER_NN`.

    The two runtimes label speakers differently — mlx-audio yields ints (0..3)
    or `speaker_0`, NeMo yields an integer speaker index (sometimes rendered as
    a string) — and downstream code keys refs and transcript tags off this
    string, so it has to be one convention.
    """
    s = str(spk)
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        return f"SPEAKER_{int(digits):02d}"
    return s


def merge_same_speaker(turns: list[SpeakerTurn], gap: float) -> list[SpeakerTurn]:
    """Merge consecutive same-speaker turns separated by <= `gap` seconds.

    Sortformer decides per frame, so one continuous utterance comes back as many
    short segments. Left unmerged they would each be a candidate ref clip and
    none would clear `min_ref_seconds`.
    """
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


def derive_overlaps(turns: list[SpeakerTurn]) -> list[OverlapInterval]:
    """Time regions where >=2 distinct speakers are simultaneously active.

    Within one speaker the merged turns are non-overlapping, so an active count
    of >=2 necessarily means >=2 distinct speakers.
    """
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
