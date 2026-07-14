"""Per-`(speaker, emotion)` reference clip picker.

Phase 2 originally picked one ref clip per speaker. Phase 2.x picks one per
`(speaker, emotion_bin)` so OmniVoice can swap to a matching emotional ref
clip per sentence. This module owns:

1. Binning each diarization turn by dominant emotion (uses the per-turn
   `EmotionPrediction` list emitted by the `emotion` stage).
2. Picking, per `(speaker, emotion)` cell, the longest turn that:
     (a) is the dominant emotion for that cell (or has any overlap with that
         emotion when there's only one matching turn — see `_dominant_bin`),
     (b) does not intersect any overlap region,
     (c) is at least `min_ref_seconds` long after trimming 100 ms inward each side.
3. Slicing the chosen window out of `speech_vocals` (cleaner than `vocals`),
   resampling to 16 kHz mono, and writing it as
   `<refs_dir>/<speaker>/<emotion>.wav`.

Fallback chain at TTS lookup time (handled by `tts/adapter.py`, not here):
`refs[(spk, emo)] → refs[(spk, "neutral")] → global default ref`.

`ref_text` is filled in later by `runner._backfill_ref_text` — ASR words
aren't available at the time this runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from patmir.pipeline.stage import (
    Emotion,
    EmotionPrediction,
    OverlapInterval,
    SpeakerRef,
    SpeakerTurn,
)

REF_SR = 16_000
EDGE_TRIM_SEC = 0.100


def _rms_db(mono: np.ndarray) -> float:
    """Mean level of a slice in dBFS. Silence → -120 (well below any floor)."""
    if len(mono) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(mono, dtype="float64"))))
    if rms <= 1e-6:
        return -120.0
    return 20.0 * float(np.log10(rms))


def _intersects_any(start: float, end: float, overlaps: list[OverlapInterval]) -> bool:
    for ov in overlaps:
        if ov.start < end and ov.end > start:
            return True
    return False


def _resample_linear(mono: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return mono.astype("float32", copy=False)
    n_target = int(len(mono) * dst_sr / src_sr)
    if n_target <= 0:
        return np.zeros(0, dtype="float32")
    x_old = np.linspace(0, 1, len(mono), endpoint=False)
    x_new = np.linspace(0, 1, n_target, endpoint=False)
    return np.interp(x_new, x_old, mono).astype("float32")


def _dominant_bin(turn: SpeakerTurn, predictions: list[EmotionPrediction]) -> Emotion:
    """Find the dominant emotion bin overlapping this turn.

    Emotion predictions are produced per-turn (they share endpoints with the
    diarizer's turns), so the simple case is a direct lookup. The general
    code path measures temporal overlap so it stays correct if the emotion
    stage is ever changed to produce windowed (non-turn-aligned) predictions.
    """
    weight: dict[Emotion, float] = {}
    for p in predictions:
        lo = max(turn.start, p.start)
        hi = min(turn.end, p.end)
        if hi > lo:
            weight[p.emotion] = weight.get(p.emotion, 0.0) + (hi - lo)
    if not weight:
        return "neutral"
    return max(weight, key=lambda e: weight[e])


def tag_turns_with_emotion(
    turns: list[SpeakerTurn],
    predictions: list[EmotionPrediction],
) -> list[SpeakerTurn]:
    """Return a new turn list with `emotion` populated from predictions.

    Mutates in-place AND returns the list for caller convenience.
    """
    for t in turns:
        t.emotion = _dominant_bin(t, predictions)
    return turns


def pick_refs(
    *,
    turns: list[SpeakerTurn],
    overlaps: list[OverlapInterval],
    predictions: list[EmotionPrediction],
    speech_vocals: Path,
    refs_dir: Path,
    min_ref_seconds: float,
    min_rms_db: float = -33.0,
) -> dict[tuple[str, Emotion], SpeakerRef]:
    """Pick + slice one reference clip per `(speaker, emotion)` cell.

    Cells with no qualifying turn are omitted — the TTS adapter's lookup
    fallback chain handles the missing case (exact → speaker-neutral →
    global default).

    `min_rms_db` is a loudness quality gate: windows quieter than the floor
    (usually far-field / bleed-dominated audio that clones badly) lose to any
    eligible window above it. If every window of a cell is below the floor the
    longest still wins — a quiet ref beats no ref.
    """
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Bin every turn by its dominant emotion (mutates turn.emotion).
    tag_turns_with_emotion(turns, predictions)

    audio, src_sr = sf.read(str(speech_vocals), dtype="float32", always_2d=True)
    mono_full = audio.mean(axis=1)

    # Group turns by (speaker, emotion).
    cells: dict[tuple[str, Emotion], list[SpeakerTurn]] = {}
    for t in turns:
        if t.emotion is None:
            continue
        cells.setdefault((t.speaker_id, t.emotion), []).append(t)

    out: dict[tuple[str, Emotion], SpeakerRef] = {}
    for (spk, emo), spk_turns in cells.items():
        eligible: list[tuple[float, float]] = []
        for t in spk_turns:
            s = t.start + EDGE_TRIM_SEC
            e = t.end - EDGE_TRIM_SEC
            if e - s < min_ref_seconds:
                continue
            if _intersects_any(s, e, overlaps):
                continue
            eligible.append((s, e))
        if not eligible:
            continue

        loud = [
            (s, e) for s, e in eligible
            if _rms_db(mono_full[max(0, int(s * src_sr)):min(len(mono_full), int(e * src_sr))]) >= min_rms_db
        ]
        s, e = max(loud or eligible, key=lambda se: se[1] - se[0])
        i0 = max(0, int(s * src_sr))
        i1 = min(len(mono_full), int(e * src_sr))
        if i1 <= i0:
            continue
        slice_mono = mono_full[i0:i1]
        slice_16k = _resample_linear(slice_mono, src_sr, REF_SR)

        spk_dir = refs_dir / spk
        spk_dir.mkdir(exist_ok=True)
        ref_path = spk_dir / f"{emo}.wav"
        sf.write(str(ref_path), slice_16k, REF_SR, subtype="PCM_16")
        out[(spk, emo)] = SpeakerRef(
            speaker_id=spk,
            ref_audio=ref_path,
            ref_text="",  # backfilled after ASR
            source_start=s,
            source_end=e,
        )
    return out
